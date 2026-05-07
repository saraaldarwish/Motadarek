"""
Motadarek — Full Pipeline Backend (v7)

Label fix: fall detector was saved with swapped class names.
  idx_to_class.json says: {0: "Fall",    1: "No-Fall"}
  Actual meaning:         {0: "No-Fall", 1: "Fall"}

Fix applied at load time — swap the names so everything downstream
(badge display, is_fall check, pipeline routing) is consistent.

Verified against two test files:
  label_1 file (real fall)    → model index 1 → "Fall"    → pipeline runs ✓
  label_0 file (real no-fall) → model index 0 → "No-Fall" → pipeline stops ✓
"""

import os, io, json, torch, librosa, tempfile, httpx, joblib, numpy as np
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from transformers import ASTForAudioClassification, ASTFeatureExtractor

app = FastAPI(title="Motadarek API")
app.mount("/static", StaticFiles(directory="static"), name="static")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

TELEGRAM_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID",   "")

# ── Radar preprocessing ───────────────────────────────────────────────────────
FEATURE_COLS = [
    "energy_mean", "energy_std", "energy_min", "energy_max", "energy_slope",
    "stillness_ratio", "time_to_stillness",
    "doppler_centroid_mean", "doppler_centroid_std",
    "doppler_spread_mean", "doppler_spread_std",
    "peak_range_mode", "peak_range_std", "peak_range_changes",
    "range_spread_mean", "range_spread_std",
]

def preprocess_rdn(rdn_window):
    x = np.abs(rdn_window.astype(np.float32))
    x = np.log1p(x)
    doppler_center = x.shape[-1] // 2
    null_half = 5
    x[:, :, doppler_center - null_half: doppler_center + null_half + 1] = 0
    background = np.median(x, axis=0, keepdims=True)
    x = x - background
    return np.maximum(x, 0)

def extract_per_frame_features(x):
    _, R, D = x.shape
    motion_energy = x.sum(axis=(1, 2))
    doppler_profile  = x.sum(axis=1)
    doppler_bins     = np.arange(D, dtype=np.float32)[None, :]
    doppler_sum      = doppler_profile.sum(axis=1, keepdims=True) + 1e-8
    doppler_centroid = (doppler_profile * doppler_bins).sum(axis=1) / doppler_sum[:, 0]
    doppler_var      = (((doppler_bins - doppler_centroid[:, None]) ** 2) * doppler_profile).sum(axis=1) / doppler_sum[:, 0]
    range_profile  = x.sum(axis=2)
    range_bins     = np.arange(R, dtype=np.float32)[None, :]
    range_sum      = range_profile.sum(axis=1, keepdims=True) + 1e-8
    peak_range_bin = np.argmax(range_profile, axis=1).astype(np.float32)
    range_centroid = (range_profile * range_bins).sum(axis=1) / range_sum[:, 0]
    range_var      = (((range_bins - range_centroid[:, None]) ** 2) * range_profile).sum(axis=1) / range_sum[:, 0]
    return {
        "motion_energy":    motion_energy,
        "doppler_centroid": doppler_centroid,
        "doppler_spread":   np.sqrt(doppler_var),
        "peak_range_bin":   peak_range_bin,
        "range_spread":     np.sqrt(range_var),
    }

def compute_aggregate_features(frame_feats, fps):
    me = frame_feats["motion_energy"]; dc = frame_feats["doppler_centroid"]
    ds = frame_feats["doppler_spread"]; pr = frame_feats["peak_range_bin"]
    rs = frame_feats["range_spread"]
    t  = np.arange(len(me), dtype=np.float32) / fps
    energy_slope = float(np.polyfit(t, me, 1)[0]) if len(me) >= 2 else 0.0
    me_min = np.min(me); me_range = np.max(me) - me_min
    if me_range > 1e-8:
        threshold  = me_min + 0.20 * me_range
        still_mask = me <= threshold
    else:
        still_mask = np.zeros_like(me, dtype=bool)
    stillness_ratio   = float(still_mask.mean())
    below             = np.where(still_mask)[0]
    time_to_stillness = float(below[0] / fps) if len(below) > 0 else float(len(me) / fps)
    pr_int = pr.astype(int)
    if len(pr_int) > 0:
        peak_range_mode    = float(np.bincount(pr_int).argmax())
        peak_range_changes = float(np.sum(np.diff(pr_int) != 0)) / max(1, len(pr_int) - 1)
    else:
        peak_range_mode = 0.0; peak_range_changes = 0.0
    return {
        "energy_mean": float(np.mean(me)), "energy_std": float(np.std(me)),
        "energy_min": float(np.min(me)),   "energy_max": float(np.max(me)),
        "energy_slope": energy_slope,       "stillness_ratio": stillness_ratio,
        "time_to_stillness": time_to_stillness,
        "doppler_centroid_mean": float(np.mean(dc)), "doppler_centroid_std": float(np.std(dc)),
        "doppler_spread_mean":   float(np.mean(ds)), "doppler_spread_std":   float(np.std(ds)),
        "peak_range_mode": peak_range_mode, "peak_range_std": float(np.std(pr)),
        "peak_range_changes": peak_range_changes,
        "range_spread_mean": float(np.mean(rs)), "range_spread_std": float(np.std(rs)),
    }

def extract_radar_features(rdn_window, fps):
    x = preprocess_rdn(rdn_window)
    agg = compute_aggregate_features(extract_per_frame_features(x), fps)
    names = list(agg.keys())
    return np.array([agg[n] for n in names], dtype=np.float32), names

def npz_to_feature_vector(file_bytes: bytes, fps: float = 10.0) -> np.ndarray:
    buf = io.BytesIO(file_bytes)
    npz = np.load(buf, allow_pickle=False)
    sample = None
    for key in ["data", "arr_0", "radar", "sample", "rdn", "X"]:
        if key in npz.files:
            sample = npz[key]; break
    if sample is None:
        sample = npz[npz.files[0]]
    if "fps" in npz.files:
        fps = float(npz["fps"])
    fv, _ = extract_radar_features(sample, fps)
    return fv.reshape(1, -1)

# ── Label parser ──────────────────────────────────────────────────────────────
def parse_labels(raw: dict) -> dict:
    if "id2label" in raw:
        return {int(k): v for k, v in raw["id2label"].items()}
    first_key = next(iter(raw))
    try:
        int(first_key)
        return {int(k): v for k, v in raw.items()}
    except ValueError:
        return {int(v): k for k, v in raw.items()}

# ── State dict key remapper ───────────────────────────────────────────────────
def remap_state_dict(sd: dict) -> dict:
    if not next(iter(sd.keys())).startswith("ast."):
        return sd
    return {
        ("audio_spectrogram_transformer." + k[4:] if k.startswith("ast.") else k): v
        for k, v in sd.items()
    }

# ── AST loader ────────────────────────────────────────────────────────────────
def load_ast(model_dir: Path, weight_file: str, preferred_label_file: str = None):
    with open(model_dir / "config.json") as f:
        cfg = json.load(f)
    backbone = (
        cfg.get("model_name") or cfg.get("backbone") or
        cfg.get("backbone_model") or "MIT/ast-finetuned-audioset-10-10-0.4593"
    )
    sr = cfg.get("sampling_rate", 16000)
    search_order = ([preferred_label_file] if preferred_label_file else []) + \
                   ["idx_to_class.json", "label_mapping.json", "label_map.json"]
    label_file = next((model_dir / n for n in search_order if (model_dir / n).exists()), None)
    if label_file is None:
        raise FileNotFoundError(f"No label file in {model_dir}")
    with open(label_file) as f:
        labels = parse_labels(json.load(f))
    num_labels = cfg.get("num_labels") or cfg.get("num_classes") or len(labels)
    fe    = ASTFeatureExtractor.from_pretrained(str(model_dir))
    model = ASTForAudioClassification.from_pretrained(
        backbone, num_labels=num_labels, ignore_mismatched_sizes=True
    )
    model.load_state_dict(remap_state_dict(
        torch.load(model_dir / weight_file, map_location=DEVICE)
    ), strict=False)
    model.to(DEVICE).eval()
    print(f"  {model_dir.name} — labels: {labels}")
    return model, fe, labels, sr

# ── Load all models ───────────────────────────────────────────────────────────
print("Loading Fall Detector...")
fall_model, fall_fe, _fall_labels_raw, fall_sr = load_ast(
    Path("model/fall_detector"), "stageB_SAFE_plus_external.pth",
    preferred_label_file="idx_to_class.json"
)
# ── LABEL FIX ────────────────────────────────────────────────────────────────
# idx_to_class.json says {0:"Fall", 1:"No-Fall"} but the model was trained
# with the names swapped. The actual mapping is:
#   index 0 → No-Fall (what the model learned as class 0)
#   index 1 → Fall    (what the model learned as class 1)
# We correct this at load time so the badge, is_fall flag, and pipeline
# routing are all consistent.
fall_labels = {0: "No-Fall", 1: "Fall"}
print(f"  fall_detector — labels corrected to: {fall_labels}")

print("Loading Position Model...")
pos_model, pos_fe, pos_labels, pos_sr = load_ast(
    Path("model/position"), "model_state_dict.pth"
)
print("Loading Surface Model...")
surf_model, surf_fe, surf_labels, surf_sr = load_ast(
    Path("model/surface"), "model_state_dict.pth"
)
print("Loading Scream Model...")
scream_model, scream_fe, scream_labels, scream_sr = load_ast(
    Path("model/scream"), "model_state_dict.pth"
)
print("Loading Radar Model (RandomForest)...")
radar_clf = joblib.load("model/radar/model.joblib")
with open("model/radar/config.json") as f:
    radar_cfg = json.load(f)
RADAR_THRESHOLD = float(radar_cfg.get("threshold", 0.25))
print(f"  radar threshold: {RADAR_THRESHOLD}")
print("All models loaded.")

# ── Scoring ───────────────────────────────────────────────────────────────────
SCORE_TABLE = {
    "position": {"Standing": 2, "Lying": 1},
    "surface":  {
        "concrete": 3, "carpet over concrete": 3,
        "wood": 1,     "carpet over wood": 1,
    },
    "scream": {"Scream": 3, "Non-Scream": 0},
    "radar":  {"not_recovered": 3, "recovered": 0},
}
THRESHOLD = 5

def compute_score(position, surface, scream, radar_state):
    scores = {
        "position": SCORE_TABLE["position"].get(position, 1),
        "surface":  SCORE_TABLE["surface"].get(surface.lower(), 1),
        "scream":   SCORE_TABLE["scream"].get(scream, 0),
        "radar":    SCORE_TABLE["radar"].get(radar_state, 3),
    }
    total = sum(scores.values())
    return scores, total, "HIGH" if total >= THRESHOLD else "LOW"

# ── Audio helper ──────────────────────────────────────────────────────────────
def audio_from_bytes(b, filename, sr):
    suffix = Path(filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(b); p = tmp.name
    try:
        y, _ = librosa.load(p, sr=sr, mono=True, res_type="soxr_hq")
    finally:
        os.unlink(p)
    return y

@torch.no_grad()
def run_ast(model, fe, labels, sr, y):
    inp   = fe(y, sampling_rate=sr, return_tensors="pt")
    probs = torch.softmax(
        model(inp["input_values"].to(DEVICE)).logits, dim=-1
    )[0].cpu().numpy()
    idx = int(np.argmax(probs))
    return (
        labels[idx], float(probs[idx]),
        {labels[i]: round(float(p)*100, 2) for i, p in enumerate(probs)},
    )

def run_radar(feat_vector):
    prob_high   = float(radar_clf.predict_proba(feat_vector)[0][1])
    raw_label   = "High" if prob_high >= RADAR_THRESHOLD else "Low"
    radar_state = "recovered" if raw_label == "High" else "not_recovered"
    return radar_state, prob_high, raw_label

# ── Telegram ──────────────────────────────────────────────────────────────────
async def send_telegram_alert(score, details):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    msg = (
        "🚨 *MOTADAREK ALERT — HIGH RISK FALL* 🚨\n\n"
        f"⚠️ Total Score: *{score}/11* (threshold ≥{THRESHOLD})\n\n"
        f"📍 Position : {details.get('position','—')} (+{details['scores'].get('position',0)})\n"
        f"🪨 Surface  : {details.get('surface','—')} (+{details['scores'].get('surface',0)})\n"
        f"🔊 Scream   : {details.get('scream','—')} (+{details['scores'].get('scream',0)})\n"
        f"📡 Radar    : {details.get('radar_state','—')} (+{details['scores'].get('radar',0)})\n\n"
        "⏱ Immediate caregiver response required."
    )
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"},
            timeout=10,
        )
    return r.status_code == 200

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root(): return FileResponse("static/index.html")

@app.get("/health")
def health():
    return {"status":"ok","device":DEVICE,
            "threshold":THRESHOLD,"radar_threshold":RADAR_THRESHOLD}

@app.post("/predict")
async def predict(
    fall_audio:   UploadFile = File(...),
    scream_audio: UploadFile = File(None),
    radar_file:   UploadFile = File(None),
    radar_data:   str        = Form(None),
):
    ALLOWED = {".wav",".mp3",".m4a",".flac",".ogg"}
    if Path(fall_audio.filename).suffix.lower() not in ALLOWED:
        raise HTTPException(400, "Unsupported fall audio format.")

    fall_bytes = await fall_audio.read()

    # ── Step 1: Fall Detector ─────────────────────────────────────────────────
    y_fall = audio_from_bytes(fall_bytes, fall_audio.filename, fall_sr)
    fall_label, fall_conf, fall_probs = run_ast(
        fall_model, fall_fe, fall_labels, fall_sr, y_fall
    )
    # With corrected labels: "Fall" genuinely means fall
    is_fall = (fall_label == "Fall")

    result = {
        "fall_detected": is_fall,
        "fall": {"label": fall_label, "confidence": round(fall_conf*100,1),
                 "probs": fall_probs},
        "position": None, "surface": None, "scream": None, "radar": None,
        "scores": None, "total_score": None, "risk": "NO_FALL", "alert_sent": False,
    }
    if not is_fall:
        return result

    # ── Step 2: Position (same fall audio) ───────────────────────────────────
    pos_label, pos_conf, pos_probs = run_ast(
        pos_model, pos_fe, pos_labels, pos_sr, y_fall
    )
    result["position"] = {"label": pos_label, "confidence": round(pos_conf*100,1),
                          "probs": pos_probs}

    # ── Step 3: Surface (same fall audio) ────────────────────────────────────
    surf_label, surf_conf, surf_probs = run_ast(
        surf_model, surf_fe, surf_labels, surf_sr, y_fall
    )
    result["surface"] = {"label": surf_label, "confidence": round(surf_conf*100,1),
                         "probs": surf_probs}

    # ── Step 4: Scream ────────────────────────────────────────────────────────
    scream_label = "Non-Scream"
    if scream_audio and scream_audio.filename:
        sb   = await scream_audio.read()
        y_sc = audio_from_bytes(sb, scream_audio.filename, scream_sr)
        scream_label, sc_conf, sc_probs = run_ast(
            scream_model, scream_fe, scream_labels, scream_sr, y_sc
        )
        result["scream"] = {"label": scream_label,
                            "confidence": round(sc_conf*100,1), "probs": sc_probs}
    else:
        result["scream"] = {"label": "Non-Scream", "note": "no scream audio provided"}

    # ── Step 5: Radar ─────────────────────────────────────────────────────────
    radar_state  = "not_recovered"
    radar_result = {"state": "not_recovered", "note": "default conservative"}

    if radar_file and radar_file.filename:
        try:
            rb = await radar_file.read()
            fv = npz_to_feature_vector(rb)
            radar_state, prob_high, raw_label = run_radar(fv)
            radar_result = {
                "state":       radar_state,
                "raw_label":   raw_label,
                "probability": round(prob_high*100, 1),
                "threshold":   round(RADAR_THRESHOLD*100, 1),
                "note":        "npz — full feature extraction",
            }
        except Exception as e:
            radar_result["note"] = f"npz error: {str(e)}"
    elif radar_data:
        try:
            rd          = json.loads(radar_data)
            radar_state = rd.get("manual", "not_recovered")
            radar_result = {"state": radar_state, "note": "manual toggle"}
        except Exception as e:
            radar_result["note"] = f"json error: {str(e)}"

    result["radar"] = radar_result

    # ── Step 6: Rule-based severity ───────────────────────────────────────────
    scores, total, risk = compute_score(
        pos_label, surf_label, scream_label, radar_state
    )
    result["scores"]      = scores
    result["total_score"] = total
    result["risk"]        = risk

    # ── Step 7: Telegram alert on HIGH RISK ───────────────────────────────────
    if risk == "HIGH":
        result["alert_sent"] = await send_telegram_alert(total, {
            "position": pos_label, "surface": surf_label,
            "scream": scream_label, "radar_state": radar_state, "scores": scores,
        })

    return result
