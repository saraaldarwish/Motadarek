---
title: Motadarek Fall Detection
alert: 🚨
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
---

# Motadarek — Detect · Assess · Alert · Protect

Multi-model fall detection and severity assessment system.

## Models
| Model | Type | Input |
|-------|------|-------|
| Fall Detector | AST Transformer | Fall audio |
| Position Model | AST Transformer | Fall audio |
| Surface Model | AST Transformer | Fall audio |
| Scream Detector | AST Transformer | Scream audio |
| Radar Recovery | scikit-learn (joblib) | Radar features |

## Scoring (threshold ≥ 5 = HIGH RISK)
| Factor | Value |
|--------|-------|
| Position: Standing | +2 |
| Position: Lying | +1 |
| Surface: Concrete/Carpet-Concrete | +3 |
| Surface: Wood/Carpet-Wood | +1 |
| Scream detected | +3 |
| Radar: not recovered | +3 |

## Environment Variables (Secrets)
- `TELEGRAM_BOT_TOKEN` — your Telegram bot token
- `TELEGRAM_CHAT_ID` — caregiver's Telegram chat ID
