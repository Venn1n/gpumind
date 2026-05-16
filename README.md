# GPUMind 🧠⛏️

**AI-Powered GPU Resource Manager** — time-share GPU antara mining dan AI inference, auto-switch berdasarkan profitability.

## Concept

GPU mahal. Mining revenue fluktuatif. AI compute demand tinggi. GPUMind optimize GPU utilization dengan:

1. **Mining Mode** — PoW mining (Keryx, etc.) when profitable
2. **AI Inference Mode** — serve LLM/vision inference when mining is down
3. **Smart Switch** — ML-based profitability predictor auto-switch modes
4. **Hybrid Mode** — split GPU: 70% mining + 30% inference simultaneously

## Architecture

```
┌─────────────────────────────────────────────┐
│                 GPUMind Daemon               │
├─────────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────────┐ │
│  │ Monitor │→ │  Brain  │→ │   Executor  │ │
│  │         │  │ (ML)    │  │             │ │
│  └─────────┘  └─────────┘  └─────────────┘ │
│       ↓            ↓              ↓         │
│  ┌─────────────────────────────────────┐    │
│  │           GPU Manager               │    │
│  │  mining ←→ inference ←→ hybrid      │    │
│  └─────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
```

## Features

- **Multi-GPU support** — manage multiple GPUs independently
- **Profitability prediction** — lightweight ML model predicts mining vs AI revenue
- **Auto-switching** — seamless transition between mining and inference
- **API server** — REST API for inference requests
- **Telegram alerts** — notifications on mode switches
- **Metrics dashboard** — track earnings, inference requests, GPU utilization

## Quick Start

```bash
# Install
cd gpumind
pip install -r requirements.txt

# Configure
cp config.example.yaml config.yaml
# Edit config.yaml with your wallet, API keys, etc.

# Run
python -m gpumind --config config.yaml
```

## Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `mining` | 100% GPU to PoW mining | When mining profitability > AI |
| `inference` | 100% GPU to AI inference | When AI requests queue builds up |
| `hybrid` | Split GPU resources | Stable mining + light AI serving |
| `burst` | Mining with instant interrupt | High-value AI requests, pause mine |

## Project Structure

```
gpumind/
├── gpumind/
│   ├── __init__.py
│   ├── core.py          # Main daemon
│   ├── brain.py         # ML profitability predictor
│   ├── gpu_manager.py   # GPU resource manager
│   ├── miners/          # Mining adapters
│   │   ├── base.py
│   │   └── keryx.py
│   ├── inference/       # AI inference server
│   │   ├── base.py
│   │   ├── llm.py       # Text generation
│   │   └── vision.py    # Image analysis
│   ├── monitor.py       # Metrics collector
│   └── api.py           # REST API
├── config.example.yaml
├── requirements.txt
└── README.md
```

## License

MIT
