# 🧪 LLMEval

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-orange.svg)](https://github.com/astral-sh/ruff)
[![Tests](https://img.shields.io/badge/tests-pytest-blue)](https://pytest.org)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

> A pluggable, async-first LLM evaluation & benchmarking framework.  
> Compare Claude, GPT-4, Gemini and more across 5 automated metrics — with a live Streamlit leaderboard and auto-generated reports.

---

## ✨ Features

| Feature | Description |
|---|---|
| **Multi-model** | Evaluate Claude, GPT-4o, Gemini side-by-side from a single YAML config |
| **5 metrics** | ROUGE, BLEU, semantic similarity, faithfulness, LLM-as-judge |
| **Async runner** | Bounded-concurrency async execution with retry & timeout handling |
| **Cost tracking** | Per-model token usage and USD cost tracked for every sample |
| **LLM-as-judge** | Use Claude or GPT-4 to score responses on a 1–5 rubric |
| **Streamlit dashboard** | Live leaderboard with score breakdown, latency, and cost charts |
| **YAML config** | Fully declarative — swap models, datasets, metrics without touching code |
| **CLI** | `llmeval run --config my_eval.yaml` — one command to run everything |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                     Input layer                      │
│  Dataset loader · Model connectors · Prompt templates │
└───────────────────────┬─────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────┐
│              Evaluation runner (async)               │
│     Batch inference · Retry logic · Cost tracking    │
└───────────────────────┬─────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────┐
│                   Metrics engine                     │
│  ROUGE · BLEU · Semantic sim · Faithfulness · Judge  │
└───────────────────────┬─────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────┐
│                Results aggregator                    │
│    Score matrix · Statistical significance · Delta   │
└───────────────────────┬─────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
  Streamlit         JSON / CSV       CLI report
  dashboard          export         (Rich table)
```

---

## 🚀 Quickstart

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/llmeval.git
cd llmeval
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### 2. Set API keys

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY and/or OPENAI_API_KEY
```

### 3. Run your first evaluation

```bash
llmeval run --config configs/example.yaml
```

Output:

```
LLMEval · run: my_first_eval · 10 samples · 2 model(s)

→ Evaluating Claude Haiku (claude-3-5-haiku-20241022)
  Claude Haiku ━━━━━━━━━━━━━━━━━━━━ 10/10 0:00:08
  ✓ Done in 8.3s · cost: $0.0012 · errors: 0

→ Evaluating GPT-4o Mini (gpt-4o-mini)
  GPT-4o Mini  ━━━━━━━━━━━━━━━━━━━━ 10/10 0:00:11
  ✓ Done in 11.1s · cost: $0.0008 · errors: 0

┏━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━┓
┃ Model         ┃ Samples ┃ Errors ┃ Avg latency┃ Total cost ┃ Score ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━┩
│ Claude Haiku  │      10 │      0 │       830ms│    $0.0012 │ 0.742 │
│ GPT-4o Mini   │      10 │      0 │      1100ms│    $0.0008 │ 0.718 │
└───────────────┴─────────┴────────┴────────────┴────────────┴───────┘

Results saved → results/my_first_eval_1711234567.json
```

### 4. Launch the dashboard (Week 4)

```bash
streamlit run dashboard/app.py
```

---

## ⚙️ Configuration

All evaluation parameters live in a single YAML file:

```yaml
run_name: "gpt_vs_claude_comparison"

models:
  - name: "Claude Haiku"
    provider: "anthropic"
    model_id: "claude-3-5-haiku-20241022"
    temperature: 0.0
    max_tokens: 256

  - name: "GPT-4o Mini"
    provider: "openai"
    model_id: "gpt-4o-mini"
    temperature: 0.0
    max_tokens: 256

dataset:
  path: "data/sample.jsonl"   # or a HuggingFace dataset name
  source: "jsonl"             # jsonl | csv | huggingface
  max_samples: 100
  input_column: "question"
  reference_column: "answer"

metrics:
  rouge: true
  bleu: true
  semantic_similarity: true
  faithfulness: true
  llm_judge: false            # set true to use Claude as judge (extra API cost)

concurrency: 5                # async requests in flight per model
output_dir: "results"
```

### Supported dataset sources

```bash
# Local JSONL (one JSON object per line)
path: "data/my_dataset.jsonl"
source: "jsonl"

# Local CSV
path: "data/my_dataset.csv"
source: "csv"

# HuggingFace dataset (auto-downloaded)
path: "rajpurkar/squad"
source: "huggingface"
split: "validation"
```

---

## 📊 Metrics

| Metric | Type | Description |
|---|---|---|
| `rouge` | Reference-based | ROUGE-1, ROUGE-2, ROUGE-L overlap with ground truth |
| `bleu` | Reference-based | BLEU-4 n-gram precision |
| `semantic_similarity` | Embedding-based | Cosine similarity via `sentence-transformers` |
| `faithfulness` | LLM-based | Is the response grounded in the provided context? |
| `llm_judge` | LLM-based | GPT-4 / Claude scores on a 1–5 rubric (opt-in) |

### LLM-as-judge rubric

When `llm_judge: true`, each prediction is scored by a judge model on:

- **Correctness** — factually accurate vs. reference answer
- **Completeness** — covers all aspects of the reference
- **Conciseness** — no unnecessary verbosity
- **Coherence** — logically structured and readable

---

## 🔌 Adding a new model provider

1. Subclass `BaseConnector` in `llmeval/connectors/`
2. Implement `generate()` and `estimate_cost()`
3. Register it in `connectors/__init__.py`'s `build_connector()` factory

```python
# llmeval/connectors/google_connector.py
class GoogleConnector(BaseConnector):
    async def generate(self, prompt: str) -> ModelResponse:
        ...
```

---

## 🗂️ Project structure

```
llmeval/
├── llmeval/
│   ├── __init__.py
│   ├── config.py          # Pydantic config models (EvalConfig, ModelConfig, ...)
│   ├── dataset.py         # DatasetLoader + EvalSample
│   ├── runner.py          # Async EvalRunner with progress + leaderboard
│   ├── cli.py             # Typer CLI (llmeval run / validate)
│   ├── connectors/
│   │   ├── base.py        # BaseConnector + ModelResponse
│   │   ├── anthropic_connector.py
│   │   └── openai_connector.py
│   └── metrics/
│       └── __init__.py    # Metrics engine (Week 2)
├── configs/
│   └── example.yaml       # Example eval config
├── data/
│   └── sample.jsonl       # 10-sample QA dataset
├── tests/
│   └── test_connectors.py # Unit tests (pytest)
├── dashboard/             # Streamlit dashboard (Week 4)
├── .env.example
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 🗺️ Roadmap

- [x] **Week 1** — Async runner, model connectors (Claude + GPT-4), dataset loader, config system, CLI
- [ ] **Week 2** — ROUGE, BLEU, semantic similarity, faithfulness, LLM-as-judge metrics
- [ ] **Week 3** — Multi-model comparison, statistical significance, delta-vs-baseline, score matrix
- [ ] **Week 4** — Streamlit leaderboard, auto-generated PDF reports, HuggingFace benchmark datasets

---

## 🧪 Running tests

```bash
pytest tests/ -v
pytest tests/ -v --cov=llmeval --cov-report=term-missing
```

---

## 🤝 Contributing

Contributions are welcome! Please open an issue before submitting a PR.

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/google-connector`
3. Commit your changes: `git commit -m 'feat: add Google Gemini connector'`
4. Push and open a PR

---



## 🙏 Acknowledgements

Built with [Anthropic Claude](https://anthropic.com), [OpenAI](https://openai.com), [sentence-transformers](https://sbert.net), [Rich](https://github.com/Textualize/rich), and [Typer](https://typer.tiangolo.com).
