# 💬 Emotion-Aware Journaling Assistant

An ML-powered conversational journaling tool that detects emotions from free-text entries and responds with empathetic, context-aware follow-ups. Built with **Streamlit** for the UI and **scikit-learn** for emotion classification.

---

## 📌 Overview

The Emotion-Aware Journaling Assistant lets users type natural-language journal entries into a chat interface. Behind the scenes, a **dual-model ensemble** classifies each entry into one of five emotions — *happy, sad, angry, anxious,* or *neutral* — and the assistant replies with tailored reflective prompts across a multi-turn conversation flow. All entries are logged, visualised on a dashboard, and can be exported as a PDF report.

---

## ✨ Features

| Feature | Description |
|---|---|
| **Emotion Detection** | Classifies text into 5 emotion classes using TF-IDF + Logistic Regression |
| **Dual-Model Ensemble** | Combines a *clean* model (DAIR-AI dataset) and a *noisy* model (GoEmotions dataset) for robust prediction |
| **Multi-Turn Chat** | Three-stage conversation flow: initial detection → follow-up question → reflective closing |
| **Emotion Dashboard** | Pie chart of emotion distribution, time-series plot, and recent entry table |
| **PDF Report Export** | One-click downloadable PDF summary of journal history |
| **Persistent Journal Log** | All entries saved to CSV with timestamps for longitudinal tracking |

---

## 🏗️ Architecture

```
User Input
    │
    ▼
┌──────────────┐     ┌──────────────────┐
│  Clean Model │     │   Noisy Model    │
│  (DAIR-AI)   │     │  (GoEmotions)    │
│  TF-IDF +    │     │  TF-IDF +        │
│  LogReg      │     │  LogReg          │
└──────┬───────┘     └───────┬──────────┘
       │                     │
       ▼                     ▼
    ┌──────────────────────────┐
    │   Ensemble Predictor     │
    │  (agreement / fallback)  │
    └────────────┬─────────────┘
                 │
                 ▼
    ┌──────────────────────────┐
    │   Response Generator     │
    │  (emotion-aware replies) │
    └────────────┬─────────────┘
                 │
                 ▼
    ┌──────────────────────────┐
    │  Streamlit Chat UI       │
    │  + Dashboard + PDF       │
    └──────────────────────────┘
```

### Dual-Model Prediction Strategy

The predictor loads two independently trained models and applies a simple ensemble rule:

1. **If both models agree** → use the agreed prediction (*high confidence*).
2. **If they disagree and the input is "messy"** (short, contains `...`, `??`, `!!`) → trust the **noisy model**, which was trained on informal Reddit text.
3. **If they disagree and the input is clean** → trust the **clean model**, which was trained on well-formed sentences.

---

## 📂 Project Structure

```
.
├── app.py                 # Streamlit application (Chat + Dashboard pages)
├── predictor.py           # Dual-model ensemble prediction logic
├── responses.py           # Emotion-specific response templates
├── storage.py             # Save journal entries to CSV
├── report.py              # PDF report generation (ReportLab)
│
├── download_data.py       # Download DAIR-AI emotion dataset from HuggingFace
├── prepare_data.py        # Consolidate to 5 emotion classes (DAIR-AI)
├── preprocess.py          # Load & clean GoEmotions dataset
├── train_dair_ai.py       # Train clean model on DAIR-AI data
├── train_goemotions.py    # Train noisy model on GoEmotions data
├── test_predictor.py      # Quick sanity-check script for predictions
│
├── model_clean.pkl        # Trained clean model (git-ignored)
├── vectorizer_clean.pkl   # TF-IDF vectorizer for clean model (git-ignored)
├── model_noisy.pkl        # Trained noisy model (git-ignored)
├── vectorizer_noisy.pkl   # TF-IDF vectorizer for noisy model (git-ignored)
│
├── journal_log.csv        # Persistent log of user entries
├── Trained.txt            # Training results & comparison log
├── data/                  # Raw & processed datasets (git-ignored)
├── results/               # Output artefacts
├── .gitignore
└── README.md
```

---

## 📊 Datasets

| Dataset | Source | Role |
|---|---|---|
| **DAIR-AI Emotion** | [HuggingFace – dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) | Clean, well-formed sentences. Labels consolidated from 6 → 5 classes (merged *love* into *happy*, dropped *surprise*). |
| **GoEmotions** | [Google Research – GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) | Noisy Reddit comments with 28 fine-grained labels, mapped down to 5 macro-emotions. Class-balanced by down-sampling *neutral*. |

### Emotion Label Mapping (GoEmotions → 5 classes)

| Target Label | Source Emotions |
|---|---|
| **happy** | joy, love, amusement, excitement, optimism, pride, caring, gratitude, relief, desire |
| **sad** | sadness, grief, disappointment, remorse, embarrassment |
| **angry** | anger, annoyance, disgust, disapproval |
| **anxious** | fear, nervousness, confusion |
| **neutral** | neutral, curiosity, realization, approval, admiration, surprise |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/rdavid27/emotion-aware-journaling-assistant.git
cd emotion-aware-journaling-assistant

# Install dependencies
pip install streamlit pandas matplotlib scikit-learn joblib reportlab datasets
```

### Training the Models (optional — pre-trained `.pkl` files are provided)

```bash
# Step 1: Download the DAIR-AI dataset
python download_data.py

# Step 2: Prepare the 5-class version
python prepare_data.py

# Step 3: Preprocess GoEmotions data (place goemotions_1/2/3.csv in data/)
python preprocess.py

# Step 4: Train both models
python train_dair_ai.py      # → model_clean.pkl, vectorizer_clean.pkl
python train_goemotions.py   # → model_noisy.pkl, vectorizer_noisy.pkl
```

### Running the App

```bash
streamlit run app.py
```

The app will open in your browser with two pages accessible from the sidebar:

- **Chat** — type journal entries and receive emotion-aware responses.
- **Dashboard** — view emotion distribution charts, timeline, and download a PDF report.

---

## 🧪 Model Performance

Results from training on the GoEmotions dataset (noisy model):

| Metric | Score |
|---|---|
| **Accuracy** | ~55% |
| **Macro Avg F1** | ~0.50 |

Results from training on the DAIR-AI dataset (clean model) — evaluated on its own test split — are comparable. The dual-model ensemble improves robustness on mixed-style inputs by routing predictions to the model best suited for the input's formality level.

> **Note:** These are lightweight TF-IDF + Logistic Regression models optimised for speed and interpretability rather than state-of-the-art accuracy. For production use, a fine-tuned transformer (e.g. BERT) would significantly improve performance.

---

## 🛠️ Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io/)
- **ML Pipeline:** [scikit-learn](https://scikit-learn.org/) (TF-IDF Vectorizer + Logistic Regression)
- **Data Handling:** [pandas](https://pandas.pydata.org/)
- **Visualisation:** [matplotlib](https://matplotlib.org/)
- **PDF Generation:** [ReportLab](https://www.reportlab.com/)
- **Dataset Loading:** [HuggingFace Datasets](https://huggingface.co/docs/datasets/)

---

## 📄 License

This project is for academic/educational purposes.

---

## 🙏 Acknowledgements

- [DAIR-AI](https://github.com/dair-ai) for the Emotion dataset
- [Google Research](https://github.com/google-research) for the GoEmotions dataset
- [Streamlit](https://streamlit.io/) for the rapid prototyping framework
