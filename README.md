# 📈 Financial Sentiment Analyzer

> NLP pipeline classifying financial news sentiment using FinBERT — deployed as a live interactive web app.

**Author:** Gautam Kumar Kanojia &nbsp;|&nbsp; ![Python](https://img.shields.io/badge/Python-3.13-blue?style=flat-square) ![HuggingFace](https://img.shields.io/badge/HuggingFace-FinBERT-orange?style=flat-square) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red?style=flat-square) ![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-brightgreen?style=flat-square)

---

## 🚀 Live Demo

👉 [**Launch App**](https://financial-sentiment-analyzer-zn3w2zxnpurudsin5gytqr.streamlit.app/)

---

## 📸 App Preview

![App Screenshot 1](app_screenshot1.png)
![App Screenshot 2](app_screenshot2.png)

---

## Project Overview

A production-ready NLP pipeline that classifies financial news sentences as **Positive**, **Negative**, or **Neutral** using `ProsusAI/finbert` — a BERT model fine-tuned specifically on financial text. Supports single-text analysis and batch CSV processing with confidence scoring and interactive visualizations.

---

## Business Problem

Financial analysts spend hours manually reading earnings reports, news headlines, and market commentary to gauge market sentiment. This tool automates sentiment detection on financial text, enabling faster and more consistent data-driven decision-making.

---

## Dataset

| Property | Details |
|---|---|
| Name | Financial PhraseBank |
| Source | Kaggle — Ankur Sinha |
| Size | 4,846 sentences |
| Labels | Positive, Negative, Neutral |
| Evaluation Split | 500 sentences |

---

## Tech Stack

| Component | Tool |
|---|---|
| NLP Model | ProsusAI/finbert (HuggingFace Transformers) |
| Framework | PyTorch |
| Data Processing | Pandas |
| Evaluation | Scikit-learn |
| UI | Streamlit |
| Visualization | Plotly |
| Deployment | Streamlit Cloud |

---

## Preprocessing

- Loaded raw CSV with latin-1 encoding, assigned column names manually
- Truncated all inputs to 512 tokens (FinBERT max sequence length)
- Normalized labels to lowercase for consistent evaluation
- Selected first 500 rows for evaluation run

---

## Models Comparison

| Model | Type | Accuracy | Weighted F1 |
|---|---|---|---|
| ProsusAI/finbert | Pre-trained BERT (Financial) | 92.00% | 0.92 |

---

## Challenges

| Challenge | Root Cause | Solution |
|---|---|---|
| HuggingFace dataset script deprecated | `financial_phrasebank` uses old loading script format | Switched to Kaggle CSV download |
| HuggingFace Inference API cold start | Free tier spins down model after inactivity | Switched to local pipeline in Colab (T4 GPU) |
| Colab network blocked HF API | Colab restricts outbound API calls | Used local `pipeline()` instead of API calls |
| Git push rejected | GitHub repo had existing README commit | Used `--allow-unrelated-histories` + `--force` |

---

## Final Model Performance

Evaluated on 500 sentences from Financial PhraseBank:

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Positive | 0.93 | 0.99 | 0.96 | 394 |
| Neutral | 0.94 | 0.67 | 0.78 | 102 |
| Negative | 0.38 | 0.75 | 0.50 | 4 |
| **Accuracy** | | | **0.92** | **500** |
| Weighted Avg | 0.93 | 0.92 | 0.92 | 500 |

### Confusion Matrix

| | Predicted Positive | Predicted Negative | Predicted Neutral |
|---|---|---|---|
| Actual Positive | 389 | 2 | 3 |
| Actual Negative | 0 | 3 | 1 |
| Actual Neutral | 31 | 3 | 68 |

---

## Key Insights

- FinBERT achieves 92% accuracy with zero fine-tuning — strong domain-specific pretraining on financial corpora
- Positive sentiment detection is highly reliable (F1: 0.96, Recall: 0.99)
- Negative class underperforms due to severe class imbalance (only 4 negative samples in test set)
- Model picks up financial language nuance — words like "rose" and "climbed" in market reports correctly classified as positive even at marginal gains

---

## Deliverables

| File | Description |
|---|---|
| `app.py` | Streamlit web application with single + batch analysis |
| `Main.py` | Core inference + evaluation pipeline |
| `Dataset.csv` | Financial PhraseBank dataset (4,846 sentences) |
| `finbert_results.csv` | Model predictions with confidence scores |
| `requirements.txt` | Python dependencies |
| `packages.txt` | System-level packages for Streamlit Cloud |

---

## How to Run

```bash
git clone https://github.com/gautam-kumar-7590/Financial-Sentiment-Analyzer.git
cd Financial-Sentiment-Analyzer
pip install -r requirements.txt
python -m streamlit run app.py
```

---

*Built by Gautam Kumar Kanojia — [LinkedIn](https://www.linkedin.com/in/GautamKumarKanojia) | [GitHub](https://github.com/gautam-kumar-7590)*
