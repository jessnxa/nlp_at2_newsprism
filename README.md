# NewsPrism

> **University of Technology Sydney**  
> **36118 Applied Natural Language Processing**

## Overview

NewsPrism is a web-based NLP application designed to analyse and compare how different news sources cover the same political topics.

The project focuses on three key areas: **immigration, clean energy, and housing**. News articles are collected and preprocessed into a standardised corpus before being analysed using several NLP techniques:

- Political leaning classification
- Sentiment and language analysis
- Topic modelling and named entity recognition
- Framing analysis and summarisation

Selected NLP outputs are benchmarked against the **Gemini API** for comparison. The final analysis is integrated into an interactive **Streamlit** application to support accessible comparison and visualisation of news coverage.

## Project Workflow

1. **Data Collection & Preprocessing** - Collect article data and metadata, then clean and standardise the corpus.
2. **NLP Analysis** - Analyse political leaning, sentiment and language, topics, and named entities using the processed corpus.
3. **Framing & Summarisation** - Use article data and relevant NLP outputs to analyse framing and generate summaries.
4. **Application Integration** - Integrate the final analysis outputs into the Streamlit application for comparison and visualisation.

## Repository Structure

```text
nlp_at2_newsprism/
│
├── data/
│   ├── raw/                   # Raw article data
│   ├── processed/             # Cleaned and standardised corpus
│   └── enriched/              # Final corpus with NLP outputs
│
├── src/
│   ├── data_preprocessing/    # Data collection and preprocessing
│   ├── political_leaning/     # Political leaning classification
│   ├── sentiment_language/    # Sentiment and language analysis
│   ├── topics_entities/       # Topic modelling and NER
│   └── framing_summarisation/ # Framing analysis and summarisation
│
├── notebooks/
│   ├── 01_data_preprocessing/
│   ├── 02_political_leaning/
│   ├── 03_sentiment_language/
│   ├── 04_topics_entities/
│   └── 05_framing_summarisation/
│
├── app/                       # Streamlit application
├── .gitignore
├── requirements.txt           # Python dependencies
└── README.md
```

## Installation & Setup

Create and activate a virtual environment:

```bash
# macOS/Linux
python -m venv .venv
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Collaboration & Git Workflow

Each project area is developed on a separate branch before being reviewed and merged into `main`.

```text
data-preprocessing
political-leaning
sentiment-language
topics-entities
framing-summarisation
streamlit-integration
```

Team members commit changes to their respective branches with descriptive commit messages. Completed work is reviewed through pull requests before being merged into `main`, which contains the integrated version of NewsPrism.