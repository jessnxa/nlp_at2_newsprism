# Sentiment and Language Analysis

## 1. Notebooks

Sentiment_Analysis_FULL.ipynb : Complete process of model development, VADER and RoBERTa comparison, manual evaluation and language extraction.

Sentiment_Analysis_FINAL.ipynb : Final RoBERTa sentiment pipeline, publisher-level analysis, language extraction and CSV outputs.

## 2. Results and Output Files

The final notebook generates three CSV files in "Results" folder, for easier analysis and dashboard integration.

| Output file | Description |
|---|---|
| `newsprism_sentiment_results.csv` | Headline and article sentiment labels, confidence scores and article sentiment probabilities |
| `newsprism_publisher_headline_sentiment.csv` | Percentage distribution of headline sentiment by publisher and topic |
| `newsprism_language_results.csv` | Top three positive and negative words extracted from each article |
  `manual_sentiment_labels.csv` | Manually assigned positive, negative and neutral labels for 39 sampled articles, used to evaluate VADER and RoBERTa |

## 3. Installation and Setup

The sentiment analysis notebooks use Python libraries for data processing, sentiment classification and model evaluation.
Install the required packages before running the notebooks:

```bash
pip install pandas numpy nltk transformers torch scikit-learn
```
The RoBERTa model (`cardiffnlp/twitter-roberta-base-sentiment-latest`) is downloaded automatically from Hugging Face when first loaded using the Transformers library. After installation, open the notebooks in Jupyter or Google Colab and run the cells in order.

## 4. Overview

The sentiment analysis component of NewsPrism examines how Australian news articles use positive, negative and neutral language across immigration, housing and clean energy topic coverage. The aim is to help readers understand differences in the tone of reporting across news sources. Sentiment is analysed separately from political stance, as negative language does not necessarily indicate political bias. This component also extracts positive and negative words to provide examples of the language used within articles.

## 5. Approaches Considered

Two sentiment analysis approaches were tested to determine which was more suitable for news articles.

###  VADER : Lexicon Based Approach

VADER (Valence Aware Dictionary and sEntiment Reasoner) uses a predefined dictionary of words and sentiment scores.

Why it was selected:

- Simple and computationally efficient.
- Does not require model training.
- Provides a useful baseline for comparison.

However, VADER relies heavily on word-level sentiment and may struggle with context. For example, articles describing negative events in neutral language can receive misleading sentiment scores. 

### RoBERTa : Transformer Based Approach

The pretrained cardiffnlp/twitter-roberta-base-sentiment-latest model was tested as an alternative.

Why it was selected:

- Considers the surrounding context of words.
- Can capture more complex language patterns than a lexicon-based approach.
- Produces sentiment probabilities for negative, neutral and positive classes.

However, the model was originally trained on social-media text, which may affect its performance on news articles.

## 6. Methodology

- Headline and Article Sentiment

Headlines and full articles were analysed separately as headlines often emphasise attention-grabbing aspects of a story, while article bodies provide more context. For full articles, text was divided into sentence-based chunks of up to 500 tokens to fit RoBERTa's input limit. Each chunk was analysed separately, and sentiment probabilities were combined using token-count-weighted averaging to produce an overall article label. The final output includes sentiment labels, confidence scores and class probabilities.

- Evaluative Language Extraction

VADER was used to identify positive and negative words and sentences as its lexicon provides sentiment scores for individual words, making it easier to identify the language contributing to an article's tone and explain the results to readers. Common news-related terms that could be misleadingly interpreted as emotional language were adjusted in the lexicon. Words and sentences with a sentiment strength of at least 1.5 in absolute value were extracted. The top three positive and negative words were kept for the dashboard to keep the results readable and less cluttered.

## 7. Model Evaluation

For manual evaluation, the dataset was grouped by topic, and 13 articles were randomly selected from each of the three topics: immigration, housing and clean energy. This ensured equal representation across topics, producing a sample of 39 articles. A fixed random seed of 42 was used to make the sampling reproducible. Each selected article was then manually labelled as positive, negative or neutral, and these labels were used as a reference to evaluate the predictions made by VADER and RoBERTa.

Model predictions were compared against human labels using accuracy, precision, recall and F1-score to assess overall performance.

| Metric | VADER | RoBERTa |
|---|---|---|
| Accuracy | 41% | 46% |
| Macro F1-score | 0.32 | 0.46 |

## 8. Final Model Selection

RoBERTa was selected for the final NewsPrism application as it achieved higher accuracy and a better macro F1-score than VADER.
VADER was retained for evaluative language extraction, where its word-level sentiment dictionary was useful for identifying positive and negative terms. This allowed to identify the overall sentiment of articles while also showing readers the positive and negative words used in the reporting.

## 9. Key Findings

The analysis found that full articles were more commonly classified as neutral, while headlines showed greater variation in sentiment. This suggests that headlines and article bodies can convey different tones, making it useful to analyse both rather than relying on headlines alone. Language extraction also helped identify specific words contributing to positive or negative language within articles.

## 10. Limitations

- RoBERTa achieved only 46% accuracy on the manually labelled sample.
- The manual evaluation was small, including only 39 articles.
- RoBERTa was pretrained on social-media text rather than news reporting.
- Models may confuse the negativity of reported events with the journalist's writing style.
- Word-level language extraction does not always capture context, sarcasm or quotations.

Therefore, sentiment results should be treated as supporting evidence rather than definitive assessments of political bias or reporting quality.