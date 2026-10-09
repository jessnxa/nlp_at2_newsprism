import json
import joblib
import numpy as np

from sentence_transformers import SentenceTransformer

# Load configuration
with open("models/political_lean_config.json", encoding="utf-8") as f:
    config = json.load(f)

# Load embedding model
embedding_model = SentenceTransformer("models/mpnet_embedder")

# Load trained classifier
classifier = joblib.load("models/political_lean_lr.joblib")

def chunk_text(text, chunk_size=200):
    words = str(text).split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def embed_article(text):
    chunks = chunk_text(text, chunk_size=config["chunk_size"])

    if not chunks:
        return np.zeros(embedding_model.get_embedding_dimension())

    chunk_embeddings = embedding_model.encode(chunks, normalize_embeddings=True)
    article_embedding = chunk_embeddings.mean(axis=0)
    norm = np.linalg.norm(article_embedding)

    if norm > 0:
        article_embedding = article_embedding / norm

    return article_embedding

def predict_political_leaning(text):
    embedding = embed_article(text)

    prediction = classifier.predict(embedding.reshape(1, -1))[0]

    probabilities = classifier.predict_proba(embedding.reshape(1, -1))[0]

    confidence = probabilities.max()

    return {"label": prediction, "confidence": float(confidence)}