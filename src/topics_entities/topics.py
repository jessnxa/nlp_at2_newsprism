"""Subtopic modelling within each broad topic (clean_energy, housing, immigration).

Pipeline: clean text -> ~100-word sentence chunks -> embeddings -> per-topic clustering ->
keywords -> human review decisions applied -> subtopic mix per article and per passage.
Settings are in config.py; `pipeline.run` is the entry point.
"""
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .cleanup import SPLIT, strip_boilerplate
from .config import (
    LEMMA_MODEL,
    CHUNK_CAP,
    CHUNK_MIN_TAIL,
    CHUNK_TARGET,
    EMBED_MODEL,
    EXPECTED_SUBTOPICS,
    FIRST_TARGET_YEAR,
    GENERIC_KEYWORDS,
    HUMAN_COLS,
    MAX_CLUSTER_SHARE,
    MAX_CROSS_SHARE,
    MAX_TOP_ARTICLE_SHARE,
    MIN_ARTICLES_PER_SUBTOPIC,
    MIN_SUGGEST_OVERLAP,
    MMR_DIVERSITY,
    NEWS_STOPWORDS,
    NUMBER,
    N_EXAMPLES,
    N_KEYWORDS,
    N_KEYWORD_CANDIDATES,
    REQUIRED_COLUMNS,
    SEED,
    SPELLING_VARIANTS,
)


# --- data and chunking ---------------------------------------------------

def prepare_corpus(corpus):
    """Validate the required columns and return a copy with in-memory boilerplate-free `model_text`."""
    missing = [c for c in REQUIRED_COLUMNS if c not in corpus.columns]
    if missing:
        raise ValueError(f'corpus is missing required columns: {missing}')
    df = corpus.copy()
    df['model_text'] = strip_boilerplate(df['clean_text'].tolist())
    return df


def split_sentences(text):
    return [s for s in SPLIT.split(text) if s.strip()]


def chunk_fixed(text, target=CHUNK_TARGET, cap=CHUNK_CAP, min_tail=CHUNK_MIN_TAIL):
    """Group whole sentences into ~target-word chunks, never exceeding cap words."""
    sents = []
    for s in split_sentences(text):
        words = s.split()
        # a single sentence longer than cap is split by words so nothing is truncated
        sents += [' '.join(words[i:i + cap]) for i in range(0, len(words), cap)]

    chunks, cur, n = [], [], 0
    for s in sents:
        w = len(s.split())
        if cur and n + w > cap:
            chunks.append((' '.join(cur), n))
            cur, n = [], 0
        cur.append(s)
        n += w
        if n >= target:
            chunks.append((' '.join(cur), n))
            cur, n = [], 0
    if cur:
        # merge a short tail into the previous chunk only if it still fits under cap
        if chunks and n < min_tail and chunks[-1][1] + n <= cap:
            prev, pn = chunks.pop()
            chunks.append((prev + ' ' + ' '.join(cur), pn + n))
        else:
            chunks.append((' '.join(cur), n))
    return [c for c, _ in chunks]


def chunk_rows(art, texts):
    """Chunk records for one article; article_id is kept so results roll up per article."""
    return [{
        'chunk_id': f'{art.article_id}_{i:03d}',
        'article_id': art.article_id,
        'topic': art.topic,
        'publisher': art.publisher,
        'chunk_idx': i,
        'text': text,
        'n_words': len(text.split()),
    } for i, text in enumerate(texts)]


def build_chunks(df, target=CHUNK_TARGET, cap=CHUNK_CAP, min_tail=CHUNK_MIN_TAIL):
    rows = []
    for art in df.itertuples(index=False):
        rows += chunk_rows(art, chunk_fixed(art.model_text, target, cap, min_tail))
    return pd.DataFrame(rows)


# --- embeddings ----------------------------------------------------------

def embed_chunks(chunks, model_name=EMBED_MODEL):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return model.encode(chunks['text'].tolist(), batch_size=32, normalize_embeddings=True,
                        show_progress_bar=False)


def cached_embeddings(chunks, cache_dir=None, model_name=EMBED_MODEL):
    """Embeddings, optionally cached under cache_dir keyed by model and exact chunk texts."""
    if cache_dir is None:
        return embed_chunks(chunks, model_name)
    cache_dir = Path(cache_dir)
    key = hashlib.sha1('\n'.join([model_name, *chunks['text']]).encode()).hexdigest()[:16]
    path = cache_dir / f"{model_name.split('/')[-1]}_{key}.npy"
    if path.exists():
        return np.load(path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    emb = embed_chunks(chunks, model_name)
    np.save(path, emb)
    return emb


# --- clustering ----------------------------------------------------------

def topic_similarity(chunks, emb):
    """similarity of each chunk to each broad-topic centroid. Annotation only, never a filter."""
    sims = {}
    for t in sorted(chunks['topic'].unique()):
        centroid = emb[(chunks['topic'] == t).to_numpy()].mean(0)
        sims[f'sim_{t}'] = emb @ (centroid / np.linalg.norm(centroid))
    sims = pd.DataFrame(sims, index=chunks.index)
    closest = sims.idxmax(axis=1).str.removeprefix('sim_')
    return sims.assign(closest_topic=closest, cross_topic=closest != chunks['topic'])


def cluster_params(n_chunks, max_k=EXPECTED_SUBTOPICS[1]):
    """Parameters scaled to topic size: the smallest real subtopic is ~1/3 of N/max_k chunks."""
    mcs = max(5, round(n_chunks / (3 * max_k)))
    return {'min_cluster_size': mcs, 'n_neighbors': mcs, 'min_samples': 5, 'article_cap': mcs // 2}


def cap_per_article(chunks, cap):
    """mask keeping at most `cap` evenly spaced chunks per article, so no article clusters alone."""
    keep = []
    for _, g in chunks.groupby('article_id', sort=False):
        idx = g.sort_values('chunk_idx').index
        if len(idx) > cap:
            idx = idx[np.linspace(0, len(idx) - 1, cap).round().astype(int)]
        keep.extend(idx)
    return chunks.index.isin(keep)


def make_umap(n_neighbors, seed=SEED):
    from umap import UMAP
    return UMAP(n_neighbors=n_neighbors, n_components=5, min_dist=0.0, metric='cosine', random_state=seed)


def make_hdbscan(min_cluster_size, min_samples):
    from hdbscan import HDBSCAN
    return HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric='euclidean',
                   cluster_selection_method='eom', prediction_data=True)


def cluster_topic(emb, params, fit_mask, seed=SEED):
    """Chunk labels (-1 = outlier); fitted on fit_mask chunks, the rest assigned afterwards."""
    from hdbscan import approximate_predict
    umap = make_umap(params['n_neighbors'], seed).fit(emb[fit_mask])
    hdb = make_hdbscan(params['min_cluster_size'], params['min_samples']).fit(umap.embedding_)
    labels = np.full(len(emb), -1)
    labels[fit_mask] = hdb.labels_
    if (~fit_mask).any():
        labels[~fit_mask] = approximate_predict(hdb, umap.transform(emb[~fit_mask]))[0]
    return labels


def passes_article_rules(chunks):
    shares = chunks['article_id'].value_counts(normalize=True)
    return len(shares) >= MIN_ARTICLES_PER_SUBTOPIC and shares.iloc[0] <= MAX_TOP_ARTICLE_SHARE


def split_oversized(chunks, emb, labels, fit_mask, seed=SEED, ks=(2, 3, 4)):
    """split a cluster above MAX_CLUSTER_SHARE when a KMeans split passes the article rules."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    labels, splits = labels.copy(), []
    sizes = pd.Series(labels[labels >= 0]).value_counts()
    for sub, n in sizes.items():
        if n / sizes.sum() <= MAX_CLUSTER_SHARE:
            continue
        in_sub = labels == sub
        fit = in_sub & fit_mask
        best = None
        for k in ks:
            km = KMeans(k, n_init=20, random_state=seed).fit(emb[fit])
            parts = chunks[fit].assign(part=km.labels_)
            if all(passes_article_rules(g) for _, g in parts.groupby('part')):
                score = silhouette_score(emb[fit], km.labels_, metric='cosine')
                if best is None or score > best[0]:
                    best = (score, k, km)
        if best is None:
            splits.append({'subtopic': int(sub), 'share': n / sizes.sum(), 'split_into': 0})
            continue
        _, k, km = best
        part = km.predict(emb[in_sub])
        next_id = labels.max() + 1
        labels[in_sub] = np.where(part == 0, sub, next_id + part - 1)
        splits.append({'subtopic': int(sub), 'share': n / sizes.sum(), 'split_into': k})
    return labels, splits


def cluster_diagnostics(chunks, labels):
    """Per-subtopic size, article and publisher spread, with flags for likely artefacts."""
    df = chunks.assign(subtopic=labels)
    diag = df[df.subtopic >= 0].groupby('subtopic').agg(
        n_chunks=('chunk_id', 'size'),
        n_articles=('article_id', 'nunique'),
        top_article_share=('article_id', lambda s: s.value_counts(normalize=True).iloc[0]),
        n_publishers=('publisher', 'nunique'),
        top_publisher_share=('publisher', lambda s: s.value_counts(normalize=True).iloc[0]),
        cross_share=('cross_topic', 'mean'),
    )
    rules = {
        'few_articles': diag.n_articles < MIN_ARTICLES_PER_SUBTOPIC,
        'dominant_article': diag.top_article_share > MAX_TOP_ARTICLE_SHARE,
        'review_offtopic': diag.cross_share > MAX_CROSS_SHARE,
        'single_publisher': diag.n_publishers == 1,
    }
    diag['flag'] = pd.DataFrame(rules).apply(lambda r: ','.join(r.index[r]), axis=1)
    return diag.reset_index()


def run_clustering(chunks, emb, seed=SEED):
    """Cluster each broad topic separately; returns chunk labels, a per-topic summary and diagnostics."""
    labels = pd.Series(-1, index=chunks.index)
    summary, diags = [], {}
    for topic, idx in chunks.groupby('topic').groups.items():
        sub_chunks = chunks.loc[idx]
        sub_emb = emb[chunks.index.get_indexer(idx)]
        params = cluster_params(len(idx))
        fit_mask = cap_per_article(sub_chunks, params['article_cap'])
        lab = cluster_topic(sub_emb, params, fit_mask, seed)
        lab, splits = split_oversized(sub_chunks, sub_emb, lab, fit_mask, seed)
        labels.loc[idx] = lab
        diags[topic] = cluster_diagnostics(sub_chunks, lab)
        k = int(lab.max()) + 1
        lo, hi = EXPECTED_SUBTOPICS
        clustered = pd.Series(lab[lab >= 0]).value_counts(normalize=True)
        summary.append({
            'topic': topic, 'n_chunks': len(idx), 'n_fit': int(fit_mask.sum()), **params,
            'n_subtopics': k, 'in_expected_range': lo <= k <= hi,
            'outlier_rate': (lab == -1).mean(),
            'largest_share': clustered.max(),
            'splits': '; '.join(f"{s['subtopic']}->{s['split_into'] or 'kept'}" for s in splits),
            'flagged': int((diags[topic].flag != '').sum()),
        })
    return labels, pd.DataFrame(summary), diags


# --- keywords ------------------------------------------------------------

def _keep_target_year(m):
    tok = m.group()
    return tok if re.fullmatch(r'20\d\d', tok) and int(tok) >= FIRST_TARGET_YEAR else ' '


def keyword_preprocess(text):
    text = NUMBER.sub(_keep_target_year, text.lower())
    for pattern, replacement in SPELLING_VARIANTS.items():
        text = re.sub(pattern, replacement, text)
    return text


def make_vectorizer():
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
    return CountVectorizer(preprocessor=keyword_preprocess, ngram_range=(1, 3), min_df=3,
                           stop_words=sorted(ENGLISH_STOP_WORDS | NEWS_STOPWORDS))


def lemmatise(texts):
    """Lemmatised copy of texts for keyword extraction only (rents -> rent).

    Original casing goes in so proper nouns are recognised and kept ("Home Affairs" stays,
    not "home affair"); spaCy's Latin lemma "datum" for "data" is undone.
    """
    import spacy
    nlp = spacy.load(LEMMA_MODEL, disable=['parser', 'ner'])
    word = lambda t: t.text if t.pos_ == 'PROPN' else ('data' if t.lemma_ == 'datum' else t.lemma_)
    return [' '.join(word(t) for t in doc).lower() for doc in nlp.pipe(texts, batch_size=64)]


def tidy_keywords(words, n=N_KEYWORDS):
    """Drop generic words and words already contained in another candidate phrase."""
    words = [w for w in words if w not in GENERIC_KEYWORDS]
    contained = lambda w: any(w != o and re.search(rf'\b{re.escape(w)}\b', o) for o in words)
    return [w for w in words if not contained(w)][:n]


def subtopic_keywords(texts, emb, labels, embedding_model=None, top_n=N_KEYWORD_CANDIDATES, mmr=True):
    """Keyword candidates per subtopic from BERTopic c-TF-IDF, keyed by OUR subtopic IDs.

    BERTopic renumbers topics by size, so keywords are looked up through an explicit mapping.
    """
    from bertopic import BERTopic
    from bertopic.cluster import BaseCluster
    from bertopic.dimensionality import BaseDimensionalityReduction
    from bertopic.representation import MaximalMarginalRelevance

    representation = MaximalMarginalRelevance(diversity=MMR_DIVERSITY) if mmr else None
    model = BERTopic(embedding_model=embedding_model, umap_model=BaseDimensionalityReduction(),
                     hdbscan_model=BaseCluster(), vectorizer_model=make_vectorizer(),
                     representation_model=representation, top_n_words=top_n)
    bt_labels, _ = model.fit_transform(texts, embeddings=emb, y=labels)
    ours_to_bt = {int(o): int(b) for o, b in zip(labels, bt_labels)}
    if len(set(ours_to_bt.values())) != len(ours_to_bt):
        raise ValueError(f'subtopic -> BERTopic mapping is not one-to-one: {ours_to_bt}')
    return {o: [w for w, _ in model.get_topic(b)] for o, b in ours_to_bt.items() if o >= 0}


# --- review sheet --------------------------------------------------------

def pick_examples(chunks, emb, labels, sub, rng, n=N_EXAMPLES):
    """Central chunks from distinct articles, plus random non-central and cross-topic chunks."""
    members = np.flatnonzero(labels == sub)
    centroid = emb[members].mean(0)
    order = members[np.argsort(-(emb[members] @ centroid))]
    central, seen = [], set()
    for i in order:
        article = chunks['article_id'].iat[i]
        if article not in seen:
            central.append(i)
            seen.add(article)
        if len(central) == n:
            break
    rest = np.setdiff1d(members, central)
    is_cross = chunks['cross_topic'].to_numpy()
    other = rng.choice(rest[~is_cross[rest]], min(n, (~is_cross[rest]).sum()), replace=False)
    cross = rng.choice(rest[is_cross[rest]], min(n, is_cross[rest].sum()), replace=False)
    return central, other, cross


def _fmt_examples(chunks, idx, width=240):
    return ' || '.join(f"{chunks['chunk_id'].iat[i]}: {chunks['text'].iat[i][:width]}…" for i in idx)


def chunk_set_id(chunk_ids):
    # 'cs_' prefix stops spreadsheets reading the hash as a number
    return 'cs_' + hashlib.sha1('|'.join(sorted(chunk_ids)).encode()).hexdigest()[:12]


def describe_subtopics(chunks, emb, diags, seed=SEED):
    """One row per subtopic with keywords, examples and diagnostics (no human fields)."""
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer(EMBED_MODEL)
    keyword_text = pd.Series(lemmatise(chunks['text'].tolist()), index=chunks.index)
    rng = np.random.default_rng(seed)
    rows = []
    for topic, idx in chunks.groupby('topic').groups.items():
        sub_chunks = chunks.loc[idx]
        sub_emb = emb[chunks.index.get_indexer(idx)]
        labels = sub_chunks['subtopic'].to_numpy()
        keywords = subtopic_keywords(keyword_text.loc[idx].tolist(), sub_emb, labels, embedding_model)
        diag = diags[topic].set_index('subtopic')
        for sub in sorted(keywords):
            central, other, cross = pick_examples(sub_chunks, sub_emb, labels, sub, rng)
            members = sub_chunks[labels == sub]
            rows.append({
                'topic': topic, 'subtopic': sub, 'flag': diag.at[sub, 'flag'],
                'n_chunks': len(members), 'n_articles': members['article_id'].nunique(),
                'n_publishers': members['publisher'].nunique(),
                'cross_share': round(diag.at[sub, 'cross_share'], 2),
                'keywords': ', '.join(tidy_keywords(keywords[sub])),
                'example_articles': ';'.join(sub_chunks['article_id'].iat[i] for i in central),
                'central_examples': _fmt_examples(sub_chunks, central),
                'other_examples': _fmt_examples(sub_chunks, other),
                'cross_topic_examples': _fmt_examples(sub_chunks, cross),
                'chunk_set_id': chunk_set_id(members['chunk_id']),
                'article_ids': ';'.join(sorted(members['article_id'].unique())),
            })
    return pd.DataFrame(rows)


def _article_overlap(a, b):
    a, b = set(a.split(';')), set(b.split(';'))
    return len(a & b) / len(a | b)


def merge_review(current, history):
    """Reuse decisions only for identical chunk sets; otherwise suggest the closest past decision."""
    current = current.assign(status='unreviewed', suggested_from='', **{c: '' for c in HUMAN_COLS})
    if history.empty:
        return current
    reviewed = history.fillna('')
    for i, row in current.iterrows():
        same = reviewed[(reviewed.topic == row.topic) & (reviewed.chunk_set_id == row.chunk_set_id)]
        if len(same):
            current.loc[i, HUMAN_COLS] = same.iloc[0][HUMAN_COLS].to_numpy()
            current.loc[i, 'status'] = 'reviewed'
            continue
        cands = reviewed[reviewed.topic == row.topic]
        if cands.empty:
            continue
        overlap = cands.article_ids.map(lambda ids: _article_overlap(row.article_ids, ids))
        if overlap.max() >= MIN_SUGGEST_OVERLAP:
            best = cands.loc[overlap.idxmax()]
            current.loc[i, 'status'] = 'needs_review'
            current.loc[i, 'suggested_from'] = (f"{best.label} (was subtopic {best.subtopic}, "
                                                f"article overlap {overlap.max():.0%})")
    return current


# --- article subtopic mix ------------------------------------------------

def subtopic_mix(chunks):
    """One row per (article, subtopic): chunk count, weight, and the primary subtopic.

    Outlier chunks are left out; primary = most chunks, ties go to the subtopic that appears
    first (news puts the main angle up front).
    """
    mix = (chunks[chunks['subtopic'] >= 0].groupby(['article_id', 'topic', 'publisher', 'subtopic'])
           .agg(n_chunks=('chunk_id', 'size'), first_chunk=('chunk_idx', 'min')).reset_index())
    mix['weight'] = mix['n_chunks'] / mix.groupby('article_id')['n_chunks'].transform('sum')
    primary = (mix.sort_values(['article_id', 'n_chunks', 'first_chunk'], ascending=[True, False, True])
               .drop_duplicates('article_id').set_index('article_id'))
    mix['is_primary'] = mix['subtopic'].to_numpy() == primary.loc[mix['article_id'], 'subtopic'].to_numpy()
    mix['primary_n_chunks'] = primary.loc[mix['article_id'], 'n_chunks'].to_numpy()
    return mix


def article_subtopics(chunks, review):
    """subtopic_mix plus outlier share and review tags; also returns the number of unassigned articles."""
    mix = subtopic_mix(chunks)
    outlier_share = chunks.groupby('article_id')['subtopic'].apply(lambda s: (s == -1).mean())
    mix['outlier_share'] = outlier_share.loc[mix['article_id']].round(3).to_numpy()
    tags = review[['topic', 'subtopic', 'status', 'label', 'relevance', 'scope', 'scope_country']].rename(
        columns={'status': 'review_status'})
    mix = mix.merge(tags.astype({'subtopic': int}), on=['topic', 'subtopic'], how='left')
    cols = ['article_id', 'topic', 'publisher', 'subtopic', 'label', 'n_chunks', 'weight', 'is_primary',
            'primary_n_chunks', 'outlier_share', 'review_status', 'relevance', 'scope', 'scope_country']
    unassigned = chunks['article_id'].nunique() - mix['article_id'].nunique()
    return mix[cols].sort_values(['topic', 'article_id', 'weight'], ascending=[True, True, False]), unassigned


def chunk_subtopics(chunks, review):
    """One row per chunk with its subtopic (-1 = outlier) and review tags.

    Lets a consumer keep only passages of accepted subtopics, e.g. entity mentions by chunk_id.
    """
    tags = review[['topic', 'subtopic', 'label', 'relevance', 'scope', 'scope_country']].astype({'subtopic': int})
    out = chunks[['chunk_id', 'article_id', 'topic', 'publisher', 'subtopic']].merge(
        tags, on=['topic', 'subtopic'], how='left')
    return out.sort_values('chunk_id').reset_index(drop=True)
