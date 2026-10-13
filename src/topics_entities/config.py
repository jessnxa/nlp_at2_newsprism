"""Final settings of the topic and entity pipeline. Nothing here is meant to be tuned by callers;
every value was chosen by the evidence in the project's design record.
"""
import re

REQUIRED_COLUMNS = ['article_id', 'topic', 'title', 'publisher', 'clean_text']

# entity recognition: the transformer model, chosen by the accuracy check (F1 0.931 vs 0.769 for lg)
NER_MODEL = 'en_core_web_trf'
NER_DISABLE = ['tagger', 'parser', 'attribute_ruler', 'lemmatizer']  # only NER is needed
# keyword lemmatisation only (topic keywords); small model is enough for lemmas
LEMMA_MODEL = 'en_core_web_sm'

SEED = 42
EMBED_MODEL = 'BAAI/bge-small-en-v1.5'  # chosen by a 45-run configuration grid

# chunking: sentence-aware ~100-word chunks, no overlap
CHUNK_TARGET, CHUNK_CAP, CHUNK_MIN_TAIL = 100, 190, 50

# clustering
EXPECTED_SUBTOPICS = (4, 10)    # sanity range per broad topic, not a target
MIN_ARTICLES_PER_SUBTOPIC = 5
MAX_TOP_ARTICLE_SHARE = 0.5     # one article may not supply more than half a subtopic
MAX_CLUSTER_SHARE = 0.35        # larger clusters are candidates for splitting
MAX_CROSS_SHARE = 0.5           # mostly cross-topic clusters are flagged for review

# keywords: used to help a human name subtopics, never as the name itself
N_KEYWORDS, N_KEYWORD_CANDIDATES = 10, 30
MMR_DIVERSITY = 0.1             # 0.3 dropped defining words such as 'rent' and 'prices'
FIRST_TARGET_YEAR = 2027        # corpus is Jul-Sep 2026; later years are targets ("by 2030")
GENERIC_KEYWORDS = {'australia', 'australian', 'australians'}
NEWS_STOPWORDS = {
    'said', 'says', 'say', 'saying', 'told', 'tell', 'mr', 'ms', 'mrs', 'dr', 'per', 'cent',
    'year', 'years', 'according', 'also', 'would', 'could', 'like', 'just', 'new', 'one', 'two',
    'people', 'including', 'last', 'week', 'time',
}
# compound variants found in this corpus, mapped to one form
SPELLING_VARIANTS = {
    r'\bdata ?centres?\b': 'data centre',
    r'\bhome ?ownership\b': 'home ownership',
    r'\bhome ?owners?\b': 'home owner',
    r'\bhome ?buyers?\b': 'home buyer',
    r'\breal ?estate\b': 'real estate',
    r'\bwind ?farms?\b': 'wind farm',
    r'\bcoal ?mines?\b': 'coal mine',
    r'\bcoal ?mining\b': 'coal mining',
    r'\bgas ?fields?\b': 'gas field',
}
NUMBER = re.compile(r'\b\d[\d,.]*%?')

# review
HUMAN_COLS = ['label', 'relevance', 'scope', 'scope_country', 'reason']
N_EXAMPLES = 3
MIN_SUGGEST_OVERLAP = 0.3       # below this, a previous label is not suggested
