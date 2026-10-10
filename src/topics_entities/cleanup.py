"""Strip site boilerplate (newsletter/subscribe promos, 7NEWS app promos) from article text.

Run on the full corpus: repeated-sentence detection depends on all articles together.
Only the in-memory `model_text` is cleaned; the caller's data is never modified.
"""
import collections
import re

SPLIT = re.compile(r'(?<=[.!?])\s+')
PROMO = re.compile(r'(?i)\b(newsletter|sign up|subscribe|download (our|the) app|inbox)\b')
INLINE = [
    r'WATCH THE VIDEO ABOVE:.*?(?=Know the news|$)',
    r'Know the news with the 7NEWS app:\s*Download today',
    r'Stream free on\s*\S*\s*$',
    r'[^.“”"]*\bnewsletter\b[^.]*\.',
]


def norm(s):
    return re.sub(r'^[\W_]+|[\W_]+$', '', s).lower()


def strip_boilerplate(texts, min_docs=3, promo_max_words=30):
    """Remove sentences repeated across >= min_docs articles, short promo sentences, and inline site promos."""
    sents = [SPLIT.split(t) for t in texts]
    freq = collections.Counter(k for ss in sents for k in {norm(s) for s in ss})
    out = []
    for ss in sents:
        keep = [s for s in ss
                if not (freq[norm(s)] >= min_docs and len(s.split()) > 3)
                and not (PROMO.search(s) and len(s.split()) <= promo_max_words)]
        t = ' '.join(keep)
        for p in INLINE:
            t = re.sub(p, ' ', t)
        out.append(re.sub(r'\s+', ' ', t).strip())
    return out
