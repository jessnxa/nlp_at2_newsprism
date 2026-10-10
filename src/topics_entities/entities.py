"""Named entities (people, organisations, places, groups) per article and per passage.

NER runs on the same chunks as topics.py (same chunk_id), plus each title separately, so a consumer
can count entities per article or only in passages of accepted subtopics. Rules are exact-match and
fixed (aliases, type overrides, self-mention removal); each was verified in context on the corpus.
"""
import re

import pandas as pd

from .config import NER_DISABLE, NER_MODEL
from .topics import build_chunks

# spaCy label -> our type; everything else (DATE, MONEY, CARDINAL, ...) is dropped
KEEP_LABELS = {'PERSON': 'PERSON', 'ORG': 'ORG', 'GPE': 'PLACE', 'LOC': 'PLACE', 'NORP': 'NORP'}

# each outlet's own names; a mention is the outlet's own if it equals one of these or starts with
# one followed by a space ("SBS News Podcasts"), except names that are also ordinary words
PUBLISHER_NAMES = {
    'The Guardian Australia': ['Guardian', 'The Guardian', 'Guardian Australia', 'The Guardian Australia'],
    'ABC News': ['ABC', 'ABC News', 'ABC iview', 'ABC Listen', 'Australian Broadcasting Corporation'],
    'SBS News': ['SBS', 'SBS News'],
    'Sydney Morning Herald': ['Sydney Morning Herald', 'The Sydney Morning Herald', 'SMH', 'Herald'],
    'The Age': ['The Age', 'Age'],
    'News.com.au': ['News.com.au', 'news.com.au'],
    '7NEWS': ['7NEWS', '7News', '7NEWS Spotlight', 'Seven', 'Seven Network'],
}
NO_PREFIX_MATCH = {'age', 'seven', 'herald'}  # 'Age Pension' in The Age is not The Age
# unnamed institutions the annotation guidelines exclude (generic references, not names)
GENERIC_ENTITIES = {'government', 'federal government', 'state government', 'commonwealth government',
                    'commonwealth', 'opposition', 'cabinet'}
# abbreviations of statistics that the model tags as organisations
NON_ENTITIES = {'nom'}  # net overseas migration
TITLE_SUFFIX = re.compile(r'(ABC|SBS|SMH|Guardian|7NEWS|News\.com\.au|theage|The Age|abc\.net|sbs\.com)', re.I)

# names that start with "The"; every other leading "the" is dropped
THE_NAMES = {'The Age', 'The Hague', 'The Guardian', 'The Australian', 'The Sydney Morning Herald',
             'The Guardian Australia'}
HONORIFIC = re.compile(r'^(?:Mr|Mrs|Ms|Miss|Dr|Prof|Professor|Senator|Sen|Premier|Prime Minister|'
                       r'Treasurer|Minister|President|Sir|Dame|Judge|Justice)\.?\s+(?=[A-Z])')

# exact-match aliases for variants found in this corpus. "Labour" (UK party, 42 mentions) is
# deliberately not merged with Labor; "Greens" is ambiguous (Australian / UK) so left as is.
ALIASES = {
    'Labor': 'Australian Labor Party', 'Labor Party': 'Australian Labor Party', 'ALP': 'Australian Labor Party',
    'Liberals': 'Liberal Party', 'Liberal': 'Liberal Party', 'Liberal Party of Australia': 'Liberal Party',
    'Nationals': 'National Party',
    'RBA': 'Reserve Bank of Australia', 'Reserve Bank': 'Reserve Bank of Australia',
    'NSW': 'New South Wales',
    'US': 'United States', 'U.S.': 'United States', 'America': 'United States',
    'UK': 'United Kingdom', 'Britain': 'United Kingdom', 'Great Britain': 'United Kingdom',
    'AEMO': 'Australian Energy Market Operator',
    'ABS': 'Australian Bureau of Statistics',
    'Home Affairs': 'Department of Home Affairs',
    # verified in context: in "One Nation's" spaCy tags "One" as a number and keeps "Nation"
    'Nation': 'One Nation',
}

# verified corrections to the voted type, keyed by resolved entity, each with its reason
TYPE_OVERRIDES = {
    'Darwin': ('PLACE', 'all 21 mentions are the city (house-price stories); spaCy tags PERSON'),
    'New South Wales': ('PLACE', 'state; spaCy tags "NSW" as ORG 60 times'),
    'Australian Labor Party': ('ORG', 'political party'),
    'Liberal Party': ('ORG', 'political party; plural "Liberals" is tagged NORP'),
    'National Party': ('ORG', 'political party; plural "Nationals" is tagged NORP'),
    'Greens': ('ORG', 'political party; tagged NORP'),
    'One Nation': ('ORG', 'political party'),
}
TYPE_PRIORITY = ['PERSON', 'ORG', 'PLACE', 'NORP']  # tie-break for type votes


# --- name normalisation ---------------------------------------------------------

def normalise_name(text):
    """Strip quotes, possessives, a leading 'the' (except real names) and attached honorifics."""
    name = re.sub(r'\s+', ' ', text.replace('’', "'").replace('‘', "'").strip(" '\"“”"))
    name = re.sub(r"'s$", '', name)
    if name.lower().startswith('the ') and name.title() not in {n.title() for n in THE_NAMES}:
        name = name[4:]
    stripped = HONORIFIC.sub('', name)
    return stripped if stripped else name


def clean_title(title):
    """Drop the site suffix: 'X | Section | The Guardian' -> 'X', 'X - SMH.com.au' -> 'X'."""
    title = title.split(' | ')[0]
    head, sep, tail = title.rpartition(' - ')
    return head if sep and TITLE_SUFFIX.search(tail) else title


def is_self_mention(name, publisher):
    own = {normalise_name(n).lower() for n in PUBLISHER_NAMES.get(publisher, [])}
    name = name.lower()
    return name in own or any(name.startswith(o + ' ') for o in own - NO_PREFIX_MATCH)


def singular_group(name, type_):
    """One entity per group: plural NORP merges into the singular ('Australians' -> 'Australian').
    Leaves '-ss' words ('Swiss') and names with their own handling (the Greens party) unchanged."""
    if type_ == 'NORP' and name.endswith('s') and not name.endswith('ss') and name not in TYPE_OVERRIDES:
        return name[:-1]
    return name


def clean_mentions(m):
    """Normalise names, drop the outlet's own name and generic institutions, apply aliases,
    merge plural groups into the singular."""
    m = m.assign(entity=m['surface'].map(normalise_name))
    keep = ((m['entity'] != '') & ~m['entity'].str.lower().isin(GENERIC_ENTITIES | NON_ENTITIES)
            & ~m.apply(lambda r: is_self_mention(r.entity, r.publisher), axis=1))
    m = m[keep].copy()
    m['entity'] = m['entity'].replace(ALIASES)
    m['entity'] = [singular_group(e, t) for e, t in zip(m['entity'], m['type'])]
    return m


def resolve_surnames(mentions):
    """Map a one-word name to the article's full name ending in it, only if exactly one exists.

    Uses the whole article (title + body). Matches the exact last word, so 'Hanson-Young' never
    resolves to 'Hanson'. Returns the resolved names and the list of ambiguous surnames.
    """
    resolved, ambiguous = mentions['entity'].copy(), []
    for article, m in mentions.groupby('article_id'):
        full = m.loc[(m['type'] == 'PERSON') & m['entity'].str.contains(' '), 'entity'].unique()
        by_last = {}
        for name in full:
            by_last.setdefault(name.split()[-1], set()).add(name)
        single = m['entity'].str.fullmatch(r'\S+') & m['type'].isin(['PERSON', 'ORG', 'NORP'])
        for idx, name in m.loc[single, 'entity'].items():
            candidates = by_last.get(name, set())
            if len(candidates) == 1:
                resolved.at[idx] = next(iter(candidates))
            elif len(candidates) > 1:
                ambiguous.append((article, name, sorted(candidates)))
    return resolved, ambiguous


def apply_context_rules(m):
    """Rules that need the word after a mention (next_word):
    - a portfolio word in a job title is not an organisation ('Energy' in 'Energy Minister Chris Bowen',
      'Energy Secretary', 'Energy ministers')
    - a leader's surname before 'government' / 'administration' names that government
      ('Albanese government', 'Trump administration' -> ORG), only when the surname belongs to a
      full PERSON name seen in the same data
    """
    title_words = {'minister', 'ministers', 'secretary', 'secretaries'}
    m = m[~(m['next_word'].str.lower().isin(title_words) & (m['type'] != 'PERSON'))].copy()
    surnames = {n.split()[-1] for n in m.loc[(m['type'] == 'PERSON') & m['entity'].str.contains(' '), 'entity']}
    following = m['next_word'].str.lower()
    gov = following.isin(['government', 'administration']) & m['type'].isin(['PERSON', 'NORP']) & m['entity'].isin(surnames)
    m.loc[gov, 'entity'] = m.loc[gov, 'entity'] + ' ' + following[gov]
    m.loc[gov, 'type'] = 'ORG'
    return m


def resolve_org_short_forms(mentions):
    """Map a one-word organisation to the article's longer organisation name starting with it,
    only if exactly one exists ('Oakes' -> 'Oakes Energy Services')."""
    resolved = mentions['entity'].copy()
    for _, m in mentions.groupby('article_id'):
        longer = m.loc[(m['type'] == 'ORG') & m['entity'].str.contains(' '), 'entity'].unique()
        by_first = {}
        for name in longer:
            by_first.setdefault(name.split()[0], set()).add(name)
        single = m['entity'].str.fullmatch(r'\S+') & (m['type'] == 'ORG')
        for idx, name in m.loc[single, 'entity'].items():
            if len(by_first.get(name, ())) == 1:
                resolved.at[idx] = next(iter(by_first[name]))
    return resolved


def vote_types(mentions):
    """One type per entity: majority within each article, then one vote per article, then overrides."""
    rank = {t: i for i, t in enumerate(TYPE_PRIORITY)}
    per_article = (mentions.groupby(['entity', 'article_id'])['type']
                   .agg(lambda s: min(s.mode(), key=rank.get)).reset_index())
    voted = (per_article.groupby('entity')['type']
             .agg(lambda s: min(s.mode(), key=rank.get)))
    for entity, (type_, _) in TYPE_OVERRIDES.items():
        if entity in voted.index:
            voted[entity] = type_
    return mentions['entity'].map(voted)


# --- extraction -------------------------------------------------------------------

def extract_mentions(corpus, nlp=None):
    """One row per kept entity mention, with normalised, resolved entity and voted type."""
    import spacy
    nlp = nlp or spacy.load(NER_MODEL, disable=NER_DISABLE)
    chunks = build_chunks(corpus)
    titles = corpus.assign(chunk_id=corpus['article_id'] + '_title',
                           text=corpus['title'].map(clean_title))[['article_id', 'topic', 'publisher', 'chunk_id', 'text']]
    units = pd.concat([titles, chunks[['article_id', 'topic', 'publisher', 'chunk_id', 'text']]], ignore_index=True)

    rows = []
    for unit, doc in zip(units.itertuples(index=False), nlp.pipe(units['text'], batch_size=32)):
        for ent in doc.ents:
            if ent.label_ in KEEP_LABELS:
                rows.append({'article_id': unit.article_id, 'topic': unit.topic, 'publisher': unit.publisher,
                             'chunk_id': unit.chunk_id, 'surface': ent.text, 'type': KEEP_LABELS[ent.label_],
                             'next_word': doc[ent.end].text if ent.end < len(doc) else ''})
    m = apply_context_rules(clean_mentions(pd.DataFrame(rows)))
    # vote first so full names spaCy mis-typed still count as PERSON candidates, then re-vote
    m['type'] = vote_types(m)
    m['entity'], ambiguous = resolve_surnames(m)
    m['entity'] = resolve_org_short_forms(m)
    m['type'] = vote_types(m)
    m['in_title'] = m['chunk_id'].str.endswith('_title')
    return m.reset_index(drop=True), ambiguous, chunks


def article_entities(mentions):
    """One row per (article, entity): body mention count, whether it is in the title, spellings seen."""
    g = mentions.groupby(['article_id', 'topic', 'publisher', 'entity', 'type'])
    out = g.agg(n_mentions=('in_title', lambda s: int((~s).sum())), in_title=('in_title', 'any'),
                surface_forms=('surface', lambda s: '; '.join(sorted(set(s))))).reset_index()
    return out.sort_values(['topic', 'article_id', 'n_mentions'], ascending=[True, True, False])


# --- acceptance checks (A1-A5) ----------------------------------------------------

def check_entities(mentions, articles, chunks, corpus, ambiguous=None):
    """Automatic sanity checks on the output; returns {check name: passed}."""
    checks = {
        'A1 only kept types': set(mentions['type']) <= set(KEEP_LABELS.values()),
        'A2 no self-mentions': not mentions.apply(lambda r: is_self_mention(r.entity, r.publisher), axis=1).any(),
        'A4 one type per entity': mentions.groupby('entity')['type'].nunique().max() == 1,
        'A5 article_ids in input': set(mentions['article_id']) <= set(corpus['article_id']),
        'A5 chunk_ids in topics chunking': set(mentions.loc[~mentions['in_title'], 'chunk_id']) <= set(chunks['chunk_id']),
        'A5 >=95% articles have an entity': articles['article_id'].nunique() / corpus['article_id'].nunique() >= 0.95,
    }
    # A3: no article keeps a one-word name when exactly one full name ends in it
    unresolved = False
    for _, m in mentions.groupby('article_id'):
        full = m.loc[(m['type'] == 'PERSON') & m['entity'].str.contains(' '), 'entity'].unique()
        lasts = pd.Series([n.split()[-1] for n in full]).value_counts()
        if set(m.loc[m['type'] == 'PERSON', 'entity']) & set(lasts[lasts == 1].index):
            unresolved = True
            break
    checks['A3 unambiguous surnames resolved'] = not unresolved
    return checks
