"""NewsPrism: read-only Streamlit dashboard over the exported NLP results.

The app runs no models. It only reads files produced by the notebooks and
src/ modules, joined on article_id. Run from the repository root:

    streamlit run app/app.py
"""
import json
import re
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
FRAMING = ROOT / "notebooks" / "05_framing_summarisation" / "Results"
SENTIMENT = ROOT / "notebooks" / "03_sentiment_language" / "Results" / "sentiment_analysis_results.csv"
STANCE_LABELS = ROOT / "notebooks" / "05_framing_summarisation" / "stance_labels.csv"
ENTITIES = ROOT / "data" / "enriched" / "topics_entities" / "article_entities.csv"
CORPUS = ROOT / "data" / "processed" / "FINAL_newsprism_corpus.csv"

TOPICS = {"immigration": "Immigration policy", "housing": "Housing", "clean_energy": "Clean energy"}
GROUPS = ["left", "centre", "right"]
GROUP_LABEL = {"left": "Left", "centre": "Centre / moderate", "right": "Right", "unlabelled": "Not labelled"}
GROUP_COLOUR = {"left": "#619B8A", "centre": "#7699D4", "right": "#917C78"}
FRAME_LABEL = {
    "conflict": "Conflict",
    "human_interest": "Human impact",
    "economic_consequences": "Economic consequences",
    "responsibility": "Responsibility",
    "morality": "Morality",
    "security": "Security & protection",
    "policy_process": "Policy & process",
    "other": "Other",
}
FRAME_COLOUR = {
    "conflict": "#B5473A",
    "human_interest": "#D98C3F",
    "economic_consequences": "#C9A227",
    "responsibility": "#7699D4",
    "morality": "#8A6FB0",
    "security": "#5C6B80",
    "policy_process": "#619B8A",
    "other": "#C9CED6",
}
SENT_COLOUR = {"Negative": "#B5473A", "Neutral": "#7699D4", "Positive": "#3E8E5E"}
THIN_LIMIT = 5

# Source type is not in the data. This grouping is our own classification of the seven publishers.
SOURCE_TYPES = {
    "All online news": None,
    "National news": ["Sydney Morning Herald", "The Age"],
    "Digital publications": ["The Guardian Australia", "News.com.au"],
    "Broadcast news": ["ABC News", "SBS News", "7NEWS"],
}
# Days back from the latest article, inclusive. 0 = the latest day only.
TIME_RANGES = {"Past 24 hours": 0, "Past 7 days": 6, "Past 1 month": 29, "Past 3 months": 91}
REGIONS = ["Australia", "International", "Both Australia and International"]
LITERACY = ["All levels", "Beginner", "Intermediate", "Advanced"]

CSS = """
<style>
.stApp { background: #F5F5F5; }
.block-container { padding-top: 1.2rem; max-width: 1250px; }
h1, h2, h3, h4 { color: #003566; }
.np-banner {
    background: linear-gradient(90deg, #003566 0%, #7699D4 100%);
    border-radius: 14px; padding: 1.3rem 1.7rem; color: #FFFFFF; margin-bottom: 1rem;
}
.np-banner h1 { color: #FFFFFF; margin: 0; font-size: 2.3rem; }
.np-banner p { margin: 0.2rem 0 0 0; font-size: 1.05rem; opacity: 0.95; }
.np-card {
    background: #FFFFFF; border-radius: 14px; padding: 1.1rem 1.3rem;
    box-shadow: 0 1px 4px rgba(34, 52, 74, 0.08); margin-bottom: 1rem; height: 100%;
}
.np-label { font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase;
    color: #5C6B80; font-weight: 600; margin-bottom: 0.3rem; }
.np-title { font-size: 1.05rem; font-weight: 700; color: #003566; margin: 0.2rem 0 0.4rem 0; }
.np-big { font-size: 1.7rem; font-weight: 700; color: #003566; line-height: 1.1; }
.np-small { font-size: 0.8rem; color: #5C6B80; }
.np-head { border-radius: 14px 14px 0 0; padding: 0.7rem 1rem; color: #FFFFFF;
    font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; font-size: 0.8rem; }
.np-pill { display: inline-block; background: #E8EEF9; color: #003566; border-radius: 999px;
    padding: 0.2rem 0.65rem; margin: 0.15rem 0.2rem 0.15rem 0; font-size: 0.85rem; }
.np-quote { background: #F5F7F9; border-left: 3px solid #7699D4; border-radius: 6px;
    padding: 0.5rem 0.7rem; margin: 0.4rem 0; color: #3A4A5E; font-style: italic; font-size: 0.9rem; }
.np-note { font-size: 0.82rem; color: #5C6B80; }
.np-footer { text-align: center; color: #5C6B80; font-size: 0.8rem; margin-top: 1.5rem; }
</style>
"""


def _syllables(word):
    word = "".join(ch for ch in word.lower() if ch.isalpha())
    if not word:
        return 0
    groups = len(re.findall(r"[aeiouy]+", word))
    if word.endswith("e") and groups > 1 and not word.endswith("le"):
        groups -= 1
    return max(groups, 1)


def flesch(text):
    """Flesch Reading Ease, approximate (vowel-group syllable count). Higher = easier to read."""
    if not isinstance(text, str):
        return None
    sentences = [x for x in re.split(r"[.!?]+", text) if re.search(r"[A-Za-z]", x)]
    words = re.findall(r"[A-Za-z']+", text)
    if not sentences or not words:
        return None
    syllables = sum(_syllables(w) for w in words)
    return 206.835 - 1.015 * len(words) / len(sentences) - 84.6 * syllables / len(words)


@st.cache_data
def load_articles():
    articles = pd.read_csv(FRAMING / "articles.csv")
    articles["date"] = pd.to_datetime(articles["date"])
    stances = pd.read_csv(STANCE_LABELS)
    articles = articles.merge(stances, on="article_id", how="left")
    articles["stance"] = articles["label"].fillna("unlabelled")

    articles["source_type"] = "Other"
    for label, names in SOURCE_TYPES.items():
        if names:
            articles.loc[articles["publisher"].isin(names), "source_type"] = label

    corpus = pd.read_csv(CORPUS, usecols=["article_id", "clean_text"])
    articles = articles.merge(corpus, on="article_id", how="left")
    articles["reading_ease"] = articles["clean_text"].map(flesch)
    articles["literacy"] = pd.cut(
        articles["reading_ease"], [-1e9, 40, 60, 1e9], labels=["Advanced", "Intermediate", "Beginner"], right=False
    ).astype(str)

    sent = pd.read_csv(SENTIMENT)
    sent["sentiment"] = (sent["roberta_positive"] - sent["roberta_negative"]) / 100  # scale -1 to +1
    articles = articles.merge(
        sent[["article_id", "sentiment", "article_roberta_label"]], on="article_id", how="left"
    )
    return articles


@st.cache_data
def load_evidence():
    return pd.read_csv(FRAMING / "article_evidence.csv")


@st.cache_data
def load_json(name):
    path = FRAMING / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@st.cache_data
def load_sentiment():
    return pd.read_csv(SENTIMENT)


@st.cache_data
def load_entities():
    """Entity table from src/topics_entities. Not in the repo; None if the file is missing."""
    return pd.read_csv(ENTITIES) if ENTITIES.exists() else None


def topic_outlets(topic):
    return load_json("topic_outlets.json").get(topic, {})


def topic_stances(topic):
    return load_json("topic_stances.json").get(topic, {})


def model_note(text="Model-generated text. Check the source article before relying on it."):
    st.markdown(f"<div class='np-note'>{text}</div>", unsafe_allow_html=True)


def sentiment_colour(score):
    return SENT_COLOUR[sentiment_word(score)]


def sentiment_word(score):
    if score > 0.1:
        return "Positive"
    if score < -0.1:
        return "Negative"
    return "Neutral"


# ---------------------------------------------------------------- sidebar

def sidebar(articles):
    st.sidebar.markdown("## Filters")
    topic = st.sidebar.selectbox("Topic", list(TOPICS), format_func=TOPICS.get)
    scope = articles[articles["topic"] == topic]

    time_range = st.sidebar.selectbox(
        "Time range",
        list(TIME_RANGES),
        index=3,
        help="Counted back from the latest article in the corpus (30 September 2026), not from today. "
             "Article dates are day-only, so 'Past 24 hours' shows the final day.",
    )
    latest = articles["date"].max()
    scope = scope[scope["date"] >= latest - pd.Timedelta(days=TIME_RANGES[time_range])]

    source = st.sidebar.selectbox(
        "Source type",
        list(SOURCE_TYPES),
        help="Our own grouping of the seven publishers. The data does not record source type.",
    )
    if SOURCE_TYPES[source]:
        scope = scope[scope["publisher"].isin(SOURCE_TYPES[source])]

    region = st.sidebar.selectbox("Region", REGIONS, index=0, help="The report's analysis uses Australian outlets only.")
    if region == "Australia":
        scope = scope[scope["used_in_topic_views"]]
    elif region == "International":
        scope = scope[~scope["used_in_topic_views"]]

    literacy = st.sidebar.selectbox(
        "Literacy level",
        LITERACY,
        help="Reading difficulty of the article text (Flesch Reading Ease, approximate). "
             "It measures the text, not the reader.",
    )
    if literacy != "All levels":
        scope = scope[scope["literacy"] == literacy]

    publishers = sorted(scope["publisher"].dropna().unique())
    chosen = st.sidebar.multiselect("Outlets", publishers, default=publishers)

    leaning_options = GROUPS + ["unlabelled"]
    leaning = st.sidebar.multiselect(
        "Political leaning",
        leaning_options,
        default=leaning_options,
        format_func=GROUP_LABEL.get,
        help="Model-assigned, three classes. Not a judgement of any article.",
    )

    filtered = scope[scope["publisher"].isin(chosen) & scope["stance"].isin(leaning)]
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Showing {len(filtered)} of {len(articles[articles['topic'] == topic])} articles for this topic.")
    return topic, filtered


# ---------------------------------------------------------------- shared pieces

def topic_banner(topic, df):
    st.markdown(
        """
<div class='np-banner'>
  <h1>NewsPrism</h1>
  <p>See the news from every light</p>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
<div class='np-card'>
  <div class='np-label'>Topic</div>
  <div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap'>
    <div class='np-title' style='font-size:1.5rem;margin:0'>{TOPICS[topic]}</div>
    <div style='color:#003566'><b>{len(df)}</b> articles analysed across <b>{df['publisher'].nunique()}</b> publishers</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def spectrum_html(df):
    labelled = df[df["stance"].isin(GROUPS)]
    if labelled.empty:
        return "<div class='np-card'>No labelled articles in this selection.</div>"
    counts = labelled["stance"].value_counts().reindex(GROUPS).fillna(0)
    share = counts / counts.sum() * 100
    segments = "".join(
        f"<div style='width:{share[g]:.1f}%;background:{GROUP_COLOUR[g]}'></div>" for g in GROUPS if share[g] > 0
    )
    labels = "".join(
        f"<div style='flex:1;text-align:center'><div class='np-big'>{share[g]:.0f}%</div>"
        f"<div class='np-small'>{GROUP_LABEL[g]} ({int(counts[g])} articles)</div></div>"
        for g in GROUPS
    )
    return (
        "<div class='np-card'>"
        "<div class='np-label'>Coverage across the political spectrum</div>"
        f"<div style='display:flex;height:16px;border-radius:8px;overflow:hidden;margin:0.6rem 0 0.9rem 0'>{segments}</div>"
        f"<div style='display:flex'>{labels}</div>"
        "<div class='np-note' style='margin-top:0.6rem'>Share of labelled articles. Leaning is a model label, not a verdict.</div>"
        "</div>"
    )


def group_stats(df):
    rows = {}
    for g in GROUPS:
        sub = df[df["stance"] == g]
        rows[g] = {"n": len(sub), "sentiment": sub["sentiment"].mean() if len(sub) else None}
    return rows


# ---------------------------------------------------------------- tabs

def tab_overview(topic, df):
    st.markdown(spectrum_html(df), unsafe_allow_html=True)

    st.markdown("### Coverage at a glance")
    stats = group_stats(df)
    cols = st.columns(3)
    for col, g in zip(cols, GROUPS):
        s = stats[g]
        with col:
            if s["n"] < THIN_LIMIT:
                body = f"<div class='np-label'>Sentiment</div><div class='np-note'>Too few articles to compare (fewer than {THIN_LIMIT}).</div>"
            else:
                body = (
                    f"<div class='np-label'>Average sentiment</div>"
                    f"<div class='np-big' style='color:{sentiment_colour(s['sentiment'])}'>{s['sentiment']:+.2f}</div>"
                    f"<div class='np-small' style='color:{sentiment_colour(s['sentiment'])}'>{sentiment_word(s['sentiment'])} · {s['n']} articles</div>"
                )
            st.markdown(
                f"<div class='np-head' style='background:{GROUP_COLOUR[g]}'>{GROUP_LABEL[g]}</div>"
                f"<div class='np-card' style='border-radius:0 0 14px 14px'>{body}</div>",
                unsafe_allow_html=True,
            )
    st.markdown(
        "<div class='np-note'>Average of positive minus negative probability, from −1 (negative) to +1 (positive). "
        "It measures tone, not accuracy or stance.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("### Where the framing diverges")
    usable = [g for g in GROUPS if stats[g]["n"] >= THIN_LIMIT]
    sub = df[df["stance"].isin(usable)]
    if sub.empty:
        st.info("No group has enough articles (5 or more) to compare in this selection.")
    else:
        share = pd.crosstab(sub["dominant_frame"], sub["stance"], normalize="columns") * 100
        share = share.reindex(columns=usable).fillna(0)
        share.index = [FRAME_LABEL.get(i, i) for i in share.index]
        share = share.sort_index()
        st.dataframe(
            share.round(0),
            column_config={
                g: st.column_config.ProgressColumn(GROUP_LABEL[g], min_value=0, max_value=100, format="%d%%")
                for g in usable
            },
            width="stretch",
        )
        st.markdown(
            "<div class='np-note'>Share of each group's articles that the model labelled with each frame. "
            "The 'Security' frame is known to be over-assigned, so read it with care.</div>",
            unsafe_allow_html=True,
        )

    st.markdown("### Shared entities")
    ents = load_entities()
    if ents is None:
        st.info("Entity results are not yet available, so shared entities cannot be shown.")
    else:
        ents = ents[ents["article_id"].isin(df["article_id"])]
        top = ents.groupby("entity")["article_id"].nunique().sort_values(ascending=False).head(12)
        st.markdown("".join(f"<span class='np-pill'>{e} · {n}</span>" for e, n in top.items()), unsafe_allow_html=True)
        st.markdown("<div class='np-note'>Number of articles mentioning each entity.</div>", unsafe_allow_html=True)

    with st.expander(f"All {len(df)} articles in this selection"):
        table = df[["publisher", "title", "date", "stance", "dominant_frame", "article_type"]].copy()
        table["date"] = table["date"].dt.date
        table["stance"] = table["stance"].map(GROUP_LABEL)
        table["dominant_frame"] = table["dominant_frame"].map(FRAME_LABEL).fillna(table["dominant_frame"])
        st.dataframe(table, hide_index=True, width="stretch")


def tab_perspectives(topic, df, evidence):
    st.markdown("### How the same story is framed")
    st.markdown(
        "<div class='np-note'>Each column shows one example article from that group, chosen as the one with the most "
        "exact key-language quotes. It is an illustration, not the whole group.</div>",
        unsafe_allow_html=True,
    )
    st.write("")
    topic_data = topic_stances(topic)
    summaries = {g.get("stance"): g for g in topic_data.get("stance_summaries", [])}
    ents = load_entities()
    cols = st.columns(3)
    for col, g in zip(cols, GROUPS):
        sub = df[df["stance"] == g]
        with col:
            st.markdown(
                f"<div class='np-head' style='background:{GROUP_COLOUR[g]}'>{GROUP_LABEL[g]} perspective</div>",
                unsafe_allow_html=True,
            )
            if len(sub) < THIN_LIMIT:
                st.markdown(
                    f"<div class='np-card' style='border-radius:0 0 14px 14px'>"
                    f"<div class='np-note'>Too few articles to show a comparison (fewer than {THIN_LIMIT}).</div></div>",
                    unsafe_allow_html=True,
                )
                continue
            rep = sub.assign(_q=sub["quotes_exact"].fillna(0)).sort_values(["_q", "date"], ascending=False).iloc[0]
            frame = sub["dominant_frame"].value_counts()
            top_frame = FRAME_LABEL.get(frame.index[0], frame.index[0])
            share = frame.iloc[0] / len(sub) * 100
            kl = evidence[
                (evidence["article_id"] == rep["article_id"])
                & (evidence["kind"] == "key_language")
                & (evidence["quote_status"] == "exact")
            ]["quote"].head(3).tolist()
            pills = "".join(f"<div class='np-quote'>“{q}”</div>" for q in kl) or "<div class='np-note'>No checked key language.</div>"
            if ents is not None:
                e_top = ents[ents["article_id"] == rep["article_id"]]["entity"].head(4).tolist()
                ent_html = "".join(f"<span class='np-pill'>{e}</span>" for e in e_top) or "<span class='np-note'>None found</span>"
            else:
                ent_html = "<span class='np-note'>Entity results not yet available</span>"
            st.markdown(
                f"""<div class='np-card' style='border-radius:0 0 14px 14px'>
<div class='np-small'>{rep['publisher']}</div>
<div class='np-title'>{rep['title']}</div>
<div style='display:flex;gap:0.6rem;margin:0.6rem 0'>
  <div style='flex:1;background:#F5F7F9;border-radius:10px;padding:0.5rem 0.7rem'>
    <div class='np-label'>Sentiment</div>
    <div class='np-big' style='font-size:1.2rem;color:{sentiment_colour(rep['sentiment'])}'>{rep['sentiment']:+.2f}</div>
    <div class='np-small'>{sentiment_word(rep['sentiment'])}</div></div>
  <div style='flex:1;background:#F5F7F9;border-radius:10px;padding:0.5rem 0.7rem'>
    <div class='np-label'>Group's top frame</div>
    <div class='np-big' style='font-size:1.2rem'>{top_frame}</div>
    <div class='np-small'>{share:.0f}% of {len(sub)} articles</div></div>
</div>
<div class='np-label'>Top entities</div>{ent_html}
<div class='np-label' style='margin-top:0.8rem'>Key language (exact quotes)</div>{pills}
</div>""",
                unsafe_allow_html=True,
            )
            if g in summaries:
                with st.expander("Group summary (model-generated)"):
                    st.write(summaries[g].get("summary", ""))

    st.markdown("### What they agree on and where they differ")
    overview = topic_data.get("overview", "")
    if overview:
        with st.expander("Model-written comparison of the groups"):
            st.write(overview)
            model_note("Model-generated. Check against the articles before quoting it.")


def tab_outlets(topic, df):
    st.markdown("### How each outlet frames the topic")
    counts = pd.crosstab(df["publisher"], df["dominant_frame"])
    totals = counts.sum(axis=1)
    thin = totals[totals < THIN_LIMIT].index.tolist()
    keep = totals[totals >= THIN_LIMIT].index
    if thin:
        st.warning(f"Too few articles to compare reliably (fewer than {THIN_LIMIT}): {', '.join(sorted(thin))}. Hidden from the chart.")
    if len(keep) == 0:
        st.info("No outlet has enough articles to compare in this selection.")
        return

    share = counts.loc[keep].div(totals.loc[keep], axis=0) * 100
    share = share.reindex(columns=[f for f in FRAME_LABEL if f in share.columns]).fillna(0)
    chart = share.rename(columns=FRAME_LABEL)
    st.bar_chart(chart, color=[FRAME_COLOUR[f] for f in share.columns], horizontal=True)
    st.markdown(
        "<div class='np-note'>Each bar sums to 100% for one outlet. Shows the share of that outlet's articles "
        "given each frame.</div>",
        unsafe_allow_html=True,
    )
    with st.expander("Article counts behind the chart"):
        st.dataframe(totals.loc[keep].rename("articles").reset_index(), hide_index=True, width="stretch")

    data = topic_outlets(topic)
    shown = set(keep)
    patterns = [p for p in data.get("outlet_patterns", []) if p.get("publisher") in shown]
    diffs = data.get("differences", [])
    if patterns or diffs:
        st.markdown("### Patterns and differences")
        for p in patterns:
            st.markdown(f"**{p['publisher']}**: {p.get('pattern', '')}")
        for d in diffs:
            st.write(f"- {d.get('description', '')}")
        model_note()


def tab_stances(topic, df):
    st.markdown("### What each political-leaning group reports")
    data = topic_stances(topic)
    if not data:
        st.info("No stance results for this topic.")
        return
    counts = df["stance"].value_counts()
    cols = st.columns(3)
    for col, g in zip(cols, GROUPS):
        with col:
            st.metric(GROUP_LABEL[g], int(counts.get(g, 0)), help="Articles in this group, after your filters")
    st.write("")
    summaries = {g.get("stance"): g for g in data.get("stance_summaries", [])}
    for g in GROUPS:
        if g not in summaries:
            continue
        n = int(counts.get(g, 0))
        group = summaries[g]
        st.markdown(
            f"<div class='np-head' style='background:{GROUP_COLOUR[g]}'>{GROUP_LABEL[g]} group</div>",
            unsafe_allow_html=True,
        )
        with st.container(border=True):
            if n < THIN_LIMIT:
                st.caption(f"Too few articles to show this group (fewer than {THIN_LIMIT}).")
                continue
            st.write(group.get("summary", ""))
            verified = [c for c in group.get("key_claims", []) if c.get("human_review") == "yes"]
            if verified:
                st.markdown("**Key claims checked by a person**")
                for claim in verified:
                    st.write(f"- {claim.get('claim', '')}")
            else:
                st.caption("No key claims for this group have passed human review, so none are shown.")
    model_note("Stance groups use model-generated labels. Only key claims checked by a person are shown.")


def tab_sentiment(topic, df):
    st.markdown("### Tone of articles by outlet")
    counts = df["publisher"].value_counts()
    keep = counts[counts >= THIN_LIMIT].index
    thin = counts[counts < THIN_LIMIT].index.tolist()
    if thin:
        st.warning(f"Too few articles to compare reliably (fewer than {THIN_LIMIT}): {', '.join(sorted(thin))}.")
    sub = df[df["publisher"].isin(keep)]
    if sub.empty:
        st.info("No outlet has enough articles to compare in this selection.")
        return

    avg = sub.groupby("publisher")["sentiment"].mean().reset_index()
    avg["label"] = avg["sentiment"].map(sentiment_word)
    chart = (
        alt.Chart(avg)
        .mark_bar(cornerRadius=3)
        .encode(
            y=alt.Y("publisher:N", sort=alt.EncodingSortField("sentiment", order="ascending"), title=None),
            x=alt.X("sentiment:Q", scale=alt.Scale(domain=[-1, 1]), title="Average sentiment (−1 negative, +1 positive)"),
            color=alt.Color(
                "label:N",
                scale=alt.Scale(domain=list(SENT_COLOUR), range=list(SENT_COLOUR.values())),
                legend=None,
            ),
            tooltip=["publisher", alt.Tooltip("sentiment:Q", format="+.2f"), "label"],
        )
        .properties(height=max(120, 36 * len(avg)))
    )
    st.altair_chart(chart, width="stretch")
    st.markdown(
        "<div class='np-note'>Average of positive minus negative probability, from −1 to +1.</div>",
        unsafe_allow_html=True,
    )

    with st.expander("Share of articles by sentiment label"):
        mix = pd.crosstab(sub["publisher"], sub["article_roberta_label"], normalize="index") * 100
        st.dataframe(mix.round(0), width="stretch")

    st.markdown(
        "<div class='np-note'>Sentiment is tone, not political stance or accuracy. The model was checked on 39 "
        "hand-labelled articles and scored 46% accuracy, so treat small differences as unreliable.</div>",
        unsafe_allow_html=True,
    )


def tab_entities(df):
    st.markdown("### Who and what each outlet names")
    ents = load_entities()
    if ents is None:
        st.info(
            "Entity results are not yet available. They come from src/topics_entities. "
            "Once article_entities.csv is in data/enriched/topics_entities/, this tab fills in."
        )
        return
    ents = ents[ents["article_id"].isin(df["article_id"])]
    top = ents.groupby(["entity", "type"])["article_id"].nunique().sort_values(ascending=False).head(25)
    st.caption("Number of articles mentioning each entity.")
    st.dataframe(top.rename("articles").reset_index(), hide_index=True, width="stretch")


def tab_article(df, evidence):
    st.markdown("### Read one article")
    if df.empty:
        st.info("No articles match your filters.")
        return
    labels = {row.article_id: f"{row.publisher}: {row.title}" for row in df.itertuples()}
    chosen = st.selectbox("Article", list(labels), format_func=labels.get)
    row = df[df["article_id"] == chosen].iloc[0]

    st.markdown(f"#### {row['title']}")
    leaning = GROUP_LABEL.get(row["stance"], row["stance"])
    st.markdown(
        f"<span class='np-pill'>{row['publisher']}</span>"
        f"<span class='np-pill'>{row['date'].date()}</span>"
        f"<span class='np-pill'>Leaning (model): {leaning}</span>"
        f"<span class='np-pill'>Frame (model): {FRAME_LABEL.get(row['dominant_frame'], row['dominant_frame'])}</span>",
        unsafe_allow_html=True,
    )
    st.link_button("Open original article", row["url"])
    st.write(row["summary"])
    if isinstance(row.get("frame_reason"), str):
        st.caption(row["frame_reason"])

    ev = evidence[(evidence["article_id"] == chosen) & (evidence["quote_status"] == "exact")]
    points = ev[ev["kind"] == "key_point"]
    if not points.empty:
        st.markdown("**Key points with quotations**")
        for p in points.itertuples():
            st.markdown(f"- {p.text}")
            st.markdown(f"<div class='np-quote'>“{p.quote}”</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='np-note'>A quotation is checked only to confirm the words appear in the article. "
            "It is not checked to confirm it supports the point. Read the source before relying on it.</div>",
            unsafe_allow_html=True,
        )
    model_note("Summary, frame and leaning are model-generated.")


def tab_about():
    st.markdown("### About NewsPrism and its limits")
    st.markdown(
        """
NewsPrism compares how Australian outlets cover immigration, housing and clean energy.
It shows where coverage differs. It does not check whether articles are true.

- **Corpus:** 253 articles from Google News RSS, 1 July to 30 September 2026. Seven publishers.
- **Region:** the default view uses Australian outlets only, as in the report.
- **Political leaning:** a three-class model (left, centre, right). Human labels used five levels, collapsed to three.
- **Sentiment:** RoBERTa, checked on 39 hand-labelled articles (46% accuracy against a 54% majority baseline).
- **Summaries and framing:** generated by local language models and not checked for every article.
- **Quotations:** checked to appear in the article. Not checked for whether they support the point.
- **Stance key claims:** only claims checked by a person are shown.
- **Source type:** our own grouping of the seven publishers, not recorded in the data.
- **Literacy level:** approximate Flesch Reading Ease of the article text. It measures the text, not the reader.
- **Time range:** counted back from 30 September 2026, the latest article, because the corpus stops there.
        """
    )


# ---------------------------------------------------------------- main

def main():
    st.set_page_config(page_title="NewsPrism", page_icon="🔆", layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)

    articles = load_articles()
    evidence = load_evidence()
    topic, df = sidebar(articles)

    topic_banner(topic, df)
    if df.empty:
        st.info("No articles match your filters. Widen the date range, region, outlets or leaning in the sidebar.")
        return

    tabs = st.tabs(["Overview", "Perspectives", "Outlets", "Stances", "Sentiment", "Entities", "Article", "About"])
    with tabs[0]:
        tab_overview(topic, df)
    with tabs[1]:
        tab_perspectives(topic, df, evidence)
    with tabs[2]:
        tab_outlets(topic, df)
    with tabs[3]:
        tab_stances(topic, df)
    with tabs[4]:
        tab_sentiment(topic, df)
    with tabs[5]:
        tab_entities(df)
    with tabs[6]:
        tab_article(df, evidence)
    with tabs[7]:
        tab_about()

    st.markdown("<div class='np-footer'>NewsPrism · research prototype · model outputs labelled throughout</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
