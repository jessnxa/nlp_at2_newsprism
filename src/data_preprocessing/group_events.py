import re

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


MAX_DATE_DIFFERENCE_DAYS = 3

TITLE_WEIGHT = 0.65
BODY_WEIGHT = 0.35

SIMILARITY_THRESHOLD = 0.35


TOPIC_PREFIXES = {
    "immigration": "IMM",
    "clean_energy": "ENE",
    "housing": "HOU",
}


def normalise_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def prepare_text(dataframe):
    dataframe = dataframe.copy()

    dataframe["event_title_text"] = dataframe[
        "title"
    ].apply(normalise_text)

    dataframe["event_body_text"] = dataframe[
        "clean_text"
    ].apply(normalise_text)

    dataframe["published_at"] = pd.to_datetime(
        dataframe["published_at"],
        errors="coerce",
        utc=True,
    )

    return dataframe


def calculate_similarity(texts):
    if len(texts) < 2:
        return None

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
    )

    matrix = vectorizer.fit_transform(texts)

    return cosine_similarity(matrix)


def date_difference_days(date_a, date_b):
    if pd.isna(date_a) or pd.isna(date_b):
        return None

    difference = abs(
        date_a - date_b
    )

    return difference.total_seconds() / 86400


def build_similarity_matrix(dataframe):
    titles = dataframe[
        "event_title_text"
    ].tolist()

    bodies = dataframe[
        "event_body_text"
    ].tolist()

    title_similarity = calculate_similarity(
        titles
    )

    body_similarity = calculate_similarity(
        bodies
    )

    if (
        title_similarity is None
        or body_similarity is None
    ):
        return None

    combined_similarity = (
        TITLE_WEIGHT * title_similarity
        + BODY_WEIGHT * body_similarity
    )

    return combined_similarity


def find_candidate_links(
    dataframe,
    similarity_matrix,
):
    links = []

    for i in range(len(dataframe)):
        for j in range(i + 1, len(dataframe)):
            row_i = dataframe.iloc[i]
            row_j = dataframe.iloc[j]

            if row_i["topic"] != row_j["topic"]:
                continue

            date_difference = date_difference_days(
                row_i["published_at"],
                row_j["published_at"],
            )

            if date_difference is None:
                continue

            if (
                date_difference
                > MAX_DATE_DIFFERENCE_DAYS
            ):
                continue

            similarity = float(
                similarity_matrix[i, j]
            )

            if similarity < SIMILARITY_THRESHOLD:
                continue

            links.append(
                {
                    "index_a": i,
                    "index_b": j,
                    "similarity": similarity,
                    "date_difference_days":
                        date_difference,
                }
            )

    return links


def find_connected_components(
    number_of_articles,
    links,
):
    adjacency = {
        index: set()
        for index in range(
            number_of_articles
        )
    }

    for link in links:
        a = link["index_a"]
        b = link["index_b"]

        adjacency[a].add(b)
        adjacency[b].add(a)

    visited = set()
    components = []

    for start in adjacency:
        if start in visited:
            continue

        stack = [start]
        component = []

        while stack:
            current = stack.pop()

            if current in visited:
                continue

            visited.add(current)
            component.append(current)

            stack.extend(
                adjacency[current]
                - visited
            )

        components.append(
            sorted(component)
        )

    return components


def assign_event_ids(dataframe):
    dataframe = prepare_text(
        dataframe
    )

    dataframe["event_id"] = None
    dataframe["event_group_size"] = 1
    dataframe["event_review_status"] = (
        "single_article"
    )

    for topic in dataframe[
        "topic"
    ].dropna().unique():

        topic_mask = (
            dataframe["topic"] == topic
        )

        topic_data = dataframe[
            topic_mask
        ].copy()

        topic_data = topic_data.reset_index()

        if len(topic_data) == 0:
            continue

        if len(topic_data) == 1:
            original_index = topic_data.loc[
                0,
                "index",
            ]

            prefix = TOPIC_PREFIXES.get(
                topic,
                "EVT",
            )

            dataframe.at[
                original_index,
                "event_id",
            ] = f"{prefix}_001"

            continue

        similarity_matrix = (
            build_similarity_matrix(
                topic_data
            )
        )

        links = find_candidate_links(
            topic_data,
            similarity_matrix,
        )

        components = find_connected_components(
            len(topic_data),
            links,
        )

        prefix = TOPIC_PREFIXES.get(
            topic,
            "EVT",
        )

        components = sorted(
            components,
            key=lambda component: (
                topic_data.loc[
                    component,
                    "published_at",
                ]
                .min()
            ),
        )

        for event_number, component in enumerate(
            components,
            start=1,
        ):
            event_id = (
                f"{prefix}_"
                f"{event_number:03d}"
            )

            group_size = len(component)

            for local_index in component:
                original_index = topic_data.loc[
                    local_index,
                    "index",
                ]

                dataframe.at[
                    original_index,
                    "event_id",
                ] = event_id

                dataframe.at[
                    original_index,
                    "event_group_size",
                ] = group_size

                if group_size > 1:
                    dataframe.at[
                        original_index,
                        "event_review_status",
                    ] = "review_group"

    return dataframe


def create_event_summary(dataframe):
    summary = (
        dataframe.groupby(
            [
                "topic",
                "event_id",
            ],
            dropna=False,
        )
        .agg(
            article_count=(
                "title",
                "size",
            ),
            publisher_count=(
                "publisher",
                "nunique",
            ),
            earliest_date=(
                "published_at",
                "min",
            ),
            latest_date=(
                "published_at",
                "max",
            ),
        )
        .reset_index()
    )

    return summary


if __name__ == "__main__":
    print(
        "NewsPrism event grouping module"
    )
    print(
        "Candidate grouping method: "
        "topic + publication date + "
        "TF-IDF text similarity"
    )