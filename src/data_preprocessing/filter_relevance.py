import re

import pandas as pd

from src.data_preprocessing.config import (
    INTERIM_DIR,
    RELEVANCE_CONCEPTS,
)


TEST_INPUT_PATH = (
    INTERIM_DIR / "TEST3_cleaned_articles.csv"
)

TEST_OUTPUT_PATH = (
    INTERIM_DIR / "TEST4_relevance_filtered.csv"
)


def normalise_text(text):
    """
    Normalise text for relevance matching only.

    This does not modify the stored clean_text.
    """

    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def contains_term(
    text,
    term,
):
    """
    Check whether a whole word or phrase
    appears in text.
    """

    text = normalise_text(
        text
    )

    pattern = (
        rf"\b{re.escape(term.lower())}\b"
    )

    return bool(
        re.search(
            pattern,
            text,
        )
    )


def find_concept_matches(
    text,
    topic,
):
    """
    Find relevance concepts and terms
    appearing in text.
    """

    text = normalise_text(
        text
    )

    matched_concepts = []
    matched_terms = []

    concepts = RELEVANCE_CONCEPTS.get(
        topic,
        {},
    )

    for concept, terms in concepts.items():

        concept_terms = []

        for term in terms:

            if contains_term(
                text,
                term,
            ):
                concept_terms.append(
                    term
                )

        if concept_terms:

            matched_concepts.append(
                concept
            )

            matched_terms.extend(
                concept_terms
            )

    return (
        matched_concepts,
        matched_terms,
    )


def assess_relevance(row):
    """
    Assess whether an article is substantively
    relevant to its assigned NewsPrism topic.

    Acceptance rules:

    1. A topic concept appears in the title.

    OR

    2. At least two distinct topic concepts
       appear in the article body.

    Articles with weaker evidence are rejected
    as incidental topic mentions.
    """

    # -------------------------------------------------
    # Quality check
    # -------------------------------------------------

    if (
        row["quality_status"]
        != "accept"
    ):
        return (
            "reject",
            "",
            "",
            0,
            0,
            0,
            "quality_reject",
        )

    topic = row["topic"]

    # -------------------------------------------------
    # Topic validation
    # -------------------------------------------------

    if topic not in RELEVANCE_CONCEPTS:
        return (
            "reject",
            "",
            "",
            0,
            0,
            0,
            "unknown_topic",
        )

    title = row["title"]

    clean_text = row["clean_text"]

    # -------------------------------------------------
    # Title matches
    # -------------------------------------------------

    (
        title_concepts,
        title_terms,
    ) = find_concept_matches(
        title,
        topic,
    )

    # -------------------------------------------------
    # Body matches
    # -------------------------------------------------

    (
        body_concepts,
        body_terms,
    ) = find_concept_matches(
        clean_text,
        topic,
    )

    # -------------------------------------------------
    # Combined diagnostic information
    # -------------------------------------------------

    all_concepts = sorted(
        set(
            title_concepts
            + body_concepts
        )
    )

    all_terms = sorted(
        set(
            title_terms
            + body_terms
        )
    )

    concept_count = len(
        all_concepts
    )

    title_concept_count = len(
        set(
            title_concepts
        )
    )

    body_concept_count = len(
        set(
            body_concepts
        )
    )

    # -------------------------------------------------
    # Rule 1:
    # Topic concept appears in the headline.
    # -------------------------------------------------

    if title_concept_count >= 1:

        status = "accept"

        reason = (
            "topic_concept_in_title"
        )

    # -------------------------------------------------
    # Rule 2:
    # Multiple distinct topic concepts appear
    # in the article body.
    # -------------------------------------------------

    elif body_concept_count >= 2:

        status = "accept"

        reason = (
            "multiple_topic_concepts"
        )

    # -------------------------------------------------
    # Otherwise:
    # Topic evidence is too weak or incidental.
    # -------------------------------------------------

    else:

        status = "reject"

        reason = (
            "weak_or_incidental_topic_mention"
        )

    return (
        status,
        ", ".join(
            all_concepts
        ),
        ", ".join(
            all_terms
        ),
        concept_count,
        title_concept_count,
        body_concept_count,
        reason,
    )


def filter_relevance(
    dataframe,
):
    """
    Apply NewsPrism relevance filtering
    to a dataframe.
    """

    dataframe = dataframe.copy()

    required_columns = [
        "topic",
        "title",
        "clean_text",
        "quality_status",
        "rejection_reason",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column
        not in dataframe.columns
    ]

    if missing_columns:

        raise ValueError(
            "Dataset is missing columns: "
            + ", ".join(
                missing_columns
            )
        )

    results = dataframe.apply(
        assess_relevance,
        axis=1,
        result_type="expand",
    )

    results.columns = [
        "relevance_status",
        "matched_concepts",
        "matched_terms",
        "concept_match_count",
        "title_concept_count",
        "body_concept_count",
        "relevance_reason",
    ]

    for column in results.columns:

        dataframe[column] = (
            results[column]
        )

    return dataframe


def run_test():
    """
    Run relevance filtering on the small
    preprocessing test dataset.
    """

    if not TEST_INPUT_PATH.exists():

        print(
            f"Test input not found: "
            f"{TEST_INPUT_PATH}"
        )

        return

    dataframe = pd.read_csv(
        TEST_INPUT_PATH
    )

    filtered = filter_relevance(
        dataframe
    )

    filtered.to_csv(
        TEST_OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    print(
        "NewsPrism relevance "
        "filtering test"
    )

    print(
        "--------------------------"
    )

    print(
        f"\nArticles assessed: "
        f"{len(filtered)}"
    )

    print(
        "\nRelevance status:"
    )

    print(
        filtered[
            "relevance_status"
        ].value_counts(
            dropna=False
        )
    )

    print(
        "\nRelevance reasons:"
    )

    print(
        filtered[
            "relevance_reason"
        ].value_counts(
            dropna=False
        )
    )

    print(
        "\nArticle relevance results:"
    )

    display_columns = [
        "topic",
        "title",
        "matched_concepts",
        "concept_match_count",
        "title_concept_count",
        "body_concept_count",
        "relevance_status",
        "relevance_reason",
    ]

    print(
        filtered[
            display_columns
        ].to_string(
            index=False
        )
    )

    print(
        f"\nSaved to: "
        f"{TEST_OUTPUT_PATH}"
    )


if __name__ == "__main__":
    run_test()