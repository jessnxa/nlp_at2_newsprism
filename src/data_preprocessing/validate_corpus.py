import pandas as pd

from src.data_preprocessing.config import MIN_WORD_COUNT, PUBLISHERS, TOPICS


REQUIRED_COLUMNS = [
    "article_id",
    "topic",
    "title",
    "publisher",
    "domain",
    "url",
    "published_at",
    "language",
    "source_country",
    "raw_text",
    "clean_text",
    "word_count",
]


def validate_required_columns(dataframe):
    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    return missing


def validate_missing_values(dataframe):
    important_columns = [
        "article_id",
        "topic",
        "title",
        "publisher",
        "domain",
        "url",
        "clean_text",
    ]

    results = {}

    for column in important_columns:
        if column in dataframe.columns:
            results[column] = int(
                dataframe[column].isna().sum()
            )

    return results


def validate_article_ids(dataframe):
    if "article_id" not in dataframe.columns:
        return None

    return int(
        dataframe["article_id"]
        .duplicated()
        .sum()
    )


def validate_urls(dataframe):
    if "url" not in dataframe.columns:
        return None

    return int(
        dataframe["url"]
        .duplicated()
        .sum()
    )


def validate_topics(dataframe):
    if "topic" not in dataframe.columns:
        return []

    valid_topics = set(TOPICS.keys())

    found_topics = set(
        dataframe["topic"]
        .dropna()
        .unique()
    )

    return sorted(
        found_topics - valid_topics
    )


def validate_publishers(dataframe):
    if "domain" not in dataframe.columns:
        return []

    valid_domains = set(PUBLISHERS.keys())

    found_domains = set(
        dataframe["domain"]
        .dropna()
        .unique()
    )

    invalid_domains = []

    for domain in found_domains:
        domain = str(domain).lower()

        valid = any(
            domain == approved
            or domain.endswith("." + approved)
            for approved in valid_domains
        )

        if not valid:
            invalid_domains.append(domain)

    return sorted(invalid_domains)


def validate_word_counts(dataframe):
    column = None

    if "clean_word_count" in dataframe.columns:
        column = "clean_word_count"

    elif "word_count" in dataframe.columns:
        column = "word_count"

    if column is None:
        return None

    word_counts = pd.to_numeric(
        dataframe[column],
        errors="coerce",
    )

    return int(
        (word_counts < MIN_WORD_COUNT).sum()
    )


def validate_corpus(dataframe):
    print("NewsPrism corpus validation")
    print("---------------------------")

    print(f"\nArticles: {len(dataframe)}")

    missing_columns = validate_required_columns(
        dataframe
    )

    print("\nRequired columns:")

    if missing_columns:
        print(
            "FAIL - missing: "
            + ", ".join(missing_columns)
        )
    else:
        print("PASS")

    missing_values = validate_missing_values(
        dataframe
    )

    print("\nMissing important values:")

    for column, count in missing_values.items():
        print(f"  {column}: {count}")

    duplicate_ids = validate_article_ids(
        dataframe
    )

    print(
        f"\nDuplicate article IDs: "
        f"{duplicate_ids}"
    )

    duplicate_urls = validate_urls(
        dataframe
    )

    print(
        f"Duplicate URLs: "
        f"{duplicate_urls}"
    )

    invalid_topics = validate_topics(
        dataframe
    )

    print("\nInvalid topics:")

    if invalid_topics:
        for topic in invalid_topics:
            print(f"  {topic}")
    else:
        print("  None")

    invalid_publishers = validate_publishers(
        dataframe
    )

    print("\nUnapproved publisher domains:")

    if invalid_publishers:
        for domain in invalid_publishers:
            print(f"  {domain}")
    else:
        print("  None")

    short_articles = validate_word_counts(
        dataframe
    )

    print(
        f"\nArticles below "
        f"{MIN_WORD_COUNT} words: "
        f"{short_articles}"
    )

    if "topic" in dataframe.columns:
        print("\nArticles per topic:")

        print(
            dataframe["topic"]
            .value_counts()
        )

    if "publisher" in dataframe.columns:
        print("\nArticles per publisher:")

        print(
            dataframe["publisher"]
            .value_counts()
        )


if __name__ == "__main__":
    print(
        "This module validates the final "
        "NewsPrism corpus."
    )