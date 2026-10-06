import re

import pandas as pd

from src.data_preprocessing.config import INTERIM_DIR


TEST_INPUT_PATH = (
    INTERIM_DIR / "TEST4_relevance_filtered.csv"
)

TEST_OUTPUT_PATH = (
    INTERIM_DIR / "TEST5_deduplicated_articles.csv"
)


def normalise_title(title):
    if pd.isna(title):
        return ""

    title = str(title).lower().strip()

    title = re.sub(
        r"[^\w\s]",
        " ",
        title,
    )

    title = re.sub(
        r"\s+",
        " ",
        title,
    )

    return title.strip()


def normalise_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower().strip()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text


def deduplicate_articles(dataframe):
    dataframe = dataframe.copy()

    required_columns = [
        "url",
        "title",
        "clean_text",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Dataset is missing columns: "
            + ", ".join(
                missing_columns
            )
        )

    dataframe["normalised_title"] = (
        dataframe["title"]
        .apply(normalise_title)
    )

    dataframe["normalised_text"] = (
        dataframe["clean_text"]
        .apply(normalise_text)
    )

    dataframe["duplicate_status"] = "unique"
    dataframe["duplicate_reason"] = None

    seen_urls = set()
    seen_titles = set()
    seen_texts = set()

    for index, row in dataframe.iterrows():
        url = row["url"]
        title = row["normalised_title"]
        text = row["normalised_text"]

        if pd.notna(url):
            url = str(url).strip()

        if (
            pd.notna(url)
            and url
            and url in seen_urls
        ):
            dataframe.at[
                index,
                "duplicate_status",
            ] = "duplicate"

            dataframe.at[
                index,
                "duplicate_reason",
            ] = "duplicate_url"

            continue

        if text and text in seen_texts:
            dataframe.at[
                index,
                "duplicate_status",
            ] = "duplicate"

            dataframe.at[
                index,
                "duplicate_reason",
            ] = "duplicate_text"

            continue

        if title and title in seen_titles:
            dataframe.at[
                index,
                "duplicate_status",
            ] = "duplicate"

            dataframe.at[
                index,
                "duplicate_reason",
            ] = "duplicate_title"

            continue

        if pd.notna(url) and url:
            seen_urls.add(url)

        if title:
            seen_titles.add(title)

        if text:
            seen_texts.add(text)

    return dataframe


def run_test():
    if not TEST_INPUT_PATH.exists():
        print(
            f"Test input not found: "
            f"{TEST_INPUT_PATH}"
        )
        return

    dataframe = pd.read_csv(
        TEST_INPUT_PATH
    )

    deduplicated = deduplicate_articles(
        dataframe
    )

    deduplicated.to_csv(
        TEST_OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    print(
        "NewsPrism deduplication test"
    )
    print(
        "----------------------------"
    )

    print(
        f"\nArticles assessed: "
        f"{len(deduplicated)}"
    )

    print("\nDuplicate status:")

    print(
        deduplicated[
            "duplicate_status"
        ].value_counts(
            dropna=False
        )
    )

    print("\nDuplicate reasons:")

    print(
        deduplicated[
            "duplicate_reason"
        ].value_counts(
            dropna=False
        )
    )

    duplicates = deduplicated[
        deduplicated[
            "duplicate_status"
        ] == "duplicate"
    ]

    if not duplicates.empty:
        print(
            "\nDuplicates detected:"
        )

        display_columns = [
            column
            for column in [
                "topic",
                "title",
                "domain",
                "duplicate_reason",
            ]
            if column
            in duplicates.columns
        ]

        print(
            duplicates[
                display_columns
            ].to_string(
                index=False
            )
        )

    else:
        print(
            "\nNo duplicates detected."
        )

    unique_count = (
        deduplicated[
            "duplicate_status"
        ] == "unique"
    ).sum()

    duplicate_count = (
        deduplicated[
            "duplicate_status"
        ] == "duplicate"
    ).sum()

    print("\nSummary:")
    print(
        f"  Unique: {unique_count}"
    )
    print(
        f"  Duplicates: "
        f"{duplicate_count}"
    )

    print(
        f"\nSaved to: "
        f"{TEST_OUTPUT_PATH}"
    )


if __name__ == "__main__":
    run_test()