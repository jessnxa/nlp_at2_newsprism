import re
import unicodedata

import pandas as pd

from src.data_preprocessing.config import INTERIM_DIR, MIN_WORD_COUNT


TEST_INPUT_PATH = (
    INTERIM_DIR / "TEST2_extracted_articles.csv"
)

TEST_OUTPUT_PATH = (
    INTERIM_DIR / "TEST3_cleaned_articles.csv"
)


def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text)

    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    text = text.replace(
        "\u00a0",
        " ",
    )

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text,
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n\s*\n+",
        "\n",
        text,
    )

    return text.strip()


def count_words(text):
    if pd.isna(text):
        return 0

    text = str(text).strip()

    if not text:
        return 0

    return len(
        text.split()
    )


def assess_quality(row):
    extraction_status = row.get(
        "extraction_status",
        "",
    )

    if extraction_status != "success":
        return (
            "reject",
            "extraction_failed",
        )

    if not row["clean_text"]:
        return (
            "reject",
            "empty_text",
        )

    if (
        row["clean_word_count"]
        < MIN_WORD_COUNT
    ):
        return (
            "reject",
            "too_short",
        )

    return (
        "accept",
        None,
    )


def clean_articles(dataframe):
    dataframe = dataframe.copy()

    if "raw_text" not in dataframe.columns:
        raise ValueError(
            "Dataset does not contain "
            "'raw_text'."
        )

    if (
        "extraction_status"
        not in dataframe.columns
    ):
        raise ValueError(
            "Dataset does not contain "
            "'extraction_status'."
        )

    dataframe["clean_text"] = dataframe[
        "raw_text"
    ].apply(clean_text)

    dataframe["clean_word_count"] = (
        dataframe["clean_text"]
        .apply(count_words)
    )

    quality_results = dataframe.apply(
        assess_quality,
        axis=1,
        result_type="expand",
    )

    quality_results.columns = [
        "quality_status",
        "rejection_reason",
    ]

    dataframe[
        [
            "quality_status",
            "rejection_reason",
        ]
    ] = quality_results

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

    cleaned = clean_articles(
        dataframe
    )

    cleaned.to_csv(
        TEST_OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    print("NewsPrism cleaning test")
    print("-----------------------")

    print(
        f"\nArticles assessed: "
        f"{len(cleaned)}"
    )

    print("\nQuality status:")

    print(
        cleaned["quality_status"]
        .value_counts(
            dropna=False
        )
    )

    print("\nRejection reasons:")

    print(
        cleaned["rejection_reason"]
        .value_counts(
            dropna=False
        )
    )

    print(
        "\nClean word-count summary:"
    )

    print(
        cleaned["clean_word_count"]
        .describe()
    )

    print(
        f"\nSaved to: "
        f"{TEST_OUTPUT_PATH}"
    )


if __name__ == "__main__":
    run_test()