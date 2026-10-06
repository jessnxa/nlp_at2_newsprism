from pathlib import Path

import pandas as pd

from src.data_preprocessing.article_ids import add_article_ids
from src.data_preprocessing.config import PROCESSED_DIR
from src.data_preprocessing.validate_corpus import validate_corpus


FINAL_OUTPUT_PATH = (
    PROCESSED_DIR / "FINAL_newsprism_corpus.csv"
)


FINAL_COLUMNS = [
    "article_id",
    "event_id",
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


def apply_relevance_decisions(
    dataframe,
    review_decisions=None,
):
    dataframe = dataframe.copy()

    dataframe["final_relevance_status"] = (
        dataframe["relevance_status"]
    )

    if review_decisions is not None:
        if "article_id" not in dataframe.columns:
            dataframe = add_article_ids(
                dataframe
            )

        decision_map = (
            review_decisions
            .set_index("article_id")[
                "review_decision"
            ]
            .to_dict()
        )

        review_mask = (
            dataframe["relevance_status"]
            == "review"
        )

        dataframe.loc[
            review_mask,
            "final_relevance_status",
        ] = (
            dataframe.loc[
                review_mask,
                "article_id",
            ]
            .map(decision_map)
            .fillna("review")
        )

    return dataframe


def select_final_articles(dataframe):
    dataframe = dataframe.copy()

    if "final_relevance_status" not in dataframe.columns:
        raise ValueError(
            "Dataset does not contain "
            "'final_relevance_status'."
        )

    final_data = dataframe[
        dataframe["final_relevance_status"]
        == "accept"
    ].copy()

    if "duplicate_status" in final_data.columns:
        final_data = final_data[
            final_data["duplicate_status"]
            == "unique"
        ].copy()

    return final_data


def prepare_final_schema(dataframe):
    dataframe = dataframe.copy()

    if "article_id" not in dataframe.columns:
        dataframe = add_article_ids(
            dataframe
        )

    if "event_id" not in dataframe.columns:
        dataframe["event_id"] = None

    if "clean_word_count" in dataframe.columns:
        dataframe["word_count"] = dataframe[
            "clean_word_count"
        ]

    missing_columns = [
        column
        for column in FINAL_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Cannot build final corpus. "
            "Missing columns: "
            + ", ".join(missing_columns)
        )

    dataframe = dataframe[
        FINAL_COLUMNS
    ].copy()

    dataframe = dataframe.sort_values(
        by=[
            "topic",
            "published_at",
            "publisher",
        ],
        na_position="last",
    )

    dataframe = dataframe.reset_index(
        drop=True
    )

    return dataframe


def build_final_corpus(
    dataframe,
    review_decisions=None,
    output_path=FINAL_OUTPUT_PATH,
):
    dataframe = dataframe.copy()

    if "article_id" not in dataframe.columns:
        dataframe = add_article_ids(
            dataframe
        )

    dataframe = apply_relevance_decisions(
        dataframe,
        review_decisions,
    )

    dataframe = select_final_articles(
        dataframe
    )

    dataframe = prepare_final_schema(
        dataframe
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    print("NewsPrism final corpus")
    print("----------------------")

    print(
        f"\nFinal articles: "
        f"{len(dataframe)}"
    )

    print(
        f"\nSaved to: "
        f"{output_path}"
    )

    print("\nValidation:")
    print()

    validate_corpus(
        dataframe
    )

    return dataframe


def main():
    print("NewsPrism final corpus builder")
    print("------------------------------")
    print()
    print(
        "This module builds the final "
        "shared NewsPrism corpus."
    )
    print()
    print(
        "Only articles with a final "
        "relevance decision of 'accept' "
        "are included."
    )
    print()
    print(
        "Known duplicates are excluded."
    )
    print()
    print(
        "The final corpus is validated "
        "before handover."
    )


if __name__ == "__main__":
    main()