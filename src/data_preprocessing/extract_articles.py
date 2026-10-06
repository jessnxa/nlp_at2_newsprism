import time

import pandas as pd
import requests
import trafilatura

from src.data_preprocessing.config import INTERIM_DIR, RAW_DIR


TEST_INPUT_PATH = RAW_DIR / "TEST1_gdelt_candidates.csv"
TEST_OUTPUT_PATH = INTERIM_DIR / "TEST2_extracted_articles.csv"

REQUEST_TIMEOUT = 20
DELAY_BETWEEN_REQUESTS = 2

TEST_SAMPLE_SIZE = 10
TEST_RANDOM_STATE = 42


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(compatible; NewsPrismResearch/1.0)"
    )
}


def download_article(url):
    if pd.isna(url) or not str(url).strip():
        return None, "missing_url"

    try:
        response = requests.get(
            str(url),
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

        return response.text, None

    except requests.RequestException as error:
        return None, str(error)


def extract_article_text(html):
    if not html:
        return None

    text = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False,
        favor_precision=True,
    )

    if not text:
        return None

    return " ".join(
        text.split()
    )


def extract_single_article(url):
    html, download_error = download_article(
        url
    )

    if download_error:
        return {
            "raw_text": None,
            "word_count": 0,
            "extraction_status":
                "download_failed",
            "extraction_error":
                download_error,
        }

    text = extract_article_text(
        html
    )

    if not text:
        return {
            "raw_text": None,
            "word_count": 0,
            "extraction_status":
                "extraction_failed",
            "extraction_error":
                "No article text extracted",
        }

    return {
        "raw_text": text,
        "word_count": len(
            text.split()
        ),
        "extraction_status": "success",
        "extraction_error": None,
    }


def extract_articles(
    dataframe,
    delay_seconds=DELAY_BETWEEN_REQUESTS,
):
    dataframe = dataframe.copy()

    if "url" not in dataframe.columns:
        raise ValueError(
            "Dataset does not contain "
            "'url'."
        )

    results = []

    total = len(dataframe)

    for position, (_, row) in enumerate(
        dataframe.iterrows(),
        start=1,
    ):
        title = row.get(
            "title",
            "Untitled article",
        )

        print(
            f"[{position}/{total}] "
            f"{title}"
        )

        result = extract_single_article(
            row["url"]
        )

        results.append(result)

        print(
            f"  Status: "
            f"{result['extraction_status']}"
        )

        print(
            f"  Words: "
            f"{result['word_count']}"
        )

        if (
            delay_seconds > 0
            and position < total
        ):
            time.sleep(
                delay_seconds
            )

    result_data = pd.DataFrame(
        results,
        index=dataframe.index,
    )

    for column in result_data.columns:
        dataframe[column] = (
            result_data[column]
        )

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

    sample_size = min(
        TEST_SAMPLE_SIZE,
        len(dataframe),
    )

    test_data = dataframe.sample(
        n=sample_size,
        random_state=TEST_RANDOM_STATE,
    ).reset_index(
        drop=True
    )

    print("NewsPrism extraction test")
    print("-------------------------")
    print(
        f"\nArticles selected: "
        f"{len(test_data)}\n"
    )

    extracted = extract_articles(
        test_data
    )

    extracted.to_csv(
        TEST_OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    print("\nExtraction status:")

    print(
        extracted[
            "extraction_status"
        ].value_counts(
            dropna=False
        )
    )

    successful = extracted[
        extracted["extraction_status"]
        == "success"
    ]

    if not successful.empty:
        print(
            "\nSuccessful article "
            "word counts:"
        )

        print(
            successful["word_count"]
            .describe()
        )

    print(
        f"\nSaved to: "
        f"{TEST_OUTPUT_PATH}"
    )


if __name__ == "__main__":
    run_test()