"""
Resolve Google News RSS candidate URLs to original publisher URLs.

Supports custom input/output files so the same resolver can be reused
for immigration, housing and clean energy datasets.

Example:
python -m src.data_preprocessing.resolve_google_news_urls
    --input data/raw/immigration_candidates_sample.csv \
    --output data/interim/immigration_resolved_candidates.csv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from googlenewsdecoder import gnewsdecoder


REQUEST_DELAY_SECONDS = 2

OUTPUT_COLUMNS = [
    "topic",
    "keyword_group",
    "search_terms",
    "publisher",
    "domain",
    "url",
    "google_news_url",
    "title",
    "published_at",
    "language",
    "source_country",
    "collection_month",
    "discovery_source",
    "collected_at",
    "resolved_domain",
    "resolution_status",
    "resolution_method",
    "decoder_success",
    "resolution_error",
]


# ---------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------

def normalise_domain(domain: str) -> str:
    domain = str(domain or "").strip().lower()

    if domain.startswith("www."):
        domain = domain[4:]

    return domain


def domain_from_url(url: str) -> str:
    try:
        hostname = urlparse(str(url)).hostname or ""
    except Exception:
        return ""

    return normalise_domain(hostname)


def is_google_news_url(url: str) -> bool:
    domain = domain_from_url(url)

    return (
        domain == "news.google.com"
        or domain.endswith(".news.google.com")
    )


def matches_expected_domain(
    url: str,
    expected_domain: str,
) -> bool:
    actual_domain = domain_from_url(url)
    expected_domain = normalise_domain(expected_domain)

    return (
        actual_domain == expected_domain
        or actual_domain.endswith("." + expected_domain)
    )


# ---------------------------------------------------------------------
# Decoder handling
# ---------------------------------------------------------------------

def extract_decoded_url(decoded_result) -> tuple[str, str]:

    if decoded_result is None:
        return "", "Decoder returned None"

    if isinstance(decoded_result, dict):

        for key in [
            "decoded_url",
            "url",
            "source_url",
        ]:
            value = decoded_result.get(key)

            if (
                isinstance(value, str)
                and value.startswith("http")
            ):
                return value.strip(), ""

        error = (
            decoded_result.get("message")
            or decoded_result.get("error")
            or decoded_result.get("status")
            or str(decoded_result)
        )

        return "", str(error)

    if isinstance(decoded_result, str):

        value = decoded_result.strip()

        if value.startswith("http"):
            return value, ""

        return "", value

    return (
        "",
        f"Unexpected decoder result: {repr(decoded_result)}",
    )


# ---------------------------------------------------------------------
# Resolve one URL
# ---------------------------------------------------------------------

def resolve_google_news_url(
    google_news_url: str,
    expected_domain: str,
) -> dict:

    result = {
        "resolved_url": "",
        "resolved_domain": "",
        "resolution_status": "failed",
        "resolution_method": "",
        "decoder_success": False,
        "resolution_error": "",
    }

    google_news_url = str(
        google_news_url or ""
    ).strip()

    expected_domain = normalise_domain(
        expected_domain
    )

    if not google_news_url:
        result["resolution_error"] = (
            "Missing Google News URL"
        )
        return result

    # Already a publisher URL
    if not is_google_news_url(google_news_url):

        if matches_expected_domain(
            google_news_url,
            expected_domain,
        ):
            result.update(
                {
                    "resolved_url": google_news_url,
                    "resolved_domain": domain_from_url(
                        google_news_url
                    ),
                    "resolution_status": "resolved",
                    "resolution_method": (
                        "already_publisher_url"
                    ),
                    "decoder_success": True,
                    "resolution_error": "",
                }
            )

        else:
            result["resolution_error"] = (
                "Input URL does not match "
                "expected publisher domain"
            )

        return result

    # Decode Google News URL
    try:
        decoded_result = gnewsdecoder(
            google_news_url
        )

    except Exception as error:
        result["resolution_error"] = (
            f"Decoder exception: {error}"
        )
        return result

    decoded_url, decoder_error = (
        extract_decoded_url(decoded_result)
    )

    if not decoded_url:
        result["resolution_error"] = (
            f"Decoder failed: {decoder_error}"
        )
        return result

    result["decoder_success"] = True

    resolved_domain = domain_from_url(
        decoded_url
    )

    result["resolved_domain"] = (
        resolved_domain
    )

    # Validate publisher domain
    if not matches_expected_domain(
        decoded_url,
        expected_domain,
    ):
        result["resolution_error"] = (
            "Decoded URL does not match "
            "expected publisher. "
            f"Expected: {expected_domain}; "
            f"received: {resolved_domain}"
        )

        return result

    result.update(
        {
            "resolved_url": decoded_url,
            "resolved_domain": resolved_domain,
            "resolution_status": "resolved",
            "resolution_method": (
                "googlenewsdecoder"
            ),
            "decoder_success": True,
            "resolution_error": "",
        }
    )

    return result


# ---------------------------------------------------------------------
# File handling
# ---------------------------------------------------------------------

def load_candidates(
    input_file: Path,
) -> pd.DataFrame:

    if not input_file.exists():
        raise FileNotFoundError(
            f"Candidate file not found: {input_file}"
        )

    dataframe = pd.read_csv(
        input_file
    )

    required_columns = {
        "topic",
        "publisher",
        "domain",
        "url",
        "title",
    }

    missing = (
        required_columns
        - set(dataframe.columns)
    )

    if missing:
        raise ValueError(
            "Candidate file is missing required columns: "
            + ", ".join(sorted(missing))
        )

    # Ensure google_news_url exists
    if "google_news_url" not in dataframe.columns:
        dataframe["google_news_url"] = (
            dataframe["url"]
        )

    return dataframe


def load_existing_output(
    output_file: Path,
) -> pd.DataFrame:

    if not output_file.exists():
        return pd.DataFrame(
            columns=OUTPUT_COLUMNS
        )

    dataframe = pd.read_csv(
        output_file
    )

    for column in OUTPUT_COLUMNS:
        if column not in dataframe.columns:
            dataframe[column] = ""

    return dataframe[OUTPUT_COLUMNS]


def save_output(
    dataframe: pd.DataFrame,
    output_file: Path,
) -> None:

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if dataframe.empty:
        dataframe = pd.DataFrame(
            columns=OUTPUT_COLUMNS
        )

    else:
        dataframe = dataframe.copy()

        for column in OUTPUT_COLUMNS:
            if column not in dataframe.columns:
                dataframe[column] = ""

        resolved = dataframe[
            dataframe["resolution_status"]
            == "resolved"
        ].copy()

        failed = dataframe[
            dataframe["resolution_status"]
            != "resolved"
        ].copy()

        if not resolved.empty:

            resolved["url"] = (
                resolved["url"]
                .fillna("")
                .astype(str)
                .str.strip()
            )

            resolved = (
                resolved.drop_duplicates(
                    subset=["url"],
                    keep="first",
                )
            )

        dataframe = pd.concat(
            [resolved, failed],
            ignore_index=True,
        )

        dataframe = dataframe[
            OUTPUT_COLUMNS
        ]

    dataframe.to_csv(
        output_file,
        index=False,
    )


# ---------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------

def processed_google_news_urls(
    dataframe: pd.DataFrame,
) -> set[str]:

    if dataframe.empty:
        return set()

    return set(
        dataframe["google_news_url"]
        .dropna()
        .astype(str)
        .str.strip()
    )


def build_output_row(
    candidate: pd.Series,
    resolution: dict,
) -> dict:

    google_news_url = str(
        candidate.get(
            "google_news_url",
            candidate.get("url", ""),
        )
        or ""
    ).strip()

    resolved_url = str(
        resolution.get(
            "resolved_url",
            "",
        )
        or ""
    ).strip()

    return {
        "topic": candidate.get(
            "topic", ""
        ),
        "keyword_group": candidate.get(
            "keyword_group", ""
        ),
        "search_terms": candidate.get(
            "search_terms", ""
        ),
        "publisher": candidate.get(
            "publisher", ""
        ),
        "domain": candidate.get(
            "domain", ""
        ),
        "url": resolved_url,
        "google_news_url": (
            google_news_url
        ),
        "title": candidate.get(
            "title", ""
        ),
        "published_at": candidate.get(
            "published_at", ""
        ),
        "language": candidate.get(
            "language", ""
        ),
        "source_country": candidate.get(
            "source_country", ""
        ),
        "collection_month": candidate.get(
            "collection_month", ""
        ),
        "discovery_source": candidate.get(
            "discovery_source",
            "google_news",
        ),
        "collected_at": candidate.get(
            "collected_at", ""
        ),
        "resolved_domain": resolution.get(
            "resolved_domain", ""
        ),
        "resolution_status": resolution.get(
            "resolution_status",
            "failed",
        ),
        "resolution_method": resolution.get(
            "resolution_method", ""
        ),
        "decoder_success": resolution.get(
            "decoder_success",
            False,
        ),
        "resolution_error": resolution.get(
            "resolution_error", ""
        ),
    }


# ---------------------------------------------------------------------
# Main resolver
# ---------------------------------------------------------------------

def resolve_all_candidates(
    input_file: Path,
    output_file: Path,
    limit: int | None = None,
) -> pd.DataFrame:

    candidates = load_candidates(
        input_file
    )

    output = load_existing_output(
        output_file
    )

    processed = (
        processed_google_news_urls(
            output
        )
    )

    pending = candidates.copy()

    pending["_google_news_key"] = (
        pending["google_news_url"]
        .fillna(pending["url"])
        .astype(str)
        .str.strip()
    )

    pending = pending[
        ~pending["_google_news_key"].isin(
            processed
        )
    ].copy()

    if limit is not None:
        pending = pending.head(
            limit
        )

    total_candidates = len(
        candidates
    )

    already_processed = len(
        processed
    )

    to_process = len(
        pending
    )

    print()
    print(
        "NEWSPRISM GOOGLE NEWS URL RESOLVER"
    )
    print("-" * 70)
    print(
        f"Input file: {input_file}"
    )
    print(
        f"Output file: {output_file}"
    )
    print(
        f"Candidate rows: {total_candidates}"
    )
    print(
        f"Previously attempted: "
        f"{already_processed}"
    )
    print(
        f"Pending this run: "
        f"{to_process}"
    )

    if to_process == 0:
        print()
        print(
            "No unresolved candidates remain."
        )
        return output

    successful_this_run = 0
    failed_this_run = 0

    for position, (_, candidate) in enumerate(
        pending.iterrows(),
        start=1,
    ):

        google_news_url = str(
            candidate.get(
                "google_news_url",
                candidate.get(
                    "url",
                    "",
                ),
            )
            or ""
        ).strip()

        print()
        print("=" * 70)
        print(
            f"Resolving "
            f"{position}/{to_process}"
        )
        print(
            f"Topic: "
            f"{candidate.get('topic', '')}"
        )
        print(
            f"Publisher: "
            f"{candidate.get('publisher', '')}"
        )
        print(
            f"Title: "
            f"{candidate.get('title', '')}"
        )

        resolution = (
            resolve_google_news_url(
                google_news_url=(
                    google_news_url
                ),
                expected_domain=(
                    candidate.get(
                        "domain",
                        "",
                    )
                ),
            )
        )

        if (
            resolution[
                "resolution_status"
            ]
            == "resolved"
        ):
            successful_this_run += 1
        else:
            failed_this_run += 1

        print(
            "Status: "
            f"{resolution['resolution_status']}"
        )

        if resolution[
            "resolved_domain"
        ]:
            print(
                "Resolved domain: "
                f"{resolution['resolved_domain']}"
            )

        if resolution[
            "resolved_url"
        ]:
            print(
                "Resolved URL: "
                f"{resolution['resolved_url']}"
            )

        if resolution[
            "resolution_error"
        ]:
            print(
                "Error: "
                f"{resolution['resolution_error']}"
            )

        output_row = build_output_row(
            candidate,
            resolution,
        )

        output = pd.concat(
            [
                output,
                pd.DataFrame(
                    [output_row]
                ),
            ],
            ignore_index=True,
        )

        # Save after every candidate
        save_output(
            output,
            output_file,
        )

        # Reload because duplicate resolved
        # publisher URLs may be removed.
        output = load_existing_output(
            output_file
        )

        if position < to_process:
            time.sleep(
                REQUEST_DELAY_SECONDS
            )

    print()
    print("=" * 70)
    print(
        "URL RESOLUTION COMPLETE"
    )
    print("=" * 70)

    print(
        f"Attempted this run: "
        f"{to_process}"
    )
    print(
        f"Resolved this run: "
        f"{successful_this_run}"
    )
    print(
        f"Failed this run: "
        f"{failed_this_run}"
    )

    resolved_total = int(
        (
            output[
                "resolution_status"
            ]
            == "resolved"
        ).sum()
    )

    failed_total = int(
        (
            output[
                "resolution_status"
            ]
            != "resolved"
        ).sum()
    )

    print(
        f"Resolved rows saved: "
        f"{resolved_total}"
    )
    print(
        f"Failed rows saved: "
        f"{failed_total}"
    )
    print(
        f"Output file: "
        f"{output_file}"
    )

    return output


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Resolve Google News candidate URLs "
            "to original publisher URLs."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/raw/google_news_candidates.csv"
        ),
        help=(
            "Candidate CSV to resolve."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/interim/"
            "google_news_resolved_candidates.csv"
        ),
        help=(
            "CSV used to save resolved URLs."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional maximum number of "
            "currently unresolved URLs "
            "to process."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    args = parse_args()

    if (
        args.limit is not None
        and args.limit < 1
    ):
        raise SystemExit(
            "--limit must be at least 1."
        )

    resolve_all_candidates(
        input_file=args.input,
        output_file=args.output,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()