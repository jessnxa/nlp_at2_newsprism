"""
Google News RSS candidate collector for NewsPrism.

Purpose
-------
Google News RSS is used as a fallback candidate-discovery source when
GDELT is unavailable or rate-limited.

Google News is NOT used as the source of article text.

The collector discovers candidate articles from approved Australian
publishers. Actual article text is extracted later from publisher pages
using the existing NewsPrism extraction pipeline.

Collection design
-----------------
Topics:
    - immigration
    - clean energy
    - housing

Publishers:
    - approved Australian publishers defined in config.py

Project period:
    - START_DATE to END_DATE from config.py

Collection unit:
    topic x publisher x month x keyword group

Each topic's search terms are split into groups of five.

Example conceptual query:

    ("housing crisis" OR "housing affordability")
    site:abc.net.au
    after:2026-09-01
    before:2026-10-01

Important
---------
This file creates candidate data only.

Permanent article_id values are assigned later after deduplication.

event_id values are assigned later after extraction, cleaning,
relevance filtering and event grouping.
"""

from __future__ import annotations

import argparse
import html
import time
import xml.etree.ElementTree as ET

from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlencode, urlparse

import pandas as pd
import requests

from src.data_preprocessing.config import (
    END_DATE,
    PUBLISHERS,
    RAW_DIR,
    START_DATE,
    TOPICS,
)


# ---------------------------------------------------------------------
# Google News settings
# ---------------------------------------------------------------------

GOOGLE_NEWS_RSS_URL = (
    "https://news.google.com/rss/search"
)

OUTPUT_FILE = (
    RAW_DIR
    / "google_news_candidates.csv"
)

PROGRESS_FILE = (
    RAW_DIR
    / "google_news_collection_progress.csv"
)


# Australian English Google News edition.
GOOGLE_NEWS_LANGUAGE = "en-AU"

GOOGLE_NEWS_COUNTRY = "AU"

GOOGLE_NEWS_EDITION = "AU:en"


KEYWORDS_PER_GROUP = 5


# Be conservative even though RSS does not have the same behaviour
# as the GDELT API.
REQUEST_DELAY_SECONDS = 5


REQUEST_TIMEOUT = (
    20,
    60,
)


USER_AGENT = (
    "Mozilla/5.0 "
    "(compatible; NewsPrismResearch/1.0)"
)


# ---------------------------------------------------------------------
# Output schemas
# ---------------------------------------------------------------------

CANDIDATE_COLUMNS = [
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
]


PROGRESS_COLUMNS = [
    "topic",
    "publisher",
    "domain",
    "collection_month",
    "keyword_group",
    "search_terms",
    "start_date",
    "end_date",
    "status",
    "raw_results",
    "valid_results",
    "new_unique_results",
    "http_status",
    "error",
    "updated_at",
]


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------

def utc_now_string() -> str:
    """
    Return a timezone-aware UTC timestamp.
    """

    return datetime.now(
        UTC
    ).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def ensure_output_directory() -> None:
    """
    Ensure data/raw exists.
    """

    Path(
        RAW_DIR
    ).mkdir(
        parents=True,
        exist_ok=True,
    )


def normalise_domain(
    domain: str,
) -> str:
    """
    Normalise a domain.

    Example:
        www.abc.net.au
        ->
        abc.net.au
    """

    domain = str(
        domain or ""
    ).strip().lower()

    if domain.startswith(
        "www."
    ):
        domain = domain[4:]

    return domain


def domain_from_url(
    url: str,
) -> str:
    """
    Extract a normalised hostname from a URL.
    """

    try:

        hostname = (
            urlparse(
                str(url)
            ).hostname
            or ""
        )

    except Exception:

        return ""

    return normalise_domain(
        hostname
    )


def clean_text(
    value: str,
) -> str:
    """
    Clean simple RSS text fields.
    """

    value = html.unescape(
        str(
            value
            or ""
        )
    )

    return " ".join(
        value.split()
    )


# ---------------------------------------------------------------------
# Publisher configuration helpers
# ---------------------------------------------------------------------

def get_all_domains() -> list[str]:
    """
    Extract approved publisher domains from PUBLISHERS.

    Supports the different dictionary structures used by NewsPrism.
    """

    domains = []

    for key, value in PUBLISHERS.items():

        if isinstance(
            value,
            str,
        ):

            if "." in str(
                key
            ):

                domains.append(
                    normalise_domain(
                        key
                    )
                )

            elif "." in value:

                domains.append(
                    normalise_domain(
                        value
                    )
                )

        elif isinstance(
            value,
            dict,
        ):

            domain = value.get(
                "domain",
                "",
            )

            if domain:

                domains.append(
                    normalise_domain(
                        domain
                    )
                )

            elif "." in str(
                key
            ):

                domains.append(
                    normalise_domain(
                        key
                    )
                )

    return list(
        dict.fromkeys(
            domains
        )
    )


def get_publisher_name(
    domain: str,
) -> str:
    """
    Retrieve the configured publisher name.
    """

    target = normalise_domain(
        domain
    )

    for key, value in PUBLISHERS.items():

        if isinstance(
            value,
            str,
        ):

            if (
                normalise_domain(
                    key
                )
                == target
            ):

                return value

            if (
                normalise_domain(
                    value
                )
                == target
            ):

                return str(
                    key
                )

        elif isinstance(
            value,
            dict,
        ):

            configured_domain = (
                normalise_domain(
                    value.get(
                        "domain",
                        "",
                    )
                )
            )

            if (
                configured_domain
                == target
            ):

                return str(
                    value.get(
                        "publisher",
                        value.get(
                            "name",
                            key,
                        ),
                    )
                )

            if (
                normalise_domain(
                    key
                )
                == target
            ):

                return str(
                    value.get(
                        "publisher",
                        value.get(
                            "name",
                            key,
                        ),
                    )
                )

    return domain


# ---------------------------------------------------------------------
# Keyword batching
# ---------------------------------------------------------------------

def split_keywords(
    keywords: list[str],
    group_size: int = KEYWORDS_PER_GROUP,
) -> list[list[str]]:
    """
    Split topic terms into groups.

    With 15 terms:

        group 1 = terms 1-5
        group 2 = terms 6-10
        group 3 = terms 11-15
    """

    return [
        keywords[
            index:
            index + group_size
        ]
        for index
        in range(
            0,
            len(keywords),
            group_size,
        )
    ]


# ---------------------------------------------------------------------
# Date batching
# ---------------------------------------------------------------------

def get_month_batches() -> list[dict]:
    """
    Build monthly collection windows from project dates.
    """

    start = pd.Timestamp(
        START_DATE
    ).normalize()

    end = pd.Timestamp(
        END_DATE
    ).normalize()

    month_starts = (
        pd.date_range(
            start=start.replace(
                day=1
            ),
            end=end.replace(
                day=1
            ),
            freq="MS",
        )
    )

    batches = []

    for month_start in month_starts:

        natural_month_end = (
            month_start
            + pd.offsets.MonthEnd(
                0
            )
        )

        batch_start = max(
            start,
            month_start,
        )

        batch_end = min(
            end,
            natural_month_end,
        )

        # Google News before: is treated as the upper boundary.
        # Using the following day makes the intended final day
        # available to the search.
        before_date = (
            batch_end
            + pd.Timedelta(
                days=1
            )
        )

        batches.append(
            {
                "month": (
                    month_start.strftime(
                        "%Y-%m"
                    )
                ),
                "start_date": (
                    batch_start
                ),
                "end_date": (
                    batch_end
                ),
                "after": (
                    batch_start.strftime(
                        "%Y-%m-%d"
                    )
                ),
                "before": (
                    before_date.strftime(
                        "%Y-%m-%d"
                    )
                ),
            }
        )

    return batches


# ---------------------------------------------------------------------
# Query construction
# ---------------------------------------------------------------------

def build_keyword_query(
    keywords: list[str],
) -> str:
    """
    Build the topic portion of a Google News query.
    """

    terms = []

    for keyword in keywords:

        keyword = str(
            keyword
        ).strip()

        if not keyword:
            continue

        terms.append(
            f'"{keyword}"'
        )

    return (
        "("
        + " OR ".join(
            terms
        )
        + ")"
    )


def build_search_query(
    keywords: list[str],
    domain: str,
    after_date: str,
    before_date: str,
) -> str:
    """
    Build the complete Google News query.

    Example:

        ("housing crisis" OR "housing affordability")
        site:abc.net.au
        after:2026-09-01
        before:2026-10-01
    """

    keyword_query = (
        build_keyword_query(
            keywords
        )
    )

    return (
        f"{keyword_query} "
        f"site:{domain} "
        f"after:{after_date} "
        f"before:{before_date}"
    )


def build_feed_url(
    query: str,
) -> str:
    """
    Build the Australian Google News RSS URL.
    """

    params = {
        "q": query,
        "hl": (
            GOOGLE_NEWS_LANGUAGE
        ),
        "gl": (
            GOOGLE_NEWS_COUNTRY
        ),
        "ceid": (
            GOOGLE_NEWS_EDITION
        ),
    }

    return (
        GOOGLE_NEWS_RSS_URL
        + "?"
        + urlencode(
            params
        )
    )


# ---------------------------------------------------------------------
# Candidate storage
# ---------------------------------------------------------------------

def load_candidates() -> pd.DataFrame:
    """
    Load previously discovered Google News candidates.
    """

    if not OUTPUT_FILE.exists():

        return pd.DataFrame(
            columns=CANDIDATE_COLUMNS
        )

    dataframe = pd.read_csv(
        OUTPUT_FILE
    )

    for column in CANDIDATE_COLUMNS:

        if (
            column
            not in dataframe.columns
        ):

            dataframe[
                column
            ] = ""

    return dataframe[
        CANDIDATE_COLUMNS
    ]


def save_candidates(
    dataframe: pd.DataFrame,
) -> None:
    """
    Save candidate URLs.

    Exact URL duplicates are removed here.

    More sophisticated deduplication happens later.
    """

    if dataframe.empty:

        dataframe = pd.DataFrame(
            columns=CANDIDATE_COLUMNS
        )

    else:

        dataframe = (
            dataframe.copy()
        )

        dataframe[
            "url"
        ] = (
            dataframe[
                "url"
            ]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        dataframe = dataframe[
            dataframe[
                "url"
            ]
            != ""
        ]

        dataframe = (
            dataframe.drop_duplicates(
                subset=[
                    "url"
                ],
                keep="first",
            )
        )

        dataframe = dataframe[
            CANDIDATE_COLUMNS
        ]

    dataframe.to_csv(
        OUTPUT_FILE,
        index=False,
    )


# ---------------------------------------------------------------------
# Progress storage
# ---------------------------------------------------------------------

def load_progress() -> pd.DataFrame:
    """
    Load collection progress.
    """

    if not PROGRESS_FILE.exists():

        return pd.DataFrame(
            columns=PROGRESS_COLUMNS
        )

    dataframe = pd.read_csv(
        PROGRESS_FILE
    )

    for column in PROGRESS_COLUMNS:

        if (
            column
            not in dataframe.columns
        ):

            dataframe[
                column
            ] = ""

    return dataframe[
        PROGRESS_COLUMNS
    ]


def save_progress(
    dataframe: pd.DataFrame,
) -> None:
    """
    Save collection progress.
    """

    dataframe[
        PROGRESS_COLUMNS
    ].to_csv(
        PROGRESS_FILE,
        index=False,
    )


def batch_key(
    topic: str,
    domain: str,
    month: str,
    keyword_group: int,
) -> tuple:
    """
    Return a unique batch identifier.
    """

    return (
        topic,
        normalise_domain(
            domain
        ),
        month,
        int(
            keyword_group
        ),
    )


def completed_batch_keys(
    progress: pd.DataFrame,
) -> set[tuple]:
    """
    Return completed collection batches.
    """

    if progress.empty:

        return set()

    completed = progress[
        progress[
            "status"
        ]
        == "completed"
    ]

    keys = set()

    for _, row in completed.iterrows():

        try:

            group_number = int(
                row[
                    "keyword_group"
                ]
            )

        except (
            TypeError,
            ValueError,
        ):

            continue

        keys.add(
            batch_key(
                str(
                    row[
                        "topic"
                    ]
                ),
                str(
                    row[
                        "domain"
                    ]
                ),
                str(
                    row[
                        "collection_month"
                    ]
                ),
                group_number,
            )
        )

    return keys


def update_progress(
    progress: pd.DataFrame,
    record: dict,
) -> pd.DataFrame:
    """
    Store the latest result for a collection batch.
    """

    if progress.empty:

        progress = pd.DataFrame(
            columns=PROGRESS_COLUMNS
        )

    else:

        mask = (
            (
                progress[
                    "topic"
                ].astype(str)
                == str(
                    record[
                        "topic"
                    ]
                )
            )
            & (
                progress[
                    "domain"
                ].astype(str)
                == str(
                    record[
                        "domain"
                    ]
                )
            )
            & (
                progress[
                    "collection_month"
                ].astype(str)
                == str(
                    record[
                        "collection_month"
                    ]
                )
            )
            & (
                progress[
                    "keyword_group"
                ].astype(str)
                == str(
                    record[
                        "keyword_group"
                    ]
                )
            )
        )

        progress = progress[
            ~mask
        ]

    progress = pd.concat(
        [
            progress,
            pd.DataFrame(
                [
                    record
                ]
            ),
        ],
        ignore_index=True,
    )

    return progress[
        PROGRESS_COLUMNS
    ]


# ---------------------------------------------------------------------
# RSS request
# ---------------------------------------------------------------------

def request_google_news(
    query: str,
) -> tuple[
    list[dict],
    int,
    str,
]:
    """
    Request and parse one Google News RSS feed.

    Returns:
        items
        HTTP status
        error
    """

    feed_url = (
        build_feed_url(
            query
        )
    )

    headers = {
        "User-Agent": (
            USER_AGENT
        )
    }

    try:

        response = requests.get(
            feed_url,
            headers=headers,
            timeout=(
                REQUEST_TIMEOUT
            ),
        )

    except (
        requests.exceptions.Timeout,
        requests.exceptions.ConnectionError,
    ) as error:

        return (
            [],
            0,
            str(
                error
            ),
        )

    status_code = (
        response.status_code
    )

    print(
        f"  HTTP "
        f"{status_code}"
    )

    if status_code != 200:

        preview = (
            response.text[
                :300
            ]
            .replace(
                "\n",
                " ",
            )
        )

        return (
            [],
            status_code,
            (
                f"HTTP "
                f"{status_code}: "
                f"{preview}"
            ),
        )

    if not response.content:

        return (
            [],
            status_code,
            "Empty RSS response",
        )

    try:

        root = (
            ET.fromstring(
                response.content
            )
        )

    except ET.ParseError as error:

        return (
            [],
            status_code,
            (
                "Could not parse "
                f"Google News RSS: "
                f"{error}"
            ),
        )

    items = []

    for item in root.findall(
        ".//item"
    ):

        title = clean_text(
            item.findtext(
                "title",
                default="",
            )
        )

        link = clean_text(
            item.findtext(
                "link",
                default="",
            )
        )

        pub_date = clean_text(
            item.findtext(
                "pubDate",
                default="",
            )
        )

        source_element = (
            item.find(
                "source"
            )
        )

        source_name = ""

        source_url = ""

        if source_element is not None:

            source_name = (
                clean_text(
                    source_element.text
                    or ""
                )
            )

            source_url = (
                clean_text(
                    source_element.attrib.get(
                        "url",
                        "",
                    )
                )
            )

        items.append(
            {
                "title": (
                    title
                ),
                "google_news_url": (
                    link
                ),
                "published_at": (
                    pub_date
                ),
                "source_name": (
                    source_name
                ),
                "source_url": (
                    source_url
                ),
            }
        )

    return (
        items,
        status_code,
        "",
    )


# ---------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------

def parse_google_news_date(
    value: str,
) -> pd.Timestamp | None:
    """
    Parse an RSS publication date.
    """

    value = str(
        value or ""
    ).strip()

    if not value:

        return None

    parsed = pd.to_datetime(
        value,
        utc=True,
        errors="coerce",
    )

    if pd.isna(
        parsed
    ):

        return None

    return parsed


def date_in_requested_window(
    published_at: pd.Timestamp,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> bool:
    """
    Independently validate the RSS publication date.
    """

    if published_at is None:

        return False

    article_date = (
        published_at
        .tz_convert(None)
        .normalize()
    )

    return (
        start_date.normalize()
        <= article_date
        <= end_date.normalize()
    )


def source_matches_domain(
    source_url: str,
    requested_domain: str,
) -> bool:
    """
    Validate the publisher using the RSS source URL.

    Google News article links themselves point to news.google.com,
    so the <source> URL is used for publisher validation.
    """

    source_domain = (
        domain_from_url(
            source_url
        )
    )

    requested_domain = (
        normalise_domain(
            requested_domain
        )
    )

    return (
        source_domain
        == requested_domain
        or source_domain.endswith(
            "."
            + requested_domain
        )
    )


# ---------------------------------------------------------------------
# Candidate conversion
# ---------------------------------------------------------------------

def convert_items_to_rows(
    items: list[dict],
    topic: str,
    publisher: str,
    domain: str,
    month: str,
    keyword_group: int,
    keywords: list[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> list[dict]:
    """
    Convert Google News RSS items into candidate rows.

    At this stage the Google News redirect URL may still be present.

    Publisher URL resolution can be performed separately before
    article extraction.
    """

    rows = []

    for item in items:

        google_news_url = str(
            item.get(
                "google_news_url",
                "",
            )
            or ""
        ).strip()

        if not google_news_url:

            continue

        source_url = str(
            item.get(
                "source_url",
                "",
            )
            or ""
        ).strip()

        if (
            source_url
            and not source_matches_domain(
                source_url,
                domain,
            )
        ):

            continue

        published_at = (
            parse_google_news_date(
                item.get(
                    "published_at",
                    "",
                )
            )
        )

        if not date_in_requested_window(
            published_at,
            start_date,
            end_date,
        ):

            continue

        rows.append(
            {
                "topic": (
                    topic
                ),
                "keyword_group": (
                    keyword_group
                ),
                "search_terms": (
                    " | ".join(
                        keywords
                    )
                ),
                "publisher": (
                    publisher
                ),
                "domain": (
                    normalise_domain(
                        domain
                    )
                ),

                # This initially stores the Google News URL.
                # We will resolve the publisher URL before sending
                # candidates into the extraction pipeline.
                "url": (
                    google_news_url
                ),

                "google_news_url": (
                    google_news_url
                ),

                "title": (
                    clean_text(
                        item.get(
                            "title",
                            "",
                        )
                    )
                ),

                "published_at": (
                    published_at.strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    )
                ),

                "language": (
                    "English"
                ),

                "source_country": (
                    "Australia"
                ),

                "collection_month": (
                    month
                ),

                "discovery_source": (
                    "google_news"
                ),

                "collected_at": (
                    utc_now_string()
                ),
            }
        )

    return rows


# ---------------------------------------------------------------------
# One collection batch
# ---------------------------------------------------------------------

def collect_batch(
    topic: str,
    domain: str,
    month_batch: dict,
    keyword_group_number: int,
    keywords: list[str],
    candidates: pd.DataFrame,
    progress: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Collect one Google News RSS batch.
    """

    publisher = (
        get_publisher_name(
            domain
        )
    )

    month = (
        month_batch[
            "month"
        ]
    )

    query = (
        build_search_query(
            keywords=(
                keywords
            ),
            domain=(
                domain
            ),
            after_date=(
                month_batch[
                    "after"
                ]
            ),
            before_date=(
                month_batch[
                    "before"
                ]
            ),
        )
    )

    print()
    print(
        "=" * 70
    )

    print(
        f"{topic} | "
        f"{publisher} | "
        f"{domain}"
    )

    print(
        f"Month: "
        f"{month}"
    )

    print(
        f"Keyword group: "
        f"{keyword_group_number}"
    )

    print(
        "Terms: "
        + ", ".join(
            keywords
        )
    )

    print(
        f"Window: "
        f"{month_batch['start_date'].date()} "
        f"to "
        f"{month_batch['end_date'].date()}"
    )

    print(
        "=" * 70
    )

    (
        items,
        http_status,
        error,
    ) = request_google_news(
        query
    )

    raw_results = len(
        items
    )

    valid_rows = (
        convert_items_to_rows(
            items=(
                items
            ),
            topic=(
                topic
            ),
            publisher=(
                publisher
            ),
            domain=(
                domain
            ),
            month=(
                month
            ),
            keyword_group=(
                keyword_group_number
            ),
            keywords=(
                keywords
            ),
            start_date=(
                month_batch[
                    "start_date"
                ]
            ),
            end_date=(
                month_batch[
                    "end_date"
                ]
            ),
        )
    )

    valid_results = len(
        valid_rows
    )

    before_urls = set(
        candidates[
            "url"
        ]
        .dropna()
        .astype(str)
    )

    if valid_rows:

        candidates = pd.concat(
            [
                candidates,
                pd.DataFrame(
                    valid_rows
                ),
            ],
            ignore_index=True,
        )

        candidates = (
            candidates.drop_duplicates(
                subset=[
                    "url"
                ],
                keep="first",
            )
        )

    after_urls = set(
        candidates[
            "url"
        ]
        .dropna()
        .astype(str)
    )

    new_unique_results = len(
        after_urls
        - before_urls
    )

    status = (
        "failed"
        if error
        else "completed"
    )

    progress_record = {
        "topic": (
            topic
        ),
        "publisher": (
            publisher
        ),
        "domain": (
            normalise_domain(
                domain
            )
        ),
        "collection_month": (
            month
        ),
        "keyword_group": (
            keyword_group_number
        ),
        "search_terms": (
            " | ".join(
                keywords
            )
        ),
        "start_date": (
            month_batch[
                "start_date"
            ].strftime(
                "%Y-%m-%d"
            )
        ),
        "end_date": (
            month_batch[
                "end_date"
            ].strftime(
                "%Y-%m-%d"
            )
        ),
        "status": (
            status
        ),
        "raw_results": (
            raw_results
        ),
        "valid_results": (
            valid_results
        ),
        "new_unique_results": (
            new_unique_results
        ),
        "http_status": (
            http_status
        ),
        "error": (
            error
        ),
        "updated_at": (
            utc_now_string()
        ),
    }

    progress = (
        update_progress(
            progress,
            progress_record,
        )
    )

    save_candidates(
        candidates
    )

    save_progress(
        progress
    )

    print()

    print(
        f"Raw RSS results: "
        f"{raw_results}"
    )

    print(
        f"Valid results: "
        f"{valid_results}"
    )

    print(
        f"New unique candidates: "
        f"{new_unique_results}"
    )

    print(
        "Total Google News candidates: "
        f"{len(candidates)}"
    )

    print(
        f"Batch status: "
        f"{status}"
    )

    if error:

        print(
            f"Error: "
            f"{error[:300]}"
        )

    return (
        candidates,
        progress,
    )


# ---------------------------------------------------------------------
# Collection controller
# ---------------------------------------------------------------------

def collect(
    topics_to_run: list[str],
    domains_to_run: list[str],
    months_to_run: list[str] | None = None,
    keyword_groups_to_run: list[int] | None = None,
) -> None:
    """
    Run resumable Google News candidate collection.
    """

    ensure_output_directory()

    candidates = (
        load_candidates()
    )

    progress = (
        load_progress()
    )

    completed = (
        completed_batch_keys(
            progress
        )
    )

    month_batches = (
        get_month_batches()
    )

    if months_to_run:

        month_batches = [
            batch
            for batch
            in month_batches
            if (
                batch[
                    "month"
                ]
                in months_to_run
            )
        ]

    total_batches = 0

    skipped_batches = 0

    attempted_batches = 0

    for topic in topics_to_run:

        if topic not in TOPICS:

            print(
                f"Unknown topic: "
                f"{topic}"
            )

            continue

        keyword_groups = (
            split_keywords(
                TOPICS[
                    topic
                ]
            )
        )

        for domain in domains_to_run:

            for month_batch in month_batches:

                for (
                    group_index,
                    keywords,
                ) in enumerate(
                    keyword_groups,
                    start=1,
                ):

                    if (
                        keyword_groups_to_run
                        and group_index
                        not in keyword_groups_to_run
                    ):

                        continue

                    total_batches += 1

                    key = (
                        batch_key(
                            topic,
                            domain,
                            month_batch[
                                "month"
                            ],
                            group_index,
                        )
                    )

                    if key in completed:

                        skipped_batches += 1

                        print(
                            "Skipping completed: "
                            f"{topic} | "
                            f"{domain} | "
                            f"{month_batch['month']} | "
                            f"group "
                            f"{group_index}"
                        )

                        continue

                    attempted_batches += 1

                    (
                        candidates,
                        progress,
                    ) = collect_batch(
                        topic=(
                            topic
                        ),
                        domain=(
                            domain
                        ),
                        month_batch=(
                            month_batch
                        ),
                        keyword_group_number=(
                            group_index
                        ),
                        keywords=(
                            keywords
                        ),
                        candidates=(
                            candidates
                        ),
                        progress=(
                            progress
                        ),
                    )

                    completed = (
                        completed_batch_keys(
                            progress
                        )
                    )

                    print()

                    print(
                        f"Waiting "
                        f"{REQUEST_DELAY_SECONDS} "
                        "seconds before "
                        "the next request..."
                    )

                    time.sleep(
                        REQUEST_DELAY_SECONDS
                    )

    print()
    print(
        "=" * 70
    )

    print(
        "GOOGLE NEWS COLLECTION RUN COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Batches considered: "
        f"{total_batches}"
    )

    print(
        "Already completed/skipped: "
        f"{skipped_batches}"
    )

    print(
        f"Attempted this run: "
        f"{attempted_batches}"
    )

    print(
        "Unique candidates saved: "
        f"{len(candidates)}"
    )

    print(
        f"Candidates file: "
        f"{OUTPUT_FILE}"
    )

    print(
        f"Progress file: "
        f"{PROGRESS_FILE}"
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """
    Command-line interface.
    """

    parser = (
        argparse.ArgumentParser(
            description=(
                "Collect NewsPrism "
                "candidate article URLs "
                "from Google News RSS."
            )
        )
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Run all topics and all "
            "approved publishers."
        ),
    )

    parser.add_argument(
        "--topic",
        choices=list(
            TOPICS.keys()
        ),
        help=(
            "Run one topic."
        ),
    )

    parser.add_argument(
        "--domain",
        help=(
            "Run one approved publisher."
        ),
    )

    parser.add_argument(
        "--month",
        choices=[
            batch[
                "month"
            ]
            for batch
            in get_month_batches()
        ],
        help=(
            "Optional month filter."
        ),
    )

    parser.add_argument(
        "--group",
        type=int,
        choices=[
            1,
            2,
            3,
        ],
        help=(
            "Optional keyword-group "
            "filter."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    """
    Run Google News collection.
    """

    args = (
        parse_args()
    )

    all_domains = (
        get_all_domains()
    )

    if not all_domains:

        raise RuntimeError(
            "No approved publisher "
            "domains found in config.py."
        )

    if args.all:

        topics_to_run = list(
            TOPICS.keys()
        )

        domains_to_run = (
            all_domains
        )

    else:

        if not args.topic:

            raise SystemExit(
                "Choose --all or provide "
                "--topic."
            )

        topics_to_run = [
            args.topic
        ]

        if args.domain:

            requested_domain = (
                normalise_domain(
                    args.domain
                )
            )

            if (
                requested_domain
                not in all_domains
            ):

                raise SystemExit(
                    "Domain is not in the "
                    "approved PUBLISHERS "
                    "configuration: "
                    f"{requested_domain}"
                )

            domains_to_run = [
                requested_domain
            ]

        else:

            domains_to_run = (
                all_domains
            )

    months_to_run = (
        [
            args.month
        ]
        if args.month
        else None
    )

    groups_to_run = (
        [
            args.group
        ]
        if args.group
        else None
    )

    print()

    print(
        "NEWSPRISM GOOGLE NEWS COLLECTION"
    )

    print(
        "-" * 70
    )

    print(
        "Topics: "
        + ", ".join(
            topics_to_run
        )
    )

    print(
        "Publishers: "
        + ", ".join(
            domains_to_run
        )
    )

    if args.month:

        print(
            f"Month filter: "
            f"{args.month}"
        )

    if args.group:

        print(
            "Keyword group filter: "
            f"{args.group}"
        )

    print(
        f"Project period: "
        f"{START_DATE} "
        f"to "
        f"{END_DATE}"
    )

    print(
        f"Keyword group size: "
        f"{KEYWORDS_PER_GROUP}"
    )

    print(
        "Google News edition: "
        "Australia / English"
    )

    print()

    collect(
        topics_to_run=(
            topics_to_run
        ),
        domains_to_run=(
            domains_to_run
        ),
        months_to_run=(
            months_to_run
        ),
        keyword_groups_to_run=(
            groups_to_run
        ),
    )


if __name__ == "__main__":
    main()