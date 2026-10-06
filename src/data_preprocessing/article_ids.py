import hashlib
import re

import pandas as pd


def normalise_url(url):
    if pd.isna(url):
        return ""

    url = str(url).strip().lower()

    url = re.sub(
        r"[?#].*$",
        "",
        url,
    )

    return url.rstrip("/")


def create_article_id(url):
    normalised_url = normalise_url(url)

    if not normalised_url:
        return None

    digest = hashlib.sha256(
        normalised_url.encode("utf-8")
    ).hexdigest()

    return f"ART_{digest[:12].upper()}"


def add_article_ids(dataframe):
    dataframe = dataframe.copy()

    dataframe["article_id"] = dataframe[
        "url"
    ].apply(create_article_id)

    return dataframe


if __name__ == "__main__":
    examples = [
        "https://www.abc.net.au/news/example-story",
        "https://www.abc.net.au/news/example-story/",
        "https://www.abc.net.au/news/example-story?utm_source=test",
    ]

    print("NewsPrism article ID test")
    print("-------------------------")

    for url in examples:
        print()
        print(f"URL: {url}")
        print(f"ID:  {create_article_id(url)}")