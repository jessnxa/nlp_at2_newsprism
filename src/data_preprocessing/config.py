from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = DATA_DIR / "reports"


START_DATE = "2026-07-01"
END_DATE = "2026-09-30"

TARGET_ARTICLES_PER_TOPIC = 100
MIN_WORD_COUNT = 200


TOPICS = {
    "immigration": [
        "immigration",
        "migration policy",
        "immigration policy",
        "skilled migration",
        "skilled migrants",
        "migration levels",
        "migration intake",
        "visa policy",
        "student visa",
        "temporary visa",
        "permanent migration",
        "migrant workers",
        "asylum seeker",
        "refugees",
        "border policy",
    ],

    "clean_energy": [
        "clean energy",
        "renewable energy",
        "renewable power",
        "energy transition",
        "net zero",
        "solar energy",
        "solar power",
        "wind energy",
        "wind power",
        "green energy",
        "clean electricity",
        "renewable electricity",
        "battery storage",
        "energy storage",
        "emissions reduction",
    ],

    "housing": [
        "housing crisis",
        "housing affordability",
        "housing shortage",
        "housing supply",
        "housing policy",
        "affordable housing",
        "social housing",
        "rental crisis",
        "rental affordability",
        "rental prices",
        "rent prices",
        "house prices",
        "property prices",
        "home ownership",
        "first home buyers",
    ],
}


RELEVANCE_CONCEPTS = {
    "immigration": {
        "migration": [
            "immigration",
            "migration",
            "migrant",
            "migrants",
        ],
        "visa": [
            "visa",
            "visas",
        ],
        "asylum_refugee": [
            "asylum seeker",
            "asylum seekers",
            "refugee",
            "refugees",
        ],
        "border": [
            "border policy",
            "border control",
        ],
    },

    "clean_energy": {
        "renewables": [
            "renewable energy",
            "renewable power",
            "renewables",
            "clean energy",
            "green energy",
        ],
        "solar": [
            "solar energy",
            "solar power",
            "solar farm",
            "solar farms",
        ],
        "wind": [
            "wind energy",
            "wind power",
            "wind farm",
            "wind farms",
        ],
        "transition": [
            "energy transition",
            "net zero",
            "emissions reduction",
        ],
        "storage": [
            "battery storage",
            "energy storage",
            "battery",
            "batteries",
        ],
    },

    "housing": {
        "housing": [
            "housing",
            "home ownership",
            "homeowner",
            "homeowners",
        ],
        "renting": [
            "rent",
            "rents",
            "rental",
            "renters",
        ],
        "prices": [
            "house prices",
            "property prices",
            "home prices",
            "housing prices",
        ],
        "supply": [
            "housing supply",
            "housing shortage",
            "housing stock",
        ],
        "buyers": [
            "first home buyer",
            "first home buyers",
            "home buyer",
            "home buyers",
        ],
    },
}


PUBLISHERS = {
    "abc.net.au": "ABC News",
    "sbs.com.au": "SBS News",
    "smh.com.au": "Sydney Morning Herald",
    "theage.com.au": "The Age",
    "theguardian.com": "The Guardian Australia",
    "9news.com.au": "9News",
    "7news.com.au": "7NEWS",
    "news.com.au": "News.com.au",
    "skynews.com.au": "Sky News Australia",
}


if __name__ == "__main__":
    print(f"Collection period: {START_DATE} to {END_DATE}")
    print(f"Target per topic: {TARGET_ARTICLES_PER_TOPIC}")
    print(f"Minimum word count: {MIN_WORD_COUNT}")
    print(f"Topics: {len(TOPICS)}")
    print(f"Publishers: {len(PUBLISHERS)}")

    for topic, keywords in TOPICS.items():
        print(f"\n{topic} ({len(keywords)} search keywords):")

        for keyword in keywords:
            print(f"  - {keyword}")

        print(
            f"Relevance concepts: "
            f"{len(RELEVANCE_CONCEPTS[topic])}"
        )