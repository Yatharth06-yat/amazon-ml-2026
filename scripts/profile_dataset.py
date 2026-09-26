from pathlib import Path
from collections import Counter
import json
import re

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"

CHUNK_SIZE = 200_000

FILES = {
    "train_source1": DATA_DIR / "train_source1.tsv",
    "train_source2": DATA_DIR / "train_source2.tsv",
    "train_source3": DATA_DIR / "train_source3.tsv",
    "train_ground_truth": DATA_DIR / "train_ground_truth.tsv",
    "test_source1": DATA_DIR / "test_source1.tsv",
    "test_source2": DATA_DIR / "test_source2.tsv",
    "test_source3": DATA_DIR / "test_source3.tsv",
}

TEXT_COLUMNS = [
    "business_name",
    "business_address",
    "country",
]


# ============================================================
# HELPERS
# ============================================================

def clean_text(series: pd.Series) -> pd.Series:
    """Basic missing/whitespace handling for EDA only."""
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
    )


def update_numeric_stats(stats, values):
    """Online min/max/sum/sumsq/count update."""
    if len(values) == 0:
        return

    values = values.astype("float64")

    stats["count"] += len(values)
    stats["sum"] += float(values.sum())
    stats["sumsq"] += float((values * values).sum())
    stats["min"] = min(stats["min"], float(values.min()))
    stats["max"] = max(stats["max"], float(values.max()))


def finalize_numeric_stats(stats):
    if stats["count"] == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }

    mean = stats["sum"] / stats["count"]

    variance = (
        stats["sumsq"] / stats["count"]
        - mean * mean
    )

    variance = max(variance, 0.0)

    return {
        "count": stats["count"],
        "mean": round(mean, 3),
        "std": round(variance ** 0.5, 3),
        "min": stats["min"],
        "max": stats["max"],
    }


def top_items(counter, n=20):
    return [
        {
            "value": str(value),
            "count": int(count),
        }
        for value, count in counter.most_common(n)
    ]


# ============================================================
# SOURCE PROFILING
# ============================================================

def profile_source(path: Path):
    print(f"\n{'=' * 70}")
    print(f"Profiling: {path.name}")
    print(f"{'=' * 70}")

    if not path.exists():
        print(f"WARNING: File not found: {path}")
        return None

    row_count = 0

    missing_counts = Counter()
    country_counter = Counter()

    # We only keep chunk-local top candidates to avoid huge RAM use.
    name_counter = Counter()
    address_counter = Counter()

    name_length_stats = {
        "count": 0,
        "sum": 0.0,
        "sumsq": 0.0,
        "min": float("inf"),
        "max": float("-inf"),
    }

    address_length_stats = {
        "count": 0,
        "sum": 0.0,
        "sumsq": 0.0,
        "min": float("inf"),
        "max": float("-inf"),
    }

    try:
        reader = pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            chunksize=CHUNK_SIZE,
            keep_default_na=False,
            on_bad_lines="warn",
        )

        columns = None

        for chunk_no, chunk in enumerate(reader, start=1):
            if columns is None:
                columns = list(chunk.columns)

            rows = len(chunk)
            row_count += rows

            # ------------------------------------------------
            # Missingness
            # ------------------------------------------------
            for col in columns:
                if col not in chunk.columns:
                    continue

                values = clean_text(chunk[col])

                missing = int((values == "").sum())
                missing_counts[col] += missing

            # ------------------------------------------------
            # Country
            # ------------------------------------------------
            if "country" in chunk.columns:
                countries = clean_text(chunk["country"])
                countries = countries[countries != ""]
                country_counter.update(countries.tolist())

            # ------------------------------------------------
            # Name statistics
            # ------------------------------------------------
            if "business_name" in chunk.columns:
                names = clean_text(chunk["business_name"])

                non_empty_names = names[names != ""]

                if len(non_empty_names):
                    lengths = non_empty_names.str.len().astype("float64")
                    update_numeric_stats(
                        name_length_stats,
                        lengths
                    )

                    # Chunk-local candidates
                    vc = non_empty_names.value_counts().head(100)
                    name_counter.update(vc.to_dict())

            # ------------------------------------------------
            # Address statistics
            # ------------------------------------------------
            if "business_address" in chunk.columns:
                addresses = clean_text(chunk["business_address"])

                non_empty_addresses = addresses[addresses != ""]

                if len(non_empty_addresses):
                    lengths = (
                        non_empty_addresses
                        .str.len()
                        .astype("float64")
                    )

                    update_numeric_stats(
                        address_length_stats,
                        lengths
                    )

                    vc = non_empty_addresses.value_counts().head(100)
                    address_counter.update(vc.to_dict())

            if chunk_no % 5 == 0:
                print(
                    f"  Processed ~{row_count:,} rows..."
                )

        # ----------------------------------------------------
        # Final stats
        # ----------------------------------------------------

        missingness = {}

        for col in columns:
            missing = missing_counts[col]

            missingness[col] = {
                "missing_or_empty": missing,
                "missing_pct": round(
                    100.0 * missing / row_count,
                    4
                ) if row_count else 0.0,
            }

        country_distribution = []

        for country, count in country_counter.most_common():
            country_distribution.append(
                {
                    "country": country,
                    "count": int(count),
                    "percentage": round(
                        100.0 * count / row_count,
                        4
                    ) if row_count else 0.0,
                }
            )

        result = {
            "file": path.name,
            "rows": row_count,
            "columns": columns,
            "missingness": missingness,
            "name_length": finalize_numeric_stats(
                name_length_stats
            ),
            "address_length": finalize_numeric_stats(
                address_length_stats
            ),
            "top_names": top_items(name_counter, 20),
            "top_addresses": top_items(address_counter, 20),
            "country_distribution": country_distribution,
        }

        print(f"\nRows: {row_count:,}")
        print(f"Columns: {columns}")

        print("\nMissingness:")
        for col, data in missingness.items():
            print(
                f"  {col}: "
                f"{data['missing_or_empty']:,} "
                f"({data['missing_pct']:.2f}%)"
            )

        print("\nCountries:")
        for row in country_distribution[:15]:
            print(
                f"  {row['country']}: "
                f"{row['count']:,} "
                f"({row['percentage']:.2f}%)"
            )

        return result

    except Exception as exc:
        print(f"ERROR while profiling {path.name}: {exc}")
        return None


# ============================================================
# GROUND TRUTH PROFILING
# ============================================================

def profile_ground_truth(path: Path):
    print(f"\n{'=' * 70}")
    print("Profiling ground truth")
    print(f"{'=' * 70}")

    if not path.exists():
        print(f"WARNING: File not found: {path}")
        return None

    total_rows = 0
    match_count_distribution = Counter()
    s2_match_distribution = Counter()
    s3_match_distribution = Counter()

    singleton_count = 0
    empty_match_strings = 0

    try:
        reader = pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            chunksize=CHUNK_SIZE,
            keep_default_na=False,
            on_bad_lines="warn",
        )

        for chunk_no, chunk in enumerate(reader, start=1):

            total_rows += len(chunk)

            matches = clean_text(
                chunk["matched_entity_ids"]
            )

            for value in matches:

                if value == "":
                    total_matches = 0
                    s2_count = 0
                    s3_count = 0

                    singleton_count += 1
                    empty_match_strings += 1

                else:
                    ids = [
                        x.strip()
                        for x in value.split(",")
                        if x.strip()
                    ]

                    total_matches = len(ids)

                    s2_count = sum(
                        x.startswith("S2-")
                        for x in ids
                    )

                    s3_count = sum(
                        x.startswith("S3-")
                        for x in ids
                    )

                match_count_distribution[total_matches] += 1
                s2_match_distribution[s2_count] += 1
                s3_match_distribution[s3_count] += 1

            if chunk_no % 5 == 0:
                print(
                    f"  Processed ~{total_rows:,} rows..."
                )

        result = {
            "rows": total_rows,
            "singleton_count": singleton_count,
            "singleton_percentage": round(
                100.0 * singleton_count / total_rows,
                4
            ) if total_rows else 0.0,
            "empty_match_strings": empty_match_strings,
            "match_count_distribution": {
                str(k): int(v)
                for k, v in sorted(
                    match_count_distribution.items()
                )
            },
            "s2_match_distribution": {
                str(k): int(v)
                for k, v in sorted(
                    s2_match_distribution.items()
                )
            },
            "s3_match_distribution": {
                str(k): int(v)
                for k, v in sorted(
                    s3_match_distribution.items()
                )
            },
        }

        print(f"\nGround-truth rows: {total_rows:,}")

        print(
            f"True singletons: "
            f"{singleton_count:,} "
            f"({result['singleton_percentage']:.2f}%)"
        )

        print("\nTotal match-count distribution:")
        for k, v in sorted(match_count_distribution.items()):
            print(f"  {k} matches: {v:,}")

        return result

    except Exception as exc:
        print(f"ERROR while profiling ground truth: {exc}")
        return None


# ============================================================
# REPORT WRITING
# ============================================================

def write_reports(results):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    json_path = REPORT_DIR / "dataset_profile.json"

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False
        )

    # ---------------------------------------------
    # Human-readable summary
    # ---------------------------------------------

    txt_path = REPORT_DIR / "dataset_profile.txt"

    with open(
        txt_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "AMAZON ML CHALLENGE 2026\n"
            "BUSINESS ENTITY RESOLUTION\n"
            "DATASET PROFILE / EDA\n"
            "=" * 70 + "\n\n"
        )

        for name, data in results["sources"].items():

            if not data:
                continue

            f.write(f"\n{name}\n")
            f.write("-" * 50 + "\n")

            f.write(
                f"Rows: {data['rows']:,}\n"
            )

            f.write(
                f"Columns: "
                f"{', '.join(data['columns'])}\n"
            )

            f.write("\nMissingness:\n")

            for col, stats in data["missingness"].items():
                f.write(
                    f"  {col}: "
                    f"{stats['missing_or_empty']:,} "
                    f"({stats['missing_pct']:.2f}%)\n"
                )

            f.write("\nCountry distribution:\n")

            for row in data[
                "country_distribution"
            ][:20]:
                f.write(
                    f"  {row['country']}: "
                    f"{row['count']:,} "
                    f"({row['percentage']:.2f}%)\n"
                )

            f.write("\nName length:\n")
            f.write(
                json.dumps(
                    data["name_length"],
                    indent=2
                )
                + "\n"
            )

            f.write("\nAddress length:\n")
            f.write(
                json.dumps(
                    data["address_length"],
                    indent=2
                )
                + "\n"
            )

            f.write("\nTop names (chunk-based candidates):\n")
            for row in data["top_names"]:
                f.write(
                    f"  {row['count']:,}  "
                    f"{row['value']}\n"
                )

        gt = results["ground_truth"]

        if gt:
            f.write(
                "\n\nGROUND TRUTH\n"
                + "=" * 70
                + "\n"
            )

            f.write(
                f"Rows: {gt['rows']:,}\n"
            )

            f.write(
                f"True singletons: "
                f"{gt['singleton_count']:,} "
                f"({gt['singleton_percentage']:.2f}%)\n"
            )

            f.write(
                "\nMatch-count distribution:\n"
            )

            for k, v in gt[
                "match_count_distribution"
            ].items():
                f.write(
                    f"  {k} matches: {v:,}\n"
                )

    # ---------------------------------------------
    # CSV outputs
    # ---------------------------------------------

    country_rows = []

    for source_name, data in results["sources"].items():

        if not data:
            continue

        for row in data[
            "country_distribution"
        ]:
            country_rows.append(
                {
                    "source": source_name,
                    **row,
                }
            )

    if country_rows:
        pd.DataFrame(country_rows).to_csv(
            REPORT_DIR / "country_distribution.csv",
            index=False,
        )

    missing_rows = []

    for source_name, data in results["sources"].items():

        if not data:
            continue

        for col, stats in data["missingness"].items():
            missing_rows.append(
                {
                    "source": source_name,
                    "column": col,
                    **stats,
                }
            )

    if missing_rows:
        pd.DataFrame(missing_rows).to_csv(
            REPORT_DIR / "missingness.csv",
            index=False,
        )

    gt = results["ground_truth"]

    if gt:

        gt_rows = [
            {
                "match_count": int(k),
                "count": int(v),
            }
            for k, v in gt[
                "match_count_distribution"
            ].items()
        ]

        pd.DataFrame(gt_rows).to_csv(
            REPORT_DIR / "ground_truth_match_distribution.csv",
            index=False,
        )

    print("\nReports written to:")
    print(f"  {json_path}")
    print(f"  {txt_path}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AMAZON ML CHALLENGE 2026")
    print("BUSINESS ENTITY RESOLUTION")
    print("LARGE-SCALE DATASET PROFILING")
    print("=" * 70)

    results = {
        "sources": {},
        "ground_truth": None,
    }

    # Profile all source files
    for name, path in FILES.items():

        if name == "train_ground_truth":
            continue

        results["sources"][name] = profile_source(path)

    # Ground truth
    results["ground_truth"] = profile_ground_truth(
        FILES["train_ground_truth"]
    )

    # Save
    write_reports(results)

    print("\nEDA profiling complete.")


if __name__ == "__main__":
    main()