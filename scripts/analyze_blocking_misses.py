from __future__ import annotations

import argparse
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import duckdb
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from normalize import normalize_dataframe_chunk


DEFAULT_S1 = ROOT / "data" / "train_source1.tsv"
DEFAULT_GT = ROOT / "data" / "train_ground_truth.tsv"
DEFAULT_DB = ROOT / "artifacts" / "train_indexes.duckdb"

DEFAULT_SUMMARY = (
    ROOT / "reports" / "blocking_miss_summary.tsv"
)

DEFAULT_EXAMPLES = (
    ROOT / "reports" / "blocking_miss_examples.tsv"
)


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze true S1-S2/S3 pairs missed by the "
            "current V1 blocking system."
        )
    )

    parser.add_argument(
        "--sample",
        type=int,
        default=10_000,
        help="Number of S1 rows to analyze. 0 = full dataset.",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2_000,
    )

    parser.add_argument(
        "--max-examples",
        type=int,
        default=500,
        help="Maximum missed-pair examples to save.",
    )

    parser.add_argument(
        "--source1",
        type=Path,
        default=DEFAULT_S1,
    )

    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=DEFAULT_GT,
    )

    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
    )

    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY,
    )

    parser.add_argument(
        "--examples-output",
        type=Path,
        default=DEFAULT_EXAMPLES,
    )

    return parser.parse_args()


# ============================================================
# HELPERS
# ============================================================

def split_pipe(value: str) -> set[str]:
    if not value:
        return set()

    return {
        x.strip()
        for x in str(value).split("|")
        if x.strip()
    }


def split_space(value: str) -> set[str]:
    if not value:
        return set()

    return {
        x.strip()
        for x in str(value).split()
        if x.strip()
    }


def safe_ratio(a: int, b: int) -> float:
    if b == 0:
        return 0.0

    return a / b


def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0

    return SequenceMatcher(
        None,
        str(a),
        str(b),
    ).ratio()


# ============================================================
# PREPARE GROUND TRUTH
# ============================================================

def prepare_ground_truth(
    con: duckdb.DuckDBPyConnection,
    path: Path,
) -> None:

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE ground_truth AS

        SELECT
            CAST(source1_entity_id AS VARCHAR)
                AS source1_entity_id,

            COALESCE(
                CAST(matched_entity_ids AS VARCHAR),
                ''
            ) AS matched_entity_ids

        FROM read_csv(
            ?,
            delim='\\t',
            header=true,
            auto_detect=true
        )
        """,
        [str(path)],
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE true_pairs AS

        SELECT DISTINCT

            gt.source1_entity_id,

            TRIM(match_item) AS true_entity_id

        FROM ground_truth gt

        CROSS JOIN UNNEST(
            string_split(
                NULLIF(
                    gt.matched_entity_ids,
                    ''
                ),
                ','
            )
        ) AS match_tbl(match_item)

        WHERE TRIM(match_item) <> ''
        """
    )


# ============================================================
# BUILD INDEX KEY SETS FOR MISSED-PAIR ANALYSIS
# ============================================================

def load_index_key_sets(con):
    """
    Load rare-token keys into Python sets.

    This is only for diagnostic analysis of a small sample,
    NOT for the production blocker.
    """

    rows = con.execute(
        """
        SELECT lookup_key
        FROM rare_name_token_index
        """
    ).fetchall()

    rare_name_keys = {
        row[0]
        for row in rows
    }

    rows = con.execute(
        """
        SELECT lookup_key
        FROM rare_address_token_index
        """
    ).fetchall()

    rare_address_keys = {
        row[0]
        for row in rows
    }

    return (
        rare_name_keys,
        rare_address_keys,
    )


# ============================================================
# BUILD S1 SAMPLE
# ============================================================

def load_s1_sample(
    path: Path,
    sample: int,
    chunk_size: int,
) -> pd.DataFrame:

    chunks = []
    collected = 0

    reader = pd.read_csv(
        path,
        sep="\t",
        dtype="string",
        chunksize=chunk_size,
    )

    for raw in reader:

        if sample > 0:
            remaining = sample - collected

            if remaining <= 0:
                break

            if len(raw) > remaining:
                raw = raw.iloc[:remaining].copy()

        normalized = normalize_dataframe_chunk(
            raw
        )

        columns = [
            "entity_id",
            "country_key",
            "name_nfkc",
            "name_compact",
            "name_core_compact",
            "name_tokens",
            "address_nfkc",
            "address_informative_tokens",
            "address_numbers",
            "postal_candidates",
        ]

        available = [
            c for c in columns
            if c in normalized.columns
        ]

        chunks.append(
            normalized[available].copy()
        )

        collected += len(raw)

        if (
            sample > 0
            and collected >= sample
        ):
            break

    if not chunks:
        raise RuntimeError(
            "No Source1 rows loaded."
        )

    return pd.concat(
        chunks,
        ignore_index=True,
    )


# ============================================================
# IDENTIFY WHETHER A TRUE PAIR WAS RETRIEVED BY V1
# ============================================================

def build_retrieval_mask(
    s1: dict,
    target: dict,
    rare_name_keys: set[str],
    rare_address_keys: set[str],
) -> dict:

    country = str(
        s1["country_key"] or ""
    )

    target_id = str(
        target["entity_id"]
    )

    # --------------------------------------------------------
    # Exact / deterministic representations
    # --------------------------------------------------------

    exact_name = (
        bool(s1["name_nfkc"])
        and
        s1["name_nfkc"]
        == target["name_nfkc"]
    )

    compact_name = (
        bool(s1["name_compact"])
        and
        s1["name_compact"]
        == target["name_compact"]
    )

    core_name = (
        bool(s1["name_core_compact"])
        and
        s1["name_core_compact"]
        == target["name_core_compact"]
    )

    exact_address = (
        bool(s1["address_nfkc"])
        and
        s1["address_nfkc"]
        == target["address_nfkc"]
    )

    # --------------------------------------------------------
    # Postal overlap
    # --------------------------------------------------------

    s1_postal = split_pipe(
        s1["postal_candidates"]
    )

    target_postal = split_pipe(
        target["postal_candidates"]
    )

    postal_overlap = (
        s1_postal
        & target_postal
    )

    postal_match = bool(
        postal_overlap
    )

    # --------------------------------------------------------
    # Name token overlap
    # --------------------------------------------------------

    s1_name_tokens = split_space(
        s1["name_tokens"]
    )

    target_name_tokens = split_space(
        target["name_tokens"]
    )

    shared_name_tokens = (
        s1_name_tokens
        & target_name_tokens
    )

    # Check whether at least one shared name token
    # actually exists in the rare-name index.

    indexed_name_tokens = []

    for token in shared_name_tokens:

        key = (
            country
            + "\x1f"
            + token
        )

        if key in rare_name_keys:
            indexed_name_tokens.append(
                token
            )

    rare_name_match = bool(
        indexed_name_tokens
    )

    # --------------------------------------------------------
    # Address token overlap
    # --------------------------------------------------------

    s1_addr_tokens = split_space(
        s1["address_informative_tokens"]
    )

    target_addr_tokens = split_space(
        target["address_informative_tokens"]
    )

    shared_address_tokens = (
        s1_addr_tokens
        & target_addr_tokens
    )

    indexed_address_tokens = []

    for token in shared_address_tokens:

        key = (
            country
            + "\x1f"
            + token
        )

        if key in rare_address_keys:
            indexed_address_tokens.append(
                token
            )

    rare_address_match = bool(
        indexed_address_tokens
    )

    # --------------------------------------------------------
    # Address numbers
    # --------------------------------------------------------

    s1_numbers = split_pipe(
        s1["address_numbers"]
    )

    target_numbers = split_pipe(
        target["address_numbers"]
    )

    shared_numbers = (
        s1_numbers
        & target_numbers
    )

    return {
        "target_entity_id": target_id,

        "exact_name": exact_name,
        "compact_name": compact_name,
        "core_name": core_name,
        "exact_address": exact_address,
        "postal_match": postal_match,

        "shared_name_tokens":
            shared_name_tokens,

        "indexed_name_tokens":
            set(indexed_name_tokens),

        "rare_name_match":
            rare_name_match,

        "shared_address_tokens":
            shared_address_tokens,

        "indexed_address_tokens":
            set(indexed_address_tokens),

        "rare_address_match":
            rare_address_match,

        "shared_numbers":
            shared_numbers,
    }


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():

    args = parse_args()

    print(
        "Loading Source1 sample..."
    )

    s1_df = load_s1_sample(
        args.source1,
        args.sample,
        args.chunk_size,
    )

    print(
        f"S1 rows loaded: "
        f"{len(s1_df):,}"
    )

    con = duckdb.connect(
        str(args.db),
        read_only=False,
    )

    con.execute(
        "PRAGMA threads=4"
    )

    prepare_ground_truth(
        con,
        args.ground_truth,
    )

    # --------------------------------------------------------
    # Register S1 sample
    # --------------------------------------------------------

    con.register(
        "s1_sample",
        s1_df,
    )

    # --------------------------------------------------------
    # Get true pairs for sample
    # --------------------------------------------------------

    pair_df = con.execute(
        """
        SELECT
            t.source1_entity_id,
            t.true_entity_id

        FROM true_pairs t

        INNER JOIN s1_sample s

            ON s.entity_id =
               t.source1_entity_id
        """
    ).df()

    print(
        f"True pairs in sample: "
        f"{len(pair_df):,}"
    )

    if pair_df.empty:
        print(
            "No true pairs found."
        )
        con.close()
        return

    # --------------------------------------------------------
    # Load target entity information
    # --------------------------------------------------------

    target_ids = pair_df[
        "true_entity_id"
    ].astype(str).tolist()

    con.register(
        "sample_true_ids",
        pd.DataFrame(
            {
                "entity_id":
                    target_ids
            }
        )
    )

    target_df = con.execute(
        """
        SELECT DISTINCT

            e.entity_id,

            e.country_key,

            e.business_name,

            e.business_address,

            e.name_nfkc,

            e.name_compact,

            e.name_core_compact,

            e.name_tokens,

            e.address_nfkc,

            e.address_informative_tokens,

            e.address_numbers,

            e.postal_candidates

        FROM entities e

        INNER JOIN sample_true_ids t

            ON t.entity_id =
               e.entity_id
        """
    ).df()

    target_map = {
        row["entity_id"]: row
        for _, row in target_df.iterrows()
    }

    s1_map = {
        row["entity_id"]: row
        for _, row in s1_df.iterrows()
    }

    # --------------------------------------------------------
    # Rare token index keys
    # --------------------------------------------------------

    print(
        "Loading rare-token diagnostic sets..."
    )

    (
        rare_name_keys,
        rare_address_keys,
    ) = load_index_key_sets(
        con
    )

    # --------------------------------------------------------
    # Analyze pairs
    # --------------------------------------------------------

    missed_records = []

    total_pairs = 0
    recovered_pairs = 0
    missed_pairs = 0

    reason_counter = Counter()

    for _, pair in pair_df.iterrows():

        s1_id = str(
            pair["source1_entity_id"]
        )

        target_id = str(
            pair["true_entity_id"]
        )

        if (
            s1_id not in s1_map
            or
            target_id not in target_map
        ):
            continue

        s1 = s1_map[s1_id]
        target = target_map[target_id]

        evidence = build_retrieval_mask(
            s1,
            target,
            rare_name_keys,
            rare_address_keys,
        )

        total_pairs += 1

        retrieved = any(
            [
                evidence["exact_name"],
                evidence["compact_name"],
                evidence["core_name"],
                evidence["exact_address"],
                evidence["postal_match"],
                evidence["rare_name_match"],
                evidence["rare_address_match"],
            ]
        )

        if retrieved:

            recovered_pairs += 1

            continue

        missed_pairs += 1

        shared_name = (
            evidence["shared_name_tokens"]
        )

        indexed_name = (
            evidence["indexed_name_tokens"]
        )

        shared_address = (
            evidence["shared_address_tokens"]
        )

        indexed_address = (
            evidence["indexed_address_tokens"]
        )

        # ----------------------------------------------------
        # Diagnose miss
        # ----------------------------------------------------

        if (
            shared_name
            and
            not indexed_name
            and
            shared_address
            and
            not indexed_address
        ):
            reason = (
                "shared_name_and_address_tokens_but_all_suppressed"
            )

        elif (
            shared_name
            and
            not indexed_name
        ):
            reason = (
                "shared_name_tokens_but_suppressed"
            )

        elif (
            shared_address
            and
            not indexed_address
        ):
            reason = (
                "shared_address_tokens_but_suppressed"
            )

        elif shared_name:
            reason = (
                "name_token_shared_but_not_indexed"
            )

        elif shared_address:
            reason = (
                "address_token_shared_but_not_indexed"
            )

        else:
            reason = (
                "no_exact_or_indexed_token_evidence"
            )

        reason_counter[
            reason
        ] += 1

        # ----------------------------------------------------
        # Similarity evidence
        # ----------------------------------------------------

        name_similarity = similarity(
            str(s1["name_compact"]),
            str(target["name_compact"]),
        )

        address_similarity = similarity(
            str(s1["address_nfkc"]),
            str(target["address_nfkc"]),
        )

        shared_numbers = (
            evidence["shared_numbers"]
        )

        missed_records.append(
            {
                "source1_entity_id":
                    s1_id,

                "true_entity_id":
                    target_id,

                "country":
                    s1["country_key"],

                "source1_business_name":
                    s1["name_nfkc"],

                "true_business_name":
                    target["business_name"],

                "source1_address":
                    s1["address_nfkc"],

                "true_address":
                    target["business_address"],

                "name_similarity":
                    round(
                        name_similarity,
                        4,
                    ),

                "address_similarity":
                    round(
                        address_similarity,
                        4,
                    ),

                "exact_name":
                    evidence["exact_name"],

                "compact_name":
                    evidence["compact_name"],

                "core_name":
                    evidence["core_name"],

                "exact_address":
                    evidence["exact_address"],

                "postal_match":
                    evidence["postal_match"],

                "shared_name_tokens":
                    "|".join(
                        sorted(shared_name)
                    ),

                "indexed_name_tokens":
                    "|".join(
                        sorted(indexed_name)
                    ),

                "shared_address_tokens":
                    "|".join(
                        sorted(shared_address)
                    ),

                "indexed_address_tokens":
                    "|".join(
                        sorted(indexed_address)
                    ),

                "shared_address_numbers":
                    "|".join(
                        sorted(shared_numbers)
                    ),

                "name_token_overlap_count":
                    len(shared_name),

                "address_token_overlap_count":
                    len(shared_address),

                "address_number_overlap_count":
                    len(shared_numbers),

                "reason":
                    reason,
            }
        )

    # ========================================================
    # RESULTS
    # ========================================================

    final_recall = safe_ratio(
        recovered_pairs,
        total_pairs,
    )

    miss_rate = safe_ratio(
        missed_pairs,
        total_pairs,
    )

    print()
    print(
        "=" * 70
    )
    print(
        "BLOCKING MISS ANALYSIS"
    )
    print(
        "=" * 70
    )

    print(
        f"S1 rows analyzed           : "
        f"{len(s1_df):,}"
    )

    print(
        f"True pairs analyzed        : "
        f"{total_pairs:,}"
    )

    print(
        f"Recovered by V1            : "
        f"{recovered_pairs:,}"
    )

    print(
        f"Missed by V1               : "
        f"{missed_pairs:,}"
    )

    print(
        f"Measured V1 recall         : "
        f"{final_recall:.4%}"
    )

    print(
        f"Miss rate                  : "
        f"{miss_rate:.4%}"
    )

    print()
    print(
        "MISS REASONS"
    )
    print(
        "-" * 70
    )

    for reason, count in (
        reason_counter.most_common()
    ):

        print(
            f"{reason:55s} "
            f"{count:,} "
            f"({safe_ratio(count, missed_pairs):.2%})"
        )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary_rows = []

    for reason, count in (
        reason_counter.most_common()
    ):

        summary_rows.append(
            {
                "reason": reason,

                "missed_pairs":
                    count,

                "share_of_misses":
                    safe_ratio(
                        count,
                        missed_pairs,
                    ),

                "share_of_all_true_pairs":
                    safe_ratio(
                        count,
                        total_pairs,
                    ),
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    args.summary_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_df.to_csv(
        args.summary_output,
        sep="\t",
        index=False,
    )

    # --------------------------------------------------------
    # Save examples
    # --------------------------------------------------------

    example_df = pd.DataFrame(
        missed_records
    )

    if not example_df.empty:

        example_df = (
            example_df
            .sort_values(
                [
                    "reason",
                    "name_similarity",
                ],
                ascending=[
                    True,
                    False,
                ],
            )
            .head(
                args.max_examples
            )
        )

    example_df.to_csv(
        args.examples_output,
        sep="\t",
        index=False,
    )

    print()
    print(
        f"Summary report              : "
        f"{args.summary_output}"
    )

    print(
        f"Missed-pair examples        : "
        f"{args.examples_output}"
    )

    print(
        "=" * 70
    )

    con.unregister(
        "s1_sample"
    )

    con.unregister(
        "sample_true_ids"
    )

    con.close()


if __name__ == "__main__":
    main()