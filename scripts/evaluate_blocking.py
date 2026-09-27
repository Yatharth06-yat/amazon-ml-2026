from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from normalize import normalize_dataframe_chunk


DEFAULT_S1 = ROOT / "data" / "train_source1.tsv"
DEFAULT_GT = ROOT / "data" / "train_ground_truth.tsv"
DEFAULT_DB = ROOT / "artifacts" / "train_indexes.duckdb"
DEFAULT_REPORT = ROOT / "reports" / "blocking_eval_per_s1.tsv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate current blocking recall against training ground truth."
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
        "--output",
        type=Path,
        default=DEFAULT_REPORT,
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
    )

    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Number of S1 rows to evaluate. 0 = full dataset.",
    )

    return parser.parse_args()


def create_ground_truth_table(
    con: duckdb.DuckDBPyConnection,
    ground_truth_path: Path,
) -> None:
    print("Loading ground truth into DuckDB...")

    con.execute("DROP TABLE IF EXISTS ground_truth")

    con.execute(
        """
        CREATE TABLE ground_truth AS
        SELECT
            CAST(source1_entity_id AS VARCHAR) AS source1_entity_id,
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
        [str(ground_truth_path)],
    )

    row_count = con.execute(
        "SELECT COUNT(*) FROM ground_truth"
    ).fetchone()[0]

    print(f"Ground-truth rows loaded: {row_count:,}")


def build_candidate_query() -> str:
    return """
    WITH candidates AS (

        /* ---------------------------------------------------------
           1. Exact normalized name
           --------------------------------------------------------- */
        SELECT
            s.source1_entity_id,
            i.entity_id AS candidate_entity_id
        FROM s1_chunk s
        JOIN exact_name_index i
          ON i.lookup_key =
             s.country_key || chr(31) || s.name_nfkc
        WHERE s.name_nfkc <> ''

        UNION ALL

        /* ---------------------------------------------------------
           2. Compact name
           --------------------------------------------------------- */
        SELECT
            s.source1_entity_id,
            i.entity_id AS candidate_entity_id
        FROM s1_chunk s
        JOIN compact_name_index i
          ON i.lookup_key =
             s.country_key || chr(31) || s.name_compact
        WHERE s.name_compact <> ''

        UNION ALL

        /* ---------------------------------------------------------
           3. Core name
           --------------------------------------------------------- */
        SELECT
            s.source1_entity_id,
            i.entity_id AS candidate_entity_id
        FROM s1_chunk s
        JOIN core_name_index i
          ON i.lookup_key =
             s.country_key || chr(31) || s.name_core_compact
        WHERE s.name_core_compact <> ''

        UNION ALL

        /* ---------------------------------------------------------
           4. Exact normalized address
           --------------------------------------------------------- */
        SELECT
            s.source1_entity_id,
            i.entity_id AS candidate_entity_id
        FROM s1_chunk s
        JOIN exact_address_index i
          ON i.lookup_key =
             s.country_key || chr(31) || s.address_nfkc
        WHERE s.address_nfkc <> ''

        UNION ALL

        /* ---------------------------------------------------------
           5. Postal / PIN
           --------------------------------------------------------- */
        SELECT
            s.source1_entity_id,
            i.entity_id AS candidate_entity_id
        FROM s1_chunk s
        CROSS JOIN UNNEST(
            string_split(
                NULLIF(s.postal_candidates, ''),
                '|'
            )
        ) AS p(postal_value)
        JOIN postal_index i
          ON i.lookup_key =
             s.country_key || chr(31) || p.postal_value
        WHERE p.postal_value <> ''

        UNION ALL

        /* ---------------------------------------------------------
           6. Rare name token
           --------------------------------------------------------- */
        SELECT
            s.source1_entity_id,
            i.entity_id AS candidate_entity_id
        FROM s1_chunk s
        CROSS JOIN UNNEST(
            string_split(
                NULLIF(s.name_tokens, ''),
                ' '
            )
        ) AS t(token_value)
        JOIN rare_name_token_index i
          ON i.lookup_key =
             s.country_key || chr(31) || t.token_value
        WHERE t.token_value <> ''

        UNION ALL

        /* ---------------------------------------------------------
           7. Rare informative address token
           --------------------------------------------------------- */
        SELECT
            s.source1_entity_id,
            i.entity_id AS candidate_entity_id
        FROM s1_chunk s
        CROSS JOIN UNNEST(
            string_split(
                NULLIF(s.address_informative_tokens, ''),
                ' '
            )
        ) AS t(token_value)
        JOIN rare_address_token_index i
          ON i.lookup_key =
             s.country_key || chr(31) || t.token_value
        WHERE t.token_value <> ''
    ),

    candidate_distinct AS (
        SELECT DISTINCT
            source1_entity_id,
            candidate_entity_id
        FROM candidates
    ),

    candidate_counts AS (
        SELECT
            source1_entity_id,
            COUNT(*) AS candidate_count
        FROM candidate_distinct
        GROUP BY source1_entity_id
    ),

    true_pairs AS (
        SELECT
            gt.source1_entity_id,
            TRIM(u.match_id) AS true_entity_id
        FROM ground_truth gt
        CROSS JOIN UNNEST(
            string_split(
                NULLIF(gt.matched_entity_ids, ''),
                ','
            )
        ) AS u(match_id)
        WHERE TRIM(u.match_id) <> ''
    ),

    true_counts AS (
        SELECT
            source1_entity_id,
            COUNT(*) AS true_count
        FROM true_pairs
        GROUP BY source1_entity_id
    ),

    recovered_counts AS (
        SELECT
            c.source1_entity_id,
            COUNT(*) AS recovered_count
        FROM candidate_distinct c
        JOIN true_pairs g
          ON g.source1_entity_id = c.source1_entity_id
         AND g.true_entity_id = c.candidate_entity_id
        GROUP BY c.source1_entity_id
    )

    SELECT
        s.source1_entity_id,

        COALESCE(tc.true_count, 0) AS true_count,

        COALESCE(cc.candidate_count, 0) AS candidate_count,

        COALESCE(rc.recovered_count, 0) AS recovered_count

    FROM s1_chunk s

    LEFT JOIN true_counts tc
      ON tc.source1_entity_id = s.source1_entity_id

    LEFT JOIN candidate_counts cc
      ON cc.source1_entity_id = s.source1_entity_id

    LEFT JOIN recovered_counts rc
      ON rc.source1_entity_id = s.source1_entity_id
    """


def print_running_stats(
    processed: int,
    true_pairs: int,
    recovered_pairs: int,
) -> None:

    recall = (
        recovered_pairs / true_pairs
        if true_pairs
        else 0.0
    )

    print(
        f"Processed: {processed:,} | "
        f"True pairs: {true_pairs:,} | "
        f"Recovered: {recovered_pairs:,} | "
        f"Recall: {recall:.4%}"
    )


def evaluate(
    source1_path: Path,
    ground_truth_path: Path,
    db_path: Path,
    output_path: Path,
    chunk_size: int,
    sample: int,
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if output_path.exists():
        output_path.unlink()

    con = duckdb.connect(
        str(db_path),
        read_only=False,
    )

    con.execute("PRAGMA threads=4")

    create_ground_truth_table(
        con,
        ground_truth_path,
    )

    candidate_query = build_candidate_query()

    use_header = True

    total_s1 = 0
    total_true_pairs = 0
    total_recovered_pairs = 0

    singleton_total = 0
    singleton_zero_candidate = 0

    multi_total_entities = 0
    multi_true_pairs = 0
    multi_recovered_pairs = 0

    reader = pd.read_csv(
        source1_path,
        sep="\t",
        dtype="string",
        chunksize=chunk_size,
    )

    for chunk_number, raw_chunk in enumerate(reader, start=1):

        if sample and total_s1 >= sample:
            break

        if sample:
            remaining = sample - total_s1

            if len(raw_chunk) > remaining:
                raw_chunk = raw_chunk.iloc[:remaining].copy()

        print(
            f"\nProcessing chunk {chunk_number} "
            f"({len(raw_chunk):,} rows)"
        )

        normalized = normalize_dataframe_chunk(
            raw_chunk
        )

        s1_chunk = pd.DataFrame(
            {
                "source1_entity_id":
                    normalized["entity_id"].astype("string"),

                "country_key":
                    normalized["country_key"].astype("string"),

                "name_nfkc":
                    normalized["name_nfkc"].astype("string"),

                "name_compact":
                    normalized["name_compact"].astype("string"),

                "name_core_compact":
                    normalized["name_core_compact"].astype("string"),

                "name_tokens":
                    normalized["name_tokens"].astype("string"),

                "address_nfkc":
                    normalized["address_nfkc"].astype("string"),

                "address_informative_tokens":
                    normalized[
                        "address_informative_tokens"
                    ].astype("string"),

                "postal_candidates":
                    normalized["postal_candidates"].astype("string"),
            }
        )

        con.register(
            "s1_chunk",
            s1_chunk,
        )

        result = con.execute(
            candidate_query
        ).df()

        result["true_count"] = (
            result["true_count"]
            .fillna(0)
            .astype("int64")
        )

        result["candidate_count"] = (
            result["candidate_count"]
            .fillna(0)
            .astype("int64")
        )

        result["recovered_count"] = (
            result["recovered_count"]
            .fillna(0)
            .astype("int64")
        )

        result.to_csv(
            output_path,
            sep="\t",
            index=False,
            mode="w" if use_header else "a",
            header=use_header,
        )

        use_header = False

        chunk_s1 = len(result)

        chunk_true = int(
            result["true_count"].sum()
        )

        chunk_recovered = int(
            result["recovered_count"].sum()
        )

        total_s1 += chunk_s1
        total_true_pairs += chunk_true
        total_recovered_pairs += chunk_recovered

        singleton_mask = (
            result["true_count"] == 0
        )

        singleton_total += int(
            singleton_mask.sum()
        )

        singleton_zero_candidate += int(
            (
                singleton_mask
                & (result["candidate_count"] == 0)
            ).sum()
        )

        multi_mask = (
            result["true_count"] >= 2
        )

        multi_total_entities += int(
            multi_mask.sum()
        )

        multi_true_pairs += int(
            result.loc[
                multi_mask,
                "true_count"
            ].sum()
        )

        multi_recovered_pairs += int(
            result.loc[
                multi_mask,
                "recovered_count"
            ].sum()
        )

        print_running_stats(
            total_s1,
            total_true_pairs,
            total_recovered_pairs,
        )

    con.unregister("s1_chunk")

    print("\nCalculating final candidate statistics...")

    stats = con.execute(
        """
        SELECT
            COUNT(*) AS s1_entities,

            AVG(candidate_count)
                AS avg_candidates,

            QUANTILE_CONT(
                candidate_count,
                0.50
            ) AS median_candidates,

            QUANTILE_CONT(
                candidate_count,
                0.95
            ) AS p95_candidates,

            MAX(candidate_count)
                AS max_candidates,

            SUM(
                CASE
                    WHEN true_count > 0
                    AND recovered_count = true_count
                    THEN 1
                    ELSE 0
                END
            ) AS fully_recovered_entities,

            SUM(
                CASE
                    WHEN true_count > 0
                    THEN 1
                    ELSE 0
                END
            ) AS matched_s1_entities

        FROM read_csv(
            ?,
            delim='\\t',
            header=true,
            auto_detect=true
        )
        """,
        [str(output_path)],
    ).df().iloc[0]

    blocking_recall = (
        total_recovered_pairs / total_true_pairs
        if total_true_pairs
        else 0.0
    )

    multi_recall = (
        multi_recovered_pairs / multi_true_pairs
        if multi_true_pairs
        else 0.0
    )

    singleton_clean_rate = (
        singleton_zero_candidate / singleton_total
        if singleton_total
        else 0.0
    )

    exact_all_match_rate = (
        float(stats["fully_recovered_entities"])
        / float(stats["matched_s1_entities"])
        if stats["matched_s1_entities"] > 0
        else 0.0
    )

    print("\n")
    print("=" * 70)
    print("BLOCKING EVALUATION COMPLETE")
    print("=" * 70)

    print(f"S1 entities evaluated        : {total_s1:,}")
    print(f"True match pairs             : {total_true_pairs:,}")
    print(f"Recovered true pairs         : {total_recovered_pairs:,}")
    print(f"Blocking pair recall         : {blocking_recall:.4%}")
    print()

    print(
        f"Average candidates / S1     : "
        f"{float(stats['avg_candidates']):,.2f}"
    )

    print(
        f"Median candidates / S1     : "
        f"{float(stats['median_candidates']):,.0f}"
    )

    print(
        f"P95 candidates / S1         : "
        f"{float(stats['p95_candidates']):,.0f}"
    )

    print(
        f"Max candidates / S1         : "
        f"{int(stats['max_candidates']):,}"
    )

    print()

    print(
        f"True singleton entities     : "
        f"{singleton_total:,}"
    )

    print(
        f"Singletons with 0 candidates: "
        f"{singleton_zero_candidate:,}"
    )

    print(
        f"Singleton clean rate        : "
        f"{singleton_clean_rate:.4%}"
    )

    print()

    print(
        f"Multi-match S1 entities     : "
        f"{multi_total_entities:,}"
    )

    print(
        f"Multi-match true pairs      : "
        f"{multi_true_pairs:,}"
    )

    print(
        f"Multi-match recovered       : "
        f"{multi_recovered_pairs:,}"
    )

    print(
        f"Multi-match pair recall     : "
        f"{multi_recall:.4%}"
    )

    print()

    print(
        f"Matched S1 fully recovered  : "
        f"{int(stats['fully_recovered_entities']):,}"
    )

    print(
        f"Matched S1 entities         : "
        f"{int(stats['matched_s1_entities']):,}"
    )

    print(
        f"All-matches-per-S1 rate     : "
        f"{exact_all_match_rate:.4%}"
    )

    print()

    print(f"Per-S1 report               : {output_path}")
    print("=" * 70)

    con.close()


def main() -> None:
    args = parse_args()

    evaluate(
        source1_path=args.source1,
        ground_truth_path=args.ground_truth,
        db_path=args.db,
        output_path=args.output,
        chunk_size=args.chunk_size,
        sample=args.sample,
    )


if __name__ == "__main__":
    main()