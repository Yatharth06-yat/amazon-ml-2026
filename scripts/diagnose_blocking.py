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
DEFAULT_OUTPUT = ROOT / "reports" / "blocking_diagnostics.tsv"


BLOCKS = {
    "exact_name": """
        SELECT
            s.source1_entity_id,
            i.entity_id AS candidate_entity_id
        FROM s1_chunk s
        JOIN exact_name_index i
          ON i.lookup_key =
             s.country_key || chr(31) || s.name_nfkc
        WHERE s.name_nfkc <> ''
    """,

    "compact_name": """
        SELECT
            s.source1_entity_id,
            i.entity_id AS candidate_entity_id
        FROM s1_chunk s
        JOIN compact_name_index i
          ON i.lookup_key =
             s.country_key || chr(31) || s.name_compact
        WHERE s.name_compact <> ''
    """,

    "core_name": """
        SELECT
            s.source1_entity_id,
            i.entity_id AS candidate_entity_id
        FROM s1_chunk s
        JOIN core_name_index i
          ON i.lookup_key =
             s.country_key || chr(31) || s.name_core_compact
        WHERE s.name_core_compact <> ''
    """,

    "exact_address": """
        SELECT
            s.source1_entity_id,
            i.entity_id AS candidate_entity_id
        FROM s1_chunk s
        JOIN exact_address_index i
          ON i.lookup_key =
             s.country_key || chr(31) || s.address_nfkc
        WHERE s.address_nfkc <> ''
    """,

    "postal": """
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
    """,

    "rare_name_token": """
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
    """,

    "rare_address_token": """
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
    """,
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sample",
        type=int,
        default=10_000,
        help="Number of S1 rows to diagnose. Default=10000.",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2_000,
    )

    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
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
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    return parser.parse_args()


def prepare_ground_truth(con, path):
    con.execute("DROP TABLE IF EXISTS ground_truth")

    con.execute(
        """
        CREATE TEMP TABLE ground_truth AS
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

    con.execute("DROP TABLE IF EXISTS true_pairs")

    con.execute(
        """
        CREATE TEMP TABLE true_pairs AS
        SELECT DISTINCT
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
        """
    )


def make_s1_chunk(normalized: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
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
                normalized[
                    "postal_candidates"
                ].astype("string"),
        }
    )


def get_block_stats(con):
    return con.execute(
        """
        SELECT
            COUNT(*) AS candidate_pairs,

            COUNT(
                DISTINCT source1_entity_id
            ) AS s1_with_candidates,

            COUNT(*) FILTER (
                WHERE EXISTS (
                    SELECT 1
                    FROM true_pairs t
                    WHERE t.source1_entity_id =
                          c.source1_entity_id
                      AND t.true_entity_id =
                          c.candidate_entity_id
                )
            ) AS recovered_true_pairs

        FROM block_candidates c
        """
    ).fetchone()


def get_cumulative_stats(con):
    return con.execute(
        """
        SELECT
            COUNT(*) AS candidate_pairs,

            COUNT(
                DISTINCT source1_entity_id
            ) AS s1_with_candidates,

            COUNT(*) FILTER (
                WHERE EXISTS (
                    SELECT 1
                    FROM true_pairs t
                    WHERE t.source1_entity_id =
                          c.source1_entity_id
                      AND t.true_entity_id =
                          c.candidate_entity_id
                )
            ) AS recovered_true_pairs

        FROM cumulative_candidates c
        """
    ).fetchone()


def main():
    args = parse_args()

    con = duckdb.connect(
        str(args.db),
        read_only=False,
    )

    con.execute("PRAGMA threads=4")

    prepare_ground_truth(
        con,
        args.ground_truth,
    )

    print("Ground truth prepared.")

    reader = pd.read_csv(
        args.source1,
        sep="\t",
        dtype="string",
        chunksize=args.chunk_size,
    )

    results = []

    processed = 0
    total_true_pairs = 0

    for chunk_no, raw in enumerate(
        reader,
        start=1,
    ):
        if processed >= args.sample:
            break

        remaining = args.sample - processed

        if len(raw) > remaining:
            raw = raw.iloc[:remaining].copy()

        normalized = normalize_dataframe_chunk(raw)

        s1_chunk = make_s1_chunk(
            normalized
        )

        con.register(
            "s1_chunk",
            s1_chunk,
        )

        current_true = con.execute(
            """
            SELECT COUNT(*)
            FROM true_pairs t
            JOIN s1_chunk s
              ON s.source1_entity_id =
                 t.source1_entity_id
            """
        ).fetchone()[0]

        total_true_pairs += current_true

        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE cumulative_candidates(
                source1_entity_id VARCHAR,
                candidate_entity_id VARCHAR
            )
            """
        )

        for block_name, block_sql in BLOCKS.items():

            print(
                f"Chunk {chunk_no}: {block_name}"
            )

            con.execute(
                f"""
                CREATE OR REPLACE TEMP TABLE block_candidates AS
                SELECT DISTINCT
                    source1_entity_id,
                    candidate_entity_id
                FROM (
                    {block_sql}
                )
                """
            )

            candidate_pairs, s1_with_candidates, recovered = (
                get_block_stats(con)
            )

            con.execute(
                """
                INSERT INTO cumulative_candidates

                SELECT
                    source1_entity_id,
                    candidate_entity_id

                FROM block_candidates
                """
            )

            con.execute(
                """
                CREATE OR REPLACE TEMP TABLE cumulative_candidates AS

                SELECT DISTINCT
                    source1_entity_id,
                    candidate_entity_id

                FROM cumulative_candidates
                """
            )

            cum_pairs, cum_s1, cum_recovered = (
                get_cumulative_stats(con)
            )

            individual_recall = (
                recovered / current_true
                if current_true
                else 0.0
            )

            cumulative_recall = (
                cum_recovered / total_true_pairs
                if total_true_pairs
                else 0.0
            )

            results.append(
                {
                    "chunk": chunk_no,
                    "block": block_name,
                    "candidate_pairs": candidate_pairs,
                    "s1_with_candidates":
                        s1_with_candidates,
                    "individual_recovered_true_pairs":
                        recovered,
                    "individual_recall":
                        individual_recall,
                    "cumulative_candidate_pairs":
                        cum_pairs,
                    "cumulative_s1_with_candidates":
                        cum_s1,
                    "cumulative_recovered_true_pairs":
                        cum_recovered,
                    "cumulative_recall":
                        cumulative_recall,
                }
            )

        processed += len(s1_chunk)

        print(
            f"Processed {processed:,} S1 rows."
        )

        con.unregister(
            "s1_chunk"
        )

    result_df = pd.DataFrame(results)

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        args.output,
        sep="\t",
        index=False,
    )

    print()
    print("=" * 70)
    print("BLOCKING DIAGNOSTIC COMPLETE")
    print("=" * 70)

    summary = (
        result_df
        .groupby("block")
        .agg(
            candidate_pairs=(
                "candidate_pairs",
                "sum",
            ),
            recovered_true_pairs=(
                "individual_recovered_true_pairs",
                "sum",
            ),
        )
        .reset_index()
    )

    summary["individual_recall"] = (
        summary["recovered_true_pairs"]
        / total_true_pairs
    )

    print()
    print(summary.to_string(index=False))

    print()
    print(
        "Per-block details written to:"
    )
    print(args.output)

    print("=" * 70)

    con.close()


if __name__ == "__main__":
    main()