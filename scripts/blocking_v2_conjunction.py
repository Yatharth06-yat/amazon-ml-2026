from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from normalize import normalize_dataframe_chunk


DEFAULT_S1 = ROOT / "data" / "train_source1.tsv"
DEFAULT_GT = ROOT / "data" / "train_ground_truth.tsv"
DEFAULT_DB = ROOT / "artifacts" / "train_indexes.duckdb"
DEFAULT_OUTPUT = ROOT / "reports" / "blocking_v2_diagnostics.tsv"


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Blocking V2 diagnostic for Amazon ML Challenge 2026. "
            "Uses strong exact blocks plus name-token AND "
            "address-token conjunction."
        )
    )

    parser.add_argument(
        "--sample",
        type=int,
        default=10_000,
        help="Number of S1 rows to evaluate. 0 = full dataset.",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2_000,
        help="Number of S1 rows processed per chunk.",
    )

    parser.add_argument(
        "--source1",
        type=Path,
        default=DEFAULT_S1,
        help="Path to train_source1.tsv",
    )

    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=DEFAULT_GT,
        help="Path to train_ground_truth.tsv",
    )

    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help="Path to train_indexes.duckdb",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output diagnostic TSV.",
    )

    return parser.parse_args()


# ============================================================
# GROUND TRUTH
# ============================================================

def prepare_ground_truth(
    con: duckdb.DuckDBPyConnection,
    gt_path: Path,
) -> None:

    print("Loading ground truth...")

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
        [str(gt_path)],
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE true_pairs AS
        SELECT DISTINCT
            gt.source1_entity_id,

            TRIM(
                match_item
            ) AS true_entity_id

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

    gt_rows = con.execute(
        "SELECT COUNT(*) FROM ground_truth"
    ).fetchone()[0]

    true_rows = con.execute(
        "SELECT COUNT(*) FROM true_pairs"
    ).fetchone()[0]

    print(
        f"Ground-truth S1 rows : {gt_rows:,}"
    )

    print(
        f"Ground-truth pairs   : {true_rows:,}"
    )


# ============================================================
# NORMALIZED S1 CHUNK
# ============================================================

def build_s1_chunk(
    normalized: pd.DataFrame,
) -> pd.DataFrame:

    return pd.DataFrame(
        {
            "source1_entity_id":
                normalized[
                    "entity_id"
                ].astype("string"),

            "country_key":
                normalized[
                    "country_key"
                ].astype("string"),

            "name_nfkc":
                normalized[
                    "name_nfkc"
                ].astype("string"),

            "name_compact":
                normalized[
                    "name_compact"
                ].astype("string"),

            "name_core_compact":
                normalized[
                    "name_core_compact"
                ].astype("string"),

            "name_tokens":
                normalized[
                    "name_tokens"
                ].astype("string"),

            "address_nfkc":
                normalized[
                    "address_nfkc"
                ].astype("string"),

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


# ============================================================
# STRONG BLOCKS
#
# Important:
# We deliberately DO NOT use a single rare-name-token
# or single rare-address-token block.
# ============================================================

STRONG_BLOCKS: dict[str, str] = {

    "exact_name": """
        SELECT DISTINCT
            s.source1_entity_id,
            i.entity_id AS candidate_entity_id

        FROM s1_chunk s

        INNER JOIN exact_name_index i
            ON i.lookup_key =
               s.country_key
               || chr(31)
               || s.name_nfkc

        WHERE s.name_nfkc <> ''
    """,

    "compact_name": """
        SELECT DISTINCT
            s.source1_entity_id,
            i.entity_id AS candidate_entity_id

        FROM s1_chunk s

        INNER JOIN compact_name_index i
            ON i.lookup_key =
               s.country_key
               || chr(31)
               || s.name_compact

        WHERE s.name_compact <> ''
    """,

    "core_name": """
        SELECT DISTINCT
            s.source1_entity_id,
            i.entity_id AS candidate_entity_id

        FROM s1_chunk s

        INNER JOIN core_name_index i
            ON i.lookup_key =
               s.country_key
               || chr(31)
               || s.name_core_compact

        WHERE s.name_core_compact <> ''
    """,

    "exact_address": """
        SELECT DISTINCT
            s.source1_entity_id,
            i.entity_id AS candidate_entity_id

        FROM s1_chunk s

        INNER JOIN exact_address_index i
            ON i.lookup_key =
               s.country_key
               || chr(31)
               || s.address_nfkc

        WHERE s.address_nfkc <> ''
    """,

    "postal": """
        SELECT DISTINCT
            s.source1_entity_id,
            i.entity_id AS candidate_entity_id

        FROM s1_chunk s

        CROSS JOIN UNNEST(
            string_split(
                NULLIF(
                    s.postal_candidates,
                    ''
                ),
                '|'
            )
        ) AS postal_tbl(postal_value)

        INNER JOIN postal_index i
            ON i.lookup_key =
               s.country_key
               || chr(31)
               || postal_tbl.postal_value

        WHERE postal_tbl.postal_value <> ''
    """,
}


# ============================================================
# TOKEN CONJUNCTION
#
# Candidate must satisfy:
#
#     at least one rare name token
#                    AND
#     at least one rare informative address token
#
# This replaces the old single-token blocks.
# ============================================================

def build_name_address_conjunction_sql() -> str:

    return """
        WITH name_candidates AS (

            SELECT DISTINCT
                s.source1_entity_id,
                i.entity_id AS candidate_entity_id

            FROM s1_chunk s

            CROSS JOIN UNNEST(
                string_split(
                    NULLIF(
                        s.name_tokens,
                        ''
                    ),
                    ' '
                )
            ) AS name_tok(token_value)

            INNER JOIN rare_name_token_index i
                ON i.lookup_key =
                   s.country_key
                   || chr(31)
                   || name_tok.token_value

            WHERE name_tok.token_value <> ''
        ),

        address_candidates AS (

            SELECT DISTINCT
                s.source1_entity_id,
                i.entity_id AS candidate_entity_id

            FROM s1_chunk s

            CROSS JOIN UNNEST(
                string_split(
                    NULLIF(
                        s.address_informative_tokens,
                        ''
                    ),
                    ' '
                )
            ) AS addr_tok(token_value)

            INNER JOIN rare_address_token_index i
                ON i.lookup_key =
                   s.country_key
                   || chr(31)
                   || addr_tok.token_value

            WHERE addr_tok.token_value <> ''
        )

        SELECT DISTINCT
            n.source1_entity_id,
            n.candidate_entity_id

        FROM name_candidates n

        INNER JOIN address_candidates a
            ON a.source1_entity_id =
               n.source1_entity_id

           AND a.candidate_entity_id =
               n.candidate_entity_id
    """


# ============================================================
# BLOCK STATISTICS
# ============================================================

def get_block_statistics(
    con: duckdb.DuckDBPyConnection,
) -> tuple[int, int, int]:

    candidate_pairs = con.execute(
        """
        SELECT COUNT(*)
        FROM block_candidates
        """
    ).fetchone()[0]

    s1_with_candidates = con.execute(
        """
        SELECT COUNT(DISTINCT source1_entity_id)
        FROM block_candidates
        """
    ).fetchone()[0]

    recovered_true_pairs = con.execute(
        """
        SELECT COUNT(*)

        FROM block_candidates b

        INNER JOIN true_pairs t

            ON t.source1_entity_id =
               b.source1_entity_id

           AND t.true_entity_id =
               b.candidate_entity_id

        INNER JOIN s1_chunk s

            ON s.source1_entity_id =
               b.source1_entity_id
        """
    ).fetchone()[0]

    return (
        int(candidate_pairs),
        int(s1_with_candidates),
        int(recovered_true_pairs),
    )


# ============================================================
# CUMULATIVE STATISTICS
# ============================================================

def get_cumulative_statistics(
    con: duckdb.DuckDBPyConnection,
) -> tuple[int, int, int, int]:

    candidate_pairs = con.execute(
        """
        SELECT COUNT(*)
        FROM cumulative_distinct
        """
    ).fetchone()[0]

    s1_with_candidates = con.execute(
        """
        SELECT COUNT(DISTINCT source1_entity_id)
        FROM cumulative_distinct
        """
    ).fetchone()[0]

    recovered_true_pairs = con.execute(
        """
        SELECT COUNT(*)

        FROM cumulative_distinct c

        INNER JOIN true_pairs t

            ON t.source1_entity_id =
               c.source1_entity_id

           AND t.true_entity_id =
               c.candidate_entity_id

        INNER JOIN s1_chunk s

            ON s.source1_entity_id =
               c.source1_entity_id
        """
    ).fetchone()[0]

    fully_recovered_entities = con.execute(
        """
        WITH candidate_counts AS (

            SELECT
                s.source1_entity_id,

                COUNT(
                    cd.candidate_entity_id
                ) AS candidate_count

            FROM s1_chunk s

            LEFT JOIN cumulative_distinct cd

                ON cd.source1_entity_id =
                   s.source1_entity_id

            GROUP BY
                s.source1_entity_id
        ),

        true_counts AS (

            SELECT
                s.source1_entity_id,

                COUNT(t.true_entity_id)
                    AS true_count

            FROM s1_chunk s

            LEFT JOIN true_pairs t

                ON t.source1_entity_id =
                   s.source1_entity_id

            GROUP BY
                s.source1_entity_id
        ),

        recovered_counts AS (

            SELECT
                s.source1_entity_id,

                COUNT(
                    cd.candidate_entity_id
                ) AS recovered_count

            FROM s1_chunk s

            LEFT JOIN cumulative_distinct cd

                ON cd.source1_entity_id =
                   s.source1_entity_id

            LEFT JOIN true_pairs t

                ON t.source1_entity_id =
                   cd.source1_entity_id

               AND t.true_entity_id =
                   cd.candidate_entity_id

            WHERE t.true_entity_id IS NOT NULL

            GROUP BY
                s.source1_entity_id
        )

        SELECT COUNT(*)

        FROM true_counts tc

        LEFT JOIN recovered_counts rc

            ON rc.source1_entity_id =
               tc.source1_entity_id

        WHERE tc.true_count > 0

          AND COALESCE(
              rc.recovered_count,
              0
          ) = tc.true_count
        """
    ).fetchone()[0]

    return (
        int(candidate_pairs),
        int(s1_with_candidates),
        int(recovered_true_pairs),
        int(fully_recovered_entities),
    )


# ============================================================
# MAIN EVALUATION
# ============================================================

def evaluate() -> None:

    args = parse_args()

    if not args.source1.exists():
        raise FileNotFoundError(
            f"Source1 not found: {args.source1}"
        )

    if not args.ground_truth.exists():
        raise FileNotFoundError(
            f"Ground truth not found: {args.ground_truth}"
        )

    if not args.db.exists():
        raise FileNotFoundError(
            f"DuckDB index not found: {args.db}"
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.output.exists():
        args.output.unlink()

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

    reader = pd.read_csv(
        args.source1,
        sep="\t",
        dtype="string",
        chunksize=args.chunk_size,
    )

    processed_rows = 0

    total_true_pairs = 0

    total_cumulative_candidates = 0

    total_cumulative_recovered = 0

    total_fully_recovered_entities = 0

    diagnostic_rows = []

    for chunk_no, raw_chunk in enumerate(
        reader,
        start=1,
    ):

        # --------------------------------------------------------
        # Stop condition
        # --------------------------------------------------------

        if (
            args.sample > 0
            and processed_rows >= args.sample
        ):
            break

        if args.sample > 0:

            remaining = (
                args.sample
                - processed_rows
            )

            if len(raw_chunk) > remaining:

                raw_chunk = raw_chunk.iloc[
                    :remaining
                ].copy()

        # --------------------------------------------------------
        # Normalize S1
        # --------------------------------------------------------

        normalized = normalize_dataframe_chunk(
            raw_chunk
        )

        s1_chunk = build_s1_chunk(
            normalized
        )

        con.register(
            "s1_chunk",
            s1_chunk
        )

        # --------------------------------------------------------
        # True pairs for this chunk
        # --------------------------------------------------------

        current_true_pairs = con.execute(
            """
            SELECT COUNT(*)

            FROM true_pairs t

            INNER JOIN s1_chunk s

                ON s.source1_entity_id =
                   t.source1_entity_id
            """
        ).fetchone()[0]

        current_true_pairs = int(
            current_true_pairs
        )

        total_true_pairs += (
            current_true_pairs
        )

        # --------------------------------------------------------
        # Empty cumulative candidate table
        # --------------------------------------------------------

        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE
            cumulative_candidates (
                source1_entity_id VARCHAR,
                candidate_entity_id VARCHAR
            )
            """
        )

        print()
        print(
            f"Chunk {chunk_no} "
            f"({len(s1_chunk):,} S1 rows)"
        )

        # ========================================================
        # STRONG BLOCKS
        # ========================================================

        for block_name, block_sql in (
            STRONG_BLOCKS.items()
        ):

            print(
                f"  Running {block_name}..."
            )

            con.execute(
                f"""
                CREATE OR REPLACE TEMP TABLE
                block_candidates AS

                {block_sql}
                """
            )

            (
                candidate_pairs,
                s1_with_candidates,
                recovered_true_pairs,
            ) = get_block_statistics(con)

            individual_recall = (
                recovered_true_pairs
                / current_true_pairs
                if current_true_pairs > 0
                else 0.0
            )

            # Add to cumulative set.
            con.execute(
                """
                INSERT INTO
                cumulative_candidates

                SELECT
                    source1_entity_id,
                    candidate_entity_id

                FROM block_candidates
                """
            )

            diagnostic_rows.append(
                {
                    "chunk": chunk_no,
                    "block": block_name,
                    "candidate_pairs":
                        candidate_pairs,
                    "s1_with_candidates":
                        s1_with_candidates,
                    "recovered_true_pairs":
                        recovered_true_pairs,
                    "individual_recall":
                        individual_recall,
                }
            )

            print(
                f"    candidates = "
                f"{candidate_pairs:,} | "
                f"recovered = "
                f"{recovered_true_pairs:,} | "
                f"recall = "
                f"{individual_recall:.4%}"
            )

        # ========================================================
        # V2 CONJUNCTION
        # ========================================================

        print(
            "  Running "
            "name_AND_address_token..."
        )

        conjunction_sql = (
            build_name_address_conjunction_sql()
        )

        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE
            block_candidates AS

            {conjunction_sql}
            """
        )

        (
            candidate_pairs,
            s1_with_candidates,
            recovered_true_pairs,
        ) = get_block_statistics(con)

        individual_recall = (
            recovered_true_pairs
            / current_true_pairs
            if current_true_pairs > 0
            else 0.0
        )

        con.execute(
            """
            INSERT INTO
            cumulative_candidates

            SELECT
                source1_entity_id,
                candidate_entity_id

            FROM block_candidates
            """
        )

        diagnostic_rows.append(
            {
                "chunk": chunk_no,
                "block":
                    "name_AND_address_token",
                "candidate_pairs":
                    candidate_pairs,
                "s1_with_candidates":
                    s1_with_candidates,
                "recovered_true_pairs":
                    recovered_true_pairs,
                "individual_recall":
                    individual_recall,
            }
        )

        print(
            f"    candidates = "
            f"{candidate_pairs:,} | "
            f"recovered = "
            f"{recovered_true_pairs:,} | "
            f"recall = "
            f"{individual_recall:.4%}"
        )

        # ========================================================
        # DEDUP CUMULATIVE CANDIDATES
        # ========================================================

        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE
            cumulative_distinct AS

            SELECT DISTINCT
                source1_entity_id,
                candidate_entity_id

            FROM cumulative_candidates
            """
        )

        (
            cumulative_pairs,
            cumulative_s1,
            cumulative_recovered,
            fully_recovered_entities,
        ) = get_cumulative_statistics(con)

        cumulative_recall = (
            cumulative_recovered
            / current_true_pairs
            if current_true_pairs > 0
            else 0.0
        )

        total_cumulative_candidates += (
            cumulative_pairs
        )

        total_cumulative_recovered += (
            cumulative_recovered
        )

        total_fully_recovered_entities += (
            fully_recovered_entities
        )

        diagnostic_rows.append(
            {
                "chunk": chunk_no,
                "block":
                    "V2_CUMULATIVE_UNION",
                "candidate_pairs":
                    cumulative_pairs,
                "s1_with_candidates":
                    cumulative_s1,
                "recovered_true_pairs":
                    cumulative_recovered,
                "individual_recall":
                    cumulative_recall,
            }
        )

        print(
            "  V2 cumulative:"
            f" candidates = "
            f"{cumulative_pairs:,} |"
            f" recovered = "
            f"{cumulative_recovered:,} |"
            f" recall = "
            f"{cumulative_recall:.4%}"
        )

        processed_rows += len(
            s1_chunk
        )

        print(
            f"Processed total: "
            f"{processed_rows:,}"
        )

        con.unregister(
            "s1_chunk"
        )

    # ============================================================
    # SAVE DIAGNOSTIC REPORT
    # ============================================================

    diagnostic_df = pd.DataFrame(
        diagnostic_rows
    )

    diagnostic_df.to_csv(
        args.output,
        sep="\t",
        index=False,
    )

    # ============================================================
    # FINAL SUMMARY
    # ============================================================

    final_recall = (
        total_cumulative_recovered
        / total_true_pairs
        if total_true_pairs > 0
        else 0.0
    )

    average_candidates = (
        total_cumulative_candidates
        / processed_rows
        if processed_rows > 0
        else 0.0
    )

    print()
    print("=" * 70)
    print("BLOCKING V2 DIAGNOSTIC COMPLETE")
    print("=" * 70)

    print(
        f"S1 rows evaluated            : "
        f"{processed_rows:,}"
    )

    print(
        f"True match pairs              : "
        f"{total_true_pairs:,}"
    )

    print(
        f"V2 recovered true pairs      : "
        f"{total_cumulative_recovered:,}"
    )

    print(
        f"V2 blocking pair recall      : "
        f"{final_recall:.4%}"
    )

    print()

    print(
        f"V2 candidate pairs           : "
        f"{total_cumulative_candidates:,}"
    )

    print(
        f"Average candidates / S1     : "
        f"{average_candidates:,.2f}"
    )

    print()

    print(
        f"Fully recovered matched S1  : "
        f"{total_fully_recovered_entities:,}"
    )

    print()

    print(
        f"Diagnostic report            : "
        f"{args.output}"
    )

    print("=" * 70)

    con.close()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    evaluate()