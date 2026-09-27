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
DEFAULT_DB = ROOT / "artifacts" / "train_indexes_df10k.duckdb"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Blocking V3 diagnostic"
    )

    parser.add_argument(
        "--sample",
        type=int,
        default=10000,
        help="Number of S1 rows. 0 = full dataset.",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2000,
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

    return parser.parse_args()


def prepare_ground_truth(con, path):
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
            TRIM(m.match_id) AS true_entity_id
        FROM ground_truth gt
        CROSS JOIN UNNEST(
            string_split(
                NULLIF(
                    gt.matched_entity_ids,
                    ''
                ),
                ','
            )
        ) AS m(match_id)
        WHERE TRIM(m.match_id) <> ''
        """
    )


def make_s1_chunk(normalized):

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


# ============================================================
# STRONG BLOCKS
# ============================================================

STRONG_SQL = """

    SELECT DISTINCT
        s.source1_entity_id,
        i.entity_id AS candidate_entity_id
    FROM s1_chunk s
    INNER JOIN exact_name_index i
        ON i.lookup_key =
           s.country_key || chr(31) || s.name_nfkc
    WHERE s.name_nfkc <> ''

    UNION

    SELECT DISTINCT
        s.source1_entity_id,
        i.entity_id AS candidate_entity_id
    FROM s1_chunk s
    INNER JOIN compact_name_index i
        ON i.lookup_key =
           s.country_key || chr(31) || s.name_compact
    WHERE s.name_compact <> ''

    UNION

    SELECT DISTINCT
        s.source1_entity_id,
        i.entity_id AS candidate_entity_id
    FROM s1_chunk s
    INNER JOIN core_name_index i
        ON i.lookup_key =
           s.country_key || chr(31) || s.name_core_compact
    WHERE s.name_core_compact <> ''

    UNION

    SELECT DISTINCT
        s.source1_entity_id,
        i.entity_id AS candidate_entity_id
    FROM s1_chunk s
    INNER JOIN exact_address_index i
        ON i.lookup_key =
           s.country_key || chr(31) || s.address_nfkc
    WHERE s.address_nfkc <> ''

    UNION

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
    ) AS p(postal_value)
    INNER JOIN postal_index i
        ON i.lookup_key =
           s.country_key
           || chr(31)
           || p.postal_value
    WHERE p.postal_value <> ''
"""


# ============================================================
# TOKEN EVIDENCE
# ============================================================

NAME_TOKEN_SQL = """

    SELECT
        s.source1_entity_id,
        i.entity_id AS candidate_entity_id,
        nt.token_value

    FROM s1_chunk s

    CROSS JOIN UNNEST(
        string_split(
            NULLIF(
                s.name_tokens,
                ''
            ),
            ' '
        )
    ) AS nt(token_value)

    INNER JOIN rare_name_token_index i
        ON i.lookup_key =
           s.country_key
           || chr(31)
           || nt.token_value

    WHERE nt.token_value <> ''
"""


ADDRESS_TOKEN_SQL = """

    SELECT
        s.source1_entity_id,
        i.entity_id AS candidate_entity_id,
        atok.token_value

    FROM s1_chunk s

    CROSS JOIN UNNEST(
        string_split(
            NULLIF(
                s.address_informative_tokens,
                ''
            ),
            ' '
        )
    ) AS atok(token_value)

    INNER JOIN rare_address_token_index i
        ON i.lookup_key =
           s.country_key
           || chr(31)
           || atok.token_value

    WHERE atok.token_value <> ''
"""


def run():

    args = parse_args()

    if not args.db.exists():
        raise FileNotFoundError(
            f"Database not found: {args.db}"
        )

    if not args.source1.exists():
        raise FileNotFoundError(
            f"Source1 not found: {args.source1}"
        )

    if not args.ground_truth.exists():
        raise FileNotFoundError(
            f"Ground truth not found: {args.ground_truth}"
        )

    print("=" * 70)
    print("BLOCKING V3")
    print("=" * 70)
    print(f"Database : {args.db}")
    print(f"Sample   : {args.sample}")
    print()

    con = duckdb.connect(
        str(args.db),
        read_only=False,
    )

    con.execute("PRAGMA threads=4")

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

    total_processed = 0
    total_true_pairs = 0
    total_recovered = 0
    total_candidates = 0

    for chunk_no, raw in enumerate(
        reader,
        start=1,
    ):

        if (
            args.sample > 0
            and total_processed >= args.sample
        ):
            break

        if args.sample > 0:

            remaining = (
                args.sample
                - total_processed
            )

            if len(raw) > remaining:
                raw = raw.iloc[
                    :remaining
                ].copy()

        normalized = normalize_dataframe_chunk(
            raw
        )

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
            INNER JOIN s1_chunk s
                ON s.source1_entity_id =
                   t.source1_entity_id
            """
        ).fetchone()[0]

        current_true = int(current_true)

        total_true_pairs += current_true

        print(
            f"Chunk {chunk_no} "
            f"({len(s1_chunk):,} S1)"
        )

        # ========================================================
        # STRONG BLOCKS
        # ========================================================

        print("  Building strong candidates...")

        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE
            strong_candidates AS
            {STRONG_SQL}
            """
        )

        # ========================================================
        # NAME TOKEN HIT COUNTS
        # ========================================================

        print("  Building name token evidence...")

        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE
            name_hits AS

            SELECT DISTINCT
                source1_entity_id,
                candidate_entity_id,
                token_value

            FROM (
                {NAME_TOKEN_SQL}
            )
            """
        )

        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE
            name_counts AS

            SELECT
                source1_entity_id,
                candidate_entity_id,
                COUNT(DISTINCT token_value)
                    AS name_overlap

            FROM name_hits

            GROUP BY
                source1_entity_id,
                candidate_entity_id
            """
        )

        # ========================================================
        # ADDRESS TOKEN HIT COUNTS
        # ========================================================

        print(
            "  Building address token evidence..."
        )

        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE
            address_hits AS

            SELECT DISTINCT
                source1_entity_id,
                candidate_entity_id,
                token_value

            FROM (
                {ADDRESS_TOKEN_SQL}
            )
            """
        )

        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE
            address_counts AS

            SELECT
                source1_entity_id,
                candidate_entity_id,
                COUNT(DISTINCT token_value)
                    AS address_overlap

            FROM address_hits

            GROUP BY
                source1_entity_id,
                candidate_entity_id
            """
        )

        # ========================================================
        # V3 FINAL CANDIDATES
        # ========================================================

        print(
            "  Constructing V3 candidate set..."
        )

        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE
            final_candidates AS

            /* ----------------------------------------------
               Strong deterministic candidates
               ---------------------------------------------- */

            SELECT
                source1_entity_id,
                candidate_entity_id

            FROM strong_candidates

            UNION

            /* ----------------------------------------------
               Multiple name tokens
               ---------------------------------------------- */

            SELECT
                source1_entity_id,
                candidate_entity_id

            FROM name_counts

            WHERE name_overlap >= 2

            UNION

            /* ----------------------------------------------
               Multiple address tokens
               ---------------------------------------------- */

            SELECT
                source1_entity_id,
                candidate_entity_id

            FROM address_counts

            WHERE address_overlap >= 2

            UNION

            /* ----------------------------------------------
               Name + address token evidence
               ---------------------------------------------- */

            SELECT
                n.source1_entity_id,
                n.candidate_entity_id

            FROM name_counts n

            INNER JOIN address_counts a

                ON a.source1_entity_id =
                   n.source1_entity_id

               AND a.candidate_entity_id =
                   n.candidate_entity_id

            WHERE
                n.name_overlap >= 1

                AND

                a.address_overlap >= 1
            """
        )

        # ========================================================
        # STATS
        # ========================================================

        candidate_count = con.execute(
            """
            SELECT COUNT(*)
            FROM final_candidates
            """
        ).fetchone()[0]

        recovered = con.execute(
            """
            SELECT COUNT(*)

            FROM final_candidates c

            INNER JOIN true_pairs t

                ON t.source1_entity_id =
                   c.source1_entity_id

               AND t.true_entity_id =
                   c.candidate_entity_id
            """
        ).fetchone()[0]

        candidate_s1 = con.execute(
            """
            SELECT COUNT(
                DISTINCT source1_entity_id
            )
            FROM final_candidates
            """
        ).fetchone()[0]

        candidate_count = int(
            candidate_count
        )

        recovered = int(
            recovered
        )

        candidate_s1 = int(
            candidate_s1
        )

        total_candidates += candidate_count
        total_recovered += recovered

        chunk_recall = (
            recovered / current_true
            if current_true
            else 0.0
        )

        total_processed += len(
            s1_chunk
        )

        cumulative_recall = (
            total_recovered
            / total_true_pairs
            if total_true_pairs
            else 0.0
        )

        print(
            f"  Candidates           : "
            f"{candidate_count:,}"
        )

        print(
            f"  S1 with candidates   : "
            f"{candidate_s1:,}"
        )

        print(
            f"  Recovered true pairs : "
            f"{recovered:,}"
        )

        print(
            f"  Chunk recall         : "
            f"{chunk_recall:.4%}"
        )

        print(
            f"  Cumulative recall    : "
            f"{cumulative_recall:.4%}"
        )

        print(
            f"  Processed            : "
            f"{total_processed:,}"
        )

        con.unregister(
            "s1_chunk"
        )

    # ============================================================
    # FINAL SUMMARY
    # ============================================================

    final_recall = (
        total_recovered
        / total_true_pairs
        if total_true_pairs
        else 0.0
    )

    avg_candidates = (
        total_candidates
        / total_processed
        if total_processed
        else 0.0
    )

    print()
    print("=" * 70)
    print("BLOCKING V3 COMPLETE")
    print("=" * 70)

    print(
        f"S1 rows evaluated       : "
        f"{total_processed:,}"
    )

    print(
        f"True match pairs        : "
        f"{total_true_pairs:,}"
    )

    print(
        f"Recovered true pairs    : "
        f"{total_recovered:,}"
    )

    print(
        f"Blocking recall         : "
        f"{final_recall:.4%}"
    )

    print(
        f"Candidate pairs         : "
        f"{total_candidates:,}"
    )

    print(
        f"Average candidates/S1   : "
        f"{avg_candidates:,.2f}"
    )

    print("=" * 70)

    con.close()


if __name__ == "__main__":
    run()