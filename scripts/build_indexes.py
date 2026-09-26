"""
Amazon ML Challenge 2026
Business Entity Resolution

FINAL INDEX BUILD PIPELINE

    train_source2.tsv
            +
    train_source3.tsv
            |
            v
    chunked reading
            |
            v
    normalize_dataframe_chunk()
            |
            v
    shared DuckDB table: entities
            |
            +----------------------+
            |                      |
            v                      v
      exact indexes          token indexes
            |                      |
            +-----------+----------+
                        |
                        v
                  postal index
                        |
                        v
             optional n-gram index
                        |
                        v
              train_indexes.duckdb

Important:
- Source 2 and Source 3 are stored in one shared `entities` table.
- Files are processed chunk-by-chunk.
- Full TSV files are never loaded into RAM at once.
- N-gram index is OFF by default.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import pandas as pd


# ============================================================================
# PROJECT ROOT
# ============================================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.normalize import normalize_dataframe_chunk


# ============================================================================
# DEFAULT PATHS
# ============================================================================

DEFAULT_SOURCE2 = (
    ROOT / "data" / "train_source2.tsv"
)

DEFAULT_SOURCE3 = (
    ROOT / "data" / "train_source3.tsv"
)

DEFAULT_OUTPUT = (
    ROOT / "artifacts" / "train_indexes.duckdb"
)


# ============================================================================
# DEFAULT CONFIGURATION
# ============================================================================

DEFAULT_CHUNK_SIZE = 100_000
DEFAULT_THREADS = 4

DEFAULT_MAX_NAME_TOKEN_DF = 2_000
DEFAULT_MAX_ADDRESS_TOKEN_DF = 2_000
DEFAULT_MAX_NGRAM_DF = 1_000


# ============================================================================
# ENTITY SCHEMA
# ============================================================================

ENTITY_COLUMNS = [
    "source",
    "entity_id",

    "business_name",
    "business_address",
    "country",

    "country_key",

    "name_nfkc",
    "name_compact",
    "name_core_compact",
    "name_tokens",
    "name_latin_aux",

    "address_nfkc",
    "address_compact",
    "address_informative_tokens",
    "address_latin_aux",
    "address_numbers",
    "postal_candidates",
]


# ============================================================================
# PRINT HELPERS
# ============================================================================

def print_header(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


# ============================================================================
# DATABASE INITIALIZATION
# ============================================================================

def create_entities_table(
    con: duckdb.DuckDBPyConnection,
) -> None:
    """
    Create the shared entities table exactly once.

    Both S2 and S3 are appended to this table.
    """

    con.execute(
        """
        CREATE TABLE entities (
            source VARCHAR,
            entity_id VARCHAR,

            business_name VARCHAR,
            business_address VARCHAR,
            country VARCHAR,

            country_key VARCHAR,

            name_nfkc VARCHAR,
            name_compact VARCHAR,
            name_core_compact VARCHAR,
            name_tokens VARCHAR,
            name_latin_aux VARCHAR,

            address_nfkc VARCHAR,
            address_compact VARCHAR,
            address_informative_tokens VARCHAR,
            address_latin_aux VARCHAR,
            address_numbers VARCHAR,
            postal_candidates VARCHAR
        )
        """
    )


# ============================================================================
# LOAD + NORMALIZE ONE SOURCE
# ============================================================================

def append_source(
    con: duckdb.DuckDBPyConnection,
    path: Path,
    source_name: str,
    chunk_size: int,
) -> int:
    """
    Read one TSV file chunk-by-chunk, normalize it, and append it
    to the already-created shared `entities` table.
    """

    print_header(
        f"Loading {source_name}: {path.name}"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file does not exist:\n{path}"
        )

    total_rows = 0

    reader = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=chunk_size,
        keep_default_na=False,
        on_bad_lines="warn",
    )

    for chunk_number, chunk in enumerate(
        reader,
        start=1,
    ):

        # ---------------------------------------------------------------
        # Normalize chunk
        # ---------------------------------------------------------------

        normalized = normalize_dataframe_chunk(
            chunk,
            include_phonetic=False,
        )

        # ---------------------------------------------------------------
        # Add source identifier
        # ---------------------------------------------------------------

        normalized.insert(
            0,
            "source",
            source_name,
        )

        # ---------------------------------------------------------------
        # Keep only required columns
        # ---------------------------------------------------------------

        normalized = normalized[
            ENTITY_COLUMNS
        ].copy()

        # ---------------------------------------------------------------
        # Force all columns to strings
        # ---------------------------------------------------------------

        for column in ENTITY_COLUMNS:
            normalized[column] = (
                normalized[column]
                .fillna("")
                .astype(str)
            )

        # ---------------------------------------------------------------
        # ALWAYS APPEND
        # ---------------------------------------------------------------

        con.append(
            "entities",
            normalized,
            by_name=True,
        )

        total_rows += len(
            normalized
        )

        # ---------------------------------------------------------------
        # Progress
        # ---------------------------------------------------------------

        if (
            chunk_number == 1
            or chunk_number % 5 == 0
        ):
            print(
                f"[{source_name}] "
                f"Processed "
                f"{total_rows:,} rows"
            )

    print(
        f"\n[{source_name}] Complete: "
        f"{total_rows:,} rows"
    )

    return total_rows


# ============================================================================
# EXACT INDEX BUILDER
# ============================================================================

def build_exact_index(
    con: duckdb.DuckDBPyConnection,
    value_column: str,
    index_table: str,
    frequency_table: str,
) -> None:
    """
    Build exact lookup postings.

    lookup_key:
        country + separator + normalized value

    posting:
        entity_id
    """

    print(
        f"Building {index_table}..."
    )

    # ------------------------------------------------------------------
    # Posting table
    # ------------------------------------------------------------------

    con.execute(
        f"""
        CREATE TABLE {index_table} AS

        SELECT
            country_key
                || chr(31)
                || {value_column}
                AS lookup_key,

            entity_id

        FROM entities

        WHERE
            country_key <> ''
            AND {value_column} <> ''
        """
    )

    # ------------------------------------------------------------------
    # Frequency table
    # ------------------------------------------------------------------

    con.execute(
        f"""
        CREATE TABLE {frequency_table} AS

        SELECT
            lookup_key,
            COUNT(*) AS doc_freq

        FROM {index_table}

        GROUP BY lookup_key
        """
    )

    # ------------------------------------------------------------------
    # ART index for fast equality lookup
    # ------------------------------------------------------------------

    con.execute(
        f"""
        CREATE INDEX {index_table}_lookup_idx
        ON {index_table}(lookup_key)
        """
    )

    posting_count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM {index_table}
        """
    ).fetchone()[0]

    unique_keys = con.execute(
        f"""
        SELECT COUNT(*)
        FROM {frequency_table}
        """
    ).fetchone()[0]

    print(
        f"  postings    : {posting_count:,}"
    )

    print(
        f"  unique keys : {unique_keys:,}"
    )


# ============================================================================
# POSTAL INDEX
# ============================================================================

def build_postal_index(
    con: duckdb.DuckDBPyConnection,
) -> None:
    """
    Build country + postal/PIN -> entity_id index.
    """

    print(
        "Building postal_index..."
    )

    con.execute(
        """
        CREATE TABLE postal_index AS

        SELECT DISTINCT

            country_key
                || chr(31)
                || postal
                AS lookup_key,

            entity_id

        FROM (

            SELECT
                country_key,
                entity_id,

                UNNEST(
                    string_split(
                        postal_candidates,
                        '|'
                    )
                ) AS postal

            FROM entities

            WHERE
                country_key <> ''
                AND postal_candidates <> ''
        )

        WHERE
            postal <> ''
        """
    )

    con.execute(
        """
        CREATE TABLE postal_frequency AS

        SELECT
            lookup_key,
            COUNT(*) AS doc_freq

        FROM postal_index

        GROUP BY lookup_key
        """
    )

    con.execute(
        """
        CREATE INDEX postal_index_lookup_idx
        ON postal_index(lookup_key)
        """
    )

    posting_count = con.execute(
        """
        SELECT COUNT(*)
        FROM postal_index
        """
    ).fetchone()[0]

    unique_keys = con.execute(
        """
        SELECT COUNT(*)
        FROM postal_frequency
        """
    ).fetchone()[0]

    print(
        f"  postings    : {posting_count:,}"
    )

    print(
        f"  unique keys : {unique_keys:,}"
    )


# ============================================================================
# RARE TOKEN INDEX
# ============================================================================

def build_token_index(
    con: duckdb.DuckDBPyConnection,
    source_column: str,
    output_table: str,
    max_df: int,
) -> None:
    """
    Build an inverted token index.

    Only tokens with document frequency <= max_df
    are retained.

    This prevents extremely common words from creating
    huge candidate sets.
    """

    print(
        f"Building {output_table}..."
    )

    temp_postings = (
        output_table
        + "_all"
    )

    temp_frequency = (
        output_table
        + "_frequency"
    )

    # ------------------------------------------------------------------
    # Tokenize / postings
    # ------------------------------------------------------------------

    con.execute(
        f"""
        CREATE TEMP TABLE {temp_postings} AS

        SELECT DISTINCT

            country_key
                || chr(31)
                || token
                AS lookup_key,

            entity_id

        FROM (

            SELECT
                country_key,
                entity_id,

                UNNEST(
                    string_split(
                        {source_column},
                        ' '
                    )
                ) AS token

            FROM entities

            WHERE
                country_key <> ''
                AND {source_column} <> ''
        )

        WHERE
            token <> ''
        """
    )

    # ------------------------------------------------------------------
    # Frequency
    # ------------------------------------------------------------------

    con.execute(
        f"""
        CREATE TEMP TABLE {temp_frequency} AS

        SELECT
            lookup_key,
            COUNT(*) AS doc_freq

        FROM {temp_postings}

        GROUP BY lookup_key
        """
    )

    # ------------------------------------------------------------------
    # Keep selective tokens
    # ------------------------------------------------------------------

    con.execute(
        f"""
        CREATE TABLE {output_table} AS

        SELECT
            p.lookup_key,
            p.entity_id

        FROM {temp_postings} p

        INNER JOIN {temp_frequency} f
            USING (lookup_key)

        WHERE
            f.doc_freq <= ?

        ORDER BY
            p.lookup_key
        """,
        [max_df],
    )

    posting_count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM {output_table}
        """
    ).fetchone()[0]

    unique_keys = con.execute(
        f"""
        SELECT COUNT(DISTINCT lookup_key)
        FROM {output_table}
        """
    ).fetchone()[0]

    print(
        f"  retained postings: "
        f"{posting_count:,}"
    )

    print(
        f"  retained tokens  : "
        f"{unique_keys:,}"
    )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    con.execute(
        f"DROP TABLE {temp_frequency}"
    )

    con.execute(
        f"DROP TABLE {temp_postings}"
    )


# ============================================================================
# OPTIONAL CHARACTER N-GRAM INDEX
# ============================================================================

def build_ngram_index(
    con: duckdb.DuckDBPyConnection,
    max_df: int,
) -> None:
    """
    Build selective 3-gram and 4-gram name index.

    This is optional because it can become large.
    """

    print(
        "Building name_ngram_index..."
    )

    # ------------------------------------------------------------------
    # Generate n-grams
    # ------------------------------------------------------------------

    con.execute(
        """
        CREATE TEMP TABLE ngrams_all AS

        WITH base AS (

            SELECT
                entity_id,
                country_key,

                '^'
                || name_compact
                || '$'
                AS value

            FROM entities

            WHERE
                country_key <> ''
                AND name_compact <> ''
        ),

        grams3 AS (

            SELECT
                entity_id,
                country_key,

                substr(
                    value,
                    position,
                    3
                ) AS gram

            FROM base,

            range(
                1,
                length(value) - 1
            ) AS r(position)

            WHERE
                length(value) >= 3
        ),

        grams4 AS (

            SELECT
                entity_id,
                country_key,

                substr(
                    value,
                    position,
                    4
                ) AS gram

            FROM base,

            range(
                1,
                length(value) - 2
            ) AS r(position)

            WHERE
                length(value) >= 4
        )

        SELECT DISTINCT

            country_key
                || chr(31)
                || gram
                AS lookup_key,

            entity_id

        FROM (

            SELECT * FROM grams3

            UNION ALL

            SELECT * FROM grams4

        )

        WHERE
            gram <> ''
        """
    )

    # ------------------------------------------------------------------
    # Frequency
    # ------------------------------------------------------------------

    con.execute(
        """
        CREATE TEMP TABLE ngram_frequency AS

        SELECT
            lookup_key,
            COUNT(*) AS doc_freq

        FROM ngrams_all

        GROUP BY lookup_key
        """
    )

    # ------------------------------------------------------------------
    # Retain selective n-grams
    # ------------------------------------------------------------------

    con.execute(
        """
        CREATE TABLE name_ngram_index AS

        SELECT
            n.lookup_key,
            n.entity_id

        FROM ngrams_all n

        INNER JOIN ngram_frequency f
            USING (lookup_key)

        WHERE
            f.doc_freq <= ?

        ORDER BY
            n.lookup_key
        """,
        [max_df],
    )

    count = con.execute(
        """
        SELECT COUNT(*)
        FROM name_ngram_index
        """
    ).fetchone()[0]

    print(
        f"  retained postings: "
        f"{count:,}"
    )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    con.execute(
        "DROP TABLE ngram_frequency"
    )

    con.execute(
        "DROP TABLE ngrams_all"
    )


# ============================================================================
# METADATA
# ============================================================================

def create_metadata(
    con: duckdb.DuckDBPyConnection,
    source2_rows: int,
    source3_rows: int,
    chunk_size: int,
    threads: int,
    enable_ngrams: bool,
    max_name_token_df: int,
    max_address_token_df: int,
) -> None:
    """
    Store build configuration inside the database.
    """

    con.execute(
        """
        CREATE TABLE metadata (
            key VARCHAR,
            value VARCHAR
        )
        """
    )

    metadata = [
        (
            "source2_rows",
            str(source2_rows),
        ),
        (
            "source3_rows",
            str(source3_rows),
        ),
        (
            "total_entities",
            str(
                source2_rows
                + source3_rows
            ),
        ),
        (
            "chunk_size",
            str(chunk_size),
        ),
        (
            "threads",
            str(threads),
        ),
        (
            "enable_ngrams",
            str(enable_ngrams),
        ),
        (
            "max_name_token_df",
            str(max_name_token_df),
        ),
        (
            "max_address_token_df",
            str(max_address_token_df),
        ),
        (
            "normalization_version",
            "evidence-preserving-v1",
        ),
        (
            "index_version",
            "duckdb-hybrid-v1",
        ),
    ]

    con.executemany(
        """
        INSERT INTO metadata
        VALUES (?, ?)
        """,
        metadata,
    )


# ============================================================================
# MAIN BUILD PIPELINE
# ============================================================================

def build_indexes(
    source2_path: Path,
    source3_path: Path,
    output_path: Path,
    chunk_size: int,
    threads: int,
    enable_ngrams: bool,
    max_name_token_df: int,
    max_address_token_df: int,
) -> None:
    """
    Build complete training retrieval database.
    """

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------

    if not source2_path.exists():
        raise FileNotFoundError(
            f"Source 2 not found:\n{source2_path}"
        )

    if not source3_path.exists():
        raise FileNotFoundError(
            f"Source 3 not found:\n{source3_path}"
        )

    # ------------------------------------------------------------------
    # Create output directory
    # ------------------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Remove old database
    # ------------------------------------------------------------------

    if output_path.exists():

        print_header(
            "REMOVING PREVIOUS INDEX DATABASE"
        )

        print(
            output_path
        )

        output_path.unlink()

    # ------------------------------------------------------------------
    # Create DuckDB
    # ------------------------------------------------------------------

    con = duckdb.connect(
        str(output_path)
    )

    try:

        # ==============================================================
        # DuckDB runtime
        # ==============================================================

        con.execute(
            f"PRAGMA threads={threads}"
        )

        # ==============================================================
        # CREATE SHARED ENTITIES TABLE ONCE
        # ==============================================================

        print_header(
            "CREATING SHARED ENTITIES TABLE"
        )

        create_entities_table(
            con
        )

        print(
            "entities table created."
        )

        # ==============================================================
        # SOURCE 2
        # ==============================================================

        source2_rows = append_source(
            con=con,
            path=source2_path,
            source_name="S2",
            chunk_size=chunk_size,
        )

        # ==============================================================
        # SOURCE 3
        # ==============================================================

        source3_rows = append_source(
            con=con,
            path=source3_path,
            source_name="S3",
            chunk_size=chunk_size,
        )

        # ==============================================================
        # STORAGE SUMMARY
        # ==============================================================

        print_header(
            "ENTITY STORAGE COMPLETE"
        )

        total_entities = con.execute(
            """
            SELECT COUNT(*)
            FROM entities
            """
        ).fetchone()[0]

        print(
            f"S2 rows : {source2_rows:,}"
        )

        print(
            f"S3 rows : {source3_rows:,}"
        )

        print(
            f"Total   : {total_entities:,}"
        )

        # ==============================================================
        # ANALYZE
        # ==============================================================

        print()
        print(
            "Analyzing entities table..."
        )

        con.execute(
            "ANALYZE entities"
        )

        # ==============================================================
        # EXACT INDEXES
        # ==============================================================

        print_header(
            "BUILDING EXACT INDEXES"
        )

        build_exact_index(
            con=con,
            value_column="name_nfkc",
            index_table="exact_name_index",
            frequency_table="exact_name_frequency",
        )

        build_exact_index(
            con=con,
            value_column="name_compact",
            index_table="compact_name_index",
            frequency_table="compact_name_frequency",
        )

        build_exact_index(
            con=con,
            value_column="name_core_compact",
            index_table="core_name_index",
            frequency_table="core_name_frequency",
        )

        build_exact_index(
            con=con,
            value_column="address_nfkc",
            index_table="exact_address_index",
            frequency_table="exact_address_frequency",
        )

        # ==============================================================
        # POSTAL INDEX
        # ==============================================================

        print_header(
            "BUILDING POSTAL INDEX"
        )

        build_postal_index(
            con
        )

        # ==============================================================
        # RARE NAME TOKENS
        # ==============================================================

        print_header(
            "BUILDING RARE NAME TOKEN INDEX"
        )

        build_token_index(
            con=con,
            source_column="name_tokens",
            output_table="rare_name_token_index",
            max_df=max_name_token_df,
        )

        # ==============================================================
        # RARE ADDRESS TOKENS
        # ==============================================================

        print_header(
            "BUILDING RARE ADDRESS TOKEN INDEX"
        )

        build_token_index(
            con=con,
            source_column="address_informative_tokens",
            output_table="rare_address_token_index",
            max_df=max_address_token_df,
        )

        # ==============================================================
        # OPTIONAL N-GRAMS
        # ==============================================================

        if enable_ngrams:

            print_header(
                "BUILDING CHARACTER N-GRAM INDEX"
            )

            build_ngram_index(
                con=con,
                max_df=DEFAULT_MAX_NGRAM_DF,
            )

        else:

            print_header(
                "CHARACTER N-GRAM INDEX"
            )

            print(
                "OFF"
            )

            print(
                "It will be evaluated later after "
                "blocking benchmark."
            )

        # ==============================================================
        # METADATA
        # ==============================================================

        create_metadata(
            con=con,
            source2_rows=source2_rows,
            source3_rows=source3_rows,
            chunk_size=chunk_size,
            threads=threads,
            enable_ngrams=enable_ngrams,
            max_name_token_df=max_name_token_df,
            max_address_token_df=max_address_token_df,
        )

        # ==============================================================
        # CHECKPOINT
        # ==============================================================

        print()
        print(
            "Checkpointing database..."
        )

        con.execute(
            "CHECKPOINT"
        )

        # ==============================================================
        # FINAL TABLE LIST
        # ==============================================================

        print_header(
            "INDEX BUILD COMPLETE"
        )

        print(
            f"Database:"
            f"\n  {output_path}"
        )

        if output_path.exists():

            size_mb = (
                output_path.stat().st_size
                / (1024 * 1024)
            )

            print(
                f"\nDatabase size:"
                f" {size_mb:,.2f} MB"
            )

        print()
        print(
            "Tables:"
        )

        tables = con.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'main'
            ORDER BY table_name
            """
        ).fetchall()

        for row in tables:
            print(
                f"  {row[0]}"
            )

        print()
        print(
            "SUCCESS."
        )

    finally:

        con.close()


# ============================================================================
# CLI
# ============================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Build DuckDB retrieval indexes "
            "for Amazon ML Challenge 2026."
        )
    )

    parser.add_argument(
        "--source2",
        type=Path,
        default=DEFAULT_SOURCE2,
    )

    parser.add_argument(
        "--source3",
        type=Path,
        default=DEFAULT_SOURCE3,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=(
            "Rows per pandas processing chunk. "
            "Default: 100000"
        ),
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_THREADS,
        help=(
            "DuckDB execution threads. "
            "Default: 4"
        ),
    )

    parser.add_argument(
        "--max-name-token-df",
        type=int,
        default=DEFAULT_MAX_NAME_TOKEN_DF,
        help=(
            "Maximum document frequency for "
            "name tokens. Default: 2000"
        ),
    )

    parser.add_argument(
        "--max-address-token-df",
        type=int,
        default=DEFAULT_MAX_ADDRESS_TOKEN_DF,
        help=(
            "Maximum document frequency for "
            "address tokens. Default: 2000"
        ),
    )

    parser.add_argument(
        "--enable-ngrams",
        action="store_true",
        help=(
            "Build optional character 3/4-gram "
            "name index."
        ),
    )

    return parser.parse_args()


# ============================================================================
# MAIN
# ============================================================================

def main():

    args = parse_args()

    print_header(
        "AMAZON ML CHALLENGE 2026"
    )

    print(
        "Business Entity Resolution"
    )

    print(
        "\nConfiguration:"
    )

    print(
        f"  Source 2       : {args.source2}"
    )

    print(
        f"  Source 3       : {args.source3}"
    )

    print(
        f"  Output         : {args.output}"
    )

    print(
        f"  Chunk size     : {args.chunk_size:,}"
    )

    print(
        f"  Threads        : {args.threads}"
    )

    print(
        f"  Name token DF  : {args.max_name_token_df:,}"
    )

    print(
        f"  Address token DF: "
        f"{args.max_address_token_df:,}"
    )

    print(
        f"  N-grams        : "
        f"{'ON' if args.enable_ngrams else 'OFF'}"
    )

    build_indexes(
        source2_path=args.source2,
        source3_path=args.source3,
        output_path=args.output,
        chunk_size=args.chunk_size,
        threads=args.threads,
        enable_ngrams=args.enable_ngrams,
        max_name_token_df=args.max_name_token_df,
        max_address_token_df=args.max_address_token_df,
    )


if __name__ == "__main__":
    main()