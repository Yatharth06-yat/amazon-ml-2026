"""
Amazon ML Challenge 2026
Business Entity Resolution

Disk-backed indexing layer using DuckDB.

Purpose:
    Fast candidate retrieval from normalized Source 2 / Source 3 data.

Index types:
    1. Exact normalized name
    2. Compact name
    3. Core name
    4. Exact normalized address
    5. Postal / PIN
    6. Rare name tokens
    7. Rare address tokens
    8. Optional character n-grams

Design:
    - DuckDB database on disk
    - Single-column exact lookup indexes
    - Rare-token postings to control candidate explosion
    - Country-aware lookup keys
    - No giant Python dictionaries
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb


# ============================================================================
# CONFIGURATION
# ============================================================================

SEPARATOR = chr(31)

# Starting values.
# These MUST be tuned later using blocking evaluation.
MAX_EXACT_NAME_DF = 5000
MAX_COMPACT_NAME_DF = 5000
MAX_CORE_NAME_DF = 3000
MAX_ADDRESS_DF = 3000
MAX_POSTAL_DF = 5000

# Rare token maximum document frequency.
MAX_NAME_TOKEN_DF = 2000
MAX_ADDRESS_TOKEN_DF = 2000

# Character n-grams are optional because they can create a large postings
# table at 10M+ row scale.
ENABLE_NGRAMS_BY_DEFAULT = False

MAX_NGRAM_DF = 1000


# ============================================================================
# HELPERS
# ============================================================================

def _safe_string(value: Any) -> str:
    if value is None:
        return ""

    try:
        if value != value:
            return ""
    except Exception:
        pass

    return str(value)


def _split_tokens(value: Any) -> list[str]:
    text = _safe_string(value).strip()

    if not text:
        return []

    return [
        token
        for token in text.split()
        if token
    ]


def _split_pipe(value: Any) -> list[str]:
    text = _safe_string(value).strip()

    if not text:
        return []

    return [
        token
        for token in text.split("|")
        if token
    ]


def _make_lookup_key(
    country_key: str,
    value: str,
) -> str:
    if not country_key or not value:
        return ""

    return (
        country_key
        + SEPARATOR
        + value
    )


def _parameter_placeholders(
    count: int,
) -> str:
    return ",".join(
        ["?"] * count
    )


# ============================================================================
# INDEX STORE
# ============================================================================

class IndexStore:
    """
    Read/query interface for the generated DuckDB indexes.
    """

    def __init__(
        self,
        db_path: str | Path,
    ) -> None:

        self.db_path = str(
            Path(db_path)
            .resolve()
        )

        self.con = duckdb.connect(
            self.db_path
        )

    # ---------------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------------

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> "IndexStore":
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        self.close()

    # ---------------------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------------------

    def tables(self) -> list[str]:
        rows = self.con.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'main'
            ORDER BY table_name
            """
        ).fetchall()

        return [
            row[0]
            for row in rows
        ]

    def table_exists(
        self,
        table_name: str,
    ) -> bool:

        return table_name in self.tables()

    # ---------------------------------------------------------------------
    # Generic lookup
    # ---------------------------------------------------------------------

    def _lookup_single(
        self,
        index_table: str,
        frequency_table: str,
        lookup_key: str,
        max_frequency: int,
    ) -> set[str]:

        if not lookup_key:
            return set()

        if not self.table_exists(
            index_table
        ):
            return set()

        # Frequency-aware candidate control.
        if self.table_exists(
            frequency_table
        ):

            row = self.con.execute(
                f"""
                SELECT doc_freq
                FROM {frequency_table}
                WHERE lookup_key = ?
                LIMIT 1
                """,
                [lookup_key],
            ).fetchone()

            if row is None:
                return set()

            doc_freq = int(
                row[0]
            )

            # Extremely common keys are not safe as standalone
            # blocking keys.
            if doc_freq > max_frequency:
                return set()

        rows = self.con.execute(
            f"""
            SELECT entity_id
            FROM {index_table}
            WHERE lookup_key = ?
            """,
            [lookup_key],
        ).fetchall()

        return {
            row[0]
            for row in rows
        }

    def _lookup_many(
        self,
        index_table: str,
        keys: list[str],
        max_frequency: int | None = None,
        frequency_table: str | None = None,
    ) -> set[str]:

        keys = [
            key
            for key in keys
            if key
        ]

        if not keys:
            return set()

        if not self.table_exists(
            index_table
        ):
            return set()

        keys = list(set(keys))

        allowed_keys = keys

        # -------------------------------------------------------------
        # Frequency filtering
        # -------------------------------------------------------------

        if (
            max_frequency is not None
            and frequency_table is not None
            and self.table_exists(
                frequency_table
            )
        ):

            placeholders = (
                _parameter_placeholders(
                    len(keys)
                )
            )

            frequency_rows = self.con.execute(
                f"""
                SELECT lookup_key
                FROM {frequency_table}
                WHERE lookup_key IN ({placeholders})
                  AND doc_freq <= ?
                """,
                keys + [max_frequency],
            ).fetchall()

            allowed_keys = [
                row[0]
                for row in frequency_rows
            ]

        if not allowed_keys:
            return set()

        placeholders = (
            _parameter_placeholders(
                len(allowed_keys)
            )
        )

        rows = self.con.execute(
            f"""
            SELECT entity_id
            FROM {index_table}
            WHERE lookup_key IN ({placeholders})
            """,
            allowed_keys,
        ).fetchall()

        return {
            row[0]
            for row in rows
        }

    # ---------------------------------------------------------------------
    # Exact name
    # ---------------------------------------------------------------------

    def get_exact_name_candidates(
        self,
        country_key: str,
        name_nfkc: str,
    ) -> set[str]:

        key = _make_lookup_key(
            country_key,
            name_nfkc,
        )

        return self._lookup_single(
            "exact_name_index",
            "exact_name_frequency",
            key,
            MAX_EXACT_NAME_DF,
        )

    # ---------------------------------------------------------------------
    # Compact name
    # ---------------------------------------------------------------------

    def get_compact_name_candidates(
        self,
        country_key: str,
        name_compact: str,
    ) -> set[str]:

        key = _make_lookup_key(
            country_key,
            name_compact,
        )

        return self._lookup_single(
            "compact_name_index",
            "compact_name_frequency",
            key,
            MAX_COMPACT_NAME_DF,
        )

    # ---------------------------------------------------------------------
    # Core name
    # ---------------------------------------------------------------------

    def get_core_name_candidates(
        self,
        country_key: str,
        name_core_compact: str,
    ) -> set[str]:

        key = _make_lookup_key(
            country_key,
            name_core_compact,
        )

        return self._lookup_single(
            "core_name_index",
            "core_name_frequency",
            key,
            MAX_CORE_NAME_DF,
        )

    # ---------------------------------------------------------------------
    # Exact address
    # ---------------------------------------------------------------------

    def get_exact_address_candidates(
        self,
        country_key: str,
        address_nfkc: str,
    ) -> set[str]:

        key = _make_lookup_key(
            country_key,
            address_nfkc,
        )

        return self._lookup_single(
            "exact_address_index",
            "exact_address_frequency",
            key,
            MAX_ADDRESS_DF,
        )

    # ---------------------------------------------------------------------
    # Postal / PIN
    # ---------------------------------------------------------------------

    def get_postal_candidates(
        self,
        country_key: str,
        postal_values: list[str],
    ) -> set[str]:

        keys = [
            _make_lookup_key(
                country_key,
                postal,
            )
            for postal in postal_values
            if postal
        ]

        return self._lookup_many(
            "postal_index",
            keys,
            max_frequency=MAX_POSTAL_DF,
            frequency_table="postal_frequency",
        )

    # ---------------------------------------------------------------------
    # Rare name tokens
    # ---------------------------------------------------------------------

    def get_rare_name_token_candidates(
        self,
        country_key: str,
        tokens: list[str],
    ) -> set[str]:

        keys = [
            _make_lookup_key(
                country_key,
                token,
            )
            for token in tokens
            if token
        ]

        return self._lookup_many(
            "rare_name_token_index",
            keys,
        )

    # ---------------------------------------------------------------------
    # Rare address tokens
    # ---------------------------------------------------------------------

    def get_rare_address_token_candidates(
        self,
        country_key: str,
        tokens: list[str],
    ) -> set[str]:

        keys = [
            _make_lookup_key(
                country_key,
                token,
            )
            for token in tokens
            if token
        ]

        return self._lookup_many(
            "rare_address_token_index",
            keys,
        )

    # ---------------------------------------------------------------------
    # Optional character n-grams
    # ---------------------------------------------------------------------

    def get_ngram_candidates(
        self,
        country_key: str,
        ngrams: list[str],
    ) -> set[str]:

        if not self.table_exists(
            "name_ngram_index"
        ):
            return set()

        keys = [
            _make_lookup_key(
                country_key,
                gram,
            )
            for gram in ngrams
            if gram
        ]

        return self._lookup_many(
            "name_ngram_index",
            keys,
        )

    # ---------------------------------------------------------------------
    # Multi-pass retrieval
    # ---------------------------------------------------------------------

    def retrieve_candidates(
        self,
        record: dict[str, Any],
        enable_ngrams: bool = False,
    ) -> dict[str, set[str]]:
        """
        Return:

            {
                entity_id: {
                    "exact_name",
                    "rare_name_token",
                    ...
                }
            }

        The provenance is important because later feature engineering
        can use the number/types of retrieval signals.
        """

        country_key = _safe_string(
            record.get(
                "country_key",
                "",
            )
        )

        name_nfkc = _safe_string(
            record.get(
                "name_nfkc",
                "",
            )
        )

        name_compact = _safe_string(
            record.get(
                "name_compact",
                "",
            )
        )

        name_core_compact = _safe_string(
            record.get(
                "name_core_compact",
                "",
            )
        )

        address_nfkc = _safe_string(
            record.get(
                "address_nfkc",
                "",
            )
        )

        name_tokens = _split_tokens(
            record.get(
                "name_tokens",
                "",
            )
        )

        address_tokens = _split_tokens(
            record.get(
                "address_informative_tokens",
                "",
            )
        )

        postal_values = _split_pipe(
            record.get(
                "postal_candidates",
                "",
            )
        )

        candidate_methods: dict[
            str,
            set[str]
        ] = {}

        def add_candidates(
            entity_ids: set[str],
            method: str,
        ) -> None:

            for entity_id in entity_ids:

                candidate_methods.setdefault(
                    entity_id,
                    set(),
                ).add(method)

        # =============================================================
        # PASS 1: EXACT NAME
        # =============================================================

        add_candidates(
            self.get_exact_name_candidates(
                country_key,
                name_nfkc,
            ),
            "exact_name",
        )

        # =============================================================
        # PASS 2: COMPACT NAME
        # =============================================================

        add_candidates(
            self.get_compact_name_candidates(
                country_key,
                name_compact,
            ),
            "compact_name",
        )

        # =============================================================
        # PASS 3: CORE NAME
        # =============================================================

        add_candidates(
            self.get_core_name_candidates(
                country_key,
                name_core_compact,
            ),
            "core_name",
        )

        # =============================================================
        # PASS 4: RARE NAME TOKENS
        # =============================================================

        add_candidates(
            self.get_rare_name_token_candidates(
                country_key,
                name_tokens,
            ),
            "rare_name_token",
        )

        # =============================================================
        # PASS 5: EXACT ADDRESS
        # =============================================================

        add_candidates(
            self.get_exact_address_candidates(
                country_key,
                address_nfkc,
            ),
            "exact_address",
        )

        # =============================================================
        # PASS 6: POSTAL / PIN
        # =============================================================

        add_candidates(
            self.get_postal_candidates(
                country_key,
                postal_values,
            ),
            "postal",
        )

        # =============================================================
        # PASS 7: INFORMATIVE ADDRESS TOKENS
        # =============================================================

        add_candidates(
            self.get_rare_address_token_candidates(
                country_key,
                address_tokens,
            ),
            "rare_address_token",
        )

        # =============================================================
        # PASS 8: OPTIONAL CHARACTER N-GRAM
        # =============================================================

        if enable_ngrams:

            try:
                from src.normalize import char_ngrams

                grams = char_ngrams(
                    name_compact
                )

                add_candidates(
                    self.get_ngram_candidates(
                        country_key,
                        grams,
                    ),
                    "name_ngram",
                )

            except Exception:
                # N-gram retrieval is optional.
                pass

        return candidate_methods


# ============================================================================
# SIMPLE TEST
# ============================================================================

def test_index(
    db_path: str | Path,
) -> None:

    with IndexStore(
        db_path
    ) as store:

        print(
            "\nAvailable tables:"
        )

        for table in store.tables():
            print(
                f"  {table}"
            )

        print(
            "\nIndex store opened successfully."
        )


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--db",
        required=True,
        help="Path to DuckDB index database.",
    )

    args = parser.parse_args()

    test_index(
        args.db
    )