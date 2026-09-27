import argparse
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import unicodedata


SEP = chr(31)

LEGAL_SUFFIXES = {
    "inc",
    "incorporated",
    "llc",
    "ltd",
    "limited",
    "corp",
    "corporation",
    "co",
    "company",
    "plc",
    "pvt",
    "private",
    "llp",
}


def primary_norm(value: str) -> str:
    if value is None:
        return ""

    s = unicodedata.normalize("NFKC", str(value)).casefold()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def compact_norm(value: str) -> str:
    s = primary_norm(value)
    return re.sub(r"[^\w]+", "", s, flags=re.UNICODE)


def core_norm(value: str) -> str:
    s = primary_norm(value)

    tokens = re.findall(r"\w+", s, flags=re.UNICODE)

    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()

    return "".join(tokens)


def country_norm(value: str) -> str:
    return primary_norm(value)


def make_keys(df: pd.DataFrame) -> pd.DataFrame:
    country = df["country"].map(country_norm)

    name_primary = df["business_name"].map(primary_norm)
    name_compact = df["business_name"].map(compact_norm)
    name_core = df["business_name"].map(core_norm)

    address_primary = df["business_address"].map(primary_norm)

    return pd.DataFrame(
        {
            "s1_entity_id": df["entity_id"].astype(str),
            "country_key": country,
            "exact_name_key": country + SEP + name_primary,
            "compact_name_key": country + SEP + name_compact,
            "core_name_key": country + SEP + name_core,
            "exact_address_key": country + SEP + address_primary,
        }
    )


def split_train_valid(ids: pd.Series) -> np.ndarray:
    h = pd.util.hash_pandas_object(ids.astype(str), index=False).astype("uint64")
    return (h % 10) == 0


def parse_match_ids(value: str):
    if not value:
        return []

    parts = re.split(r"[|,;\s]+", str(value).strip())

    return [x.strip() for x in parts if x.strip()]


def select_s1_rows(
    source1_path: str,
    gt_map: dict,
    max_train: int,
    max_valid: int,
    chunk_size: int,
):
    train_parts = []
    valid_parts = []

    train_count = 0
    valid_count = 0

    for chunk in pd.read_csv(
        source1_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=chunk_size,
    ):
        chunk["entity_id"] = chunk["entity_id"].astype(str)

        valid_mask = split_train_valid(chunk["entity_id"])

        valid_chunk = chunk[valid_mask].copy()
        train_chunk = chunk[~valid_mask].copy()

        if train_count < max_train:
            take = min(max_train - train_count, len(train_chunk))

            if take > 0:
                train_parts.append(train_chunk.iloc[:take].copy())
                train_count += take

        if valid_count < max_valid:
            take = min(max_valid - valid_count, len(valid_chunk))

            if take > 0:
                valid_parts.append(valid_chunk.iloc[:take].copy())
                valid_count += take

        if train_count >= max_train and valid_count >= max_valid:
            break

    train_df = pd.concat(train_parts, ignore_index=True)
    valid_df = pd.concat(valid_parts, ignore_index=True)

    train_df["matched_entity_ids"] = train_df["entity_id"].map(gt_map).fillna("")
    valid_df["matched_entity_ids"] = valid_df["entity_id"].map(gt_map).fillna("")

    return train_df, valid_df


def get_positive_pairs(con, s1_df, positives_per_s1):
    records = []

    for row in s1_df.itertuples(index=False):
        ids = parse_match_ids(row.matched_entity_ids)

        for candidate_id in ids[:positives_per_s1]:
            records.append(
                {
                    "s1_entity_id": str(row.entity_id),
                    "candidate_entity_id": str(candidate_id),
                    "label": 1,
                    "s1_name": row.business_name,
                    "s1_address": row.business_address,
                    "s1_country": row.country,
                }
            )

    if not records:
        return pd.DataFrame(
            columns=[
                "s1_entity_id",
                "candidate_source",
                "candidate_entity_id",
                "label",
                "s1_name",
                "s1_address",
                "s1_country",
                "candidate_name",
                "candidate_address",
                "candidate_country",
            ]
        )

    pos_df = pd.DataFrame(records)

    con.register("positive_pairs", pos_df)

    result = con.execute(
        """
        SELECT
            p.s1_entity_id,
            e.source AS candidate_source,
            e.entity_id AS candidate_entity_id,
            1 AS label,
            p.s1_name,
            p.s1_address,
            p.s1_country,
            e.business_name AS candidate_name,
            e.business_address AS candidate_address,
            e.country AS candidate_country
        FROM positive_pairs p
        JOIN entities e
          ON e.entity_id = p.candidate_entity_id
        """
    ).df()

    con.unregister("positive_pairs")

    return result


def get_hard_negatives(
    con,
    s1_df,
    negatives_per_s1,
):
    if len(s1_df) == 0:
        return pd.DataFrame()

    keys = make_keys(s1_df)

    positive_records = []

    for row in s1_df.itertuples(index=False):
        ids = parse_match_ids(row.matched_entity_ids)

        for candidate_id in ids:
            positive_records.append(
                {
                    "s1_entity_id": str(row.entity_id),
                    "candidate_entity_id": str(candidate_id),
                }
            )

    if positive_records:
        pos_df = pd.DataFrame(positive_records)
    else:
        pos_df = pd.DataFrame(
            columns=["s1_entity_id", "candidate_entity_id"]
        )

    con.register("s1_keys", keys)
    con.register("positive_ids", pos_df)

    query = f"""
    WITH candidates AS (

        SELECT DISTINCT
            k.s1_entity_id,
            e.source AS candidate_source,
            e.entity_id AS candidate_entity_id
        FROM s1_keys k
        JOIN exact_name_index i
          ON i.lookup_key = k.exact_name_key
        JOIN entities e
          ON e.entity_id = i.entity_id

        UNION

        SELECT DISTINCT
            k.s1_entity_id,
            e.source AS candidate_source,
            e.entity_id AS candidate_entity_id
        FROM s1_keys k
        JOIN compact_name_index i
          ON i.lookup_key = k.compact_name_key
        JOIN entities e
          ON e.entity_id = i.entity_id

        UNION

        SELECT DISTINCT
            k.s1_entity_id,
            e.source AS candidate_source,
            e.entity_id AS candidate_entity_id
        FROM s1_keys k
        JOIN core_name_index i
          ON i.lookup_key = k.core_name_key
        JOIN entities e
          ON e.entity_id = i.entity_id

        UNION

        SELECT DISTINCT
            k.s1_entity_id,
            e.source AS candidate_source,
            e.entity_id AS candidate_entity_id
        FROM s1_keys k
        JOIN exact_address_index i
          ON i.lookup_key = k.exact_address_key
        JOIN entities e
          ON e.entity_id = i.entity_id
    ),

    filtered AS (
        SELECT
            c.*
        FROM candidates c
        WHERE NOT EXISTS (
            SELECT 1
            FROM positive_ids p
            WHERE p.s1_entity_id = c.s1_entity_id
              AND p.candidate_entity_id = c.candidate_entity_id
        )
    ),

    ranked AS (
        SELECT
            *,
            ROW_NUMBER() OVER (
                PARTITION BY s1_entity_id
                ORDER BY hash(
                    CAST(candidate_source AS VARCHAR)
                    || '{SEP}'
                    || CAST(candidate_entity_id AS VARCHAR)
                )
            ) AS rn
        FROM filtered
    )

    SELECT
        r.s1_entity_id,
        r.candidate_source,
        r.candidate_entity_id,
        0 AS label,
        s.business_name AS s1_name,
        s.business_address AS s1_address,
        s.country AS s1_country,
        e.business_name AS candidate_name,
        e.business_address AS candidate_address,
        e.country AS candidate_country
    FROM ranked r
    JOIN s1_keys k
      ON k.s1_entity_id = r.s1_entity_id
    JOIN (
        SELECT
            entity_id,
            business_name,
            business_address,
            country
        FROM entities
    ) e
      ON e.entity_id = r.candidate_entity_id
    JOIN (
        SELECT
            entity_id,
            business_name,
            business_address,
            country
        FROM s1_keys
        JOIN (
            SELECT
                entity_id,
                business_name,
                business_address,
                country
            FROM (
                VALUES
                {",".join(["(NULL,NULL,NULL,NULL)"])}
            )
        ) dummy
          ON FALSE
    ) x
      ON FALSE
    WHERE r.rn <= {int(negatives_per_s1)}
    """

    # DuckDB cannot use the above synthetic S1 projection directly.
    # Re-register the real S1 table and run a simpler query.
    con.unregister("s1_keys")

    s1_small = s1_df[
        ["entity_id", "business_name", "business_address", "country"]
    ].copy()

    s1_small = s1_small.rename(columns={"entity_id": "s1_entity_id"})

    con.register("s1_small", s1_small)
    con.register("s1_keys", keys)

    query = f"""
    WITH candidates AS (

        SELECT DISTINCT
            k.s1_entity_id,
            e.source AS candidate_source,
            e.entity_id AS candidate_entity_id
        FROM s1_keys k
        JOIN exact_name_index i
          ON i.lookup_key = k.exact_name_key
        JOIN entities e
          ON e.entity_id = i.entity_id

        UNION

        SELECT DISTINCT
            k.s1_entity_id,
            e.source AS candidate_source,
            e.entity_id AS candidate_entity_id
        FROM s1_keys k
        JOIN compact_name_index i
          ON i.lookup_key = k.compact_name_key
        JOIN entities e
          ON e.entity_id = i.entity_id

        UNION

        SELECT DISTINCT
            k.s1_entity_id,
            e.source AS candidate_source,
            e.entity_id AS candidate_entity_id
        FROM s1_keys k
        JOIN core_name_index i
          ON i.lookup_key = k.core_name_key
        JOIN entities e
          ON e.entity_id = i.entity_id

        UNION

        SELECT DISTINCT
            k.s1_entity_id,
            e.source AS candidate_source,
            e.entity_id AS candidate_entity_id
        FROM s1_keys k
        JOIN exact_address_index i
          ON i.lookup_key = k.exact_address_key
        JOIN entities e
          ON e.entity_id = i.entity_id
    ),

    filtered AS (
        SELECT c.*
        FROM candidates c
        WHERE NOT EXISTS (
            SELECT 1
            FROM positive_ids p
            WHERE p.s1_entity_id = c.s1_entity_id
              AND p.candidate_entity_id = c.candidate_entity_id
        )
    ),

    ranked AS (
        SELECT
            *,
            ROW_NUMBER() OVER (
                PARTITION BY s1_entity_id
                ORDER BY hash(
                    CAST(candidate_source AS VARCHAR)
                    || '{SEP}'
                    || CAST(candidate_entity_id AS VARCHAR)
                )
            ) AS rn
        FROM filtered
    )

    SELECT
        r.s1_entity_id,
        r.candidate_source,
        r.candidate_entity_id,
        0 AS label,
        s.business_name AS s1_name,
        s.business_address AS s1_address,
        s.country AS s1_country,
        e.business_name AS candidate_name,
        e.business_address AS candidate_address,
        e.country AS candidate_country
    FROM ranked r
    JOIN s1_small s
      ON s.s1_entity_id = r.s1_entity_id
    JOIN entities e
      ON e.entity_id = r.candidate_entity_id
    WHERE r.rn <= {int(negatives_per_s1)}
    """

    result = con.execute(query).df()

    con.unregister("s1_keys")
    con.unregister("s1_small")
    con.unregister("positive_ids")

    return result


def write_parquet(df, path, writer):
    if len(df) == 0:
        return writer

    table = pa.Table.from_pandas(
        df,
        preserve_index=False,
    )

    if writer is None:
        writer = pq.ParquetWriter(
            path,
            table.schema,
            compression="zstd",
        )

    writer.write_table(table)

    return writer


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--db", required=True)
    parser.add_argument("--source1", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--out-dir", required=True)

    parser.add_argument("--max-train-s1", type=int, default=200_000)
    parser.add_argument("--max-valid-s1", type=int, default=30_000)

    parser.add_argument("--positives-per-s1", type=int, default=2)
    parser.add_argument("--negatives-per-s1", type=int, default=4)

    parser.add_argument("--batch-size", type=int, default=10_000)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = str(out_dir / "train_pairs.parquet")
    valid_path = str(out_dir / "valid_pairs.parquet")

    print("Loading ground truth...")

    gt = pd.read_csv(
        args.ground_truth,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=["source1_entity_id", "matched_entity_ids"],
    )

    gt["source1_entity_id"] = gt["source1_entity_id"].astype(str)

    gt_map = dict(
        zip(
            gt["source1_entity_id"],
            gt["matched_entity_ids"],
        )
    )

    print(f"Ground truth rows: {len(gt):,}")

    print("Selecting S1 rows...")

    train_s1, valid_s1 = select_s1_rows(
        args.source1,
        gt_map,
        args.max_train_s1,
        args.max_valid_s1,
        args.batch_size,
    )

    print(f"Train S1: {len(train_s1):,}")
    print(f"Valid S1: {len(valid_s1):,}")

    con = duckdb.connect(args.db)
    con.execute("PRAGMA threads=4")

    train_writer = None
    valid_writer = None

    for split_name, split_df, output_path, writer in [
        ("train", train_s1, train_path, train_writer),
        ("valid", valid_s1, valid_path, valid_writer),
    ]:

        print(f"\nBuilding {split_name} pairs...")

        for start in range(0, len(split_df), args.batch_size):
            batch = split_df.iloc[
                start : start + args.batch_size
            ].copy()

            print(
                f"{split_name}: "
                f"{start:,} -> "
                f"{start + len(batch):,}"
            )

            positive_df = get_positive_pairs(
                con,
                batch,
                args.positives_per_s1,
            )

            negative_df = get_hard_negatives(
                con,
                batch,
                args.negatives_per_s1,
            )

            combined = pd.concat(
                [
                    positive_df,
                    negative_df,
                ],
                ignore_index=True,
            )

            if len(combined):
                writer = write_parquet(
                    combined,
                    output_path,
                    writer,
                )

                print(
                    f"pairs written: {len(combined):,}"
                )

        if split_name == "train":
            train_writer = writer
        else:
            valid_writer = writer

    if train_writer is not None:
        train_writer.close()

    if valid_writer is not None:
        valid_writer.close()

    con.close()

    print("\nDONE")
    print(f"Train: {train_path}")
    print(f"Valid: {valid_path}")


if __name__ == "__main__":
    main()