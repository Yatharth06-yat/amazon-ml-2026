"""
Amazon ML Challenge 2026
Business Entity Resolution

Evidence-preserving, multilingual normalization.

Pipeline:

    RAW
      ↓
    Unicode NFKC + casefold
      ↓
    Native / accent-free / transliteration views
      ↓
    Token / compact / core-name views
      ↓
    Numbers / postal candidates
      ↓
    Blocking + feature engineering

Design principles:
- Preserve raw values.
- Never rely on one normalized string.
- Transliteration is auxiliary only.
- Accent-free representation is auxiliary only.
- Legal suffix removal creates an auxiliary core-name view.
- Address and name have separate normalization pipelines.
- Designed for chunk-based processing on multi-million-row data.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

import pandas as pd


# ============================================================================
# OPTIONAL DEPENDENCIES
# ============================================================================

try:
    from unidecode import unidecode
except ImportError:
    unidecode = None


try:
    from metaphone import doublemetaphone
except ImportError:
    doublemetaphone = None


# ============================================================================
# CONFIGURATION
# ============================================================================

ENABLE_TRANSLITERATION = True

# Keep phonetic matching disabled initially.
# We will benchmark it later during blocking experiments.
ENABLE_PHONETIC = False


# ============================================================================
# LEGAL SUFFIXES
# ============================================================================

LEGAL_SUFFIXES = [
    # US / UK / India
    "private limited",
    "private ltd",
    "pvt limited",
    "pvt ltd",
    "public limited",
    "public ltd",
    "limited liability company",
    "limited liability co",
    "limited",
    "ltd",
    "llc",
    "llp",
    "incorporated",
    "inc",
    "corporation",
    "corp",
    "company",
    "co",

    # India
    "private",
    "pvt",

    # France / Europe
    "societe anonyme",
    "société anonyme",
    "sarl",
    "sas",
    "s.a.s",
    "sa",

    # Germany / Europe
    "gmbh",
    "ag",

    # Other
    "plc",
    "bv",
    "nv",
    "pte ltd",
    "pty ltd",
]


# ============================================================================
# GENERIC ADDRESS TOKENS
# ============================================================================

GENERIC_ADDRESS_TOKENS = {
    "street",
    "st",
    "road",
    "rd",
    "avenue",
    "ave",
    "boulevard",
    "blvd",
    "lane",
    "ln",
    "drive",
    "dr",
    "highway",
    "hwy",
    "parkway",
    "pkwy",
    "place",
    "pl",
    "way",
    "square",
    "sq",
    "building",
    "bldg",
    "floor",
    "fl",
    "unit",
    "suite",
    "ste",
}


# ============================================================================
# COUNTRY / POSTAL CONFIG
# ============================================================================

US_STATE_CODES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE",
    "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS",
    "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS",
    "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY",
    "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV",
    "WI", "WY", "DC",
}

US_STATE_CODES_LOWER = {x.lower() for x in US_STATE_CODES}


# Generic patterns only.
# Context-aware extraction happens inside extract_postal_candidates().
US_ZIP_RE = re.compile(r"\b\d{5}(?:-\d{4})?\b")
INDIA_PIN_RE = re.compile(r"\b[1-9]\d{5}\b")
FRANCE_POSTAL_RE = re.compile(r"\b\d{5}\b")


# ============================================================================
# BASIC TEXT UTILITIES
# ============================================================================

def safe_text(value: Any) -> str:
    """
    Convert a scalar safely to string.

    None / NaN / pandas missing values become "".
    """
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value)


def nfkc_casefold(text: Any) -> str:
    """
    Primary Unicode-safe normalization.

    Example:
        "  ABC  Pvt. Ltd. "
        ->
        "abc pvt. ltd."
    """
    value = safe_text(text)

    if not value:
        return ""

    value = unicodedata.normalize(
        "NFKC",
        value,
    )

    value = value.casefold()

    value = re.sub(
        r"\s+",
        " ",
        value,
    ).strip()

    return value


def accent_free(text: Any) -> str:
    """
    Auxiliary accent-insensitive representation.

    Example:
        "Président"
        ->
        "president"
    """
    value = nfkc_casefold(text)

    if not value:
        return ""

    decomposed = unicodedata.normalize(
        "NFKD",
        value,
    )

    result = "".join(
        char
        for char in decomposed
        if not unicodedata.combining(char)
    )

    return unicodedata.normalize(
        "NFKC",
        result,
    )


def transliterate(text: Any) -> str:
    """
    Auxiliary Latin-script representation.

    Native/original representation is always preserved.
    """
    value = nfkc_casefold(text)

    if not value:
        return ""

    # ASCII doesn't need transliteration.
    if value.isascii():
        return value

    if (
        not ENABLE_TRANSLITERATION
        or unidecode is None
    ):
        return value

    result = unidecode(value)

    return nfkc_casefold(result)


def tokenize(text: Any) -> list[str]:
    """
    Unicode-safe tokenization.

    Example:
        "ABC-Pvt. Ltd."
        ->
        ["abc", "pvt", "ltd"]
    """
    value = nfkc_casefold(text)

    if not value:
        return []

    # Treat underscore as separator.
    value = value.replace(
        "_",
        " ",
    )

    # Keep Unicode letters/digits.
    value = re.sub(
        r"[^\w]+",
        " ",
        value,
        flags=re.UNICODE,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    ).strip()

    if not value:
        return []

    return value.split()


def tokens_to_text(
    tokens: list[str],
) -> str:
    """Convert tokens into a space-separated string."""
    return " ".join(tokens)


def compact_text(text: Any) -> str:
    """
    Remove separators while preserving Unicode letters/digits.

    Example:
        "ABC Pvt. Ltd."
        ->
        "abcpvtltd"
    """
    value = nfkc_casefold(text)

    if not value:
        return ""

    value = value.replace(
        "_",
        "",
    )

    return re.sub(
        r"[^\w]+",
        "",
        value,
        flags=re.UNICODE,
    )


# ============================================================================
# LEGAL SUFFIX PREPARATION
# ============================================================================

def _prepare_suffixes() -> list[tuple[str, ...]]:
    """
    Prepare legal suffix token sequences.

    This function is called only after nfkc_casefold()
    and tokenize() are defined.
    """
    prepared: list[tuple[str, ...]] = []

    for suffix in LEGAL_SUFFIXES:

        normalized = nfkc_casefold(
            suffix
        )

        suffix_tokens = tuple(
            tokenize(normalized)
        )

        if suffix_tokens:
            prepared.append(
                suffix_tokens
            )

    # Longest suffix first.
    prepared.sort(
        key=len,
        reverse=True,
    )

    return prepared


# IMPORTANT:
# This initialization happens AFTER nfkc_casefold() and tokenize().
PREPARED_SUFFIXES = _prepare_suffixes()


# ============================================================================
# NAME UTILITIES
# ============================================================================

def sorted_unique_tokens(
    tokens: list[str],
) -> list[str]:
    """Sorted unique token representation."""
    return sorted(set(tokens))


def remove_trailing_legal_suffix(
    tokens: list[str],
) -> tuple[list[str], list[str]]:
    """
    Remove only trailing corporate/legal suffix sequences.

    Example:
        ["abc", "private", "limited"]

        ->
        core:
            ["abc"]

        suffix:
            ["private limited"]
    """
    remaining = list(tokens)

    removed_suffixes: list[str] = []

    changed = True

    while changed and remaining:

        changed = False

        for suffix_tokens in PREPARED_SUFFIXES:

            n = len(suffix_tokens)

            if n > len(remaining):
                continue

            if tuple(
                remaining[-n:]
            ) == suffix_tokens:

                suffix_text = " ".join(
                    suffix_tokens
                )

                removed_suffixes.insert(
                    0,
                    suffix_text,
                )

                remaining = remaining[:-n]

                changed = True

                break

    return (
        remaining,
        removed_suffixes,
    )


def phonetic_key(
    text: Any,
) -> str:
    """
    Auxiliary Double Metaphone representation.

    Disabled by default.
    """
    if not ENABLE_PHONETIC:
        return ""

    if doublemetaphone is None:
        return ""

    value = transliterate(text)

    if not value:
        return ""

    primary, secondary = (
        doublemetaphone(value)
    )

    keys = [
        key
        for key in (
            primary,
            secondary,
        )
        if key
    ]

    return "|".join(keys)


def char_ngrams(
    text: Any,
    n_min: int = 3,
    n_max: int = 4,
) -> list[str]:
    """
    Generate character n-grams.

    These are generated on demand for indexes.py.
    We intentionally do not store them in every dataframe row.
    """
    value = compact_text(text)

    if not value:
        return []

    grams: set[str] = set()

    padded = "^" + value + "$"

    for n in range(
        n_min,
        n_max + 1,
    ):

        if len(padded) < n:
            continue

        for i in range(
            len(padded) - n + 1
        ):
            grams.add(
                padded[i:i + n]
            )

    return sorted(grams)


# ============================================================================
# ADDRESS UTILITIES
# ============================================================================

def extract_address_numbers(
    text: Any,
    postal_candidates: list[str] | None = None,
) -> list[str]:
    """
    Extract address house/unit/plot numbers.

    Confirmed postal candidates are excluded.

    Examples:
        "126-B New Line Road"
        -> ["126-b"]

        "123/4 Main Street"
        -> ["123/4"]
    """
    value = nfkc_casefold(text)

    if not value:
        return []

    matches = re.findall(
        r"\b\d+[a-z]?(?:[-/]\d+[a-z]?)?\b",
        value,
        flags=re.IGNORECASE,
    )

    postal_set = set(
        postal_candidates or []
    )

    return sorted(
        {
            match
            for match in matches
            if match not in postal_set
        }
    )


def _is_probable_postal_position(
    value: str,
    candidate: str,
    country_key: str,
) -> bool:
    """
    Conservative context check to reduce the chance of treating a
    house/building number as a postal code.
    """
    tokens = tokenize(value)

    if not tokens:
        return False

    # Find token positions containing the candidate.
    candidate_positions = [
        i
        for i, token in enumerate(tokens)
        if token == candidate
    ]

    if not candidate_positions:
        return False

    pos = candidate_positions[0]

    # ---------------------------------------------------------------
    # US
    # ---------------------------------------------------------------
    if country_key == "us":

        # Strong signal:
        # state code immediately before ZIP.
        if pos > 0:
            previous = tokens[pos - 1]

            if previous in US_STATE_CODES_LOWER:
                return True

        # Another useful signal:
        # ZIP appears near the end of the address.
        if pos >= len(tokens) - 2:
            return True

        return False

    # ---------------------------------------------------------------
    # India
    # ---------------------------------------------------------------
    if country_key == "india":

        # PIN codes are commonly near the end.
        if pos >= len(tokens) - 3:
            return True

        return False

    # ---------------------------------------------------------------
    # France
    # ---------------------------------------------------------------
    if country_key == "france":

        # French postal codes generally occur after street details
        # and before/around locality information.
        #
        # A 5-digit value at the very end is also plausible.
        if pos >= 1:
            if pos >= len(tokens) - 3:
                return True

        return False

    # ---------------------------------------------------------------
    # Unknown country
    # ---------------------------------------------------------------

    # Conservative fallback:
    # only accept a 5+ digit candidate near the end.
    if candidate.isdigit():
        if len(candidate) >= 5:
            return pos >= len(tokens) - 2

    return False


def extract_postal_candidates(
    address: Any,
    country: Any = "",
) -> list[str]:
    """
    Extract high-confidence postal/PIN candidates.

    Important:
        We intentionally do NOT classify every 4/5/6 digit number
        as a postal code.

    This prevents common US house numbers such as 17560 or 5559
    from automatically becoming postal features.
    """
    value = nfkc_casefold(address)

    if not value:
        return []

    country_key = nfkc_casefold(
        country
    )

    matches: set[str] = set()

    # ---------------------------------------------------------------
    # US
    # ---------------------------------------------------------------

    if country_key == "us":

        candidates = set(
            US_ZIP_RE.findall(value)
        )

        for candidate in candidates:

            if _is_probable_postal_position(
                value,
                candidate,
                country_key,
            ):
                matches.add(candidate)

    # ---------------------------------------------------------------
    # India
    # ---------------------------------------------------------------

    elif country_key == "india":

        candidates = set(
            INDIA_PIN_RE.findall(value)
        )

        for candidate in candidates:

            if _is_probable_postal_position(
                value,
                candidate,
                country_key,
            ):
                matches.add(candidate)

    # ---------------------------------------------------------------
    # France
    # ---------------------------------------------------------------

    elif country_key == "france":

        candidates = set(
            FRANCE_POSTAL_RE.findall(value)
        )

        for candidate in candidates:

            if _is_probable_postal_position(
                value,
                candidate,
                country_key,
            ):
                matches.add(candidate)

    # ---------------------------------------------------------------
    # Unknown country
    # ---------------------------------------------------------------

    else:

        candidates = re.findall(
            r"\b\d{5,10}\b",
            value,
        )

        for candidate in candidates:

            if _is_probable_postal_position(
                value,
                candidate,
                country_key,
            ):
                matches.add(candidate)

    return sorted(matches)


def informative_address_tokens(
    tokens: list[str],
) -> list[str]:
    """
    Auxiliary address representation.

    Generic road/building terms are excluded.

    Full address token representation is preserved separately.
    """
    return [
        token
        for token in tokens
        if token not in GENERIC_ADDRESS_TOKENS
    ]


# ============================================================================
# NAME NORMALIZATION BUNDLE
# ============================================================================

def normalize_name(
    value: Any,
) -> dict[str, Any]:
    """
    Produce all evidence-preserving name views.
    """
    raw = safe_text(value)

    native = nfkc_casefold(raw)

    accent = accent_free(
        native
    )

    latin = transliterate(
        native
    )

    tokens = tokenize(
        native
    )

    core_tokens, suffixes = (
        remove_trailing_legal_suffix(
            tokens
        )
    )

    return {
        "name_raw": raw,

        "name_nfkc": native,

        "name_accent_free": accent,

        "name_latin_aux": latin,

        "name_tokens": tokens_to_text(
            tokens
        ),

        "name_tokens_sorted": (
            tokens_to_text(
                sorted_unique_tokens(tokens)
            )
        ),

        "name_compact": compact_text(
            native
        ),

        "name_alnum": compact_text(
            native
        ),

        "name_core": tokens_to_text(
            core_tokens
        ),

        "name_core_compact": compact_text(
            tokens_to_text(core_tokens)
        ),

        "name_legal_suffixes": "|".join(
            suffixes
        ),

        "name_phonetic": phonetic_key(
            latin
        ),
    }


# ============================================================================
# ADDRESS NORMALIZATION BUNDLE
# ============================================================================

def normalize_address(
    value: Any,
    country: Any = "",
) -> dict[str, Any]:
    """
    Produce all evidence-preserving address views.
    """
    raw = safe_text(value)

    native = nfkc_casefold(
        raw
    )

    latin = transliterate(
        native
    )

    tokens = tokenize(
        native
    )

    informative_tokens = (
        informative_address_tokens(
            tokens
        )
    )

    postal = extract_postal_candidates(
        native,
        country,
    )

    numbers = extract_address_numbers(
        native,
        postal,
    )

    return {
        "address_raw": raw,

        "address_nfkc": native,

        "address_latin_aux": latin,

        "address_tokens": tokens_to_text(
            tokens
        ),

        "address_tokens_sorted": (
            tokens_to_text(
                sorted_unique_tokens(tokens)
            )
        ),

        "address_informative_tokens": (
            tokens_to_text(
                informative_tokens
            )
        ),

        "address_compact": compact_text(
            native
        ),

        "address_numbers": "|".join(
            numbers
        ),

        "postal_candidates": "|".join(
            postal
        ),
    }


# ============================================================================
# COMPLETE RECORD NORMALIZATION
# ============================================================================

def normalize_record(
    business_name: Any,
    business_address: Any,
    country: Any,
) -> dict[str, Any]:
    """
    Normalize one complete business record.
    """
    name_bundle = normalize_name(
        business_name
    )

    address_bundle = normalize_address(
        business_address,
        country,
    )

    country_raw = safe_text(
        country
    )

    country_key = nfkc_casefold(
        country_raw
    )

    return {
        **name_bundle,
        **address_bundle,
        "country_raw": country_raw,
        "country_key": country_key,
    }


# ============================================================================
# FAST VECTORIZED PRIMARY NORMALIZATION
# ============================================================================

def _vectorized_nfkc_casefold(
    series: pd.Series,
) -> pd.Series:
    """
    Fast pandas-based Unicode normalization.
    """
    series = (
        series
        .fillna("")
        .astype("string")
    )

    return (
        series
        .str.normalize("NFKC")
        .str.casefold()
        .str.replace(
            r"\s+",
            " ",
            regex=True,
        )
        .str.strip()
    )


# ============================================================================
# CHUNK-LEVEL NORMALIZATION
# ============================================================================

def normalize_dataframe_chunk(
    df: pd.DataFrame,
    include_phonetic: bool | None = None,
) -> pd.DataFrame:
    """
    Normalize one dataframe chunk.

    Intended for multi-million-row processing.

    Example:

        for chunk in pd.read_csv(
            path,
            sep="\\t",
            dtype=str,
            chunksize=200_000,
        ):
            normalized = normalize_dataframe_chunk(
                chunk
            )
    """
    global ENABLE_PHONETIC

    old_phonetic = ENABLE_PHONETIC

    if include_phonetic is not None:
        ENABLE_PHONETIC = (
            include_phonetic
        )

    try:

        result = df.copy()

        # ================================================================
        # COUNTRY
        # ================================================================

        if "country" in result.columns:

            result["country_raw"] = (
                result["country"]
                .fillna("")
                .astype("string")
                .str.strip()
            )

            result["country_key"] = (
                result["country_raw"]
                .str.normalize("NFKC")
                .str.casefold()
            )

        else:

            result["country_raw"] = ""
            result["country_key"] = ""

        # ================================================================
        # NAME
        # ================================================================

        names = (
            result["business_name"]
            .fillna("")
            .astype("string")
        )

        name_nfkc = (
            _vectorized_nfkc_casefold(
                names
            )
        )

        result["name_raw"] = names

        result["name_nfkc"] = (
            name_nfkc
        )

        # Compact form.
        result["name_compact"] = (
            name_nfkc
            .str.replace(
                "_",
                "",
                regex=False,
            )
            .str.replace(
                r"[^\w]+",
                "",
                regex=True,
            )
        )

        # Accent-free view.
        result["name_accent_free"] = (
            name_nfkc.map(
                accent_free
            )
        )

        # Transliteration view.
        result["name_latin_aux"] = (
            name_nfkc.map(
                transliterate
            )
        )

        # ================================================================
        # NAME TOKENS
        # ================================================================

        name_token_lists = (
            name_nfkc.map(tokenize)
        )

        result["name_tokens"] = (
            name_token_lists.map(
                tokens_to_text
            )
        )

        result["name_tokens_sorted"] = (
            name_token_lists.map(
                lambda tokens:
                tokens_to_text(
                    sorted_unique_tokens(
                        tokens
                    )
                )
            )
        )

        # ================================================================
        # CORE NAME
        # ================================================================

        core_results = (
            name_token_lists.map(
                remove_trailing_legal_suffix
            )
        )

        result["name_core"] = (
            core_results.map(
                lambda x:
                tokens_to_text(x[0])
            )
        )

        result["name_core_compact"] = (
            result["name_core"].map(
                compact_text
            )
        )

        result["name_legal_suffixes"] = (
            core_results.map(
                lambda x:
                "|".join(x[1])
            )
        )

        # ================================================================
        # PHONETIC
        # ================================================================

        if ENABLE_PHONETIC:

            result["name_phonetic"] = (
                result["name_latin_aux"]
                .map(
                    phonetic_key
                )
            )

        else:

            result["name_phonetic"] = ""

        # ================================================================
        # ADDRESS
        # ================================================================

        addresses = (
            result["business_address"]
            .fillna("")
            .astype("string")
        )

        address_nfkc = (
            _vectorized_nfkc_casefold(
                addresses
            )
        )

        result["address_raw"] = (
            addresses
        )

        result["address_nfkc"] = (
            address_nfkc
        )

        result["address_compact"] = (
            address_nfkc
            .str.replace(
                "_",
                "",
                regex=False,
            )
            .str.replace(
                r"[^\w]+",
                "",
                regex=True,
            )
        )

        result["address_latin_aux"] = (
            address_nfkc.map(
                transliterate
            )
        )

        # ================================================================
        # ADDRESS TOKENS
        # ================================================================

        address_token_lists = (
            address_nfkc.map(tokenize)
        )

        result["address_tokens"] = (
            address_token_lists.map(
                tokens_to_text
            )
        )

        result["address_tokens_sorted"] = (
            address_token_lists.map(
                lambda tokens:
                tokens_to_text(
                    sorted_unique_tokens(
                        tokens
                    )
                )
            )
        )

        result["address_informative_tokens"] = (
            address_token_lists.map(
                lambda tokens:
                tokens_to_text(
                    informative_address_tokens(
                        tokens
                    )
                )
            )
        )

        # ================================================================
        # POSTAL CANDIDATES
        # ================================================================

        result["postal_candidates"] = [
            "|".join(
                extract_postal_candidates(
                    address,
                    country,
                )
            )
            for address, country in zip(
                address_nfkc,
                result["country_key"],
            )
        ]

        # ================================================================
        # ADDRESS NUMBERS
        # ================================================================

        result["address_numbers"] = [
            "|".join(
                extract_address_numbers(
                    address,
                    [
                        x
                        for x in postal.split("|")
                        if x
                    ],
                )
            )
            for address, postal in zip(
                address_nfkc,
                result["postal_candidates"],
            )
        ]

        return result

    finally:

        ENABLE_PHONETIC = (
            old_phonetic
        )


# ============================================================================
# DEMO TESTS
# ============================================================================

def demo() -> None:
    """
    Small local normalization tests.

    This does NOT process the full dataset.
    """

    examples = [
        {
            "name": "ABC Pvt. Ltd.",
            "address": (
                "126-B Main Street, "
                "Delhi 110001"
            ),
            "country": "India",
        },
        {
            "name": "ABC Private Limited",
            "address": (
                "126 B Main St, "
                "Delhi 110001"
            ),
            "country": "India",
        },
        {
            "name": "Ptit Àmicale",
            "address": (
                "175 Boulevard du "
                "Président Franklin Roosevelt "
                "33000 Bordeaux"
            ),
            "country": "France",
        },
        {
            "name": "नागपुर मार्केटिंग",
            "address": (
                "123 मुख्य मार्ग "
                "नागपुर 440001"
            ),
            "country": "India",
        },
        {
            "name": "Prime Money",
            "address": (
                "17560 Ellis Road, "
                "Tahlequah, OK"
            ),
            "country": "US",
        },
        {
            "name": "Custom Wealth Services LLC",
            "address": (
                "OH, Columbus, "
                "5559 Orville Avenue"
            ),
            "country": "US",
        },
    ]

    for item in examples:

        print()
        print("=" * 90)

        print("INPUT")
        print(item)

        output = normalize_record(
            item["name"],
            item["address"],
            item["country"],
        )

        print()
        print("NORMALIZED")

        for key, value in output.items():

            print(
                f"{key:35} : {value}"
            )


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    demo()