"""
Shared normalization utilities for the Amazon ML Business Entity Resolution task.

This module:
- preserves original source columns
- creates normalized business-name/address/country representations
- preserves multilingual scripts
- preserves digits and address numbers
- provides a reusable DataFrame-level normalization function

The functions are designed to work with the chunked TSV loading strategy
used by the project.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]


# Common legal/business suffixes.
#
# These are NOT removed from name_normalized.
# They are only removed from name_core, so we retain both representations.
BUSINESS_SUFFIXES = (
    "private limited",
    "private ltd",
    "pvt limited",
    "pvt ltd",
    "public limited",
    "public ltd",
    "limited liability company",
    "limited liability partnership",
    "llc",
    "llp",
    "limited",
    "ltd",
    "incorporated",
    "inc",
    "corporation",
    "corp",
    "company",
    "co",
    "plc",
    "gmbh",
    "sarl",
    "sas",
    "srl",
    "spa",
    "ag",
    "bv",
    "nv",
)


NULL_MARKERS = {
    "",
    "null",
    "none",
    "nan",
    "na",
    "n/a",
}


def _as_string(value: object) -> str:
    """Convert a value to a safe string without introducing 'nan'."""
    if value is None:
        return ""

    if isinstance(value, float) and pd.isna(value):
        return ""

    return str(value)


def _unicode_normalize(value: str) -> str:
    """
    Apply Unicode compatibility normalization and case folding.

    NFKC helps normalize compatibility variants while preserving the
    underlying script. casefold() provides stronger case-insensitive
    normalization than lower().
    """
    value = unicodedata.normalize("NFKC", value)
    value = value.casefold()
    value = unicodedata.normalize("NFKC", value)
    return value


def _replace_ampersand(value: str) -> str:
    """Normalize the common business-name '&' variant."""
    return value.replace("&", " and ")


def _clean_characters(value: str) -> str:
    """
    Replace punctuation/symbols with spaces while preserving:

    - Unicode letters
    - Unicode combining marks
    - Unicode numbers
    - whitespace

    This deliberately does NOT strip non-ASCII characters.
    """
    output: list[str] = []

    for char in value:
        category = unicodedata.category(char)

        if char.isspace():
            output.append(" ")
        elif category[0] in {"L", "M", "N"}:
            # L = Letter
            # M = Mark
            # N = Number
            output.append(char)
        else:
            # Punctuation and symbols become separators.
            output.append(" ")

    return "".join(output)


def _collapse_whitespace(value: str) -> str:
    """Collapse repeated whitespace and trim the result."""
    return re.sub(r"\s+", " ", value).strip()


def normalize_text(value: object) -> str:
    """
    General conservative text normalization.

    This is the base representation used by names and addresses.
    """
    text = _as_string(value)

    if not text:
        return ""

    text = _unicode_normalize(text)
    text = _replace_ampersand(text)
    text = _clean_characters(text)
    text = _collapse_whitespace(text)

    return text


def normalize_name(value: object) -> str:
    """Normalize a business name while preserving all scripts and digits."""
    return normalize_text(value)


def _strip_trailing_suffixes(name: str) -> str:
    """
    Remove recognized business/legal suffixes from the END of a name.

    This function intentionally does not remove suffixes from the middle
    of names. The original normalized name remains available separately.
    """
    if not name:
        return ""

    tokens = name.split()

    # Longest suffixes first so that "private limited" is checked before
    # "limited".
    suffix_tokens = sorted(
        (suffix.split() for suffix in BUSINESS_SUFFIXES),
        key=len,
        reverse=True,
    )

    changed = True

    while changed and tokens:
        changed = False

        for suffix in suffix_tokens:
            suffix_len = len(suffix)

            if len(tokens) >= suffix_len:
                if tokens[-suffix_len:] == suffix:
                    tokens = tokens[:-suffix_len]
                    changed = True
                    break

    return " ".join(tokens)


def normalize_name_core(value: object) -> str:
    """
    Return the normalized business name with trailing legal/business
    suffixes removed.

    The suffix is NOT removed from name_normalized.
    """
    normalized = normalize_name(value)
    return _strip_trailing_suffixes(normalized)


def _remove_null_tokens(value: str) -> str:
    """
    Remove standalone null-like address tokens.

    Example:
        '12 mg road null bengaluru'
        -> '12 mg road bengaluru'

    Substrings are never removed. For example, 'nullify' is untouched.
    """
    if not value:
        return ""

    tokens = value.split()
    tokens = [token for token in tokens if token not in NULL_MARKERS]

    return " ".join(tokens)


def normalize_address(value: object) -> str:
    """
    Normalize a business address.

    Important:
    - numbers are preserved
    - non-Latin scripts are preserved
    - missing/null components are not invented
    """
    normalized = normalize_text(value)
    normalized = _remove_null_tokens(normalized)
    return _collapse_whitespace(normalized)

def normalize_address_numbers(value: object) -> str:
    """
    Extract numeric/alphanumeric address components and canonicalize
    leading zeros.

    Examples:
        '0337 Oakland Avenue' -> '337'
        '005559 Orville Avenue' -> '5559'
        '0017560 ELLIS ROAD' -> '17560'
        '24A Main Street' -> '24a'
    """
    normalized = normalize_address(value)
    if not normalized:
        return ""

    numbers: list[str] = []

    for token in normalized.split():
        match = re.fullmatch(r"0*(\d+)([a-z])?", token)
        if not match:
            continue

        digits = match.group(1)
        suffix = match.group(2) or ""

        # Convert numeric part to an integer string so leading
        # zeros do not prevent otherwise identical addresses
        # from sharing a blocking key.
        canonical_digits = str(int(digits))

        numbers.append(f"{canonical_digits}{suffix}")

    return " ".join(dict.fromkeys(numbers))


def tokenize_address(value: object) -> str:
    """
    Produce a deterministic token representation of an address.

    Tokens are sorted and deduplicated so this can later support
    token-based blocking without changing the original address.
    """
    normalized = normalize_address(value)

    if not normalized:
        return ""

    tokens = sorted(set(normalized.split()))
    return " ".join(tokens)


def normalize_country(value: object) -> str:
    """
    Normalize country values conservatively.

    We do not hard-code the competition's countries here because the
    challenge explicitly includes an additional country in test data.
    """
    return normalize_text(value)


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add shared normalized columns to a source DataFrame.

    Original columns are left untouched.

    Added columns:
        name_normalized
        name_core
        address_normalized
        address_tokens
        country_normalized
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    result = df.copy()

    result["name_normalized"] = result["business_name"].map(normalize_name)
    result["name_core"] = result["business_name"].map(normalize_name_core)

    result["address_normalized"] = result["business_address"].map(
    normalize_address
    )
    result["address_tokens"] = result["business_address"].map(
    tokenize_address
    )
    result["address_numbers_normalized"] = result["business_address"].map(
    normalize_address_numbers
    )

    result["country_normalized"] = result["country"].map(normalize_country)

    return result


def normalize_chunks(
    chunks: Iterable[pd.DataFrame],
) -> Iterable[pd.DataFrame]:
    """
    Normalize an iterable of DataFrame chunks.

    This allows the existing 100k-row TSV reader to remain chunked.
    """
    for chunk in chunks:
        yield normalize_dataframe(chunk)