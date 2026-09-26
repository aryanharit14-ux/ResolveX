"""
Candidate generation / blocking for Business Entity Resolution.

The blocking stage is recall-oriented:
multiple independent blocking strategies are OR-ed together.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from typing import Dict, List, Set

import pandas as pd


# ============================================================
# Configuration
# ============================================================

REQUIRED_COLUMNS = [
    "entity_id",
    "name_normalized",
    "name_core",
    "address_normalized",
    "country_normalized",
]

OPTIONAL_COLUMNS = [
    "address_numbers_normalized",
]

VALID_TARGET_SOURCES = {"S2", "S3"}

MIN_NAME_TOKEN_LENGTH = 3
MIN_ADDRESS_TOKEN_LENGTH = 4


# ------------------------------------------------------------
# Address frequency limits
# ------------------------------------------------------------

MAX_RARE_ADDRESS_TOKEN_FREQUENCY = 500
MAX_RARE_ADDRESS_NUMBER_FREQUENCY = 50

MAX_FINAL_TOKEN_BLOCK_FREQUENCY = 500
MAX_FINAL_ADDRESS_TOKEN_BLOCK_FREQUENCY = 500

# New: allow useful combinations even when individual
# address tokens are not rare.
MAX_ADDRESS_PAIR_FREQUENCY = 250

# New: number + location combination limit.
MAX_ADDRESS_NUMBER_LOCATION_FREQUENCY = 250
MAX_LEETSPEAK_NAME_FREQUENCY = 150

# ------------------------------------------------------------
# Name frequency limits
# ------------------------------------------------------------

MAX_RARE_NAME_TOKEN_FREQUENCY = 100
MAX_RARE_NAME_NGRAM_FREQUENCY = 100

NAME_NGRAM_SIZE = 3

# New controlled name-prefix rule.
MIN_NAME_PREFIX_LENGTH = 4


# ============================================================
# Generic address tokens
# ============================================================

GENERIC_ADDRESS_TOKENS = {
    "road",
    "rd",
    "street",
    "st",
    "avenue",
    "ave",
    "drive",
    "dr",
    "lane",
    "ln",
    "boulevard",
    "blvd",
    "highway",
    "hwy",
    "way",
    "unit",
    "apartment",
    "apt",
    "floor",
    "fl",
    "block",
    "plot",
    "po",
    "box",
    "north",
    "south",
    "east",
    "west",
    "northeast",
    "northwest",
    "southeast",
    "southwest",
}


# ============================================================
# Validation
# ============================================================

def _validate_columns(df: pd.DataFrame) -> None:
    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required normalized columns: {missing}"
        )


# ============================================================
# Basic utilities
# ============================================================

def _safe_string(value: object) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()


def _tokens(value: object) -> List[str]:
    text = _safe_string(value)

    if not text:
        return []

    return [
        token
        for token in text.split()
        if token
    ]


# ============================================================
# Name utilities
# ============================================================

def _informative_name_tokens(name: object) -> List[str]:
    tokens = _tokens(name)

    informative = [
        token
        for token in tokens
        if len(token) >= MIN_NAME_TOKEN_LENGTH
    ]

    return informative or tokens


def _name_token_key(name: object) -> str:
    tokens = _informative_name_tokens(name)

    if not tokens:
        return ""

    tokens = sorted(
        set(tokens),
        key=lambda token: (-len(token), token),
    )

    return "|".join(sorted(tokens[:2]))


def _name_prefix_key(name: object) -> str:
    """
    Return the first useful name token.

    Examples:
        Williams Silicon
            -> williams

        Global Institute
            -> global

        Sunrise Tech
            -> sunrise
    """

    tokens = _informative_name_tokens(name)

    if not tokens:
        return ""

    for token in tokens:
        token = token.casefold()

        if len(token) >= MIN_NAME_PREFIX_LENGTH:
            return token

    return ""


def _name_subset_key(name: object) -> str:
    """
    Return the first two informative name tokens.

    Examples:

        Sunrise Tech
        Sunrise Tech Limited Service

    both produce:

        sunrise|tech
    """

    tokens = _informative_name_tokens(name)

    if not tokens:
        return ""

    tokens = [
        token.casefold()
        for token in tokens
        if len(token) >= MIN_NAME_TOKEN_LENGTH
    ]

    if not tokens:
        return ""

    return "|".join(tokens[:2])


# ============================================================
# Accent folding
# ============================================================

def _accent_fold(value: object) -> str:
    """
    Create an accent-insensitive representation.
    """

    text = _safe_string(value)

    if not text:
        return ""

    normalized = unicodedata.normalize(
        "NFKD",
        text,
    )

    return "".join(
        char
        for char in normalized
        if not unicodedata.combining(char)
    )


def _accent_fold_name(value: object) -> str:
    """
    Create a compact accent-insensitive name key.

    Spaces/punctuation are removed.
    """

    text = _accent_fold(value)

    if not text:
        return ""

    return "".join(
        char.casefold()
        for char in text
        if char.isalnum()
    )

def _leetspeak_name(value: object) -> str:
    """
    Create a compact leetspeak-normalized name key.

    Common OCR/typing substitutions are normalized.
    """

    text = _accent_fold_name(value)

    if not text:
        return ""

    replacements = str.maketrans({
        "0": "o",
        "1": "i",
        "3": "e",
        "4": "a",
        "5": "s",
        "7": "t",
    })

    return text.translate(replacements)

# ============================================================
# Name character n-grams
# ============================================================

def _name_ngrams(
    name: object,
    n: int = NAME_NGRAM_SIZE,
) -> Set[str]:
    """
    Generate character n-grams from a business name.
    """

    compact = _accent_fold_name(name)

    if not compact:
        return set()

    if len(compact) < n:
        return set()

    return {
        compact[index:index + n]
        for index in range(
            len(compact) - n + 1
        )
    }


# ============================================================
# Address utilities
# ============================================================

def _informative_address_tokens(address: object) -> List[str]:
    tokens = {
        token
        for token in _tokens(address)
        if (
            len(token) >= MIN_ADDRESS_TOKEN_LENGTH
            and token not in GENERIC_ADDRESS_TOKENS
            and not token.isdigit()
        )
    }

    return sorted(
        tokens,
        key=lambda token: (-len(token), token),
    )


def extract_address_numbers(address: object) -> tuple[str, ...]:
    """
    Extract normalized address numbers.

    Supports:

        12       -> 12
        024A     -> 24a
        0337     -> 337
        142/4    -> 142/4
        142/4A   -> 142/4a
    """

    text = _safe_string(address)

    if not text:
        return ()

    values: List[str] = []

    for token in text.casefold().split():

        # Compound address number.
        compound = re.fullmatch(
            r"0*(\d+)/0*(\d+)([a-z])?",
            token,
        )

        if compound:
            first = str(int(compound.group(1)))
            second = str(int(compound.group(2)))
            suffix = compound.group(3) or ""

            values.append(
                f"{first}/{second}{suffix}"
            )

            continue

        # Normal number.
        match = re.fullmatch(
            r"0*(\d+)([a-z])?",
            token,
        )

        if not match:
            continue

        digits = match.group(1)
        suffix = match.group(2) or ""

        canonical_digits = str(int(digits))

        values.append(
            f"{canonical_digits}{suffix}"
        )

    return tuple(dict.fromkeys(values))


def _address_numbers(row: pd.Series) -> tuple[str, ...]:
    value = row.get(
        "address_numbers_normalized",
        "",
    )

    text = _safe_string(value)

    if text:
        return tuple(
            token
            for token in text.split()
            if token
        )

    return extract_address_numbers(
        row.get(
            "address_normalized",
            "",
        )
    )


def _address_base_numbers(row: pd.Series) -> tuple[str, ...]:
    """
    Convert:

        908c  -> 908
        100a  -> 100
        7130c -> 7130

    Compound numbers remain compound.
    """

    numbers = _address_numbers(row)

    bases = []

    for number in numbers:

        compound = re.fullmatch(
            r"(\d+)/(\d+)([a-z])?",
            number.casefold(),
        )

        if compound:
            bases.append(
                f"{compound.group(1)}/{compound.group(2)}"
            )

            # Also retain the first component of a compound
            # address number for more flexible matching.
            bases.append(
                compound.group(1)
            )

            continue

        normal = re.fullmatch(
            r"(\d+)[a-z]?",
            number.casefold(),
        )

        if normal:
            bases.append(
                normal.group(1)
            )

    return tuple(dict.fromkeys(bases))


def _address_pair_key(address: object) -> str:
    tokens = _informative_address_tokens(address)

    if len(tokens) < 2:
        return ""

    selected = sorted(tokens[:2])

    return "|".join(selected)


def _address_token_pairs(
    address: object,
) -> Set[str]:
    """
    Generate several informative address-token pairs.

    Only the strongest six address tokens are considered
    to prevent uncontrolled candidate explosion.
    """

    tokens = _informative_address_tokens(address)

    if len(tokens) < 2:
        return set()

    selected = tokens[:6]

    pairs: Set[str] = set()

    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):

            first = selected[i]
            second = selected[j]

            pairs.add(
                "|".join(
                    sorted(
                        [
                            first,
                            second,
                        ]
                    )
                )
            )

    return pairs


def _address_number_location_pairs(
    row: pd.Series,
) -> Set[str]:
    """
    Generate address-number + address-token combinations.

    Example:

        402, Thane, Maharashtra

    creates combinations such as:

        402|thane
        402|maharashtra
    """

    numbers = _address_base_numbers(row)

    address = _safe_string(
        row.get(
            "address_normalized",
            "",
        )
    )

    tokens = _informative_address_tokens(
        address
    )

    if not numbers or not tokens:
        return set()

    pairs: Set[str] = set()

    for number in numbers:

        for token in tokens:

            pairs.add(
                f"{number}|{token}"
            )

    return pairs


# ============================================================
# Frequency analysis
# ============================================================

def _build_address_token_frequency(
    target_df: pd.DataFrame,
) -> Counter:

    frequency = Counter()

    for address in target_df[
        "address_normalized"
    ]:

        tokens = set(
            _informative_address_tokens(address)
        )

        for token in tokens:
            frequency[token] += 1

    return frequency


def _build_address_number_frequency(
    target_df: pd.DataFrame,
) -> Counter:

    frequency = Counter()

    for row in target_df.itertuples(
        index=False
    ):

        row_series = pd.Series(
            row._asdict()
        )

        numbers = _address_base_numbers(
            row_series
        )

        for number in set(numbers):
            frequency[number] += 1

    return frequency


def _build_address_pair_frequency(
    target_df: pd.DataFrame,
) -> Counter:

    frequency = Counter()

    for address in target_df[
        "address_normalized"
    ]:

        pairs = _address_token_pairs(
            address
        )

        for pair in pairs:
            frequency[pair] += 1

    return frequency


def _build_address_number_location_frequency(
    target_df: pd.DataFrame,
) -> Counter:

    frequency = Counter()

    for row in target_df.itertuples(
        index=False
    ):

        row_series = pd.Series(
            row._asdict()
        )

        pairs = _address_number_location_pairs(
            row_series
        )

        for pair in pairs:
            frequency[pair] += 1

    return frequency


# ============================================================
# Name frequency
# ============================================================

def _build_name_token_frequency(
    target_df: pd.DataFrame,
) -> Counter:

    frequency = Counter()

    for name in target_df[
        "name_core"
    ]:

        tokens = set(
            _informative_name_tokens(name)
        )

        for token in tokens:
            frequency[
                token.casefold()
            ] += 1

    return frequency


def _build_name_ngram_frequency(
    target_df: pd.DataFrame,
) -> Counter:

    frequency = Counter()

    for name in target_df[
        "name_core"
    ]:

        ngrams = _name_ngrams(name)

        for ngram in ngrams:
            frequency[ngram] += 1

    return frequency


def _build_leetspeak_name_frequency(
    target_df: pd.DataFrame,
) -> Counter:
    frequency = Counter()

    for name in target_df[
        "name_core"
    ]:

        leet_name = _leetspeak_name(name)

        if leet_name:
            frequency[leet_name] += 1

    return frequency


# ============================================================
# Rare address helpers
# ============================================================

def _rare_address_tokens(
    address: object,
    address_token_frequency: Counter,
) -> List[str]:

    tokens = _informative_address_tokens(
        address
    )

    return [
        token
        for token in tokens
        if address_token_frequency.get(
            token,
            0,
        )
        <= MAX_RARE_ADDRESS_TOKEN_FREQUENCY
    ]


def _rare_address_numbers(
    row: pd.Series,
    address_number_frequency: Counter,
) -> List[str]:

    numbers = _address_base_numbers(row)

    return [
        number
        for number in numbers
        if address_number_frequency.get(
            number,
            0,
        )
        <= MAX_RARE_ADDRESS_NUMBER_FREQUENCY
    ]


def _rare_address_token_pairs(
    address: object,
    address_pair_frequency: Counter,
) -> List[str]:

    pairs = _address_token_pairs(
        address
    )

    return [
        pair
        for pair in pairs
        if address_pair_frequency.get(
            pair,
            0,
        )
        <= MAX_ADDRESS_PAIR_FREQUENCY
    ]


def _rare_address_number_location_pairs(
    row: pd.Series,
    frequency: Counter,
) -> List[str]:

    pairs = _address_number_location_pairs(
        row
    )

    return [
        pair
        for pair in pairs
        if frequency.get(
            pair,
            0,
        )
        <= MAX_ADDRESS_NUMBER_LOCATION_FREQUENCY
    ]


# ============================================================
# Rare name helpers
# ============================================================

def _rare_name_tokens(
    name: object,
    name_token_frequency: Counter,
) -> List[str]:

    tokens = _informative_name_tokens(
        name
    )

    result = []

    for token in set(tokens):

        normalized_token = token.casefold()

        if (
            len(normalized_token)
            >= MIN_NAME_TOKEN_LENGTH
            and name_token_frequency.get(
                normalized_token,
                0,
            )
            <= MAX_RARE_NAME_TOKEN_FREQUENCY
        ):
            result.append(
                normalized_token
            )

    return result


def _rare_name_ngrams(
    name: object,
    name_ngram_frequency: Counter,
) -> List[str]:

    ngrams = _name_ngrams(name)

    return [
        ngram
        for ngram in ngrams
        if name_ngram_frequency.get(
            ngram,
            0,
        )
        <= MAX_RARE_NAME_NGRAM_FREQUENCY
    ]


def _final_recall_keys(
    name_core: str,
    name: str,
    address_normalized: str,
) -> set[str]:
    """
    Final high-recall blocking keys.

    These keys are intentionally broad and are only used with
    frequency limits in generate_block_keys().
    """
    keys: set[str] = set()

    # ---------------------------------------------------------
    # 1. Individual meaningful name tokens
    #    Helps:
    #    OV Foods <-> OV Private Foods
    #    NS Consulting <-> NS Corp Consulting
    #    QR Brothers <-> QR Private Limited Services
    # ---------------------------------------------------------
    name_tokens = {
        token
        for token in re.findall(r"[^\W\d_]+", name_core or name, flags=re.UNICODE)
        if len(token) >= MIN_NAME_TOKEN_LENGTH
    }

    for token in name_tokens:
        keys.add(f"final_name_token:{token}")

    # ---------------------------------------------------------
    # 2. Character trigrams from the complete normalized name
    #    Helps small spelling/transposition errors.
    # ---------------------------------------------------------
    compact_name = re.sub(
        r"[^\w]",
        "",
        _accent_fold_name(name_core or name),
        flags=re.UNICODE,
    )

    if len(compact_name) >= 4:
        for i in range(len(compact_name) - 2):
            keys.add(
                f"final_name_tri:{compact_name[i:i + 3]}"
            )

    # ---------------------------------------------------------
    # 3. Address tokens
    #    Helps multilingual names where address is the bridge.
    # ---------------------------------------------------------
    address_tokens = {
        token
        for token in re.findall(
            r"[^\W\d_]+",
            address_normalized or "",
            flags=re.UNICODE,
        )
        if len(token) >= MIN_ADDRESS_TOKEN_LENGTH
    }

    for token in address_tokens:
        keys.add(f"final_address_token:{token}")

    # ---------------------------------------------------------
    # 4. Numeric address fragments
    #    Handles:
    #       408 <-> 08
    #       1410 <-> 14-10
    #       207 2nd <-> 207 2th
    # ---------------------------------------------------------
    numbers = re.findall(
        r"\d+[a-zA-Z]?",
        address_normalized or "",
    )

    for number in numbers:
        number = number.lower()

        keys.add(f"final_address_number:{number}")

        digits = re.sub(r"[^0-9]", "", number)

        if len(digits) >= 2:
            keys.add(
                f"final_address_number_suffix:{digits[-2:]}"
            )

        if len(digits) >= 3:
            keys.add(
                f"final_address_number_suffix:{digits[-3:]}"
            )

    return keys

# ============================================================
# Blocking key generation
# ============================================================

def generate_block_keys(
    row: pd.Series,
    address_token_frequency: Counter | None = None,
    address_number_frequency: Counter | None = None,
    name_token_frequency: Counter | None = None,
    name_ngram_frequency: Counter | None = None,
    address_pair_frequency: Counter | None = None,
    address_number_location_frequency: Counter | None = None,
    leetspeak_name_frequency: Counter | None = None,
) -> Set[str]:

    country = _safe_string(
        row.get(
            "country_normalized",
            "",
        )
    )

    name = _safe_string(
        row.get(
            "name_normalized",
            "",
        )
    )

    name_core = _safe_string(
        row.get(
            "name_core",
            "",
        )
    )

    address = _safe_string(
        row.get(
            "address_normalized",
            "",
        )
    )

    keys: Set[str] = set()

    # ========================================================
    # NAME BLOCKING
    # ========================================================

    # --------------------------------------------------------
    # 1. Exact normalized name
    # --------------------------------------------------------

    if name:

        if country:
            keys.add(
                f"country_name::{country}::{name}"
            )

        keys.add(
            f"name::{name}"
        )

    # --------------------------------------------------------
    # 2. Name core
    # --------------------------------------------------------

    if name_core:

        if country:
            keys.add(
                f"country_core::{country}::{name_core}"
            )

        keys.add(
            f"core::{name_core}"
        )

    # --------------------------------------------------------
    # 3. Name token pair
    # --------------------------------------------------------

    token_key = _name_token_key(
        name_core or name
    )

    if country and token_key:

        keys.add(
            f"country_name_tokens::{country}::{token_key}"
        )

    # --------------------------------------------------------
    # 3b. Name prefix
    # --------------------------------------------------------

    name_prefix = _name_prefix_key(
        name_core or name
    )

    if country and name_prefix:

        keys.add(
            f"country_name_prefix::{country}::{name_prefix}"
        )

    # --------------------------------------------------------
    # 3c. Name subset / first two tokens
    # --------------------------------------------------------

    name_subset = _name_subset_key(
        name_core or name
    )

    if country and name_subset:

        keys.add(
            f"country_name_subset::{country}::{name_subset}"
        )

    # --------------------------------------------------------
    # 4. Accent-insensitive name
    # --------------------------------------------------------

    accent_name = _accent_fold_name(
        name_core or name
    )

    if country and accent_name:

        keys.add(
            f"country_accent_name::{country}::{accent_name}"
        )

    # --------------------------------------------------------
    # 4b. Controlled leetspeak-normalized name
    # --------------------------------------------------------

    if (
        country
        and leetspeak_name_frequency is not None
    ):

        leet_name = _leetspeak_name(
            name_core or name
        )

        if (
            leet_name
            and leetspeak_name_frequency.get(
                leet_name,
                0,
            ) <= MAX_LEETSPEAK_NAME_FREQUENCY
        ):

            keys.add(
                "country_leetspeak_name::"
                f"{country}::{leet_name}"
            )

    # --------------------------------------------------------
    # 5. Rare individual name tokens
    # --------------------------------------------------------

    if (
        country
        and name_token_frequency is not None
    ):

        rare_tokens = _rare_name_tokens(
            name_core or name,
            name_token_frequency,
        )

        for token in rare_tokens:

            keys.add(
                "country_rare_name_token::"
                f"{country}::{token}"
            )

    # --------------------------------------------------------
    # 6. Rare character n-grams
    # --------------------------------------------------------

    if (
        country
        and name_ngram_frequency is not None
    ):

        rare_ngrams = _rare_name_ngrams(
            name_core or name,
            name_ngram_frequency,
        )

        for ngram in rare_ngrams:

            keys.add(
                "country_name_ngram::"
                f"{country}::{ngram}"
            )

    # ========================================================
    # ADDRESS BLOCKING
    # ========================================================

    address_numbers = _address_numbers(
        row
    )

    address_base_numbers = _address_base_numbers(
        row
    )

    informative_tokens = (
        _informative_address_tokens(
            address
        )
    )

    # --------------------------------------------------------
    # 7. Address number + token
    # --------------------------------------------------------

    if country:

        for number in address_numbers:

            for token in informative_tokens[:3]:

                keys.add(
                    "country_address_number_token::"
                    f"{country}::{number}::{token}"
                )

        # ----------------------------------------------------
        # Base number + token
        # ----------------------------------------------------

        for base_number in address_base_numbers:

            for token in informative_tokens[:3]:

                keys.add(
                    "country_address_base_number_token::"
                    f"{country}::{base_number}::{token}"
                )

    # --------------------------------------------------------
    # 8. Original address token pair
    # --------------------------------------------------------

    address_pair = _address_pair_key(
        address
    )

    if country and address_pair:

        keys.add(
            "country_address_pair::"
            f"{country}::{address_pair}"
        )

    # --------------------------------------------------------
    # 8b. Multiple address-token pairs
    # --------------------------------------------------------

    if (
        country
        and address_pair_frequency is not None
    ):

        rare_pairs = _rare_address_token_pairs(
            address,
            address_pair_frequency,
        )

        for pair in rare_pairs:

            keys.add(
                "country_address_token_pair::"
                f"{country}::{pair}"
            )

    # --------------------------------------------------------
    # 9. Rare address token
    # --------------------------------------------------------

    if (
        country
        and address_token_frequency is not None
    ):

        for token in _rare_address_tokens(
            address,
            address_token_frequency,
        ):

            keys.add(
                "country_rare_address_token::"
                f"{country}::{token}"
            )

    # --------------------------------------------------------
    # 10. Rare address number
    # --------------------------------------------------------

    if (
        country
        and address_number_frequency is not None
    ):

        rare_numbers = _rare_address_numbers(
            row,
            address_number_frequency,
        )

        for number in rare_numbers:

            keys.add(
                "country_rare_address_number::"
                f"{country}::{number}"
            )

            # Stronger number + address-token strategy.
            for token in informative_tokens[:3]:

                keys.add(
                    "country_rare_address_number_token::"
                    f"{country}::{number}::{token}"
                )

    # --------------------------------------------------------
    # 11. Rare address number + location
    # --------------------------------------------------------

    if (
        country
        and address_number_location_frequency
        is not None
    ):

        rare_pairs = (
            _rare_address_number_location_pairs(
                row,
                address_number_location_frequency,
            )
        )

        for pair in rare_pairs:

            keys.add(
                "country_address_number_location::"
                f"{country}::{pair}"
            )

    return keys


# ============================================================
# Block index
# ============================================================

def build_block_index(
    source_df: pd.DataFrame,
    address_token_frequency: Counter | None = None,
    address_number_frequency: Counter | None = None,
    name_token_frequency: Counter | None = None,
    name_ngram_frequency: Counter | None = None,
    address_pair_frequency: Counter | None = None,
    address_number_location_frequency: Counter | None = None,
    leetspeak_name_frequency: Counter | None = None,
) -> Dict[str, List[str]]:

    _validate_columns(source_df)

    # --------------------------------------------------------
    # Frequency statistics
    # --------------------------------------------------------

    if address_token_frequency is None:

        address_token_frequency = (
            _build_address_token_frequency(
                source_df
            )
        )

    if address_number_frequency is None:

        address_number_frequency = (
            _build_address_number_frequency(
                source_df
            )
        )

    if name_token_frequency is None:

        name_token_frequency = (
            _build_name_token_frequency(
                source_df
            )
        )

    if name_ngram_frequency is None:

        name_ngram_frequency = (
            _build_name_ngram_frequency(
                source_df
            )
        )

    if address_pair_frequency is None:

        address_pair_frequency = (
            _build_address_pair_frequency(
                source_df
            )
        )

        if (address_number_location_frequency is None):
            address_number_location_frequency = (
                _build_address_number_location_frequency(
                    source_df
                    )
                    )

    if leetspeak_name_frequency is None:

        leetspeak_name_frequency = (
            _build_leetspeak_name_frequency(
                source_df
            )
        )

    # --------------------------------------------------------
    # Build index
    # --------------------------------------------------------

    index: Dict[str, List[str]] = defaultdict(list)

    for row in source_df.itertuples(
        index=False
    ):

        row_dict = row._asdict()

        entity_id = str(
            row_dict["entity_id"]
        )

        row_series = pd.Series(
            row_dict
        )

        keys = generate_block_keys(
            row_series,
            address_token_frequency=(
                address_token_frequency
            ),
            address_number_frequency=(
                address_number_frequency
            ),
            name_token_frequency=(
                name_token_frequency
            ),
            name_ngram_frequency=(
                name_ngram_frequency
            ),
            address_pair_frequency=(
                address_pair_frequency
            ),
            address_number_location_frequency=(
                address_number_location_frequency
                ),
                leetspeak_name_frequency=(
                    leetspeak_name_frequency
                    ),
                    )

        for key in keys:

            index[key].append(
                entity_id
            )

    return dict(index)


# ============================================================
# Candidate generation
# ============================================================

def generate_candidate_pairs(
    source1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    target_source: str,
) -> pd.DataFrame:

    _validate_columns(source1_df)
    _validate_columns(target_df)

    if target_source not in VALID_TARGET_SOURCES:
        raise ValueError(
            "target_source must be 'S2' or 'S3'"
        )

    # --------------------------------------------------------
    # Target-source-specific frequencies
    # --------------------------------------------------------

    address_token_frequency = (
        _build_address_token_frequency(
            target_df
        )
    )

    address_number_frequency = (
        _build_address_number_frequency(
            target_df
        )
    )

    name_token_frequency = (
        _build_name_token_frequency(
            target_df
        )
    )

    name_ngram_frequency = (
        _build_name_ngram_frequency(
            target_df
        )
    )

    address_pair_frequency = (
        _build_address_pair_frequency(
            target_df
        )
    )

    address_number_location_frequency = (
        _build_address_number_location_frequency(
            target_df
        )
    )

    leetspeak_name_frequency = (
        _build_leetspeak_name_frequency(
            target_df
            )
            )

    # --------------------------------------------------------
    # Build target block index
    # --------------------------------------------------------

    index = build_block_index(
        target_df,
        address_token_frequency=(
            address_token_frequency
        ),
        address_number_frequency=(
            address_number_frequency
        ),
        name_token_frequency=(
            name_token_frequency
        ),
        name_ngram_frequency=(
            name_ngram_frequency
        ),
        address_pair_frequency=(
            address_pair_frequency
        ),
        address_number_location_frequency=(
            address_number_location_frequency
        ),
        leetspeak_name_frequency=(
            leetspeak_name_frequency
        ),
    )

    candidates: Set[
        tuple[str, str, str]
    ] = set()

    # --------------------------------------------------------
    # Generate candidates from Source 1
    # --------------------------------------------------------

    for row in source1_df.itertuples(
        index=False
    ):

        row_dict = row._asdict()

        s1_id = str(
            row_dict["entity_id"]
        )

        row_series = pd.Series(
            row_dict
        )

        keys = generate_block_keys(
            row_series,
            address_token_frequency=(
                address_token_frequency
            ),
            address_number_frequency=(
                address_number_frequency
            ),
            name_token_frequency=(
                name_token_frequency
            ),
            name_ngram_frequency=(
                name_ngram_frequency
            ),
            address_pair_frequency=(
                address_pair_frequency
            ),
            address_number_location_frequency=(
                address_number_location_frequency
            ),
            leetspeak_name_frequency=(
                leetspeak_name_frequency
            ),
        )

        for key in keys:

            for candidate_id in index.get(
                key,
                [],
            ):

                candidates.add(
                    (
                        s1_id,
                        candidate_id,
                        target_source,
                    )
                )

    result = pd.DataFrame(
        list(candidates),
        columns=[
            "source1_entity_id",
            "candidate_entity_id",
            "source",
        ],
    )

    if result.empty:

        return pd.DataFrame(
            columns=[
                "source1_entity_id",
                "candidate_entity_id",
                "source",
            ]
        )

    return (
        result
        .sort_values(
            [
                "source1_entity_id",
                "source",
                "candidate_entity_id",
            ]
        )
        .reset_index(drop=True)
    )


# ============================================================
# Both target sources
# ============================================================

def generate_candidates_for_both_sources(
    source1_df: pd.DataFrame,
    source2_df: pd.DataFrame,
    source3_df: pd.DataFrame,
) -> pd.DataFrame:

    candidates_s2 = generate_candidate_pairs(
        source1_df=source1_df,
        target_df=source2_df,
        target_source="S2",
    )

    candidates_s3 = generate_candidate_pairs(
        source1_df=source1_df,
        target_df=source3_df,
        target_source="S3",
    )

    result = pd.concat(
        [
            candidates_s2,
            candidates_s3,
        ],
        ignore_index=True,
    )

    if result.empty:

        return pd.DataFrame(
            columns=[
                "source1_entity_id",
                "candidate_entity_id",
                "source",
            ]
        )

    return (
        result
        .drop_duplicates(
            subset=[
                "source1_entity_id",
                "candidate_entity_id",
                "source",
            ]
        )
        .sort_values(
            [
                "source1_entity_id",
                "source",
                "candidate_entity_id",
            ]
        )
        .reset_index(drop=True)
    )


# ============================================================
# Submission-format conversion
# ============================================================

def create_candidate_pairs_output(
    candidate_pairs: pd.DataFrame,
    source1_df: pd.DataFrame | None = None,
) -> pd.DataFrame:

    if candidate_pairs.empty:

        grouped = pd.DataFrame(
            columns=[
                "source1_entity_id",
                "candidate_entity_ids",
            ]
        )

    else:

        grouped = (
            candidate_pairs
            .groupby(
                "source1_entity_id",
                sort=True,
            )["candidate_entity_id"]
            .apply(
                lambda values: ",".join(
                    sorted(
                        set(
                            str(value)
                            for value in values
                        )
                    )
                )
            )
            .reset_index()
        )

        grouped = grouped.rename(
            columns={
                "candidate_entity_id":
                    "candidate_entity_ids"
            }
        )

    if source1_df is not None:

        required = pd.DataFrame(
            {
                "source1_entity_id":
                    source1_df["entity_id"]
                    .astype(str)
                    .drop_duplicates()
            }
        )

        grouped = required.merge(
            grouped,
            on="source1_entity_id",
            how="left",
        )

        grouped[
            "candidate_entity_ids"
        ] = grouped[
            "candidate_entity_ids"
        ].fillna("")

    return grouped


def save_candidate_pairs(
    candidate_pairs: pd.DataFrame,
    output_path: str,
    source1_df: pd.DataFrame | None = None,
) -> None:

    output = create_candidate_pairs_output(
        candidate_pairs,
        source1_df=source1_df,
    )

    output.to_csv(
        output_path,
        sep="\t",
        index=False,
        encoding="utf-8",
    )
