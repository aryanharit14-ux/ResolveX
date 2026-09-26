import pandas as pd

from business_entity_resolution.src.normalization import (
    normalize_address,
    normalize_country,
    normalize_dataframe,
    normalize_name,
    normalize_name_core,
    normalize_address_numbers,
)

def test_normalize_address_numbers_removes_leading_zeros():
    assert normalize_address_numbers("0337 Oakland Avenue") == "337"


def test_normalize_address_numbers_handles_long_leading_zeros():
    assert normalize_address_numbers("005559 Orville Avenue") == "5559"
    assert normalize_address_numbers("0017560 Ellis Road") == "17560"


def test_normalize_address_numbers_preserves_alpha_suffix():
    assert normalize_address_numbers("024A Main Street") == "24a"


def test_normalize_dataframe_adds_address_numbers_normalized():
    df = pd.DataFrame(
        {
            "entity_id": ["S1-test"],
            "business_name": ["Example Business"],
            "business_address": ["0017560 Ellis Road"],
            "country": ["US"],
        }
    )

    result = normalize_dataframe(df)

    assert result.loc[0, "address_numbers_normalized"] == "17560"
    
def test_name_normalization():
    assert normalize_name("  ACME   Technologies, Inc.  ") == (
        "acme technologies inc"
    )


def test_ampersand_normalization():
    assert normalize_name("A & B Enterprises") == (
        "a and b enterprises"
    )


def test_unicode_and_casefold():
    assert normalize_name("GÜRZENICHSTRAẞE") == (
        "gürzenichstrasse"
    )


def test_indic_script_is_preserved():
    value = "राम मार्केटिंग प्राइवेट लिमिटेड"
    result = normalize_name(value)

    assert "राम" in result
    assert "मार्केटिंग" in result


def test_address_preserves_numbers():
    result = normalize_address("12-A, MG Road, Bengaluru")

    assert "12" in result
    assert "a" in result
    assert "mg" in result
    assert "road" in result
    assert "bengaluru" in result


def test_address_removes_standalone_null():
    result = normalize_address("12 MG Road null Bengaluru")

    assert result == "12 mg road bengaluru"


def test_name_core_removes_trailing_suffix():
    assert normalize_name_core("Acme Technologies Pvt Ltd") == (
        "acme technologies"
    )


def test_country_normalization():
    assert normalize_country("  INDIA ") == "india"


def test_dataframe_preserves_original_columns():
    df = pd.DataFrame(
        {
            "entity_id": ["S1-1"],
            "business_name": ["ACME Technologies Pvt Ltd"],
            "business_address": ["12 MG Road"],
            "country": ["India"],
        }
    )

    result = normalize_dataframe(df)

    assert result.loc[0, "business_name"] == "ACME Technologies Pvt Ltd"
    assert result.loc[0, "business_address"] == "12 MG Road"
    assert result.loc[0, "country"] == "India"

    assert result.loc[0, "name_normalized"] == (
        "acme technologies pvt ltd"
    )
    assert result.loc[0, "name_core"] == "acme technologies"
    assert result.loc[0, "address_normalized"] == "12 mg road"
    assert result.loc[0, "country_normalized"] == "india"


def test_dataframe_requires_expected_columns():
    df = pd.DataFrame(
        {
            "entity_id": ["S1-1"],
            "business_name": ["ACME"],
        }
    )

    try:
        normalize_dataframe(df)
    except ValueError as exc:
        assert "business_address" in str(exc)
        assert "country" in str(exc)
    else:
        raise AssertionError("Expected ValueError")