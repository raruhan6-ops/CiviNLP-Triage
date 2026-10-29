import pandas as pd

from civinlp.data.download_nbi import load_nbi_data
from civinlp.data.download_roads_311 import download_roads_311
from civinlp.data.schema import RawRoadSchema
from civinlp.utils.io import load_parquet


def test_road_schema_validation():
    df = pd.DataFrame(
        {
            "unique_key": ["123"],
            "created_date": ["2023-01-01"],
            "complaint_type": ["Pothole"],
        }
    )
    assert RawRoadSchema.validate_columns(df) == []

    df_bad = pd.DataFrame({"Wrong": [1]})
    missing = RawRoadSchema.validate_columns(df_bad)
    assert "unique_key" in missing


def test_download_roads_mock(tmp_path, monkeypatch):
    """Test the download logic by mocking pandas read_csv."""
    output_file = tmp_path / "roads.parquet"

    # Mock data
    df_mock = pd.DataFrame(
        {
            "unique_key": ["1"],
            "created_date": ["2023-01-01"],
            "complaint_type": ["Pothole"],
        }
    )

    def mock_read_csv(*args, **kwargs):
        return df_mock

    monkeypatch.setattr(pd, "read_csv", mock_read_csv)

    download_roads_311(str(output_file), source_url_or_path="dummy_url")

    assert output_file.exists()
    df_loaded = load_parquet(output_file)
    assert len(df_loaded) == 1
    assert df_loaded.iloc[0]["complaint_type"] == "Pothole"


def test_load_nbi_mock(tmp_path, monkeypatch):
    input_csv = tmp_path / "nbi.csv"
    output_parquet = tmp_path / "nbi.parquet"

    df_mock = pd.DataFrame({"STRUCTURE_NUMBER_008": ["B123"], "OTHER_COL": [1]})
    df_mock.to_csv(input_csv, index=False)

    load_nbi_data(str(input_csv), str(output_parquet))

    assert output_parquet.exists()
    df_loaded = load_parquet(output_parquet)
    assert len(df_loaded) == 1


def test_build_311_url_filters_roads_and_sets_limit():
    from urllib.parse import parse_qs, urlsplit

    from civinlp.data.download_roads_311 import NYC_311_RESOURCE, build_311_url

    params = parse_qs(urlsplit(build_311_url(NYC_311_RESOURCE, 500)).query)
    assert params["$limit"] == ["500"]
    assert "'Street Condition'" in params["$where"][0]
    assert params["$order"] == ["created_date DESC,unique_key DESC"]

    # An existing $limit is overridden, other params kept, and no stray "&"
    url = build_311_url(NYC_311_RESOURCE + "?$limit=2000&borough=BRONX", 50)
    params = parse_qs(urlsplit(url).query)
    assert params["$limit"] == ["50"]
    assert params["borough"] == ["BRONX"]
    assert "?&" not in url


def test_download_passes_limit_to_url(tmp_path, monkeypatch):
    seen = {}

    def mock_read_csv(source, **kwargs):
        seen["source"] = source
        seen["kwargs"] = kwargs
        return pd.DataFrame(
            {
                "unique_key": ["1"],
                "created_date": ["2023-01-01"],
                "complaint_type": ["Street Condition"],
            }
        )

    import civinlp.data.download_roads_311 as mod

    monkeypatch.setattr(mod, "fetch_csv", mock_read_csv)
    download_roads_311(str(tmp_path / "r.parquet"), limit=10)
    assert "%24limit=10" in seen["source"] or "$limit=10" in seen["source"]
    assert "nrows" not in seen["kwargs"]
    assert seen["kwargs"]["dtype"] == {"incident_zip": str}


def test_area_split_with_explicit_test_areas():
    from civinlp.data.split import area_based_split

    df = pd.DataFrame({"area_id": ["A"] * 8 + ["B"] * 2, "x": range(10)})
    train, val, test = area_based_split(df, test_areas=["A"], val_frac=0.1)
    assert len(test) == 8
    assert len(val) == 1 and len(train) == 1
    assert set(train["area_id"]) == {"B"}


def test_socrata_download_is_paged(tmp_path, monkeypatch):
    from urllib.parse import parse_qs, urlsplit

    import civinlp.data.download_roads_311 as mod

    calls = []

    def mock_read_csv(url, **kwargs):
        q = parse_qs(urlsplit(url).query)
        calls.append((int(q["$limit"][0]), int(q.get("$offset", ["0"])[0])))
        n = int(q["$limit"][0]) if len(calls) < 3 else 1  # third page is short
        return pd.DataFrame(
            {
                "unique_key": [str(len(calls))] * n,
                "created_date": ["2023-01-01"] * n,
                "complaint_type": ["Street Condition"] * n,
            }
        )

    monkeypatch.setattr(mod, "fetch_csv", mock_read_csv)
    monkeypatch.setattr(mod, "PAGE_SIZE", 4)
    mod.download_roads_311(str(tmp_path / "r.parquet"), limit=20)
    assert calls == [(4, 0), (4, 4), (4, 8)]
    assert len(load_parquet(tmp_path / "r.parquet")) == 9
