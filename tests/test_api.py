import pytest
from fastapi.testclient import TestClient

from abalone.api.app import create_app
from abalone.data import split_xy
from abalone.models import build_model

SAMPLE = {
    "sex": "M",
    "length": 0.455,
    "diameter": 0.365,
    "height": 0.095,
    "whole_weight": 0.514,
    "shucked_weight": 0.2245,
    "viscera_weight": 0.101,
    "shell_weight": 0.15,
}


@pytest.fixture()
def client(synthetic_df, small_cfg):
    X, y = split_xy(synthetic_df)
    model = build_model("ridge", small_cfg).fit(X, y)
    with TestClient(create_app(model=model, metadata={"model_name": "ridge"})) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_predict_single(client):
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_rings"] > 0
    assert body["predicted_age_years"] == pytest.approx(body["predicted_rings"] + 1.5)


def test_predict_batch_matches_single(client):
    single = client.post("/predict", json=SAMPLE).json()["predicted_rings"]
    other = {**SAMPLE, "sex": "I", "length": 0.2}
    r = client.post("/predict/batch", json={"instances": [SAMPLE, other]})
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert len(preds) == 2 and preds[0]["predicted_rings"] == pytest.approx(single)
    assert r.json()["model_name"] == "ridge"


@pytest.mark.parametrize(
    "patch", [{"sex": "X"}, {"length": -0.1}, {"whole_weight": "heavy"}, {"shell_weight": None}]
)
def test_invalid_payloads_rejected(client, patch):
    assert client.post("/predict", json={**SAMPLE, **patch}).status_code == 422


def test_empty_batch_rejected(client):
    assert client.post("/predict/batch", json={"instances": []}).status_code == 422


def test_model_info(client):
    assert client.get("/model/info").json()["model_name"] == "ridge"


def test_503_without_model(tmp_path):
    app = create_app(model_path=tmp_path / "missing.joblib")
    with TestClient(app) as c:
        assert c.get("/health").json()["model_loaded"] is False
        assert c.post("/predict", json=SAMPLE).status_code == 503


def test_loads_model_from_disk(tmp_path, synthetic_df, small_cfg):
    from abalone.persistence import save_model

    X, y = split_xy(synthetic_df)
    path = save_model(
        build_model("linear", small_cfg).fit(X, y), tmp_path / "m.joblib", {"model_name": "linear"}
    )
    with TestClient(create_app(model_path=path)) as c:
        assert c.post("/predict", json=SAMPLE).status_code == 200
        assert c.get("/model/info").json()["model_name"] == "linear"
