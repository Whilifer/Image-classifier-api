import pytest

from app.config import settings


@pytest.mark.parametrize("asset_file", [("dog.jpg", "image/jpeg")], indirect=True)
def test_predict(client, asset_file):
    response = client.post("/predict/", files={"file": asset_file})

    assert response.status_code == 200

    body = response.json()

    assert "predictions" in body
    assert isinstance(body["predictions"], list)
    assert len(body["predictions"]) > 0

    prediction = body["predictions"][0]

    assert "class_name" in prediction
    assert "confidence" in prediction

    assert isinstance(prediction["class_name"], str)
    assert isinstance(prediction["confidence"], float)

    assert 0 <= prediction["confidence"] <= 1


def test_predict_without_file(client):
    response = client.post("/predict")

    assert response.status_code == 422


@pytest.mark.parametrize("asset_file", [("test.txt", "text/plain")], indirect=True)
def test_predict_text_file(client, asset_file):
    response = client.post("/predict/", files={"file": asset_file})
    assert response.status_code == 400


@pytest.mark.parametrize("asset_file", [("broken.jpg", "image/jpeg")], indirect=True)
def test_predict_broken_image(client, asset_file):
    response = client.post("/predict/", files={"file": asset_file})
    assert response.status_code == 400


def test_predict_file_too_large(client):
    content = b"0" * (settings.MAX_UPLOAD_SIZE + 1)

    response = client.post(
        "/predict/",
        files={
            "file": (
                "large.jpg",
                content,
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 413


# Batch tests


@pytest.mark.parametrize(
    "batch_assets_count",
    [settings.MAX_BATCH_SIZE],
    indirect=True,
)
def test_batch_max_files(client, batch_assets_count):
    files = [
        ("files", (name, content, mime)) for name, content, mime in batch_assets_count
    ]

    response = client.post(
        "/predict/batch",
        files=files,
    )

    assert response.status_code == 200

    body = response.json()

    assert "results" in body
    assert len(body["results"]) == settings.MAX_BATCH_SIZE


@pytest.mark.parametrize(
    "batch_assets",
    [
        [
            ("dog.jpg", "image/jpeg"),
            ("bird.jpg", "image/jpeg"),
        ],
    ],
    indirect=True,
)
def test_batch_predict(client, batch_assets):
    files = [("files", (name, content, mime)) for name, content, mime in batch_assets]

    response = client.post(
        "/predict/batch",
        files=files,
    )

    assert response.status_code == 200

    body = response.json()

    assert "results" in body
    assert len(body["results"]) == 2

    for result in body["results"]:
        assert "predictions" in result
        assert len(result["predictions"]) > 0

        prediction = result["predictions"][0]

        assert "class_name" in prediction
        assert "confidence" in prediction

        assert isinstance(prediction["class_name"], str)
        assert isinstance(prediction["confidence"], float)

        assert 0 <= prediction["confidence"] <= 1


@pytest.mark.parametrize(
    "batch_assets",
    [
        [
            ("dog.jpg", "image/jpeg"),
            ("test.txt", "text/plain"),
        ],
    ],
    indirect=True,
)
def test_batch_predict_text_file(client, batch_assets):
    files = [("files", (name, content, mime)) for name, content, mime in batch_assets]

    response = client.post(
        "/predict/batch",
        files=files,
    )

    assert response.status_code == 400


@pytest.mark.parametrize(
    "batch_assets",
    [
        [
            ("dog.jpg", "image/jpeg"),
            ("broken.jpg", "image/jpeg"),
        ],
    ],
    indirect=True,
)
def test_batch_predict_broken_image(client, batch_assets):
    files = [("files", (name, content, mime)) for name, content, mime in batch_assets]

    response = client.post(
        "/predict/batch",
        files=files,
    )

    assert response.status_code == 400


@pytest.mark.parametrize(
    "batch_assets",
    [
        [
            ("dog.jpg", "image/jpeg"),
            ("empty.jpg", "image/jpeg"),
        ],
    ],
    indirect=True,
)
def test_batch_predict_empty_file(client, batch_assets):
    files = [("files", (name, content, mime)) for name, content, mime in batch_assets]

    response = client.post(
        "/predict/batch",
        files=files,
    )

    assert response.status_code == 400


def test_batch_predict_file_too_large(client):
    content = b"0" * (settings.MAX_UPLOAD_SIZE + 1)

    files = [
        (
            "files",
            ("large.jpg", content, "image/jpeg"),
        )
    ]

    response = client.post(
        "/predict/batch",
        files=files,
    )

    assert response.status_code == 413
