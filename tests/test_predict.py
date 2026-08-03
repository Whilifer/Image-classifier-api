


import pytest


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
