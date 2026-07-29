from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


def test_predict():

    with TestClient(app) as client:

        image_path = Path("tests/assets/dog.jpg")

        with image_path.open("rb") as image:

            response = client.post(
                "/predict/",
                files={
                    "file": (
                        image_path.name,
                        image,
                        "image/jpeg"
                    )
                }
            )

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