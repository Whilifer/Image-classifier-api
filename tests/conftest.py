from fastapi.testclient import TestClient

from app.main import app

import pytest

from pathlib import Path


@pytest.fixture
def client():

    with TestClient(app) as client:
        yield client

@pytest.fixture
def getfile():
    def _getfile(filename: str) -> Path:
        return Path(f"tests/assets/{filename}")
    return _getfile

@pytest.fixture
def asset_file(request, getfile):

    filename, mime = request.param    
    path = getfile(filename)

    with path.open("rb") as f:

        yield (path.name, f, mime)