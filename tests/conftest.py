from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app


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


@pytest.fixture
def batch_assets(request, getfile):
    file_specs = request.param
    files = []
    for filename, mime in file_specs:
        path = getfile(filename)
        with path.open("rb") as f:
            content = f.read()
        files.append((filename, content, mime))
    yield files


@pytest.fixture
def batch_assets_count(request, getfile):
    """
    Возвращает список кортежей (filename, content, mime) длиной request.param.
    Все файлы берутся из dog.jpg.
    """
    count = request.param
    path = getfile("dog.jpg")
    with path.open("rb") as f:
        content = f.read()
    files = [(path.name, content, "image/jpeg") for _ in range(count)]
    return files
