"""
Pytest fixtures for tests.
Centralizes creation of mock datasets and shared configs.
"""

import pytest
from PIL import Image


@pytest.fixture(scope="session")
def mock_covid_data(tmp_path_factory):
    """Create mock COVID Radiography dataset."""
    root = tmp_path_factory.mktemp("covid_radiography")
    class_sample_counts = {
        "COVID": 200,
        "NORMAL": 200,
        "LUNG_OPACITY": 200,
        "VIRAL_PNEUMONIA": 200,
    }
    for cls, n_samples in class_sample_counts.items():
        img_dir = root / cls / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_samples):
            Image.new("L", (64, 64), color=i * 20).save(img_dir / f"{cls}-{i}.png")

    return root
