"""
Pytest fixtures for MedVisNet tests.
Centralizes creation of mock datasets and shared configs.
"""

import pytest
from PIL import Image


@pytest.fixture(scope="session")
def mock_covid_data(tmp_path_factory):
    """Create mock COVID Radiography dataset once per session."""
    root = tmp_path_factory.mktemp("covid_radiography")
    for cls in ["COVID", "NORMAL", "LUNG_OPACITY", "VIRAL_PNEUMONIA"]:
        img_dir = root / cls / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        for i in range(
            {"COVID": 3, "NORMAL": 8, "LUNG_OPACITY": 5, "VIRAL_PNEUMONIA": 2}[cls]
        ):
            Image.new("RGB", (64, 64), color=(i * 20, 0, 0)).save(
                img_dir / f"{cls}-{i}.png"
            )
    return root
