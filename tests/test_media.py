import numpy as np
import pytest
from PIL import Image

from qcv.models.media import load_media, uniform_frame_indices


def test_uniform_frame_indices():
    assert uniform_frame_indices(3, 8).tolist() == [0, 1, 2]
    assert uniform_frame_indices(9, 3).tolist() == [0, 4, 8]
    with pytest.raises(ValueError):
        uniform_frame_indices(0, 4)
    with pytest.raises(ValueError):
        uniform_frame_indices(4, 0)


def test_load_media_image_frames_and_frame_directory(tmp_path):
    paths = []
    for index in range(5):
        path = tmp_path / f"frame_{index:02d}.png"
        Image.fromarray(np.full((8, 8, 3), 40 * index, dtype=np.uint8)).save(path)
        paths.append(str(path))
    kind, image = load_media({"type": "image", "path": paths[0]})
    assert kind == "image" and image.mode == "RGB" and image.size == (8, 8)
    kind, frames = load_media({"type": "frames", "paths": paths[:2]})
    assert kind == "video" and len(frames) == 2
    kind, frames = load_media({"type": "video", "path": str(tmp_path), "frame_count": 3})
    assert kind == "video"
    assert [frame.getpixel((0, 0))[0] for frame in frames] == [0, 80, 160]


def test_load_media_rejects_bad_records(tmp_path):
    with pytest.raises(ValueError):
        load_media({"type": "image"})
    with pytest.raises(ValueError):
        load_media({"type": "audio", "path": "x"})
    with pytest.raises(ValueError):
        load_media({"type": "frames", "path": str(tmp_path), "paths": []})
