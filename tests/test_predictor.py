from pathlib import Path
from PIL import Image
import numpy as np
import torch
from tasdf.models import TASDFNet
from tasdf.engine.predictor import Predictor


def test_predictor_pure_inference(tmp_path: Path):
    img_path = tmp_path / "sample.png"
    Image.fromarray(np.uint8(np.random.rand(64, 64) * 255)).save(img_path)

    model = TASDFNet(in_channels=1, base_channels=16, num_classes=4, k_iters=2)
    predictor = Predictor(image_size=32, device=torch.device("cpu"), branch="ensemble")
    res = predictor.predict(model, source=img_path, output_dir=tmp_path / "out", save=True)

    assert len(res) == 1
    assert res[0]["mask"].shape == (64, 64)