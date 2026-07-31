"""Dependency-free checks connecting implementation and manuscript contract."""

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    python_files = [
        path for path in ROOT.rglob("*.py")
        if "notebooks" not in path.parts
    ]
    for path in python_files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    sca = (ROOT / "lwcamo" / "models" / "sca.py").read_text(encoding="utf-8")
    loss = (ROOT / "lwcamo" / "losses" / "idb.py").read_text(encoding="utf-8")
    model = (ROOT / "lwcamo" / "models" / "lwcamo.py").read_text(encoding="utf-8")
    assert "attended = merged * attention" in sca
    assert "BatchNormalization" in sca
    assert "tf.split(inputs, 4" in sca
    assert "Conv2DTranspose" in sca and "strides=2" in sca
    assert "merged + attention" not in sca
    assert "attended +" not in sca
    assert "0.5 * bce_loss + 2.0" in loss
    assert "decoder_projection" in model and "range(1, 6)" in model
    assert '"small": "nvidia/mit-b0"' in model
    assert '"medium": "nvidia/mit-b2"' in model
    assert '"large": "nvidia/mit-b5"' in model
    for variant in ("small", "medium", "large"):
        config = json.loads((ROOT / "configs" / f"lwcamo_{variant}.json").read_text())
        assert config["variant"] == variant
        assert config["image_size"] == 512
        assert config["batch_size"] == 8
        assert config["epochs"] == 400
        assert config["learning_rate"] == 0.00006
    required = [
        ROOT / "datasets" / "splits" / "train_camo_cod10k.txt",
        ROOT / "datasets" / "splits" / "valid_camo_cod10k.txt",
        ROOT / "docs" / "Figure_2_LWCamo_Architecture.png",
        ROOT / "docs" / "Figure_3_SCA_Block.png",
    ]
    assert all(path.is_file() for path in required)
    assert not (ROOT / "notebooks").exists()
    assert not any((ROOT / "weights").glob("*.h5"))
    assert not any((ROOT / "weights").glob("*.keras"))
    print(f"Repository contract: PASS ({len(python_files)} Python files parsed)")


if __name__ == "__main__":
    main()
