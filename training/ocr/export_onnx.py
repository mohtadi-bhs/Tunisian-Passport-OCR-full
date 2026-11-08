import torch
from pathlib import Path
from training.ocr.model_crnn import CRNN
from training.utils.common import ROOT, ensure_dir


def main(ckpt="ocr_best.pth", img_h=32, img_w=320):
    models = ROOT / "models"
    ensure_dir(models)
    ckpt_path = Path(ckpt)
    state = torch.load(ckpt_path, map_location="cpu")
    nclass = state.get("nclass", 37)
    model = CRNN(nclass=nclass)
    model.load_state_dict(state["model"])
    model.eval()
    dummy = torch.randn(1, 1, img_h, img_w)
    out = models / "ocr_extractor.onnx"
    torch.onnx.export(model, dummy, out, input_names=["input"], output_names=["logits"], opset_version=12, dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch", 1: "time"}})
    print("Exported to", out)


if __name__ == "__main__":
    main()
