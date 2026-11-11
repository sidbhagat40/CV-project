"""Measure training throughput and peak GPU memory with random tensors, to size experiments for your GPU.

  python scripts/benchmark.py --what mae seg det --arch vit_small
"""

import argparse
import time

import torch
import torch.nn.functional as F

from xdssl.data.taxonomy import DET_CLASSES, SEG_CLASSES
from xdssl.models.detection import DetectionModel
from xdssl.models.mae import MAE
from xdssl.models.segmentation import SegmentationModel
from xdssl.models.vit import create_vit
from xdssl.utils.misc import autocast


def bench(name, model, make_batch, loss_fn, steps, device, amp):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    torch.cuda.reset_peak_memory_stats()
    for i in range(steps + 3):
        if i == 3:
            torch.cuda.synchronize()
            t0 = time.time()
        batch = make_batch()
        with autocast(device, amp):
            loss = loss_fn(model, batch)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / steps
    print(f"{name:40s} {dt * 1000:7.1f} ms/step   peak mem {torch.cuda.max_memory_allocated() / 2**30:5.2f} GB")
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", nargs="+", default=["mae", "seg", "det"])
    ap.add_argument("--arch", default="vit_small")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--amp", default="bf16")
    ap.add_argument("--mae-batch", type=int, default=128)
    ap.add_argument("--seg-batch", type=int, default=8)
    ap.add_argument("--det-batch", type=int, default=4)
    a = ap.parse_args()
    dev = torch.device("cuda")
    print(torch.cuda.get_device_name(0))

    if "mae" in a.what:
        m = MAE(create_vit(a.arch, (224, 224))).to(dev).train()
        dt = bench(f"MAE {a.arch} 224px bs{a.mae_batch}", m, lambda: torch.randn(a.mae_batch, 3, 224, 224, device=dev),
                   lambda m, x: m(x)[0]["loss"], a.steps, dev, a.amp)
        print(f"    -> {a.mae_batch / dt:.0f} img/s; 30k steps x 256 imgs ~ {30000 * 256 / (a.mae_batch / dt) / 3600:.1f} h (compute only)")
        del m
        torch.cuda.empty_cache()
    if "seg" in a.what:
        m = SegmentationModel(create_vit(a.arch), len(SEG_CLASSES)).to(dev).train()
        dt = bench(f"Seg FPN {a.arch} 512x512 bs{a.seg_batch}", m,
                   lambda: (torch.randn(a.seg_batch, 3, 512, 512, device=dev), torch.randint(0, 19, (a.seg_batch, 512, 512), device=dev)),
                   lambda m, b: F.cross_entropy(m(b[0]).float(), b[1]), a.steps, dev, a.amp)
        print(f"    -> 16k steps ~ {16000 * dt / 3600:.1f} h")
        del m
        torch.cuda.empty_cache()
    if "det" in a.what:
        m = DetectionModel(create_vit(a.arch), len(DET_CLASSES)).to(dev).train()

        def batch():
            imgs = [torch.rand(3, 720, 1280, device=dev) for _ in range(a.det_batch)]
            tg = [{"boxes": torch.tensor([[100., 300., 300., 450.], [600., 350., 640., 450.]], device=dev),
                   "labels": torch.tensor([1, 2], device=dev)} for _ in range(a.det_batch)]
            return imgs, tg

        dt = bench(f"FasterRCNN {a.arch} 576x1024 bs{a.det_batch}", m, batch, lambda m, b: sum(m(*b).values()), a.steps, dev, a.amp)
        print(f"    -> 20k steps ~ {20000 * dt / 3600:.1f} h")


if __name__ == "__main__":
    main()
