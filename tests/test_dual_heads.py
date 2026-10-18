import torch
from tasdf.models.dual_heads import PixelHead, TopoHead, DualHeads


def test_pixel_head_shape():
    head = PixelHead(in_channels=64, num_classes=5)
    f = torch.randn(2, 64, 128, 128)
    p = head(f)
    assert p.shape == (2, 5, 128, 128)


def test_topo_head_shape_and_range():
    head = TopoHead(in_channels=64, topo_channels=4, k_iters=5)
    f = torch.randn(2, 64, 128, 128)
    t = head(f)
    assert t.shape == (2, 4, 128, 128)
    assert t.min() >= 0.0 and t.max() <= 1.0


def test_dual_heads_gradient_flow():
    dual_heads = DualHeads(in_channels=64, num_classes=5, k_iters=5)
    dual_heads.train()
    f = torch.randn(1, 64, 64, 64, requires_grad=True)
    p, t = dual_heads(f)
    loss = p.sum() + t.sum()
    loss.backward()
    assert dual_heads.pixel_head.conv.weight.grad is not None
    assert dual_heads.topo_head.proj.weight.grad is not None
    assert f.grad is not None and torch.any(f.grad != 0)