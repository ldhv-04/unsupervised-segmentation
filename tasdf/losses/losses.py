import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tasdf.models.itgm import ITGM


def sigmoid_rampup(current_epoch: int, max_epochs: int, alpha: float = 1.0) -> float:
    """Hàm điều phối trọng số động Sigmoid Ramp-up w(t) theo Laine & Aila (2016)."""
    if max_epochs <= 0:
        return alpha
    current = np.clip(current_epoch, 0, max_epochs)
    phase = 1.0 - (current / max_epochs)
    return float(alpha * np.exp(-5.0 * phase * phase))


class MainLoss(nn.Module):
    """
    Hàm mất mát toàn thể TASDF (Mục 3.3):
    - L_sup = omega_0 * CE(P1, Y) + omega_1 * CE(P2, Y) [+ neo khung xương tô-pô chống sụp đổ nghiệm]
    - L_pc  = MSE(softmax(P1_u), softmax(P2_u))
    - L_tc  = MSE(T1_u, T2_u)
    - L_distillation = lambda * w(t) * L_tc + w(t) * L_pc
    - L_total = gamma * L_sup + (1 - gamma) * L_distillation
    """

    def __init__(
        self,
        lambda_topo: float = 0.10,
        gamma: float = 0.5,
        alpha: float = 1.0,
        max_epochs: int = 2000,
        omega_0: float = 1.0,
        omega_1: float = 1.0,
        topo_sup_weight: float = 0.10,
        k_iters: int = 10,
    ):
        super(MainLoss, self).__init__()
        self.lambda_topo = lambda_topo
        self.gamma = gamma
        self.alpha = alpha
        self.max_epochs = max_epochs
        self.omega_0 = omega_0
        self.omega_1 = omega_1
        self.topo_sup_weight = topo_sup_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.target_itgm = ITGM(iterations=k_iters)

    @torch.no_grad()
    def _extract_target_skeleton(
        self, targets: torch.Tensor, num_classes: int, target_channels: int
    ) -> torch.Tensor:
        """
        Trích xuất khung xương chuẩn từ mặt nạ nhãn Y cho các lớp tiền cảnh (1..C-1)
        để neo nhánh TopoHead không bị suy biến về 0.
        """
        one_hot = F.one_hot(targets.clamp(0, num_classes - 1), num_classes=num_classes)
        # Chuyển về [B, C, H, W] và lấy các lớp tiền cảnh 1..C-1
        one_hot_chw = one_hot.permute(0, 3, 1, 2).float()
        fg_masks = one_hot_chw[:, 1:, :, :] if num_classes > 1 else one_hot_chw

        if fg_masks.shape[1] != target_channels:
            # Tương thích ngược nếu bài test truyền t_labeled có 1 kênh gộp
            fg_union = (targets > 0).unsqueeze(1).float()
            fg_masks = fg_union.expand(-1, target_channels, -1, -1)

        target_skel = self.target_itgm(fg_masks) * fg_masks
        return target_skel

    def forward(
        self,
        p1_labeled: torch.Tensor,
        p2_labeled: torch.Tensor,
        targets_labeled: torch.Tensor,
        p1_unlabeled: torch.Tensor,
        p2_unlabeled: torch.Tensor,
        t1_unlabeled: torch.Tensor,
        t2_unlabeled: torch.Tensor,
        current_epoch: int,
        t1_labeled: torch.Tensor | None = None,
        t2_labeled: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        # 1. Tổn thất có giám sát mức điểm ảnh (Phương trình 8)
        loss_ce1 = self.ce_loss(p1_labeled, targets_labeled)
        loss_ce2 = self.ce_loss(p2_labeled, targets_labeled)
        loss_sup = self.omega_0 * loss_ce1 + self.omega_1 * loss_ce2

        # Neo giám sát khung xương tô-pô trên tập có nhãn để chống Mode Collapse cho TopoHead
        loss_topo_sup_val = 0.0
        if self.topo_sup_weight > 0.0 and t1_labeled is not None and t2_labeled is not None:
            num_classes = p1_labeled.shape[1]
            target_skel = self._extract_target_skeleton(
                targets_labeled, num_classes=num_classes, target_channels=t1_labeled.shape[1]
            )
            loss_topo_sup = F.mse_loss(t1_labeled, target_skel) + F.mse_loss(t2_labeled, target_skel)
            loss_sup = loss_sup + self.topo_sup_weight * loss_topo_sup
            loss_topo_sup_val = loss_topo_sup.item()

        # 2. Tổn thất nhất quán mức điểm ảnh trên dữ liệu không nhãn (Phương trình 9)
        prob1_unlabeled = F.softmax(p1_unlabeled, dim=1)
        prob2_unlabeled = F.softmax(p2_unlabeled, dim=1)
        loss_pc = F.mse_loss(prob1_unlabeled, prob2_unlabeled)

        # 3. Tổn thất nhất quán mức tô-pô trên dữ liệu không nhãn (Phương trình 10)
        loss_tc = F.mse_loss(t1_unlabeled, t2_unlabeled)

        # 4. Tổn thất tự chưng cất và Tổng tổn thất (Phương trình 11 & 12)
        w_t = sigmoid_rampup(current_epoch, self.max_epochs, self.alpha)
        loss_distill = self.lambda_topo * w_t * loss_tc + w_t * loss_pc

        total_loss = self.gamma * loss_sup + (1.0 - self.gamma) * loss_distill

        metrics = {
            "total_loss": total_loss.item(),
            "loss_sup": loss_sup.item(),
            "loss_topo_sup": loss_topo_sup_val,
            "loss_pc": loss_pc.item(),
            "loss_tc": loss_tc.item(),
            "w_t": w_t,
        }
        return total_loss, metrics