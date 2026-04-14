import torch

def inner_iou(pred_boxes: torch.Tensor, target_boxes: torch.Tensor, aux_ratio: torch.Tensor | float = 0.5, eps: float = 1e-7) -> torch.Tensor:
    """
    计算Inner-IoU，辅助边框面积为原框的aux_ratio倍，支持每对框自适应。
    
    Args:
        pred_boxes (torch.Tensor): 预测框，(N, 4)，xyxy格式。
        target_boxes (torch.Tensor): GT框，(N, 4)，xyxy格式。
        aux_ratio (Tensor | float): (N,) 每对框的辅助边框缩放比例，或标量。
        eps (float): 防止除零。
    
    Returns:
        torch.Tensor: Inner-IoU分数，(N,)
    """
    def shrink_box(box, ratio):
        x1, y1, x2, y2 = box.unbind(-1)
        w = x2 - x1
        h = y2 - y1
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        new_w = w * ratio
        new_h = h * ratio
        new_x1 = cx - new_w / 2
        new_y1 = cy - new_h / 2
        new_x2 = cx + new_w / 2
        new_y2 = cy + new_h / 2
        return torch.stack([new_x1, new_y1, new_x2, new_y2], dim=-1)

    # 处理 aux_ratio 类型和形状
    if not torch.is_tensor(aux_ratio):
        aux_ratio = torch.full((pred_boxes.shape[0],), float(aux_ratio), device=pred_boxes.device, dtype=pred_boxes.dtype)
    else:
        # 确保形状为 (N,)
        if aux_ratio.dim() > 1:
            aux_ratio = aux_ratio.squeeze(-1)

    pred_aux = shrink_box(pred_boxes, aux_ratio)
    target_aux = shrink_box(target_boxes, aux_ratio)

    inter_x1 = torch.max(pred_aux[:, 0], target_aux[:, 0])
    inter_y1 = torch.max(pred_aux[:, 1], target_aux[:, 1])
    inter_x2 = torch.min(pred_aux[:, 2], target_aux[:, 2])
    inter_y2 = torch.min(pred_aux[:, 3], target_aux[:, 3])
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h
    area1 = (pred_aux[:, 2] - pred_aux[:, 0]) * (pred_aux[:, 3] - pred_aux[:, 1])
    area2 = (target_aux[:, 2] - target_aux[:, 0]) * (target_aux[:, 3] - target_aux[:, 1])
    union_area = area1 + area2 - inter_area + eps
    inner_iou = inter_area / union_area
    return inner_iou