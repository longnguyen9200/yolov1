import torch
import torch.nn as nn
import IOU

class Yolo_Loss(nn.Module):
    def __init__(self, s=7, b=2, c=4):
        super(Yolo_Loss, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')
        self.s = s
        self.b = b
        self.c = c
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, preds, targets):
        preds = preds.reshape(-1, self.s, self.s, (self.c + self.b*5))
        
        iou_b1 = IOU.Intersection_over_union(preds[..., 5:9], targets[..., 5:9])
        iou_b2 = IOU.Intersection_over_union(preds[..., 10:14], targets[..., 5:9])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        iou_max, best_box = torch.max(ious, dim=0)

        indentity_fn = targets[..., 4].unsqueeze(3)

        box_pred = indentity_fn * (best_box * preds[..., 5:9] + (1-best_box) * preds[..., 10:14])
        box_target = indentity_fn * targets[..., 5:9]

        box_pred[..., 2:4] = torch.sign(box_pred[..., 2:4]) * torch.sqrt(torch.abs(box_pred[..., 2:4] + 1e-6))
        box_target[..., 2:4] = torch.sqrt(box_target[..., 2:4])

        box_loss = self.mse(
            torch.flatten(box_pred, end_dim=-2),
            torch.flatten(box_target, end_dim=-2)
        )

        p_pred_box = (best_box * preds[..., 9:10] + (1-best_box) * preds[..., 4:5])
        obj_loss = self.mse(
            torch.flatten(indentity_fn * p_pred_box),
            torch.flatten(indentity_fn * targets[..., 4:5])
        )

        noobj_loss = self.mse(
            torch.flatten((1-indentity_fn) * targets[..., 4:5], start_dim=1),
            torch.flatten((1-indentity_fn) * preds[..., 4:5], start_dim=1)
        )

        noobj_loss += self.mse(
            torch.flatten((1-indentity_fn) * targets[..., 4:5], start_dim=1),
            torch.flatten((1-indentity_fn) * preds[..., 9:10], start_dim=1)
        )

        class_loss = self.mse(
            torch.flatten(indentity_fn * preds[..., :4], end_dim=-2),
            torch.flatten(indentity_fn * targets[..., :4], end_dim=-2)
        )

        loss = (
            self.lambda_coord * box_loss +
            obj_loss + 
            self.lambda_noobj * noobj_loss + 
            class_loss
        )

        return loss



