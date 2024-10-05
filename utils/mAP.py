import torch
from collections import Counter
import IOU

def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, box_fomat='midpoint', num_classes=4):
    avg_precision = []
    eps = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truth = []

        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for gr_truth in true_boxes:
            if gr_truth[1] == c:
                ground_truth.append(gr_truth)

        num_bboxes = Counter([gt[0] for gt in ground_truth])

        for key, val in num_bboxes.items():
            num_bboxes[key] = torch.zeros(val)

        detections.sort(key=lambda x: x[2], reverse=True)
        tp = torch.zeros(len(detections))
        fp = torch.zeros(len(detections))
        all_true_bboxes = len(ground_truth)

        if all_true_bboxes == 0:
            continue

        for idx, detection in enumerate(detections):
            best_iou = 0
            ground_truth_img = [
                bbox for bbox in ground_truth if bbox[0] == detection[0]
            ]
            for idx_, gt in enumerate(ground_truth_img):
                iou = IOU.Intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:])
                )

                if iou > best_iou:
                    best_iou = iou
                    gt_idx = idx_
            
            if best_iou > iou_threshold:
                if num_bboxes[detection[0]][gt_idx] == 0:
                    tp[idx] = 1
                    num_bboxes[detection[0]][gt_idx] = 1
                else:
                    fp[idx] = 1
            else:
                fp[idx] = 1

        sum_tp = torch.cumsum(tp, dim=0)
        sum_fp = torch.cumsum(fp, dim=0)

        recalls = sum_tp / (all_true_bboxes + eps)
        precisions = torch.divide(sum_tp, (sum_tp + sum_fp + eps))

        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))

        avg_precision.append(torch.trapz(precisions, recalls))
    
    print(sum(avg_precision) / len(avg_precision))

    return sum(avg_precision) / len(avg_precision)



                