import torch
import IOU
import numpy as np

def non_max_suppression(bboxes, iou_threshold, threshold, box_format="midpoint"):
    
    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)
        
        chosen_box = np.array(chosen_box)

        bboxes = [
            box for box in bboxes
            if box[0] != chosen_box[0]
            or IOU.Intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:])
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms