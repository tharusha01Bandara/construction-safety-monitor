def calculate_iou(boxA, boxB):
    # Determines the intersection over union
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def is_inside(inner_box, outer_box):
    # Checks if inner_box is significantly inside outer_box
    xA = max(inner_box[0], outer_box[0])
    yA = max(inner_box[1], outer_box[1])
    xB = min(inner_box[2], outer_box[2])
    yB = min(inner_box[3], outer_box[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    inner_area = (inner_box[2] - inner_box[0]) * (inner_box[3] - inner_box[1])
    
    if inner_area == 0:
        return False
    return (interArea / inner_area) > 0.5
