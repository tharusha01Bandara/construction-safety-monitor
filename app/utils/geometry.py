def avg_conf(scores):
    return sum(scores) / len(scores) if scores else 0.0

def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)

def intersection_area(boxA, boxB):
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    if x2 <= x1 or y2 <= y1:
        return 0
    return (x2 - x1) * (y2 - y1)

def overlap_ratio(inner, outer):
    inter = intersection_area(inner, outer)
    inner_area = box_area(inner)
    if inner_area == 0:
        return 0
    return inter / inner_area

def is_valid_person(box, min_area=12000):
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    area = width * height
    return area >= min_area and width >= 40 and height >= 80

def helmet_belongs_to_person(helmet_box, person_box):
    hx1, hy1, hx2, hy2 = helmet_box
    px1, py1, px2, py2 = person_box

    cx = (hx1 + hx2) / 2
    cy = (hy1 + hy2) / 2

    upper_limit = py1 + 0.35 * (py2 - py1)
    return px1 <= cx <= px2 and py1 <= cy <= upper_limit
