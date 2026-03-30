from ..utils.geometry import is_inside
from ..config import settings

def apply_safety_rules(persons, helmets, vests):
    workers = []
    
    for i, person in enumerate(persons):
        box, conf = person
        area = (box[2] - box[0]) * (box[3] - box[1])
        if area < settings.MIN_PERSON_AREA:
            continue
            
        worker = {
            "worker_id": i + 1,
            "person_box": box,
            "person_conf": conf,
            "helmet": False,
            "helmet_conf": None,
            "vest": False,
            "vest_conf": None,
            "violations": [],
            "status": "safe",
            "review_needed": False
        }
        
        # Match helmet
        for h_box, h_conf in helmets:
            if is_inside(h_box, box):
                worker["helmet"] = True
                worker["helmet_conf"] = h_conf
                break
                
        # Match vest
        for v_box, v_conf in vests:
            if is_inside(v_box, box):
                worker["vest"] = True
                worker["vest_conf"] = v_conf
                break
                
        if not worker["helmet"]:
            worker["violations"].append("missing helmet")
        if not worker["vest"]:
            worker["violations"].append("missing vest")
            
        if worker["violations"]:
            worker["status"] = "unsafe"
            
        confs = [conf]
        if worker["helmet_conf"]: confs.append(worker["helmet_conf"])
        if worker["vest_conf"]: confs.append(worker["vest_conf"])
        worker["confidence"] = sum(confs) / len(confs)
        
        if worker["confidence"] < settings.REVIEW_CONFIDENCE_THRESHOLD:
            worker["review_needed"] = True
            
        workers.append(worker)
        
    scene_status = "unsafe" if any(w["status"] == "unsafe" for w in workers) else "safe"
    scene_conf = sum(w["confidence"] for w in workers) / len(workers) if workers else 1.0
    
    return workers, scene_status, scene_conf
