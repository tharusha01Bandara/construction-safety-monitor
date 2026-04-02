from ..utils.geometry import is_valid_person, helmet_belongs_to_person, overlap_ratio, avg_conf
from ..config import settings

def apply_safety_rules(persons, helmets, vests):
    worker_results = []
    scene_safe = True

    used_helmets = set()
    used_vests = set()

    for i_person, person_item in enumerate(persons):
        person = person_item[0]
        person_conf = person_item[1]

        # Apply worker thresholds
        if person_conf < settings.PERSON_CONF_THRESHOLD or not is_valid_person(person, settings.PERSON_MIN_AREA):
            continue

        matched_helmet = None
        best_helmet_conf = -1

        for i, h in enumerate(helmets):
            if i in used_helmets:
                continue
            h_box, h_conf = h
            if h_conf >= settings.HELMET_CONF_THRESHOLD and helmet_belongs_to_person(h_box, person):
                if h_conf > best_helmet_conf:
                    best_helmet_conf = h_conf
                    matched_helmet = i

        matched_vest = None
        best_vest_conf = -1
        best_vest_score = -1

        torso_box = [
            person[0],
            person[1] + settings.VEST_TORSO_TOP_RATIO * (person[3] - person[1]),
            person[2],
            person[1] + settings.VEST_TORSO_BOTTOM_RATIO * (person[3] - person[1]),
        ]

        for i, v in enumerate(vests):
            if i in used_vests:
                continue
            
            v_box, v_conf = v
            if v_conf < settings.VEST_CONF_THRESHOLD:
                continue

            score = overlap_ratio(v_box, torso_box)

            if score >= settings.VEST_OVERLAP_THRESHOLD and score > best_vest_score:
                best_vest_score = score
                best_vest_conf = v_conf
                matched_vest = i

        has_helmet = matched_helmet is not None
        has_vest = matched_vest is not None

        helmet_conf = best_helmet_conf if has_helmet else None
        vest_conf = best_vest_conf if has_vest else None
        
        helmet_box = helmets[matched_helmet][0] if has_helmet else None
        vest_box = vests[matched_vest][0] if has_vest else None

        if matched_helmet is not None:
            used_helmets.add(matched_helmet)

        if matched_vest is not None:
            used_vests.add(matched_vest)

        violations = []
        if not has_helmet:
            violations.append("missing helmet")
        if not has_vest:
            violations.append("missing vest")

        safe = has_helmet and has_vest
        if not safe:
            scene_safe = False

        conf_parts = [person_conf]
        if has_helmet:
            conf_parts.append(best_helmet_conf)
        if has_vest:
            conf_parts.append(best_vest_conf)

        worker_confidence = round(avg_conf(conf_parts), 3)
        review_needed = worker_confidence < settings.MANUAL_REVIEW_CONFIDENCE

        worker_results.append({
            "worker_id": len(worker_results) + 1,
            "person_box": person,
            "person_conf": person_conf,
            "helmet": has_helmet,
            "helmet_conf": helmet_conf,
            "helmet_box": helmet_box,
            "vest": has_vest,
            "vest_conf": vest_conf,
            "vest_box": vest_box,
            "status": "safe" if safe else "unsafe",
            "violations": violations,
            "confidence": worker_confidence,
            "review_needed": review_needed
        })

    scene_confidence = round(avg_conf([w["confidence"] for w in worker_results]), 3) if worker_results else 0.0

    return worker_results, "safe" if scene_safe else "unsafe", scene_confidence
