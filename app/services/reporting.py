def generate_alert_report(scene_status, scene_conf, workers):
    lines = []

    scene_status_upper = scene_status.upper()
    total_workers = len(workers)
    safe_workers = sum(1 for w in workers if w.get("status", "").lower() == "safe")
    unsafe_workers = total_workers - safe_workers

    scene_icon = "✅" if scene_status_upper == "SAFE" else "🚨"
    lines.append("=" * 60)
    lines.append(f"{scene_icon} PPE SAFETY ALERT REPORT")
    lines.append("=" * 60)
    lines.append(f"Scene Status      : {scene_status_upper}")
    lines.append(f"Scene Confidence  : {scene_conf:.2f}")
    lines.append(f"Total Workers     : {total_workers}")
    lines.append(f"Safe Workers      : {safe_workers}")
    lines.append(f"Unsafe Workers    : {unsafe_workers}")
    lines.append("-" * 60)

    for i, w in enumerate(workers, start=1):
        worker_status = w.get("status", "unknown").upper()
        worker_conf = w.get("confidence", 0.0)
        person_conf = w.get("person_conf", 0.0)

        helmet_detected = w.get("helmet", False)
        helmet_conf = w.get("helmet_conf", None)

        vest_detected = w.get("vest", False)
        vest_conf = w.get("vest_conf", None)

        violations = w.get("violations", [])
        review_needed = w.get("review_needed", False)

        status_icon = "✅" if worker_status == "SAFE" else "❌"

        lines.append(f"👷 Worker {i}")
        lines.append(f"   Status             : {status_icon} {worker_status}")
        lines.append(f"   Person Confidence  : {person_conf:.2f}")

        if helmet_detected and helmet_conf is not None:
            lines.append(f"   Helmet             : YES 🪖 ({helmet_conf:.2f})")
        else:
            lines.append("   Helmet             : NO 🪖")

        if vest_detected and vest_conf is not None:
            lines.append(f"   Vest               : YES 🦺 ({vest_conf:.2f})")
        else:
            lines.append("   Vest               : NO 🦺")

        if violations:
            lines.append(f"   Violations         : {', '.join(violations)}")
        else:
            lines.append("   Violations         : None")

        lines.append(f"   Worker Confidence  : {worker_conf:.2f}")

        if review_needed:
            lines.append("   Review             : ⚠️ Manual review recommended")

        lines.append("-" * 60)

    if unsafe_workers == 0:
        lines.append("🎉 All detected workers comply with PPE rules.")
    else:
        lines.append(f"⚠️ Action needed: {unsafe_workers} worker(s) have PPE violations.")

    lines.append("=" * 60)

    return "\n".join(lines)

