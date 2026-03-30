def generate_alert_report(scene_status, scene_conf, workers):
    total = len(workers)
    safe_count = sum(1 for w in workers if w["status"] == "safe")
    unsafe_count = total - safe_count
    
    report = [
        "PPE SAFETY ALERT REPORT",
        f"Scene Status: {scene_status.upper()}",
        f"Scene Confidence: {scene_conf:.2f}",
        f"Total Workers: {total}",
        f"Safe Workers: {safe_count}",
        f"Unsafe Workers: {unsafe_count}",
        ""
    ]
    
    for w in workers:
        status_str = f"Worker {w['worker_id']}: {w['status'].upper()}"
        if w["violations"]:
            status_str += f" - {', '.join(w['violations'])}"
        if w["review_needed"]:
            status_str += " [MANUAL REVIEW SUGGESTED]"
        report.append(status_str)
        
    return "\n".join(report)
