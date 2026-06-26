from datetime import datetime

def trust_score(candidate):
    verified_email = candidate["redrob_signals"]["verified_email"]
    verified_phone = candidate["redrob_signals"]["verified_phone"]
    interview_completion_rate = candidate["redrob_signals"]["interview_completion_rate"]
    recruiter_response_rate = candidate["redrob_signals"]["recruiter_response_rate"]

    email_score = 1 if verified_email else 0
    phone_score = 1 if verified_phone else 0
    
    final_score = (email_score * 0.15) + (phone_score * 0.15) + (interview_completion_rate * 0.4) + (recruiter_response_rate * 0.3)

    return final_score

def availability_score(candidate, work_type, job_location):
    signals = candidate["redrob_signals"]
    profile = candidate["profile"]

    notice_period = min(signals["notice_period_days"], 90)
    notice_score = 1 - (notice_period/ 90) 

    open_to_work = 1 if signals["open_to_work_flag"] else 0

    last_active = datetime.strptime(
        signals["last_active_date"],
        "%Y-%m-%d"
    )

    days_since_last_active = min((datetime.now() - last_active).days, 180)

    active_score = 1 - (days_since_last_active / 180)

    if work_type == "remote":
        return (open_to_work * 0.35) + (notice_score * 0.35) + (active_score * 0.3)
    
    candidate_location = profile["location"]   
    location_match = 1 if job_location == candidate_location else 0

    if not location_match:
        location_score = 1 if signals["willing_to_relocate"] else 0
    else:
        location_score = 1

    if work_type == "hybrid":
        return (open_to_work * 0.325) + (notice_score * 0.325) + (active_score * 0.25) + (location_score * 0.1)
    
    return (open_to_work * 0.25) + (notice_score * 0.25) + (active_score * 0.2) + (location_score * 0.3)