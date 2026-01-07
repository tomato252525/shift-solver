from __future__ import annotations

import os
from typing import List, Dict, Any, Tuple, Optional
from datetime import date, datetime, timedelta

from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel, Field
from ortools.sat.python import cp_model
from supabase import create_client, Client


# =============================
# Config
# =============================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
API_KEY = os.getenv("API_KEY")  # Excel/Webから叩くための簡易認証

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required")

sb: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

TIME_LIMIT_MANUAL_SEC = 30.0
TIME_LIMIT_AUTO_PER_DAY_SEC = 50.0  # ★GAS 6分制限を考慮


# =============================
# Auth (simple API key)
# =============================
def require_api_key(x_api_key: Optional[str] = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


# =============================
# API Models
# =============================
class SolveWeekJobRequest(BaseModel):
    week_start_date: str = Field(..., description="YYYY-MM-DD (Monday)")
    K: int = Field(..., description="7 or 8")
    objective_mode: str = Field(..., description="maximize_count or maximize_minutes")


class SolveDayJobRequest(BaseModel):
    week_start_date: str
    date: str
    K: int
    objective_mode: str


class JobResponse(BaseModel):
    ok: bool
    status: str
    saved_count: int
    details: Dict[str, Any] = {}


# =============================
# Helpers
# =============================
def normalize_time_str(t: Optional[str]) -> Optional[str]:
    # "10:00:00" -> "10:00"
    if t is None:
        return None
    if len(t) >= 5 and t[2] == ":":
        return t[:5]
    return t

def normalize_end_time(end_time: str) -> str:
    return "24:00" if end_time == "23:59" else end_time

def hhmm_to_5min_index(hhmm: str) -> int:
    try:
        h, m = map(int, hhmm.split(":"))
    except Exception:
        raise ValueError(f"Invalid time format: {hhmm}")
    total = h * 60 + m
    if total < 0 or total > 24 * 60:
        raise ValueError(f"Out of range time: {hhmm}")
    if total % 5 != 0:
        raise ValueError(f"Time is not multiple of 5 minutes: {hhmm}")
    return total // 5

def validate_day_rows(rows: List[Dict[str, Any]]) -> None:
    seen = set()
    for r in rows:
        key = (r["user_id"], r["date"])
        if key in seen:
            raise ValueError(f"Duplicate (user_id, date) detected: {key}")
        seen.add(key)

        is_av = r["is_available"]
        st = r.get("start_time")
        et = r.get("end_time")

        if not is_av and (st is not None or et is not None):
            raise ValueError(f"Inconsistent row: is_available=false but has start/end. request_id={r['id']}")
        if is_av and (st is None or et is None):
            raise ValueError(f"Inconsistent row: is_available=true but start/end is null. request_id={r['id']}")

def fetch_requests_for_day(week_start_date: str, day: str) -> List[Dict[str, Any]]:
    res = (
        sb.table("shift_requests")
        .select("id,user_id,week_start_date,date,start_time,end_time,is_available,exit_by_end_time")
        .eq("week_start_date", week_start_date)
        .eq("date", day)
        .execute()
    )
    return res.data or []

def fetch_requests_for_week(week_start_date: str) -> List[Dict[str, Any]]:
    res = (
        sb.table("shift_requests")
        .select("id,user_id,week_start_date,date,start_time,end_time,is_available,exit_by_end_time")
        .eq("week_start_date", week_start_date)
        .execute()
    )
    return res.data or []

def delete_assignments_for_day(week_start_date: str, day: str) -> None:
    sb.table("shift_assignments").delete().eq("week_start_date", week_start_date).eq("date", day).execute()

def delete_assignments_for_week(week_start_date: str) -> None:
    sb.table("shift_assignments").delete().eq("week_start_date", week_start_date).execute()

def insert_assignments(assignments: List[Dict[str, Any]], K: int, objective_mode: str, generated_by: str) -> int:
    if not assignments:
        return 0
    now = datetime.utcnow().isoformat()
    payload = []
    for a in assignments:
        payload.append({
            **a,
            "max_rooms": K,
            "objective_mode": objective_mode,
            "generated_by": generated_by,
            "generated_at": now
        })
    res = sb.table("shift_assignments").insert(payload).execute()
    return len(res.data or [])


# =============================
# Solver (one day)
# =============================
def solve_one_day(rows: List[Dict[str, Any]], K: int, objective_mode: str, time_limit_sec: float) -> Tuple[str, List[Dict[str, Any]]]:
    if K not in (7, 8):
        raise ValueError("K must be 7 or 8")
    if objective_mode not in ("maximize_count", "maximize_minutes"):
        raise ValueError("objective_mode must be maximize_count or maximize_minutes")

    validate_day_rows(rows)

    # 候補抽出（is_available=trueのみ）
    candidates = []
    for r in rows:
        if not r["is_available"]:
            continue
        st = normalize_time_str(r["start_time"])
        et = normalize_time_str(r["end_time"])
        et = normalize_end_time(et)

        s = hhmm_to_5min_index(st)
        e = hhmm_to_5min_index(et)
        dur = e - s
        if dur <= 0:
            raise ValueError(f"Invalid duration start>=end. request_id={r['id']}")
        candidates.append((r, s, e, dur))

    model = cp_model.CpModel()
    y = {}
    x = []
    durations = []
    intervals_in_room = [[] for _ in range(K)]

    for idx, (r, s, e, dur) in enumerate(candidates):
        durations.append(dur)
        y_row = []
        for j in range(K):
            pres = model.NewBoolVar(f"y_{idx}_{j}")
            y[(idx, j)] = pres
            interval = model.NewOptionalIntervalVar(s, dur, e, pres, f"int_{idx}_{j}")
            intervals_in_room[j].append(interval)
            y_row.append(pres)

        model.Add(sum(y_row) <= 1)

        xr = model.NewBoolVar(f"x_{idx}")
        model.Add(xr == sum(y_row))
        x.append(xr)

    for j in range(K):
        model.AddNoOverlap(intervals_in_room[j])

    if objective_mode == "maximize_count":
        model.Maximize(sum(x))
    else:
        model.Maximize(sum(x[i] * durations[i] for i in range(len(x))))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_sec)
    solver.parameters.num_search_workers = 8

    status = solver.Solve(model)
    status_map = {
        cp_model.OPTIMAL: "OPTIMAL",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.MODEL_INVALID: "MODEL_INVALID",
        cp_model.UNKNOWN: "UNKNOWN",
    }
    status_str = status_map.get(status, "UNKNOWN")

    assignments: List[Dict[str, Any]] = []
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for idx, (r, s, e, dur) in enumerate(candidates):
            room_no = None
            for j in range(K):
                if solver.Value(y[(idx, j)]) == 1:
                    room_no = j + 1
                    break
            if room_no is not None:
                assignments.append({
                    "week_start_date": r["week_start_date"],
                    "date": r["date"],
                    "user_id": r["user_id"],
                    "request_id": r["id"],
                    "room_no": room_no,
                    "start_time": normalize_time_str(r["start_time"]),
                    "end_time": normalize_time_str(r["end_time"]),
                    "exit_by_end_time": r.get("exit_by_end_time"),  # ★追加
                })

    return status_str, assignments


# =============================
# FastAPI
# =============================
app = FastAPI(title="Shift Solver API (Supabase)", version="1.0.0")

@app.get("/health")
def health():
    return {"ok": True, "time": datetime.utcnow().isoformat()}

@app.post("/jobs/solve_day", response_model=JobResponse, dependencies=[Depends(require_api_key)])
def job_solve_day(req: SolveDayJobRequest):
    """
    手動：指定日を全作り直し（削除→insert）
    タイムリミット固定 30秒
    """
    try:
        rows = fetch_requests_for_day(req.week_start_date, req.date)
        if not rows:
            return JobResponse(ok=True, status="NO_DATA", saved_count=0, details={"date": req.date})

        status, assignments = solve_one_day(rows, req.K, req.objective_mode, TIME_LIMIT_MANUAL_SEC)

        delete_assignments_for_day(req.week_start_date, req.date)
        saved = insert_assignments(assignments, req.K, req.objective_mode, generated_by="manual_day")

        return JobResponse(ok=True, status=status, saved_count=saved,
                           details={"date": req.date, "time_limit_sec": TIME_LIMIT_MANUAL_SEC})
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

@app.post("/jobs/solve_week", response_model=JobResponse, dependencies=[Depends(require_api_key)])
def job_solve_week(req: SolveWeekJobRequest):
    """
    自動：週を全作り直し（週削除→日別solve→insert）
    タイムリミット固定：各日50秒
    """
    try:
        week_rows = fetch_requests_for_week(req.week_start_date)
        if not week_rows:
            return JobResponse(ok=True, status="NO_DATA", saved_count=0, details={"week_start_date": req.week_start_date})

        ws = date.fromisoformat(req.week_start_date)
        days = [(ws + timedelta(days=i)).isoformat() for i in range(7)]

        delete_assignments_for_week(req.week_start_date)

        saved_total = 0
        day_statuses: Dict[str, str] = {}

        for d in days:
            day_rows = [r for r in week_rows if r["date"] == d]
            if not day_rows:
                day_statuses[d] = "NO_DATA"
                continue

            st, assignments = solve_one_day(day_rows, req.K, req.objective_mode, TIME_LIMIT_AUTO_PER_DAY_SEC)
            day_statuses[d] = st
            saved_total += insert_assignments(assignments, req.K, req.objective_mode, generated_by="auto_week")

        rank = {"OPTIMAL": 4, "FEASIBLE": 3, "UNKNOWN": 2, "INFEASIBLE": 1, "MODEL_INVALID": 0, "NO_DATA": 5}
        worst_rank = min((rank.get(s, 2) for s in day_statuses.values()), default=2)
        inv = {v: k for k, v in rank.items()}
        week_status = inv.get(worst_rank, "UNKNOWN")

        return JobResponse(
            ok=True,
            status=week_status,
            saved_count=saved_total,
            details={
                "week_start_date": req.week_start_date,
                "time_limit_per_day_sec": TIME_LIMIT_AUTO_PER_DAY_SEC,
                "day_statuses": day_statuses
            }
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")