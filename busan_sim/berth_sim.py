"""선석 배정 시뮬레이션 핵심 모듈 (연속 안벽 모델).

흐름: 선박을 도착(ready) 시각 순으로 처리 -> 풀 안에서 가장 빨리 들어갈 수 있는 자리를 고름
      -> 접안 시각 기록 -> 대기시간 = 접안 시각 - 도착 시각. 한 번 배정하면 바꾸지 않는다.
"""
import numpy as np                                         # 숫자 계산·난수
import pandas as pd                                        # 표 처리
import config                                              # 설정


def load_calls(path):
    """step2 결과(berth_2025_loa.csv)를 읽어 시뮬레이션 입력표를 만든다."""
    b = pd.read_csv(path, parse_dates=["etb", "etd"])      # 접안·출항 시각을 날짜로 읽기
    b["ready"] = b["etb"]                                  # 도착 시각: 우선 계획 접안(ETB)으로 둔다 (포트미스 박지 정보는 이후 단계에서 보강)
    b["service_h"] = (b["etd"] - b["etb"]).dt.total_seconds() / 3600   # 안벽 점유시간(시간)
    b = b[b["service_h"] > 0]                              # 출항이 접안보다 빠른 이상 행 제거
    return b.sort_values("ready").reset_index(drop=True)   # 도착 순서로 정렬


def fill_loa(calls, rng):
    """선박 길이가 비어 있는 호출을 같은 터미널의 길이 분포에서 무작위로 채워 배열로 돌려준다."""
    loa = calls["LOA"].copy()                              # 원래 길이
    for term, g in calls.groupby("terminal"):              # 터미널별로
        known = g["LOA"].dropna().to_numpy()               # 알려진 길이들
        empty = g.index[g["LOA"].isna()]                   # 비어 있는 행 번호
        if len(empty) and len(known):
            loa.loc[empty] = rng.choice(known, size=len(empty))   # 알려진 길이에서 뽑아 채운다
    return loa.fillna(loa.median()).to_numpy()             # 혹시 남은 빈칸은 전체 중앙값으로


def find_slot(busy, s, e, need, quay):
    """한 터미널 안벽에서 시간 [s, e) 동안 길이 need(m)가 들어갈 위치를 찾는다. 없으면 None."""
    segs = sorted((x0, x1) for (t0, t1, x0, x1) in busy if t0 < e and t1 > s)   # 그 시간대에 점유된 안벽 구간
    pos = 0.0                                              # 지금까지 확인한 안벽 위치
    for x0, x1 in segs:                                    # 안벽 앞쪽부터 빈틈 탐색
        if x0 - pos >= need:                               # 점유 구간 앞의 틈에 들어가면
            return pos
        pos = max(pos, x1)                                 # 못 들어가면 그 구간 뒤로 이동
    return pos if quay - pos >= need else None             # 마지막 틈 확인


def run_scenario(calls, pools, loa):
    """시나리오 한 번 실행. pools는 같은 풀로 묶인 터미널 목록의 목록."""
    t0 = calls["ready"].min()                              # 시간 기준점
    ready = (calls["ready"] - t0).dt.total_seconds().to_numpy() / 3600   # 도착 시각(시간 단위)
    hold = calls["service_h"].to_numpy() + config.CHANGEOVER_H           # 점유시간 + 교체 간격
    own = calls["terminal"].to_numpy()                     # 원래 터미널
    pool_of = {t: p for p in pools for t in p}             # 터미널 -> 속한 풀
    busy = {t: [] for t in config.TERMINALS}               # 터미널별 점유 기록 (시작, 끝, 위치0, 위치1)
    start = np.zeros(len(calls))                           # 접안 시각 결과
    assigned = np.empty(len(calls), dtype=object)          # 배정된 터미널 결과
    for j in range(len(calls)):                            # 도착 순서대로 한 척씩
        need = loa[j] * config.LOA_MARGIN                  # 필요한 안벽 길이
        cands = pool_of[own[j]]                            # 쓸 수 있는 터미널 후보
        times = {ready[j]}                                 # 접안 후보 시각: 도착 즉시 + 기존 점유가 끝나는 시각들
        for t in cands:
            busy[t] = [b for b in busy[t] if b[1] > ready[j]]   # 이미 끝난 점유는 정리
            times.update(b[1] for b in busy[t])
        placed = None
        for s in sorted(times):                            # 이른 시각부터 시도
            opts = []
            for t in cands:
                if need > config.TERMINALS[t]["quay_m"]:   # 안벽보다 긴 배는 불가
                    continue
                x = find_slot(busy[t], s, s + hold[j], need, config.TERMINALS[t]["quay_m"])
                if x is not None:
                    opts.append((0 if t == own[j] else 1, t, x))   # 원래 터미널을 우선(0)
            if opts:
                placed = (s,) + min(opts)[1:]              # 우선순위 가장 높은 후보
                break
        s, t, x = placed
        start[j], assigned[j] = s, t
        busy[t].append((s, s + hold[j], x, x + need))      # 점유 기록
    out = calls[["terminal", "vessel", "voyage", "etb", "ready", "service_h", "discharge", "load"]].copy()
    out["assigned"] = assigned                             # 배정된 터미널
    out["moved"] = out["assigned"] != out["terminal"]      # 원래와 다르면 이동
    out["sim_wait_h"] = start - ready                      # 시뮬레이션 대기시간
    out["sim_etb"] = t0 + pd.to_timedelta(start, unit="h") # 시뮬레이션 접안 시각
    return out


def monte_carlo(calls, pools, reps=config.MC_REPS):
    """길이 채우기를 reps번 다르게 해서 반복 실행한 결과를 모두 합쳐 돌려준다."""
    rng = np.random.default_rng(config.SEED)               # 재현 가능한 난수
    res = []
    for r in range(reps):
        o = run_scenario(calls, pools, fill_loa(calls, rng))
        o["rep"] = r
        res.append(o)
    return pd.concat(res, ignore_index=True)
