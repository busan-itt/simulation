"""4단계: 선석 배정이 A(기존)에서 C(통합)로 바뀔 때 HJNC 자부두 환적 컨테이너의 야드 체류시간 변화를 계산한다.

체류시간 = 반출시간 - 반입시간.
 - 모선(양하모선): 컨테이너를 내려놓는 배 -> 반입시간이 모선 접안 시각과 같이 움직인다고 본다.
 - 연결선(적하모선): 컨테이너를 싣는 배 -> 반출시간이 연결선 접안 시각과 같이 움직인다고 본다.
 새 체류시간 = 기존 체류시간 + (연결선 접안 변화) - (모선 접안 변화)
 두 배가 같은 만큼 움직이면 효과가 상쇄된다. 그래서 두 가지를 모두 보고한다.
   (가) 연결선만 이동  (나) 모선·연결선 모두 이동 (기준)
"""
import numpy as np                                         # 숫자 계산
import pandas as pd                                        # 표 처리
import config                                              # 설정
from step1_load_berth import find_file                     # 파일 찾기
from berth_sim import load_calls, run_scenario, fill_loa   # 시뮬레이션 함수


def vkey(v):
    """항차 표기를 통일한다. 'SMPL-006/2025'와 'SMPL-2025-0006'을 둘 다 'SMPL-2025-6'으로 바꾼다."""
    s = str(v).upper().strip()
    import re
    m = re.match(r"^([A-Z0-9]+)-(\d+)/(\d{4})$", s)        # 컨테이너 파일 형식: 코드-번호/연도
    if m:
        return f"{m[1]}-{m[3]}-{int(m[2])}"
    m = re.match(r"^([A-Z0-9]+)-(\d{4})-(\d+)$", s)        # 선석배정 파일 형식: 코드-연도-번호
    if m:
        return f"{m[1]}-{m[2]}-{int(m[3])}"
    return s


def sim_etb_table(calls, pools):
    """시나리오를 반복 실행해 선박(터미널·선박명·항차)별 평균 시뮬레이션 접안 시각과 이동 여부를 구한다."""
    rng = np.random.default_rng(config.SEED)
    res = []
    for r in range(config.MC_REPS):
        o = run_scenario(calls, pools, fill_loa(calls, rng))
        o["h"] = (o["sim_etb"] - calls["ready"].min()).dt.total_seconds() / 3600   # 기준점부터 시간
        res.append(o)
    r = pd.concat(res)
    r["vk"] = r["vessel"].str.upper().str.replace(r"[^A-Z0-9]", "", regex=True) + "|" + r["voyage"].map(vkey)   # 선박 키
    r = r[r["terminal"] == "HJNC"]                          # HJNC 자부두 환적이므로 HJNC 호출만
    return r.groupby("vk").agg(h=("h", "mean"), moved=("moved", "mean")), calls["ready"].min()


if __name__ == "__main__":
    calls = load_calls(config.OUT_DIR / f"berth_{config.YEAR}_loa.csv")
    A, t0 = sim_etb_table(calls, config.SCENARIOS["A_기존(터미널별)"])
    C, _ = sim_etb_table(calls, config.SCENARIOS["C_통합(신항 전체)"])
    ship = A.join(C, lsuffix="_A", rsuffix="_C")            # 선박별 A, C 시뮬레이션 접안 시각
    ship["delta_h"] = ship["h_C"] - ship["h_A"]             # 접안 시각 변화(시간). 음수 = C에서 더 일찍 접안

    ts = pd.read_csv(find_file(config.TS_FILE_KEYWORD, (".csv",)), encoding="utf-8-sig",
                     parse_dates=["반입시간", "반출시간"])     # HJNC 환적 컨테이너
    norm = lambda s: s.astype(str).str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
    ts["mk"] = norm(ts["양하모선명"]) + "|" + ts["양하모선항차"].map(vkey)   # 모선 키
    ts["ck"] = norm(ts["적하모선명"]) + "|" + ts["적하모선항차"].map(vkey)   # 연결선 키
    ts = ts.join(ship[["delta_h", "moved_C"]].add_prefix("m_"), on="mk")     # 모선의 접안 변화
    ts = ts.join(ship[["delta_h", "moved_C"]].add_prefix("c_"), on="ck")     # 연결선의 접안 변화
    ts = ts.dropna(subset=["m_delta_h", "c_delta_h"]).copy()
    print(f"환적 컨테이너 {len(ts)}개 (모선·연결선 모두 매칭된 건)")

    ts["dwell_h"] = (ts["반출시간"] - ts["반입시간"]).dt.total_seconds() / 3600    # 기존 체류시간
    move_h = config.TERMINAL_MOVE_MIN / 60 * ts["c_moved_C"]                         # 연결선이 이동한 비율만큼 이동시간 가산
    ts["new_conn_only"] = (ts["dwell_h"] + ts["c_delta_h"] + move_h).clip(lower=0)  # (가) 연결선만 이동
    ts["new_both"] = (ts["dwell_h"] + ts["c_delta_h"] - ts["m_delta_h"] + move_h).clip(lower=0)   # (나) 모선도 이동

    # 선박 쌍(모선, 연결선) 단위 체류시간 = 마지막 반출 - 첫 반입 (정의대로 집계)
    g = ts.groupby(["mk", "ck"])
    pair = pd.DataFrame({"first_in": g["반입시간"].min(), "last_out": g["반출시간"].max(), "n": g.size(),
                         "dwell_conn_only_change": g["new_conn_only"].mean() - g["dwell_h"].mean(),
                         "dwell_both_change": g["new_both"].mean() - g["dwell_h"].mean()})
    pair["dwell_h"] = (pair["last_out"] - pair["first_in"]).dt.total_seconds() / 3600
    pair.to_csv(config.OUT_DIR / "ts_dwell_pairs.csv", encoding="utf-8-sig")
    ts.to_csv(config.OUT_DIR / "ts_dwell_containers.csv", index=False, encoding="utf-8-sig")

    print("\n컨테이너 평균 체류시간(시간)")
    print(f"  기존 A            : {ts['dwell_h'].mean():.1f}")
    print(f"  (가) 연결선만 이동 : {ts['new_conn_only'].mean():.1f}  (변화 {ts['new_conn_only'].mean()-ts['dwell_h'].mean():+.2f})")
    print(f"  (나) 모선·연결선   : {ts['new_both'].mean():.1f}  (변화 {ts['new_both'].mean()-ts['dwell_h'].mean():+.2f})")
    print("\n접안 변화 분포(시간): 연결선 / 모선")
    print(ts[["c_delta_h", "m_delta_h"]].describe().round(2))
