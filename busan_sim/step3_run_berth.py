"""3단계: 시나리오 A(기존)와 C(통합)를 돌려 선박별 접안·대기시간 차이를 저장한다."""
import pandas as pd                                        # 표 처리
import config                                              # 설정
from berth_sim import load_calls, monte_carlo             # 시뮬레이션 함수

calls = load_calls(config.OUT_DIR / f"berth_{config.YEAR}_loa.csv")   # 입력표
print("호출 수:", len(calls))

results = {}
for name, pools in config.SCENARIOS.items():               # 시나리오별 실행
    results[name] = monte_carlo(calls, pools)
    r = results[name]
    print(f"{name}: 평균 대기 {r['sim_wait_h'].mean():.2f}h, 대기 발생 {(r['sim_wait_h'] > 0.5).mean()*100:.1f}%, 타 터미널 이동 {r['moved'].mean()*100:.1f}%")

# A와 C를 선박 단위로 비교: 반복 평균 접안 대기시간과 C에서의 배정 터미널 최빈값
key = ["terminal", "vessel", "voyage", "etb"]
def per_ship(r):
    return r.groupby(key, dropna=False).agg(wait=("sim_wait_h", "mean")).reset_index()
a, c = per_ship(results["A_기존(터미널별)"]), per_ship(results["C_통합(신항 전체)"])
cmp = a.merge(c, on=key, suffixes=("_A", "_C"))
cmp["wait_diff_h"] = cmp["wait_C"] - cmp["wait_A"]         # 음수면 통합으로 대기가 줄어듦
cmp.to_csv(config.OUT_DIR / "berth_compare_A_vs_C.csv", index=False, encoding="utf-8-sig")
print("\n선박별 대기 변화(C-A) 요약")
print(cmp["wait_diff_h"].describe().round(2))
