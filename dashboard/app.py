"""부산 신항 환적 컨테이너 야드 체류시간 개선 시뮬레이션 대시보드.

시뮬레이션(최종 시뮬레이션.ipynb)이 저장한 결과표(data/)를 읽어서 보여준다.
앱 안에서 시뮬레이션을 다시 돌리지 않는다. 숫자는 모두 2025년 데이터 기준의 시뮬레이션 결과다.
"""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA = Path(__file__).parent / "data"
BLUE, ORANGE, GRAY, GREEN = "#2878d6", "#e8590c", "#868e96", "#2f9e44"
SCEN_LABEL = {"A": "A 기존 (터미널별)", "B": "B 인접 터미널 공유", "C": "C 신항 전체 통합"}
SCEN_COLOR = {"A": GRAY, "B": BLUE, "C": GREEN}
ADV = [6, 12, 24, 48, 72, 96, 168]                    # 시뮬레이션한 요청 앞당김(h)

st.set_page_config(page_title="부산 신항 환적 야드 체류 개선 시뮬레이션", page_icon="🚢", layout="wide")


# ───────────────────────── 데이터 읽기 ─────────────────────────
@st.cache_data
def load():
    d = {
        "sim": pd.read_csv(DATA / "v4_stepA.csv"),
        "hold": pd.read_csv(DATA / "v4_holdout_agg.csv"),
        "final": pd.read_csv(DATA / "final_summary.csv"),
        "sens": pd.read_csv(DATA / "sensitivity_config.csv"),
        "gapbins": pd.read_csv(DATA / "gap_bins.csv"),
        "vessels": pd.read_csv(DATA / "vessel_priority.csv"),
        "val_term": pd.read_csv(DATA / "validation_terminal.csv"),
        "kpi": pd.read_csv(DATA / "kpi_ABC.csv", index_col=0),
    }
    d["meta"] = json.load(open(DATA / "meta.json", encoding="utf-8"))
    return d


def pick(sim, policy, scen, a):
    """정책·시나리오·요청 앞당김에 맞는 결과 한 행."""
    r = sim[(sim["정책"] == policy) & (sim["시나리오"] == scen) & (sim["요청 앞당김(h)"] == a)]
    return r.iloc[0]


def fmt_h(x):
    return f"{x:,.1f}h"


D = load()
SIM, META = D["sim"], D["meta"]
BASE = META["기존 평균 야드 체류(h)"]

# ───────────────────────── 머리말 ─────────────────────────
st.title("🚢 부산 신항 환적 컨테이너 야드 체류시간 개선 시뮬레이션")
st.markdown(
    "**연구 질문** · 환적 컨테이너의 야드 체류시간을 줄이기 위해 **어떤 연결선의 스케줄을 우선 조정해야 하는가?**  \n"
    "개선의 초점은 *타부두 이동 자체*가 아니라 **모선과 연결선 사이의 스케줄 간격**이다."
)
st.caption(
    f"2025년 부산 신항 7개 터미널 선박 호출 {META['선박 호출 수']:,}건, 환적 컨테이너 {META['환적 컨테이너 수']:,}건(HJNC·DGT) 기준 · "
    "모든 수치는 시뮬레이션 결과이며 실제 적용 효과를 보장하지 않는다."
)

tab_sum, tab_sim, tab_abc, tab_ves, tab_val, tab_lim = st.tabs(
    ["① 요약", "② 연결선 조정 시뮬레이터", "③ A/B/C 선석 운영의 역할", "④ 우선 조정 연결선", "⑤ 검증·근거", "⑥ 한계와 결론"]
)

# ───────────────────────── ① 요약 ─────────────────────────
with tab_sum:
    sel48 = pick(SIM, "선별", "A", 48)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("기존 평균 야드 체류", fmt_h(BASE))
    c2.metric("선별 48h 조정 후", fmt_h(BASE * (1 - sel48["감소율(%)"] / 100)), f"-{sel48['감소율(%)']:.1f}%", delta_color="inverse")
    dA, dC = pick(SIM, "선별", "A", 48)["다른 선박 지연(h)"], pick(SIM, "선별", "C", 48)["다른 선박 지연(h)"]
    c3.metric("다른 선박 접안 지연 (A → C)", f"{dA:.2f}h → {dC:.2f}h", "선석 공유로 충돌 흡수", delta_color="off")
    rA, rC = pick(SIM, "선별", "A", 48)["실현 비율(%)"], pick(SIM, "선별", "C", 48)["실현 비율(%)"]
    c4.metric("요청 대비 조정 실현율 (A → C)", f"{rA:.0f}% → {rC:.0f}%")

    st.subheader("최종 결과표 (시나리오 A 기준 감소율)")
    f = D["final"].copy()
    f["감소율(%)"] = f["감소율(%)"].map(lambda v: f"{v:.2f}")
    st.dataframe(f, hide_index=True, use_container_width=True)

    left, right = st.columns([3, 2])
    with left:
        st.subheader("핵심 메시지")
        st.markdown(
            "1. **자부두/타부두 여부보다 연결선 간격이 체류를 좌우한다.** (체류시간과 간격의 상관 "
            f"{META['체류-간격 상관']:.3f})\n"
            "2. **전체 조정보다 선별 조정이 약 2배 효율적이다.** 실리는 환적 박스가 내리는 박스보다 많은 연결선만 조정한다.\n"
            "3. **선석 통합은 체류를 직접 줄이는 수단이 아니라, 스케줄 조정이 만드는 접안 충돌을 흡수하는 지원 수단이다.**\n"
            "4. 효과는 하반기 데이터에서도 유지됐고, 머신러닝으로 입력을 보완해도 결론은 같았다."
        )
    with right:
        st.subheader("제안하는 개선 순서")
        st.markdown(
            "1. 환적 연결선 모니터링 (대기 박스, 실릴/내릴 박스, 모선-연결선 간격)\n"
            "2. 우선 조정 선박 선정 (실리는 박스 − 내리는 박스 > 0, 효과 순)\n"
            "3. 연결선 스케줄 간격 축소 (가능한 범위 내)\n"
            "4. 선석 충돌 시뮬레이션 확인\n"
            "5. 충돌 시 인접 터미널 선석 활용"
        )
    st.info("**한 줄 결론** · 효과가 큰 연결선을 데이터로 먼저 선별해 스케줄 간격을 관리하고, 그 과정의 접안 충돌은 인접 터미널 선석 공유로 줄이는 방식이 현재 분석에서 가장 현실적인 개선 방향이다.")

# ───────────────────────── ② 시뮬레이터 ─────────────────────────
with tab_sim:
    st.subheader("연결선 스케줄 간격 축소 시뮬레이터")
    st.caption("연결선의 접안 일정을 요청한 시간만큼 앞당길 수 있다고 가정한 What-if 시나리오입니다. 앞당긴 뒤에도 모선과 연결선 사이 최소 24h는 남깁니다.")
    c1, c2, c3 = st.columns([2, 2, 3])
    with c1:
        policy = st.radio("조정 방식", ["선별", "전체"], horizontal=True, help="선별: 실리는 박스 > 내리는 박스인 연결선만 / 전체: 앞당길 수 있는 모든 연결선")
    with c2:
        scen = st.radio("선석 운영", list(SCEN_LABEL), format_func=lambda k: SCEN_LABEL[k], horizontal=False)
    with c3:
        a = st.select_slider("요청 앞당김 (h)", options=ADV, value=48)
    r = pick(SIM, policy, scen, a)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("평균 야드 체류", fmt_h(BASE * (1 - r["감소율(%)"] / 100)), f"-{r['감소율(%)']:.2f}%", delta_color="inverse")
    m2.metric("95% 구간 (부트스트랩)", f"{r['부트스트랩 하한']:.1f} ~ {r['부트스트랩 상한']:.1f}%")
    m3.metric("다른 선박 평균 접안 지연", f"{r['다른 선박 지연(h)']:.2f}h")
    m4.metric("요청 대비 실현율", f"{r['실현 비율(%)']:.0f}%")

    fig = go.Figure()
    for pol, col in [("전체", GRAY), ("선별", ORANGE)]:
        s = SIM[(SIM["정책"] == pol) & (SIM["시나리오"] == scen)].sort_values("요청 앞당김(h)")
        xs = s["요청 앞당김(h)"].tolist()
        rgb = tuple(int(col[i:i + 2], 16) for i in (1, 3, 5))
        fig.add_trace(go.Scatter(x=xs + xs[::-1], y=s["부트스트랩 상한"].tolist() + s["부트스트랩 하한"].tolist()[::-1], fill="toself",
                                 fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.15)", line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=xs, y=s["감소율(%)"], mode="lines+markers", name=f"{pol} 조정", line=dict(color=col, width=3)))
    fig.add_trace(go.Scatter(x=[a], y=[r["감소율(%)"]], mode="markers", marker=dict(size=14, color=BLUE, symbol="diamond"), name="선택", hoverinfo="skip"))
    fig.update_layout(title=f"요청 앞당김별 평균 야드 체류 감소율 · {SCEN_LABEL[scen]} (음영 = 95% 구간)", xaxis_title="요청 앞당김 (h)",
                      yaxis_title="감소율 (%)", height=420, margin=dict(t=60, b=40), legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 목표 감소율을 채우려면 얼마나 앞당겨야 하나")
    target = st.select_slider("목표 감소율 (%)", options=[1, 3, 5, 10, 15, 20, 25, 30], value=10)
    rows = []
    for pol in ("선별", "전체"):
        s = SIM[(SIM["정책"] == pol) & (SIM["시나리오"] == scen)].sort_values("요청 앞당김(h)")
        mean_ok = s[s["감소율(%)"] >= target]["요청 앞당김(h)"]
        sure_ok = s[s["부트스트랩 하한"] >= target]["요청 앞당김(h)"]
        rows.append({"조정 방식": pol, "평균적으로 달성하는 최소 앞당김": f"{int(mean_ok.min())}h" if len(mean_ok) else "168h 이내 불가",
                     "95% 구간 하한까지 달성하는 최소 앞당김": f"{int(sure_ok.min())}h" if len(sure_ok) else "168h 이내 불가"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("앞당김은 6·12·24·48·72·96·168h에서 시뮬레이션했습니다. 표의 값은 그중 처음으로 목표를 넘는 값이라 실제 필요 시간은 그 이하일 수 있습니다.")

# ───────────────────────── ③ A/B/C ─────────────────────────
with tab_abc:
    st.subheader("선석 운영 방식(A/B/C)의 역할")
    st.markdown("야드 체류 **감소율**은 A/B/C에서 거의 같습니다. 차이는 **다른 선박에 주는 지연**과 **요청한 조정이 실제로 실현되는 정도**에서 납니다.")
    c1, c2 = st.columns(2)
    with c1:
        pol3 = st.radio("조정 방식", ["선별", "전체"], horizontal=True, key="pol3")
    with c2:
        a3 = st.select_slider("요청 앞당김 (h)", options=ADV, value=48, key="a3")
    sub = {s: pick(SIM, pol3, s, a3) for s in "ABC"}

    def bar(col, title, unit, fmt):
        f = go.Figure(go.Bar(x=[SCEN_LABEL[s] for s in "ABC"], y=[sub[s][col] for s in "ABC"], marker_color=[SCEN_COLOR[s] for s in "ABC"],
                             text=[fmt.format(sub[s][col]) for s in "ABC"], textposition="outside"))
        f.update_layout(title=title, yaxis_title=unit, height=330, margin=dict(t=50, b=30), showlegend=False)
        return f

    b1, b2, b3 = st.columns(3)
    b1.plotly_chart(bar("감소율(%)", "야드 체류 감소율", "%", "{:.2f}%"), use_container_width=True)
    b2.plotly_chart(bar("다른 선박 지연(h)", "다른 선박 평균 접안 지연", "h", "{:.2f}h"), use_container_width=True)
    b3.plotly_chart(bar("실현 비율(%)", "요청 대비 조정 실현율", "%", "{:.0f}%"), use_container_width=True)
    st.success("**해석** · 선별 스케줄 조정이 체류를 줄이는 주 개선책이고, 선석 공유(B·C)는 조정 때문에 생기는 접안 충돌을 줄여 요청한 조정이 그대로 실현되게 하는 지원 수단입니다.")

    st.markdown("#### 앞당김 없이 선석 운영만 바꿨을 때의 접안 대기 (100회 반복 평균)")
    k = D["kpi"].loc[["평균 대기(h)", "P95 대기(h)", "타부두 이동 호출(%)"]].round(2)
    st.dataframe(k, use_container_width=True)
    st.caption("2025년 실제 접안 데이터 기준으로는 선석 대기가 원래 짧아(평균 1h 안팎) 통합만으로 줄일 여지가 작습니다. 실제 대기의 대부분은 선석 공간이 아닌 요인 때문입니다(⑤ 참고).")

# ───────────────────────── ④ 우선 조정 연결선 ─────────────────────────
with tab_ves:
    st.subheader("우선 조정 연결선 목록")
    st.caption("기대 이득(박스·h) = (실리는 환적 박스 − 내리는 박스) × 앞당길 수 있는 시간. 박스 수와 일정으로 계산한 순위이며, 시뮬레이션 절감은 선박 접안 충돌까지 반영한 값입니다.")
    V = D["vessels"].copy()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        a4 = st.select_slider("요청 앞당김 (h)", options=[24, 48, 96], value=48, key="a4")
    with c2:
        terms = sorted(V["터미널"].unique())
        sel_t = st.multiselect("터미널", terms, default=terms)
    with c3:
        only_sel = st.checkbox("선별 대상만 (순이득 > 0, 앞당길 수 있음)", value=True)
    with c4:
        topn = st.slider("상위 N척 표시", 10, 300, 50, step=10)
    gcol = f"기대 이득 {a4}h(박스·h)"
    V = V[V["터미널"].isin(sel_t)]
    if only_sel:
        V = V[V["선별 대상"] & (V["앞당김 가능 상한(h)"] > 0)]
    V = V.sort_values(gcol, ascending=False).reset_index(drop=True)
    st.markdown(f"조건에 맞는 연결선 **{len(V):,}척** · 상위 {min(topn, len(V))}척의 기대 이득은 전체의 "
                f"**{(V[gcol].head(topn).sum() / V[gcol].sum() * 100 if V[gcol].sum() > 0 else 0):.0f}%**")
    show = ["터미널", "선박", "항차", "선사", "접안", "실리는 환적 박스", "내리는 환적 박스", "순이득 박스", "앞당김 가능 상한(h)", gcol, "시뮬 절감 48h 선별(박스·h)"]
    st.dataframe(V[show].head(topn), hide_index=True, use_container_width=True)
    st.download_button("우선순위 표 내려받기 (CSV)", V[show].to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"), "priority_vessels.csv", "text/csv")

    if len(V) > 0 and V[gcol].sum() > 0:
        cum = V[gcol].cumsum() / V[gcol].sum() * 100
        fig = go.Figure(go.Scatter(x=(V.index + 1) / len(V) * 100, y=cum, mode="lines", line=dict(color=ORANGE, width=3)))
        fig.update_layout(title="상위 선박에 집중할수록 얼마나 얻는가 (기대 이득 누적 비중)", xaxis_title="우선순위 상위 비율 (%)",
                          yaxis_title="전체 기대 이득 중 누적 비중 (%)", height=360, margin=dict(t=50, b=40))
        st.plotly_chart(fig, use_container_width=True)
    H = D["hold"]
    h_all = H[(H["시나리오"] == "A") & (H["그룹"] == "H2 선별(순이득>0 전부)")]["하반기 감소율(%)"]
    h25 = H[(H["시나리오"] == "A") & (H["그룹"] == "박스순위 25%")]["하반기 감소율(%)"]
    if len(h_all) and len(h25):
        st.info(f"**하반기 검증** · 하반기 선별 대상 전부를 48h 조정하면 {h_all.iloc[0]:.1f}% 감소, 박스 순위 상위 25%만 조정해도 {h25.iloc[0]:.1f}% 감소했습니다. "
                "모든 연결선을 바꾸기보다 효과가 큰 일부를 우선 관리하는 방식이 현실적입니다.")

# ───────────────────────── ⑤ 검증·근거 ─────────────────────────
with tab_val:
    st.subheader("① 체류시간은 연결선 간격이 좌우한다")
    G = D["gapbins"]
    fig = go.Figure()
    for kind, col in zip(G["kind"].unique(), [BLUE, ORANGE, GREEN, GRAY]):
        g = G[G["kind"] == kind]
        fig.add_trace(go.Bar(x=g["간격 구간"], y=g["평균체류h"], name=kind, marker_color=col))
    fig.update_layout(barmode="group", title="연결선 간격 구간별 평균 야드 체류시간", xaxis_title="모선-연결선 접안 간격", yaxis_title="평균 야드 체류 (h)", height=380,
                      margin=dict(t=50, b=40), legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"체류시간과 연결선 간격의 상관계수 {META['체류-간격 상관']:.3f}. 같은 간격 구간에서는 자부두/타부두 사이 차이가 작습니다.")

    st.subheader("② 모델 검증: 선석 공간만으로 실제 대기를 설명하는가")
    v1, v2, v3 = st.columns(3)
    v1.metric(f"박지 대기 {META['박지 대기 호출 수']}건 실제 평균 대기", fmt_h(META["박지 대기 관측 평균(h)"]))
    v2.metric("같은 호출의 모델 대기", fmt_h(META["박지 대기 모델 평균(h)"]))
    v3.metric("모델이 설명하는 비율", f"{META['모델 설명 비율(%)']:.1f}%")
    st.warning("본 모델은 실제 지연 전체를 예측하는 모델이 아닙니다. 기상·도선·예선·선사 일정·크레인 등은 반영하지 못했으므로 A/B/C의 **절대 대기시간은 예측값이 아니라 선석 공간 조건만 바꿨을 때의 상대 비교**로만 읽어야 합니다.")
    st.dataframe(D["val_term"].rename(columns={"terminal": "터미널"}), hide_index=True, use_container_width=True)

    st.subheader("③ 불확실성: 100회 반복 + 부트스트랩")
    s = SIM[(SIM["시나리오"] == "A") & (SIM["요청 앞당김(h)"].isin([24, 48, 96, 168]))].sort_values(["정책", "요청 앞당김(h)"])
    st.dataframe(s[["정책", "요청 앞당김(h)", "감소율(%)", "반복 5%", "반복 95%", "부트스트랩 하한", "부트스트랩 상한", "확실 달성 목표(%)"]], hide_index=True, use_container_width=True)
    st.caption("반복 간 범위는 ±0.1%p로 작습니다. 불확실성은 대부분 '어떤 연결선이 표본에 포함되느냐'에서 나오며 부트스트랩 구간이 이를 나타냅니다.")

    st.subheader("④ 상반기 학습 → 하반기 검증 (48h, 시나리오 A)")
    H = D["hold"]
    Ha = H[H["시나리오"] == "A"].copy()
    st.dataframe(Ha[["그룹", "앞당긴 배", "하반기 감소율(%)", "배당 절감(박스·h)", "다른 선박 지연(h)"]].sort_values("앞당긴 배", ascending=False), hide_index=True, use_container_width=True)
    st.caption("박스 수 순위와 예측모델 순위의 성과가 같아서, 복잡한 예측 대신 단순한 박스 수 규칙을 유지합니다. 무작위로 고른 경우는 같은 비율에서 효과가 훨씬 낮습니다.")

    st.subheader("⑤ 설정 민감도")
    st.dataframe(D["sens"], hide_index=True, use_container_width=True)
    st.caption("선박 길이 여유, 교체 간격, 선석 수 제약을 바꿔도 A/B/C의 상대 관계가 유지되는지 확인한 결과입니다.")

    st.subheader("⑥ 머신러닝 입력 보완 (보조)")
    ml = META["ML"]
    m1, m2 = st.columns(2)
    m1.metric("점유시간 예측 오차(MAE)", f"{ml['점유시간']['그래디언트 부스팅']['MAE']:.2f}h", f"기존 {ml['점유시간']['기준(터미널 중앙값)']['MAE']:.2f}h", delta_color="off")
    m2.metric("선박 길이 예측 오차(MAE)", f"{ml['선박길이']['그래디언트 부스팅']['MAE']:.1f}m", f"기존 {ml['선박길이']['무작위(현재 방식)']['MAE']:.1f}m", delta_color="off")
    st.caption("머신러닝은 정책 선정이 아니라 시뮬레이션 입력(점유시간, 선박 길이 결측) 품질을 높이는 데만 썼으며, 입력을 보완해도 핵심 결론은 바뀌지 않았습니다.")

# ───────────────────────── ⑥ 한계와 결론 ─────────────────────────
with tab_lim:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("✅ 증명한 것")
        st.markdown(
            "- 선석 공간만으로 실제 선박 대기 전체를 설명할 수 없다.\n"
            "- 연결선 스케줄 간격은 환적 야드 체류를 개선할 수 있는 유효한 운영 변수다.\n"
            "- 전체 연결선을 조정하는 것보다 선별 조정이 효율적이다.\n"
            "- 선석 공유는 다른 선박의 접안 지연을 줄이고 조정 실현율을 높인다.\n"
            "- 선별 정책의 효과는 하반기 데이터에서도 유지됐다.\n"
            "- 머신러닝으로 입력값을 보완해도 핵심 결론은 유지됐다."
        )
    with c2:
        st.subheader("⚠️ 증명하지 못한 것")
        st.markdown(
            "- 실제 항만 지연의 전체 원인\n"
            "- 스케줄 간격과 체류시간의 완전한 인과관계\n"
            "- 선사가 실제로 24~96h 스케줄을 변경할 수 있는지\n"
            "- 다른 터미널 이용 시 실제 하역 생산성 변화\n"
            "- 실제 적용 시 시뮬레이션과 같은 감소율이 나타나는지"
        )
    st.subheader("한계")
    st.markdown(
        "- 분석 기간은 2025년 1개 연도이고, 환적 컨테이너는 HJNC·DGT 자료만 사용했습니다.\n"
        "- 실제 대기에는 기상·도선·예선·항로 혼잡·선사 일정·야드·크레인 요인이 있으나 반영하지 못했습니다.\n"
        "- 터미널 이동 시 크레인 수·생산성·야드 상태 변화는 기존 서비스시간으로 대체했습니다.\n"
        "- DGT 등 일부 자료에 결측이 있습니다. (물량 결측은 터미널 중앙값으로 보완)\n"
        "- 외부 혼잡 수준(2026년 8월 부산 접안 지연 8.7h)을 가정한 스트레스 테스트에서도 방향은 유지됐으나, 기간과 정의가 달라 최종 결론에는 쓰지 않았습니다."
    )
