"""2단계: 선박 제원(길이 등)을 선석배정 표에 붙인다. 결과는 berth_2025_loa.csv로 저장된다.

선박명을 대문자·영문숫자만 남기는 방식으로 정규화한 뒤 이름이 같은 선박끼리 연결한다.
"""
import pandas as pd                                        # 표 처리
import config                                              # 프로젝트 설정
from step1_load_berth import find_file                     # 1단계의 파일 찾기 함수 재사용


def name_key(s):
    """선박명 비교용 키: 대문자로 바꾸고 영문자와 숫자만 남긴다."""
    return s.astype(str).str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)


def load_vessel_specs():
    """선박 제원 엑셀을 읽어 선박명 키와 길이(LOA) 등을 가진 표를 만든다."""
    path = find_file(config.VESSEL_SPEC_KEYWORD, (".xlsx",))                # 선박 제원 파일 찾기
    d = pd.read_excel(path, sheet_name=config.VESSEL_SPEC_SHEET)          # 제원 시트 읽기
    d.columns = ["vessel", "terms", "type", "built", "GT", "DWT", "LOA", "beam",
                 "vf", "note", "imo", "status", "src"]                    # 열 이름을 알아보기 쉽게 바꾼다 (13열 가정)
    d["key"] = name_key(d["vessel"])                                       # 선박명 키 만들기
    d = d.dropna(subset=["LOA"]).drop_duplicates("key")                    # 길이가 있는 선박만, 같은 이름은 하나만
    print(f"선박 제원: {path.name} -> 길이가 있는 선박 {len(d)}척")
    return d[["key", "LOA", "beam", "GT", "DWT"]]


def attach_specs(berth):
    """선석배정 표에 선박 길이를 붙인다."""
    specs = load_vessel_specs()                                            # 제원 표
    berth = berth.copy()                                                   # 원본을 바꾸지 않도록 복사
    berth["key"] = name_key(berth["vessel"])                               # 선석배정 표에도 같은 방식으로 키 생성
    merged = berth.merge(specs, on="key", how="left")                      # 선박명 키로 연결 (없으면 빈 값)
    return merged.drop(columns="key")


if __name__ == "__main__":
    berth = pd.read_csv(config.OUT_DIR / f"berth_{config.YEAR}.csv",
                        parse_dates=["etb", "etd", "gate_open", "gate_cutoff", "work_start", "work_end"])   # 1단계 결과 읽기
    out_table = attach_specs(berth)
    out = config.OUT_DIR / f"berth_{config.YEAR}_loa.csv"
    out_table.to_csv(out, index=False, encoding="utf-8-sig")
    print("저장:", out, f"({len(out_table)}행)")
    g = out_table.groupby("terminal")["LOA"]                               # 터미널별 길이 통계
    print(pd.DataFrame({"호출": g.size(), "길이 매칭률(%)": (g.count() / g.size() * 100).round(0),
                       "중앙값(m)": g.median().round(0), "95%(m)": g.quantile(.95).round(0)}))
