"""1단계: 7개 터미널의 선석배정 파일을 읽어 하나의 표(berth_2025.csv)로 합친다.

터미널마다 열 이름과 형식이 달라서, 터미널별로 읽는 함수를 따로 두고
모두 같은 열 이름(공통 열)으로 바꿔 놓는다.
공통 열: terminal, berth, vessel, voyage, carrier, route, gate_open, gate_cutoff,
        etb(접안 시각), etd(출항 시각), work_start, work_end, discharge(양하), load(적하), shift, status
"""
import io                                                  # 문자열을 파일처럼 읽을 때 사용
import pandas as pd                                        # 표 처리 라이브러리
import config                                              # 프로젝트 설정


def find_file(keyword, extensions, folder=config.DATA_DIR):
    """data 폴더에서 이름에 keyword가 들어 있고 확장자가 맞는 파일을 정확히 하나 찾는다."""
    hits = [p for p in folder.iterdir()                    # 폴더 안의 모든 파일을 하나씩 보면서
            if p.suffix.lower() in extensions              # 확장자가 맞고
            and keyword.lower() in p.name.lower()          # 이름에 찾는 글자가 들어 있으며
            and not p.name.startswith("~$")]               # 엑셀 임시 파일(~$로 시작)이 아닌 것만 모은다
    if len(hits) != 1:                                     # 0개이거나 2개 이상이면 어느 파일인지 알 수 없다
        raise FileNotFoundError(f"'{keyword}' 파일이 {len(hits)}개 발견됨: {[p.name for p in hits]}  (config.py의 키워드를 확인하세요)")
    return hits[0]                                         # 정확히 하나일 때만 그 파일을 돌려준다


def num(s):
    """쉼표·공백·'-'가 섞인 열을 숫자로 바꾼다. 바꿀 수 없으면 빈 값(NaN)이 된다."""
    cleaned = s.astype(str).str.replace(",", "").str.strip().replace({"null": None, "-": None, "nan": None})
    return pd.to_numeric(cleaned, errors="coerce")


def dt(s):
    """문자열 열을 날짜·시각으로 바꾼다. 바꿀 수 없으면 빈 값(NaT)이 된다."""
    return pd.to_datetime(s.replace({"-": None, "null": None}), errors="coerce")


def read_html_table(path):
    """확장자는 .xls인데 실제로는 HTML 표인 파일(HPNT, PNIT)을 읽는다. 가장 큰 표를 쓴다."""
    text = open(path, encoding="utf-8").read()             # 파일을 글자로 읽고
    tables = pd.read_html(io.StringIO(text), flavor="lxml")   # HTML 안의 표를 모두 찾아서
    return max(tables, key=len)                            # 행이 가장 많은 표를 고른다


# ---------------------------------------------------------------------------
# 터미널별 읽기 함수 (열 이름이 터미널마다 달라서 하나씩 연결한다)
# ---------------------------------------------------------------------------
def load_hjnc(path):
    d = pd.read_excel(path, header=1)                      # 첫 줄은 제목이라 두 번째 줄을 열 이름으로 사용
    return pd.DataFrame({
        "terminal": "HJNC", "berth": d["선석"].astype(str), "vessel": d["선박명"], "voyage": d["모선항차"],
        "carrier": d["선사"], "route": d["항로"],
        "gate_open": dt(d["반입 시작일시"]), "gate_cutoff": dt(d["반입 마감일시"]),
        "etb": dt(d["입항일시"]), "etd": dt(d["출항일시"]),            # HJNC는 접안 시각이 '입항일시' 열에 있다
        "work_start": dt(d["작업 시작일시"]), "work_end": dt(d["작업 완료일시"]),
        "discharge": num(d["양하"]), "load": num(d["선적"]), "shift": num(d["S/H"]), "status": None})


def load_hpnt(path):
    d = read_html_table(path)                              # HPNT는 HTML 표 형식
    return pd.DataFrame({
        "terminal": "HPNT", "berth": d["선석"].astype(str), "vessel": d["선명"], "voyage": d["모선항차"],
        "carrier": d["선사"], "route": d["ROUTE"],
        "gate_open": dt(d["반입시작일시"]), "gate_cutoff": dt(d["반입마감시한"]),
        "etb": dt(d["접안(예정)일시"]), "etd": dt(d["출항(예정)일시"]),
        "work_start": pd.NaT, "work_end": pd.NaT,           # 이 터미널에는 작업 시각 열이 없다
        "discharge": num(d["양하"]), "load": num(d["적하"]), "shift": num(d["Shift"]), "status": d["상태"]})


def load_pnit(path):
    d = read_html_table(path)                              # PNIT도 HTML 표 형식
    return pd.DataFrame({
        "terminal": "PNIT", "berth": d["선석"].astype(str), "vessel": d["선명"], "voyage": d["모선항차"],
        "carrier": d["선사"], "route": d["ROUTE"],
        "gate_open": pd.NaT, "gate_cutoff": dt(d["반입마감시한"]),
        "etb": dt(d["접안(예정)일시"]), "etd": dt(d["출항(예정)일시"]),
        "work_start": pd.NaT, "work_end": pd.NaT,
        "discharge": num(d["양하"]), "load": num(d["적하"]), "shift": num(d["Shift"]), "status": d["상태"]})


def load_bct(path):
    d = pd.read_excel(path)
    return pd.DataFrame({
        "terminal": "BCT", "berth": d["선석"].astype(str), "vessel": d["모선명"], "voyage": d["모선/항차"],
        "carrier": d["선사"], "route": d["ROUTE"],
        "gate_open": pd.NaT, "gate_cutoff": dt(d["CCT"]),
        "etb": dt(d["접안예정시간(ETB)"]), "etd": dt(d["출항예정시간(ETD)"]),
        "work_start": pd.NaT, "work_end": pd.NaT,
        "discharge": num(d["양하"]), "load": num(d["적하"]), "shift": num(d["이적"]), "status": d["상태"]})


def load_bnct(path):
    d = pd.read_excel(path)
    c = list(d.columns)                                    # BNCT는 열 이름이 길고 복잡해서 위치(순서)로 읽는다
    wl = d[c[7]].astype(str).str.split("/", expand=True)   # '양하/적하/이적' 형태로 합쳐진 열을 나눈다
    return pd.DataFrame({
        "terminal": "BNCT", "berth": d[c[0]].astype(str),
        "vessel": d[c[3]].astype(str).str.replace(r"\(.*\)$", "", regex=True).str.strip(),   # 선박명 뒤 괄호(항로) 제거
        "voyage": d[c[2]].astype(str).str.extract(r"^(\S+)")[0], "carrier": d[c[1]],
        "route": d[c[3]].astype(str).str.extract(r"\(([^()]*)\)$")[0],                       # 괄호 안 글자를 항로로 사용
        "gate_open": pd.NaT, "gate_cutoff": dt(d[c[4]]), "etb": dt(d[c[5]]), "etd": dt(d[c[6]]),
        "work_start": pd.NaT, "work_end": pd.NaT,
        "discharge": num(wl[0]), "load": num(wl[1]), "shift": num(wl[2]), "status": d[c[8]]})


def load_pnc(path):
    d = pd.read_excel(path, sheet_name="2025 선석배정현황")   # 여러 시트 중 선석배정 시트만 읽는다
    return pd.DataFrame({
        "terminal": "PNC", "berth": d["선석"].astype(str), "vessel": d["모선명"], "voyage": d["모선코드"],
        "carrier": d["운항선사"], "route": d["항로"],
        "gate_open": pd.NaT, "gate_cutoff": dt(d["반입마감일시"]),
        "etb": dt(d["접안(예정)일시"]), "etd": dt(d["출항(예정)일시"]),
        "work_start": pd.NaT, "work_end": pd.NaT,
        "discharge": num(d["양하수량"]), "load": num(d["선적수량"]), "shift": num(d["Shift"]), "status": None})


def load_dgt(path):
    d = pd.read_excel(path, sheet_name="2025 선석배정현황")
    c = list(d.columns)
    wl = d[c[12]].astype(str).str.split("/", expand=True)  # 양하/적하/이적이 합쳐진 열(13번째)
    return pd.DataFrame({
        "terminal": "DGT", "berth": d["선석"].astype(str), "vessel": d["모선명"], "voyage": d[c[2]],
        "carrier": d["선사코드"], "route": d["항로"],
        "gate_open": dt(d["반입시작시간"]), "gate_cutoff": dt(d["반입마감시간"]),
        "etb": dt(d["접안예정일시"]), "etd": dt(d["출항예정일시"]),
        "work_start": dt(d["작업시작시간"]), "work_end": dt(d["작업완료시간"]),
        "discharge": num(wl[0]), "load": num(wl[1]), "shift": num(wl[2]), "status": d["상태"]})


LOADERS = {"HJNC": load_hjnc, "HPNT": load_hpnt, "PNIT": load_pnit, "BCT": load_bct,
           "BNCT": load_bnct, "PNC": load_pnc, "DGT": load_dgt}


def build_berth_table():
    """7개 터미널을 모두 읽어 합치고 2025년 접안 건만 남긴 표를 돌려준다."""
    frames = []                                            # 터미널별 표를 모을 목록
    for term, loader in LOADERS.items():                   # 터미널을 하나씩
        path = find_file(config.BERTH_FILE_KEYWORDS[term], config.BERTH_EXTENSIONS)   # 파일을 찾고
        df = loader(path)                                  # 읽어서
        print(f"{term}: {path.name} -> {len(df)}행")        # 몇 행인지 확인 출력
        frames.append(df)
    allb = pd.concat(frames, ignore_index=True)            # 7개 표를 위아래로 이어 붙인다
    allb = allb.dropna(subset=["etb", "etd"])              # 접안·출항 시각이 없는 행은 시뮬레이션에 못 쓰므로 제외
    allb = allb[allb["etb"].dt.year == config.YEAR].copy() # 분석 대상 연도(2025년)만
    return allb.reset_index(drop=True)


if __name__ == "__main__":                                 # 이 파일을 직접 실행했을 때만 아래가 돈다
    table = build_berth_table()
    out = config.OUT_DIR / f"berth_{config.YEAR}.csv"
    table.to_csv(out, index=False, encoding="utf-8-sig")   # 한글이 깨지지 않게 utf-8-sig로 저장
    print("\n저장:", out, f"({len(table)}행)")
    print("\n터미널별 호출 수와 접안 시각 범위")
    print(table.groupby("terminal").agg(rows=("vessel", "size"), first=("etb", "min"), last=("etb", "max")))
    print("\n열별 결측률(%)")
    cols = ["gate_open", "gate_cutoff", "etb", "etd", "work_start", "work_end", "discharge", "load"]
    print((table[cols].isna().groupby(table["terminal"]).mean() * 100).round(0))
