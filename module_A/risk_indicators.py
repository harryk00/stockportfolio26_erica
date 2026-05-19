import io
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as web
import requests
import warnings
warnings.filterwarnings('ignore')

# SSL 경고 메시지 숨기기 (macOS 인증서 우회)
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ================================================
# 기본 설정
# ================================================

START_DATE = "2022-01-01"
END_DATE   = "2024-12-31"

FRED_SERIES = {
    "T10Y2Y":       "YIELD_SPREAD",  # 장단기 금리차 (10Y-2Y, 일별) [선행]
    "CPIAUCSL":     "CPI",           # 소비자물가지수 (월별 → YoY) [선행]
    "BAMLH0A0HYM2": "HY_SPREAD",     # 하이일드 스프레드 (일별)    [후행]
}

YF_SERIES = {
    "^VIX":   "VIX",    # VIX 변동성 지수  [동행]
    "^SKEW":  "SKEW",   # SKEW 지수        [동행]
    "^VIX3M": "VIX3M",  # 3개월 VIX        [동행]
}


# ================================================
# [선행] FRED 핵심 거시지표 수집
# ================================================

def get_fred_indicators(
    series: dict, start: str, end: str
) -> pd.DataFrame:
    frames = []

    for series_id, name in series.items():
        print(f"  FRED 수집 중: {name} ({series_id})")
        try:
            df = web.DataReader(series_id, "fred", start=start, end=end)
            df.columns = [name]
            df.index = pd.to_datetime(df.index).tz_localize(None)

            if name == "CPI":
                df[name] = df[name].pct_change(12) * 100
                print(f"    → CPI YoY 변화율로 변환")

            frames.append(df)
            print(f"    ✅ {name}: {len(df)}행 수집 완료")

        except Exception as e:
            print(f"    ⚠️ {name} 수집 실패: {e}")

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, axis=1)
    full_idx = pd.date_range(start=start, end=end, freq="B")
    result = result.reindex(full_idx).ffill()
    return result


# ================================================
# [동행] yfinance 시장 심리 지표 수집
# ================================================

def get_market_indicators(
    series: dict, start: str, end: str
) -> pd.DataFrame:
    frames = []

    for ticker, name in series.items():
        print(f"  시장 지표 수집 중: {name} ({ticker})")
        try:
            raw = yf.download(
                ticker, start=start, end=end,
                auto_adjust=True, progress=False
            )

            if isinstance(raw.columns, pd.MultiIndex):
                raw = raw["Close"]
                raw = raw.rename(columns={ticker: name})
            else:
                raw = raw[["Close"]].rename(columns={"Close": name})

            raw.index = pd.to_datetime(raw.index).tz_localize(None)
            frames.append(raw)
            print(f"    ✅ {name}: {len(raw)}행 수집 완료")

        except Exception as e:
            print(f"    ⚠️ {name} 수집 실패: {e}")

    return pd.concat(frames, axis=1) if frames else pd.DataFrame()


# ================================================
# [선행] OECD CLI (서버 차단으로 현재 수집 불가)
# ================================================

def get_oecd_cli(start: str, end: str) -> pd.DataFrame:
    print("  OECD CLI: 서버 차단(403)으로 건너뜁니다")
    return pd.DataFrame()


# ================================================
# [동행] VIX 백워데이션 계산
# ================================================

def calc_vix_backwardation(indicators: pd.DataFrame) -> pd.DataFrame:
    if "VIX" in indicators.columns and "VIX3M" in indicators.columns:
        indicators["VIX_BKWD"] = indicators["VIX"] - indicators["VIX3M"]
        print("  ✅ VIX 백워데이션 계산 완료")
    else:
        print("  ⚠️ VIX3M 없음, 백워데이션 계산 건너뜀")
    return indicators


# ================================================
# [선행] T5YIE + ICSA 수집 및 경제 국면 분류
# ================================================

def get_macro_regime_indicators(
    start: str = START_DATE,
    end:   str = END_DATE
) -> pd.DataFrame:
    """
    T5YIE (5년 기대인플레이션) + ICSA (신규 실업수당 청구)를
    수집하고 아래 매트릭스로 경제 국면을 분류합니다.

              T5YIE
              낮음(<1.5%)  중립(1.8~2.3%)  높음(>2.5%)
    ────────────────────────────────────────────────
    낮음       디플레       골디락스 ★      과열
    ICSA(<22만)
    높음       경기침체     둔화/주의        스태그플레이션
    ICSA(>25만)
    ────────────────────────────────────────────────
    """
    print("\n" + "=" * 50)
    print("경제 국면 지표 수집 (T5YIE + ICSA)")
    print("=" * 50)

    full_idx = pd.date_range(start=start, end=end, freq="B")

    # ------------------------------------------------
    # T5YIE (5년 기대인플레이션, 일별)
    # ------------------------------------------------
    print("\n[T5YIE] 5년 기대인플레이션")
    try:
        t5yie = web.DataReader("T5YIE", "fred", start=start, end=end)
        t5yie.columns = ["T5YIE"]
        t5yie.index = pd.to_datetime(t5yie.index).tz_localize(None)
        t5yie = t5yie.reindex(full_idx).ffill().bfill()
        print(f"  ✅ T5YIE: {len(t5yie)}행 | 최근값: {t5yie['T5YIE'].iloc[-1]:.2f}%")
    except Exception as e:
        print(f"  ⚠️ T5YIE 수집 실패: {e}")
        return pd.DataFrame()

    # ------------------------------------------------
    # ICSA (신규 실업수당 청구, 주별 → 일별)
    # FIX: reindex 전에 resample('D')로 일별 먼저 변환
    #      주별 데이터를 바로 영업일 reindex하면
    #      날짜 불일치로 NaN 발생하는 문제 해결
    # ------------------------------------------------
    print("\n[ICSA] 신규 실업수당 청구")
    try:
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=ICSA"

        # SSL 인증서 검증 우회 (macOS 환경 대응)
        response = requests.get(url, verify=False, timeout=30)
        response.raise_for_status()

        # FRED는 결측값을 '.'으로 표기 → NaN으로 처리
        icsa_raw = pd.read_csv(
            io.StringIO(response.text),
            index_col=0,
            parse_dates=True,
            na_values=['.', 'NA', 'N/A', '']
        )
        icsa_raw.columns = ["ICSA"]
        icsa_raw.index = pd.to_datetime(icsa_raw.index).tz_localize(None)

        # 기간 필터링 (Timestamp 명시적 사용)
        icsa_raw = icsa_raw.loc[
            (icsa_raw.index >= pd.Timestamp(start)) &
            (icsa_raw.index <= pd.Timestamp(end))
        ].copy()

        # 유효 데이터 확인
        valid_count = icsa_raw["ICSA"].notna().sum()
        sample_val  = icsa_raw["ICSA"].dropna().iloc[0]
        print(f"    데이터 확인: {len(icsa_raw)}주, 유효값 {valid_count}개, 샘플: {sample_val:.0f}")

        if valid_count == 0:
            raise ValueError("유효한 ICSA 데이터 없음")

        # 단위 자동 감지 후 만명 변환
        if sample_val > 1000:
            icsa_raw["ICSA"] = icsa_raw["ICSA"] / 10000  # 명 → 만명
        else:
            icsa_raw["ICSA"] = icsa_raw["ICSA"] / 10     # 천명 → 만명

        # FIX: 주별 → 일별 변환
        # 기존: reindex(영업일) → 주별 날짜가 안 맞아서 NaN 발생
        # 수정: resample('D').ffill()로 일별 먼저 채운 후 영업일 reindex
        icsa_daily = icsa_raw.resample('D').ffill()
        icsa_daily = icsa_daily.reindex(full_idx).ffill().bfill()

        print(f"  ✅ ICSA: {len(icsa_daily)}행 | 최근값: {icsa_daily['ICSA'].iloc[-1]:.1f}만명")
        icsa = icsa_daily

    except Exception as e:
        print(f"  ⚠️ ICSA 수집 실패: {e}")
        return pd.DataFrame()

    # ------------------------------------------------
    # 구간 분류 함수
    # ------------------------------------------------
    def classify_t5yie(val: float) -> str:
        if val < 1.5:    return "LOW"
        elif val <= 2.3: return "NEUTRAL"
        else:            return "HIGH"

    def classify_icsa(val: float) -> str:
        if val < 22:    return "LOW"
        elif val <= 25: return "NEUTRAL"
        else:           return "HIGH"

    def classify_regime(row) -> str:
        t = row["T5YIE_ZONE"]
        i = row["ICSA_ZONE"]
        regime_map = {
            ("LOW",     "LOW"):     "디플레",
            ("NEUTRAL", "LOW"):     "골디락스",
            ("HIGH",    "LOW"):     "과열",
            ("LOW",     "NEUTRAL"): "둔화_주의",
            ("NEUTRAL", "NEUTRAL"): "둔화_주의",
            ("HIGH",    "NEUTRAL"): "둔화_주의",
            ("LOW",     "HIGH"):    "경기침체",
            ("NEUTRAL", "HIGH"):    "둔화_주의",
            ("HIGH",    "HIGH"):    "스태그플레이션",
        }
        return regime_map.get((t, i), "불명확")

    # ------------------------------------------------
    # 국면 분류 적용
    # ------------------------------------------------
    df = pd.concat([t5yie, icsa], axis=1)
    df["T5YIE_ZONE"]  = df["T5YIE"].apply(classify_t5yie)
    df["ICSA_ZONE"]   = df["ICSA"].apply(classify_icsa)
    df["REGIME"]      = df.apply(classify_regime, axis=1)

    regime_risk = {
        "골디락스":       -1,
        "과열":            1,
        "디플레":          1,
        "둔화_주의":       2,
        "경기침체":        3,
        "스태그플레이션":  3,
    }
    df["REGIME_SCORE"] = df["REGIME"].map(regime_risk)

    # ------------------------------------------------
    # 결과 요약
    # ------------------------------------------------
    regime_counts = df["REGIME"].value_counts()
    print("\n📊 백테스팅 기간 국면 분포")
    for regime, count in regime_counts.items():
        pct = count / len(df) * 100
        print(f"   {regime:<15}: {count:>4}일 ({pct:.1f}%)")

    latest = df.iloc[-1]
    print(f"\n📊 최근 국면")
    print(f"   T5YIE:       {latest['T5YIE']:.2f}% ({latest['T5YIE_ZONE']})")
    print(f"   ICSA:        {latest['ICSA']:.1f}만명 ({latest['ICSA_ZONE']})")
    print(f"   현재 국면:    {latest['REGIME']}")
    print(f"   리스크 기여:  {latest['REGIME_SCORE']:+d}점")

    return df


# ================================================
# [후행] IG 스프레드 + STLFSI4 + 샴 룰 수집
# ================================================

def get_lagging_indicators(
    start: str = START_DATE,
    end:   str = END_DATE
) -> pd.DataFrame:
    """
    후행 지표 3개를 수집하고 임계치 기반 점수를 산출합니다.

    임계치 기준 (FRED 단위 기준)
    ─────────────────────────────────────────────────────
    IG 스프레드  정상 <1.0%  / 경계 1.0~2.5% / 위험 >2.5%
                (FRED 단위 %, 1.0% = 100bp)
    STLFSI4     정상 <0     / 경계 0~1       / 위험 >1
    샴 룰       정상 <0.3   / 경계 0.3~0.5   / 위험 ≥0.5
    ─────────────────────────────────────────────────────
    """
    print("\n" + "=" * 50)
    print("후행 지표 수집 (IG 스프레드 + STLFSI4 + 샴 룰)")
    print("=" * 50)

    full_idx = pd.date_range(start=start, end=end, freq="B")
    frames = []

    # ------------------------------------------------
    # IG 스프레드 (일별)
    # BAMLC0A0CM: ICE BofA US Corporate Index OAS
    # FRED 단위: % (예: 0.82 = 82bp)
    # ------------------------------------------------
    print("\n[1] IG 스프레드 (BAMLC0A0CM)")
    try:
        ig = web.DataReader("BAMLC0A0CM", "fred", start=start, end=end)
        ig.columns = ["IG_SPREAD"]
        ig.index = pd.to_datetime(ig.index).tz_localize(None)
        ig = ig.reindex(full_idx).ffill().bfill()

        def score_ig(val: float) -> int:
            # FRED 단위: % (0.82 = 82bp)
            # 정상: <1.0%  (100bp 미만)  → 0점
            # 경계: 1.0~2.5% (100~250bp) → 1점
            # 위험: >2.5%  (250bp 초과)  → 2점
            if val < 1.0:    return 0
            elif val <= 2.5: return 1
            else:            return 2

        ig["IG_SCORE"] = ig["IG_SPREAD"].apply(score_ig)
        frames.append(ig)

        latest_val = ig["IG_SPREAD"].iloc[-1]
        latest_scr = ig["IG_SCORE"].iloc[-1]
        status = "🟢 정상" if latest_scr == 0 else "🟡 경계" if latest_scr == 1 else "🔴 위험"
        print(f"  ✅ IG 스프레드: {len(ig)}행 | 최근값: {latest_val:.2f}% ({latest_val*100:.0f}bp) {status}")

    except Exception as e:
        print(f"  ⚠️ IG 스프레드 수집 실패: {e}")

    # ------------------------------------------------
    # STLFSI4 (주별 → 일별)
    # 세인트루이스 연준 금융 스트레스 지수
    # ------------------------------------------------
    print("\n[2] STLFSI4 (금융 스트레스 지수)")
    try:
        fsi = web.DataReader("STLFSI4", "fred", start=start, end=end)
        fsi.columns = ["STLFSI"]
        fsi.index = pd.to_datetime(fsi.index).tz_localize(None)

        # FIX: STLFSI도 주별 데이터 → resample 적용
        fsi_daily = fsi.resample('D').ffill()
        fsi_daily = fsi_daily.reindex(full_idx).ffill().bfill()

        def score_fsi(val: float) -> int:
            if val < 0:    return 0
            elif val <= 1: return 1
            else:          return 2

        fsi_daily["FSI_SCORE"] = fsi_daily["STLFSI"].apply(score_fsi)
        frames.append(fsi_daily)

        latest_val = fsi_daily["STLFSI"].iloc[-1]
        latest_scr = fsi_daily["FSI_SCORE"].iloc[-1]
        status = "🟢 정상" if latest_scr == 0 else "🟡 경계" if latest_scr == 1 else "🔴 위험"
        print(f"  ✅ STLFSI4: {len(fsi_daily)}행 | 최근값: {latest_val:.3f} {status}")

    except Exception as e:
        print(f"  ⚠️ STLFSI4 수집 실패: {e}")

    # ------------------------------------------------
    # 샴 룰 (월별 → 일별)
    # ≥ 0.5: 경기침체 공식 진입 신호
    # ------------------------------------------------
    print("\n[3] 샴 룰 (SAHMREALTIME)")
    try:
        sahm = web.DataReader("SAHMREALTIME", "fred", start=start, end=end)
        sahm.columns = ["SAHM"]
        sahm.index = pd.to_datetime(sahm.index).tz_localize(None)

        # FIX: 월별 데이터 → resample 적용
        sahm_daily = sahm.resample('D').ffill()
        sahm_daily = sahm_daily.reindex(full_idx).ffill().bfill()

        def score_sahm(val: float) -> int:
            if val < 0.3:   return 0
            elif val < 0.5: return 1
            else:           return 2

        sahm_daily["SAHM_SCORE"] = sahm_daily["SAHM"].apply(score_sahm)
        frames.append(sahm_daily)

        latest_val = sahm_daily["SAHM"].iloc[-1]
        latest_scr = sahm_daily["SAHM_SCORE"].iloc[-1]
        status = (
            "🟢 정상" if latest_scr == 0 else
            "🟡 경계" if latest_scr == 1 else
            "🔴 위험 ⚠️ 침체 신호"
        )
        print(f"  ✅ 샴 룰: {len(sahm_daily)}행 | 최근값: {latest_val:.2f} {status}")

    except Exception as e:
        print(f"  ⚠️ 샴 룰 수집 실패: {e}")

    if not frames:
        print("⚠️ 후행 지표 전체 수집 실패")
        return pd.DataFrame()

    result = pd.concat(frames, axis=1).ffill().bfill()

    score_cols = [c for c in ["IG_SCORE", "FSI_SCORE", "SAHM_SCORE"]
                  if c in result.columns]
    result["LAGGING_TOTAL"] = result[score_cols].sum(axis=1)

    print(f"\n📊 후행 종합 점수 (최근): "
          f"{result['LAGGING_TOTAL'].iloc[-1]:.0f}점 / 6점")

    return result


# ================================================
# 전체 통합 실행
# ================================================

def get_all_indicators(
    start: str = START_DATE,
    end:   str = END_DATE
) -> pd.DataFrame:
    """
    모든 리스크 지표를 수집하고 하나의 데이터프레임으로 반환합니다.

    최종 지표 구성
    ─────────────────────────────────────────
    선행 (40%): YIELD_SPREAD, CPI, T5YIE, ICSA
    동행 (35%): VIX, SKEW, VIX_BKWD
    후행 (25%): HY_SPREAD, IG_SPREAD, STLFSI, SAHM
    ─────────────────────────────────────────
    """
    print("=" * 50)
    print("전체 리스크 지표 수집 시작")
    print("=" * 50)

    print("\n[1] FRED 핵심 거시지표")
    fred_data = get_fred_indicators(FRED_SERIES, start, end)

    print("\n[2] 시장 심리 지표")
    market_data = get_market_indicators(YF_SERIES, start, end)

    print("\n[3] OECD 경기선행지수")
    oecd_data = get_oecd_cli(start, end)

    frames = [f for f in [fred_data, market_data, oecd_data]
              if not f.empty]
    all_indicators = pd.concat(frames, axis=1)

    print("\n[4] VIX 백워데이션 계산")
    all_indicators = calc_vix_backwardation(all_indicators)

    regime_df = get_macro_regime_indicators(start, end)
    if not regime_df.empty:
        all_indicators = pd.concat([all_indicators, regime_df], axis=1)

    lagging_df = get_lagging_indicators(start, end)
    if not lagging_df.empty:
        all_indicators = pd.concat([all_indicators, lagging_df], axis=1)

    all_indicators = all_indicators.ffill().bfill()

    print("\n" + "=" * 50)
    print("✅ 전체 리스크 지표 수집 완료")
    print(f"   기간: {all_indicators.index[0].date()} ~ "
          f"{all_indicators.index[-1].date()}")
    print(f"   행 수: {len(all_indicators)}일")
    print(f"\n   선행 지표: YIELD_SPREAD, CPI, T5YIE, ICSA")
    print(f"   동행 지표: VIX, SKEW, VIX_BKWD")
    print(f"   후행 지표: HY_SPREAD, IG_SPREAD, STLFSI, SAHM")

    missing = all_indicators.isnull().sum()
    if missing.sum() > 0:
        print(f"\n⚠️ 결측치 남아있음:")
        print(missing[missing > 0])
    else:
        print("\n   결측치: 없음")
    print("=" * 50)

    return all_indicators


# ================================================
# 실행
# ================================================

if __name__ == "__main__":

    indicators = get_all_indicators()

    print("\n📊 지표 미리보기 (최근 3일)")
    print(indicators.tail(3).to_string())

    print("\n📊 지표 기초 통계")
    print(indicators.describe().round(2).to_string())

    indicators.to_csv("risk_indicators.csv")
    print("\n✅ risk_indicators.csv 저장 완료")