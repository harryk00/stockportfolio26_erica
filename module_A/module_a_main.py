"""
=================================================
파트 A 최종 실행 파일
=================================================

이 파일 하나만 실행하면 A의 전체 작업이 완료됩니다.

실행 순서
1. 자산 가격 데이터 수집       (data_collector.py)
2. 리스크 지표 수집            (risk_indicators.py)
3. 리스크 점수 계산            (risk_scoring.py)
4. B·C에 넘길 출력물 최종 정리

최종 출력물
─────────────────────────────────────────────
B 파트 (포트폴리오 최적화)에게
  price_data.csv      9개 자산 원화 환산 일별 가격
  log_returns.csv     일별 로그 수익률
  output_for_B.csv    월별 리스크 점수 + BL 조정 계수

C 파트 (몬테카를로 시뮬레이션)에게
  price_data.csv      동일 (B와 공유)
  log_returns.csv     동일 (B와 공유)
  risk_scored.csv     일별 리스크 점수 전체 (글라이드 패스용)
─────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
import os

from data_collector  import get_all_prices, calc_log_returns
from risk_indicators import get_all_indicators, START_DATE, END_DATE
from risk_scoring    import (
    apply_risk_scoring,
    get_monthly_risk_score,
    get_output_for_module_b,
    print_final_summary,
    plot_risk_score,
)


# ================================================
# 설정
# ================================================

START = START_DATE   # "2022-01-01"
END   = END_DATE     # "2024-12-31"

# 출력 파일 경로
OUTPUT = {
    "price_data":    "price_data.csv",
    "log_returns":   "log_returns.csv",
    "risk_scored":   "risk_scored.csv",
    "risk_monthly":  "risk_monthly.csv",
    "output_for_B":  "output_for_B.csv",
    "output_for_C":  "output_for_C.csv",
    "chart":         "risk_score_chart.png",
}


# ================================================
# Step 1. 자산 가격 데이터 수집
# ================================================

def run_step1() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n" + "=" * 50)
    print("STEP 1. 자산 가격 데이터 수집")
    print("=" * 50)

    prices  = get_all_prices(start=START, end=END)
    returns = calc_log_returns(prices)

    prices.to_csv(OUTPUT["price_data"])
    returns.to_csv(OUTPUT["log_returns"])

    print(f"\n  ✅ price_data.csv  저장 완료 ({len(prices)}일 × {len(prices.columns)}종목)")
    print(f"  ✅ log_returns.csv 저장 완료")

    return prices, returns


# ================================================
# Step 2. 리스크 지표 수집
# ================================================

def run_step2() -> pd.DataFrame:
    print("\n" + "=" * 50)
    print("STEP 2. 리스크 지표 수집")
    print("=" * 50)

    indicators = get_all_indicators(start=START, end=END)
    return indicators


# ================================================
# Step 3. 리스크 점수 계산
# ================================================

def run_step3(indicators: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n" + "=" * 50)
    print("STEP 3. 리스크 점수 계산")
    print("=" * 50)

    scored  = apply_risk_scoring(indicators)
    monthly = get_monthly_risk_score(scored)

    scored.to_csv(OUTPUT["risk_scored"])
    monthly.to_csv(OUTPUT["risk_monthly"])

    print(f"  ✅ risk_scored.csv  저장 완료")
    print(f"  ✅ risk_monthly.csv 저장 완료")

    return scored, monthly


# ================================================
# Step 4. B·C 출력물 최종 정리
# ================================================

def run_step4(
    prices:  pd.DataFrame,
    returns: pd.DataFrame,
    scored:  pd.DataFrame,
    monthly: pd.DataFrame
) -> None:
    print("\n" + "=" * 50)
    print("STEP 4. B·C 파트 출력물 정리")
    print("=" * 50)

    # ------------------------------------------------
    # B 파트 출력물
    # 월별 리스크 점수 + 블랙-리터만 확신도 조정 계수
    # ------------------------------------------------
    b_output = get_output_for_module_b(monthly)
    b_output.to_csv(OUTPUT["output_for_B"])

    print("\n  [B 파트 전달 파일]")
    print(f"  price_data.csv     → 9개 자산 일별 원화 가격")
    print(f"  log_returns.csv    → 일별 로그 수익률")
    print(f"  output_for_B.csv   → 월별 리스크 점수 + BL 조정 계수")
    print(f"\n  output_for_B.csv 미리보기 (최근 3개월)")
    print(b_output.tail(3).to_string())

    # ------------------------------------------------
    # C 파트 출력물
    # 일별 리스크 점수 + 글라이드 패스 신호
    # ------------------------------------------------
    c_output = scored[[
        "RISK_SCORE", "SIGNAL", "ACTION",
        "REGIME", "LEADING_SCORE", "COINCIDENT_SCORE", "LAGGING_SCORE"
    ]].copy()

    # 글라이드 패스 가속 여부 (7점 이상이면 즉시 방어)
    c_output["GLIDE_ACCELERATE"] = c_output["RISK_SCORE"] >= 7

    c_output.to_csv(OUTPUT["output_for_C"])

    print(f"\n  [C 파트 전달 파일]")
    print(f"  price_data.csv     → 동일 (B와 공유)")
    print(f"  log_returns.csv    → 동일 (B와 공유)")
    print(f"  output_for_C.csv   → 일별 리스크 점수 + 글라이드 패스 신호")
    print(f"\n  output_for_C.csv 미리보기 (최근 3일)")
    print(c_output.tail(3).to_string())


# ================================================
# 최종 요약 출력
# ================================================

def print_module_a_summary(
    prices:  pd.DataFrame,
    returns: pd.DataFrame,
    scored:  pd.DataFrame,
    monthly: pd.DataFrame
) -> None:
    """
    파트 A 전체 결과를 한눈에 보여주는 최종 요약입니다.
    """
    print("\n" + "=" * 50)
    print("파트 A 완료 — 최종 요약")
    print("=" * 50)

    # 자산 데이터
    print("\n📦 수집된 자산 데이터")
    print(f"   기간:    {prices.index[0].date()} ~ {prices.index[-1].date()}")
    print(f"   영업일:  {len(prices)}일")
    print(f"   자산:    {list(prices.columns)}")

    # 현재 리스크 상태
    latest       = scored.iloc[-1]
    latest_month = monthly.iloc[-1]

    print(f"\n📊 현재 리스크 상태 ({scored.index[-1].date()})")
    print(f"   점수:    {latest['RISK_SCORE']:.2f} / 10")
    print(f"   신호:    {latest['SIGNAL']}")
    print(f"   국면:    {latest['REGIME']}")

    # 백테스팅 기간 신호 분포
    total  = len(scored)
    calm   = (scored["RISK_SCORE"] <= 3).sum()
    cloudy = ((scored["RISK_SCORE"] > 3) & (scored["RISK_SCORE"] <= 6)).sum()
    storm  = (scored["RISK_SCORE"] > 6).sum()

    print(f"\n📅 백테스팅 기간 신호 분포")
    print(f"   🟢 맑음:  {calm:>4}일 ({calm/total*100:.1f}%)")
    print(f"   🟡 흐림:  {cloudy:>4}일 ({cloudy/total*100:.1f}%)")
    print(f"   🔴 폭풍:  {storm:>4}일 ({storm/total*100:.1f}%)")

    # B 파트에 넘기는 핵심 값
    print(f"\n📤 B 파트에 넘기는 이번 달 리스크 정보")
    print(f"   RISK_SCORE:    {latest_month['RISK_SCORE']:.2f}")
    bl_multiplier = max(0.1, 1 - latest_month["RISK_SCORE"] * 0.08)
    print(f"   BL_MULTIPLIER: {bl_multiplier:.2f}")
    print(f"   (확신도 = 원래 확신도 × {bl_multiplier:.2f})")   
    # 생성된 파일 목록
    print(f"\n📁 생성된 파일")
    for name, path in OUTPUT.items():
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024
            print(f"   ✅ {path:<25} ({size:.1f} KB)")
        else:
            print(f"   ⬜ {path:<25} (미생성)")

    print("\n" + "=" * 50)
    print("파트 A 모든 작업 완료")
    print("B 파트: price_data.csv, log_returns.csv, output_for_B.csv 사용")
    print("C 파트: price_data.csv, log_returns.csv, output_for_C.csv 사용")
    print("=" * 50)


# ================================================
# 전체 실행
# ================================================

if __name__ == "__main__":

    print("=" * 50)
    print("파트 A 전체 파이프라인 시작")
    print(f"기간: {START} ~ {END}")
    print("=" * 50)

    # Step 1. 자산 가격 수집
    prices, returns = run_step1()

    # Step 2. 리스크 지표 수집
    indicators = run_step2()

    # Step 3. 리스크 점수 계산
    scored, monthly = run_step3(indicators)

    # Step 4. B·C 출력물 정리
    run_step4(prices, returns, scored, monthly)

    # 최종 요약
    print_module_a_summary(prices, returns, scored, monthly)

    # 차트 저장
    print("\n차트 생성 중...")
    plot_risk_score(scored)