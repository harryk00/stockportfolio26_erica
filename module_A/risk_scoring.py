import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'AppleGothic'  # macOS 한글 폰트
matplotlib.rcParams['axes.unicode_minus'] = False

# risk_indicators.py에서 데이터 수집 함수 가져오기
from risk_indicators import get_all_indicators, START_DATE, END_DATE


# ================================================
# 리스크 스코어링 설계
# ================================================
#
# 최종 점수: 0 ~ 10점
#
# 선행 그룹 (40%, 최대 4.0점)
# ├── YIELD_SPREAD  장단기 금리차     0~2점
# ├── CPI           물가 수준         0~2점
# └── REGIME_SCORE  T5YIE+ICSA 국면  0~3점
#     (골디락스 -1이면 최종 점수에서 0.5점 차감)
#
# 동행 그룹 (35%, 최대 3.5점)
# ├── VIX           시장 공포         0~2점
# ├── SKEW          꼬리 위험         0~2점
# └── VIX_BKWD      백워데이션        0~2점
#
# 후행 그룹 (25%, 최대 2.5점)
# ├── HY_SPREAD     하이일드 스프레드  0~2점
# ├── IG_SCORE      IG 스프레드       0~2점 (이미 계산됨)
# ├── FSI_SCORE     금융 스트레스     0~2점 (이미 계산됨)
# └── SAHM_SCORE    샴 룰             0~2점 (이미 계산됨)
#
# 행동 기준
# 0~3점  → 🟢 맑음   기본 비중 유지
# 4~6점  → 🟡 흐림   위험자산 10~20% 축소
# 7~10점 → 🔴 폭풍   공격적 방어 모드
# ================================================


# ================================================
# 개별 지표 점수 계산 함수
# ================================================

def score_yield_spread(val: float) -> int:
    """
    장단기 금리차 (10Y-2Y)
    > 0:       정상 (경기 확장)  → 0점
    -0.5~0:    경계 (역전 시작)  → 1점
    < -0.5:    위험 (역전 심화)  → 2점
    """
    if val > 0:       return 0
    elif val >= -0.5: return 1
    else:             return 2


def score_cpi(val: float) -> int:
    """
    CPI YoY 변화율
    < 2.5%:    목표 이하 (안정)  → 0점
    2.5~4.0%:  경계 (약간 높음)  → 1점
    > 4.0%:    위험 (고인플레)   → 2점
    """
    if val < 2.5:    return 0
    elif val <= 4.0: return 1
    else:            return 2


def score_vix(val: float) -> int:
    """
    VIX 변동성 지수
    < 20:    정상  → 0점
    20~30:   경계  → 1점
    > 30:    위험  → 2점
    """
    if val < 20:    return 0
    elif val <= 30: return 1
    else:           return 2


def score_skew(val: float) -> int:
    """
    SKEW 지수 (꼬리 위험 프리미엄)
    < 130:   정상               → 0점
    130~140: 하방 tail risk 큼  → 1점
    > 140:   극단적 하락 프라이싱 → 2점
    """
    if val < 130:    return 0
    elif val <= 140: return 1
    else:            return 2


def score_vix_backwardation(val: float) -> int:
    """
    VIX 백워데이션 (1M VIX - 3M VIX)
    ≤ 0:   정상 (콘탱고)         → 0점
    0~2:   약한 백워데이션        → 1점
    > 2:   강한 백워데이션 (패닉) → 2점
    """
    if val <= 0:   return 0
    elif val <= 2: return 1
    else:          return 2


def score_hy_spread(val: float) -> int:
    """
    하이일드 스프레드 (%)
    < 5%:   정상 (리스크온)      → 0점
    5~7%:   경계 (경기둔화 우려)  → 1점
    > 7%:   위험 (신용 위기)     → 2점
    """
    if val < 5:    return 0
    elif val <= 7: return 1
    else:          return 2


# ================================================
# 핵심: 리스크 점수 계산 함수 (행 단위)
# ================================================

def calculate_risk_score(row: pd.Series) -> pd.Series:
    """
    하나의 날짜(행)에 대해 리스크 점수를 계산합니다.
    각 그룹 점수와 최종 0~10점 점수를 함께 반환합니다.
    """

    # ------------------------------------------------
    # 선행 그룹 개별 점수
    # ------------------------------------------------
    s_yield  = score_yield_spread(row.get("YIELD_SPREAD", 0))
    s_cpi    = score_cpi(row.get("CPI", 2))

    # REGIME_SCORE: -1(골디락스) ~ +3(침체/스태그)
    # 음수는 0으로 클리핑 (골디락스 보너스는 최종 단계에서 적용)
    regime_raw   = row.get("REGIME_SCORE", 0)
    s_regime     = max(0, int(regime_raw))   # 0~3

    leading_raw  = s_yield + s_cpi + s_regime        # 0~7
    leading_max  = 7

    # ------------------------------------------------
    # 동행 그룹 개별 점수
    # ------------------------------------------------
    s_vix   = score_vix(row.get("VIX", 15))
    s_skew  = score_skew(row.get("SKEW", 120))
    s_bkwd  = score_vix_backwardation(row.get("VIX_BKWD", -2))

    coincident_raw = s_vix + s_skew + s_bkwd          # 0~6
    coincident_max = 6

    # ------------------------------------------------
    # 후행 그룹 개별 점수
    # (IG_SCORE, FSI_SCORE, SAHM_SCORE는 이미 계산됨)
    # ------------------------------------------------
    s_hy   = score_hy_spread(row.get("HY_SPREAD", 3))
    s_ig   = int(row.get("IG_SCORE", 0))
    s_fsi  = int(row.get("FSI_SCORE", 0))
    s_sahm = int(row.get("SAHM_SCORE", 0))

    lagging_raw = s_hy + s_ig + s_fsi + s_sahm        # 0~8
    lagging_max = 8

    # ------------------------------------------------
    # 그룹별 가중 점수 계산 (합계 0~10)
    # 선행 40%, 동행 35%, 후행 25%
    # ------------------------------------------------
    leading_score    = (leading_raw    / leading_max)    * 4.0
    coincident_score = (coincident_raw / coincident_max) * 3.5
    lagging_score    = (lagging_raw    / lagging_max)    * 2.5

    risk_score = leading_score + coincident_score + lagging_score

    # ------------------------------------------------
    # 골디락스 보너스: 최종 점수에서 0.5점 차감
    # 경제 국면이 가장 우호적일 때 리스크 점수 소폭 낮춤
    # ------------------------------------------------
    if int(regime_raw) == -1:
        risk_score = max(0.0, risk_score - 0.5)

    # ------------------------------------------------
    # 행동 신호 분류
    # ------------------------------------------------
    if risk_score <= 3:
        signal = "🟢 맑음"
        action = "기본 비중 유지"
    elif risk_score <= 6:
        signal = "🟡 흐림"
        action = "위험자산 10~20% 축소"
    else:
        signal = "🔴 폭풍"
        action = "공격적 방어 모드"

    return pd.Series({
        # 최종 점수
        "RISK_SCORE":    round(risk_score, 2),
        "SIGNAL":        signal,
        "ACTION":        action,

        # 그룹별 기여 점수
        "LEADING_SCORE":    round(leading_score, 2),
        "COINCIDENT_SCORE": round(coincident_score, 2),
        "LAGGING_SCORE":    round(lagging_score, 2),

        # 개별 지표 점수
        "S_YIELD":  s_yield,
        "S_CPI":    s_cpi,
        "S_REGIME": s_regime,
        "S_VIX":    s_vix,
        "S_SKEW":   s_skew,
        "S_BKWD":   s_bkwd,
        "S_HY":     s_hy,
        "S_IG":     s_ig,
        "S_FSI":    s_fsi,
        "S_SAHM":   s_sahm,
    })


# ================================================
# 전체 데이터프레임에 스코어링 적용
# ================================================

def apply_risk_scoring(indicators: pd.DataFrame) -> pd.DataFrame:
    """
    전체 기간의 리스크 점수를 계산합니다.
    indicators DataFrame의 각 행에 calculate_risk_score를 적용합니다.
    """
    print("리스크 점수 계산 중...")

    scores = indicators.apply(calculate_risk_score, axis=1)
    result = pd.concat([indicators, scores], axis=1)

    print(f"  ✅ 완료: {len(result)}일 점수 계산")
    return result


# ================================================
# 월별 리스크 점수 요약
# (B 담당자에게 넘길 월별 신호)
# ================================================

def get_monthly_risk_score(
    scored: pd.DataFrame
) -> pd.DataFrame:
    """
    일별 점수를 월말 기준 월별 점수로 변환합니다.
    B 파트에서 매월 블랙-리터만 확신도 조정에 사용합니다.
    """
    monthly = scored["RISK_SCORE"].resample("ME").last()
    monthly_signal = scored["SIGNAL"].resample("ME").last()
    monthly_action = scored["ACTION"].resample("ME").last()
    monthly_regime = scored["REGIME"].resample("ME").last()

    result = pd.DataFrame({
        "RISK_SCORE":  monthly,
        "SIGNAL":      monthly_signal,
        "ACTION":      monthly_action,
        "REGIME":      monthly_regime,
    })

    return result


# ================================================
# 결과 시각화
# ================================================

def plot_risk_score(scored: pd.DataFrame) -> None:
    """
    리스크 점수 시계열 차트를 그립니다.
    배경색으로 맑음/흐림/폭풍 구간을 표시합니다.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("리스크 스코어링 결과 (2022~2024)", fontsize=14, y=0.98)

    # ------------------------------------------------
    # 차트 1: 리스크 점수 (메인)
    # ------------------------------------------------
    ax1 = axes[0]
    ax1.plot(scored.index, scored["RISK_SCORE"],
             color="#2196F3", linewidth=1.5, label="리스크 점수")

    # 임계치 구간 배경색
    ax1.axhspan(0, 3,  alpha=0.08, color="green",  label="맑음 (0~3)")
    ax1.axhspan(3, 6,  alpha=0.08, color="orange", label="흐림 (4~6)")
    ax1.axhspan(6, 10, alpha=0.08, color="red",    label="폭풍 (7~10)")
    ax1.axhline(y=3, color="green",  linestyle="--", linewidth=0.8, alpha=0.6)
    ax1.axhline(y=6, color="orange", linestyle="--", linewidth=0.8, alpha=0.6)

    ax1.set_ylim(0, 10)
    ax1.set_ylabel("리스크 점수 (0~10)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ------------------------------------------------
    # 차트 2: 그룹별 기여 점수
    # ------------------------------------------------
    ax2 = axes[1]
    ax2.stackplot(
        scored.index,
        scored["LEADING_SCORE"],
        scored["COINCIDENT_SCORE"],
        scored["LAGGING_SCORE"],
        labels=["선행 (40%)", "동행 (35%)", "후행 (25%)"],
        colors=["#FF5722", "#FF9800", "#FFC107"],
        alpha=0.7
    )
    ax2.set_ylabel("그룹별 기여 점수")
    ax2.set_ylim(0, 10)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ------------------------------------------------
    # 차트 3: 핵심 지표 (VIX, HY 스프레드)
    # ------------------------------------------------
    ax3 = axes[2]
    ax3_twin = ax3.twinx()

    ax3.plot(scored.index, scored["VIX"],
             color="#9C27B0", linewidth=1.2, label="VIX (좌)", alpha=0.8)
    ax3_twin.plot(scored.index, scored["HY_SPREAD"],
                  color="#F44336", linewidth=1.2,
                  label="HY 스프레드 % (우)", linestyle="--", alpha=0.8)

    ax3.axhline(y=20, color="#9C27B0", linestyle=":", linewidth=0.8, alpha=0.5)
    ax3.axhline(y=30, color="#9C27B0", linestyle=":", linewidth=0.8, alpha=0.5)
    ax3_twin.axhline(y=5, color="#F44336", linestyle=":", linewidth=0.8, alpha=0.5)

    ax3.set_ylabel("VIX", color="#9C27B0")
    ax3_twin.set_ylabel("HY 스프레드 (%)", color="#F44336")
    ax3.legend(loc="upper left", fontsize=8)
    ax3_twin.legend(loc="upper right", fontsize=8)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("risk_score_chart.png", dpi=150, bbox_inches="tight")
    print("  ✅ risk_score_chart.png 저장 완료")
    plt.show()


# ================================================
# 최종 출력물 요약
# (B·C 파트에 넘길 형태로 정리)
# ================================================

def print_final_summary(
    scored:  pd.DataFrame,
    monthly: pd.DataFrame
) -> None:
    """
    최근 상태와 월별 요약을 출력합니다.
    """
    latest = scored.iloc[-1]

    print("\n" + "=" * 50)
    print("📊 현재 리스크 상태 (최근 영업일 기준)")
    print("=" * 50)
    print(f"  날짜:         {scored.index[-1].date()}")
    print(f"  리스크 점수:  {latest['RISK_SCORE']:.2f} / 10")
    print(f"  시장 신호:    {latest['SIGNAL']}")
    print(f"  권장 행동:    {latest['ACTION']}")
    print(f"  경제 국면:    {latest['REGIME']}")

    print(f"\n  그룹별 기여")
    print(f"  선행 (40%):  {latest['LEADING_SCORE']:.2f}점")
    print(f"  동행 (35%):  {latest['COINCIDENT_SCORE']:.2f}점")
    print(f"  후행 (25%):  {latest['LAGGING_SCORE']:.2f}점")

    print(f"\n  개별 지표 점수")
    print(f"  YIELD_SPREAD:  {latest['S_YIELD']}점  ({latest['YIELD_SPREAD']:.2f}%)")
    print(f"  CPI:           {latest['S_CPI']}점  ({latest['CPI']:.2f}%)")
    print(f"  REGIME:        {latest['S_REGIME']}점  ({latest['REGIME']})")
    print(f"  VIX:           {latest['S_VIX']}점  ({latest['VIX']:.1f})")
    print(f"  SKEW:          {latest['S_SKEW']}점  ({latest['SKEW']:.1f})")
    print(f"  VIX_BKWD:      {latest['S_BKWD']}점  ({latest['VIX_BKWD']:.2f})")
    print(f"  HY_SPREAD:     {latest['S_HY']}점  ({latest['HY_SPREAD']:.2f}%)")
    print(f"  IG_SPREAD:     {latest['S_IG']}점  ({latest['IG_SPREAD']:.2f}%)")
    print(f"  STLFSI:        {latest['S_FSI']}점  ({latest['STLFSI']:.3f})")
    print(f"  SAHM:          {latest['S_SAHM']}점  ({latest['SAHM']:.2f})")

    print("\n" + "=" * 50)
    print("📊 B 파트에 전달할 월별 리스크 점수 (최근 6개월)")
    print("=" * 50)
    print(monthly.tail(6).to_string())

    # 점수 구간별 일수 통계
    total = len(scored)
    calm   = (scored["RISK_SCORE"] <= 3).sum()
    cloudy = ((scored["RISK_SCORE"] > 3) & (scored["RISK_SCORE"] <= 6)).sum()
    storm  = (scored["RISK_SCORE"] > 6).sum()

    print(f"\n📊 백테스팅 기간 신호 분포")
    print(f"  🟢 맑음 (0~3점):  {calm:>4}일 ({calm/total*100:.1f}%)")
    print(f"  🟡 흐림 (4~6점):  {cloudy:>4}일 ({cloudy/total*100:.1f}%)")
    print(f"  🔴 폭풍 (7~10점): {storm:>4}일 ({storm/total*100:.1f}%)")
    print("=" * 50)


# ================================================
# B 파트에 넘길 최종 출력물 생성
# ================================================

def get_output_for_module_b(
    monthly: pd.DataFrame
) -> pd.DataFrame:
    """
    B 파트(포트폴리오 최적화)에 넘길 월별 리스크 점수.

    B 파트에서 이 점수를 받아 블랙-리터만 확신도를 조정합니다.
    조정 공식: 확신도 = 원래 확신도 × (1 - RISK_SCORE × 0.08)

    예시:
    RISK_SCORE=0  → 확신도 100% 유지
    RISK_SCORE=5  → 확신도 60%로 축소
    RISK_SCORE=7  → 확신도 44%로 축소
    """
    output = monthly[["RISK_SCORE", "SIGNAL", "ACTION", "REGIME"]].copy()

    # 블랙-리터만 확신도 조정 계수 추가
    output["BL_MULTIPLIER"] = (
        1 - output["RISK_SCORE"] * 0.08
    ).clip(lower=0.1)  # 최소 10% 유지

    return output


# ================================================
# 실행
# ================================================

if __name__ == "__main__":

    # 1. 지표 수집
    print("지표 데이터 로드 중...")
    indicators = get_all_indicators(START_DATE, END_DATE)

    # 2. 리스크 점수 계산
    scored = apply_risk_scoring(indicators)

    # 3. 월별 집계
    monthly = get_monthly_risk_score(scored)

    # 4. 최종 요약 출력
    print_final_summary(scored, monthly)

    # 5. 차트 저장
    print("\n차트 생성 중...")
    plot_risk_score(scored)

    # 6. 파일 저장
    scored.to_csv("risk_scored.csv")
    monthly.to_csv("risk_monthly.csv")

    # 7. B 파트 출력물
    b_output = get_output_for_module_b(monthly)
    b_output.to_csv("output_for_B.csv")

    print("\n✅ 저장 완료")
    print("   risk_scored.csv    → 일별 전체 점수")
    print("   risk_monthly.csv   → 월별 점수 요약")
    print("   output_for_B.csv   → B 파트 전달용")
    print("   risk_score_chart.png → 시각화")