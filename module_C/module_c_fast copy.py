import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch.bootstrap import CircularBlockBootstrap, StationaryBootstrap

# --- [사용자 정의 상수 유지] ---
MONTHLY_CONTRIBUTION = 1_600_000
TARGET_VALUE = 150_000_000

# 사용자가 지정한 비용 비율 그대로 유지
DOMESTIC_BUY_FEE = 0.00015
DOMESTIC_SELL_TAX = 0.0018  # 국내 매도 거래세
FOREX_FEE = 0.005
FOREIGN_BUY_FEE = 0.0025
REBALANCE_COST = 0.002

ANNUAL_CASH_RETURN = 0.02  
DAILY_CASH_RETURN = ANNUAL_CASH_RETURN / 252

def load_data():
    """모든 CSV 파일을 처음에 딱 한 번만 메모리로 로드합니다."""
    returns_df = pd.read_csv("log_returns.csv")
    if "Unnamed: 0" in returns_df.columns:
        returns_df.rename(columns={"Unnamed: 0": "Date"}, inplace=True)
        
    df_w = pd.read_csv("01. optimal_weights.csv")
    base_weights = dict(zip(df_w["Asset"], df_w["Weight"]))
    if "현금(CMA)" not in base_weights:
        base_weights["현금(CMA)"] = 0.0
        
    df_e = pd.read_csv("02. expected_returns.csv")
    expected_dict = dict(zip(df_e["Asset"], df_e["Expected_Return"]))
    
    signal_df = pd.read_csv("output_for_C.csv")
    return returns_df, base_weights, expected_dict, signal_df

def generate_return_path_fast(returns_df, circular_assets, stationary_assets, target_days, block_size=60):
    """부트스트랩 블록을 효율적으로 대량 추출하여 결합합니다."""
    # 넉넉하게 블록을 뽑아 한 번에 연산
    cbs = CircularBlockBootstrap(block_size, returns_df[circular_assets].values)
    circular_blocks = [bs[0][0] for bs in cbs.bootstrap(30)]
    circular_res = np.vstack(circular_blocks)[:target_days]

    sbs = StationaryBootstrap(block_size, returns_df[stationary_assets].values)
    stationary_blocks = [bs[0][0] for bs in sbs.bootstrap(30)]
    stationary_res = np.vstack(stationary_blocks)[:target_days]

    # 넘파이 배열 형태로 반환하여 Pandas 오버헤드 제거
    return np.hstack([circular_res, stationary_res])

def apply_glide_path(weight_dict, portfolio_value, risk_score, accelerate):
    """기존 글라이드 패스 로직 그대로 유지"""
    new_weights = weight_dict.copy()
    risky_assets = ["PLTR", "VRT", "SSINENG", "SOL_AI"]
    reduced_weight = 0

    if accelerate or risk_score >= 7:
        for asset in risky_assets:
            if asset in new_weights:
                old_weight = new_weights[asset]
                new_weights[asset] *= 0.5
                reduced_weight += (old_weight - new_weights[asset])
        new_weights["현금(CMA)"] += reduced_weight
        return new_weights

    if portfolio_value >= 135_000_000:
        for asset in risky_assets:
            if asset in new_weights:
                old_weight = new_weights[asset]
                new_weights[asset] *= 0.5
                reduced_weight += (old_weight - new_weights[asset])
        new_weights["현금(CMA)"] += reduced_weight
    elif portfolio_value >= 105_000_000:
        for asset in ["PLTR", "VRT", "SSINENG"]:
            if asset in new_weights:
                old_weight = new_weights[asset]
                new_weights[asset] *= 0.8
                reduced_weight += (old_weight - new_weights[asset])
        new_weights["현금(CMA)"] += reduced_weight

    return new_weights

def rebalance_portfolio(portfolio, current_weights, target_weights):
    """매도세 및 리밸런싱 비용을 현실적으로 반영하는 로직"""
    total_value = sum(portfolio.values())
    total_cost = 0
    new_portfolio = portfolio.copy()
    
    domestic_assets = ["SSINENG", "YUANTA", "SKITERNX", "KOSEF_TIPS", "SOL_AI"]

    for asset in target_weights:
        target_val = total_value * target_weights[asset]
        current_val = portfolio[asset]
        trade_amount = target_val - current_val  # (+)면 매수, (-)면 매도
        
        if trade_amount < 0:  # 매도 발생 시
            if asset in domestic_assets:
                # 국내 주식 매도 시 거래세(0.18%) 반영
                total_cost += abs(trade_amount) * DOMESTIC_SELL_TAX
        
        # 기본 리밸런싱 비용(0.15%~0.3% 중 설정값) 차감
        total_cost += abs(trade_amount) * REBALANCE_COST
        new_portfolio[asset] = target_val

    new_portfolio["현금(CMA)"] -= total_cost
    return new_portfolio

def run_dca_simulation(returns_df, base_weights, expected_dict, signal_df, asset_order, circular_assets, stationary_assets):
    target_days = 5 * 252
    path_matrix = generate_return_path_fast(returns_df, circular_assets, stationary_assets, target_days)
    
    portfolio = {asset: 0.0 for asset in base_weights}
    portfolio["현금(CMA)"] = 0.0

    # [최적화 핵심] 일일 루프 외부에서 5년 치 리스크 점수를 한 번에 넘파이 무작위 인덱싱으로 추출
    random_indices = np.random.choice(len(signal_df), size=target_days, replace=True)
    risk_scores = signal_df["RISK_SCORE"].values[random_indices]
    accelerates = signal_df["GLIDE_ACCELERATE"].values[random_indices]
    
    # 일일 기대수익률 배열 변환
    daily_expecteds = np.array([expected_dict[asset] / 252 for asset in asset_order])
    
    # 월별 잔고를 기록할 리스트 (총 60개월 관측)
    monthly_history = []
    current_weights = base_weights.copy()

    for day in range(target_days):
        portfolio_value = sum(portfolio.values())
        
        # 글라이드 패스 적용
        risk_score = risk_scores[day]
        accelerate = accelerates[day]
        weight_dict = apply_glide_path(base_weights, portfolio_value, risk_score, accelerate)

        # 21영업일 주기 DCA 투자
        if day % 21 == 0:
            for asset in weight_dict:
                invest_amount = MONTHLY_CONTRIBUTION * weight_dict[asset]
                if asset == "현금(CMA)":
                    portfolio[asset] += invest_amount
                elif asset in ["SSINENG", "YUANTA", "SKITERNX", "KOSEF_TIPS", "SOL_AI"]:
                    portfolio[asset] += invest_amount * (1 - DOMESTIC_BUY_FEE)
                else:
                    portfolio[asset] += invest_amount * (1 - FOREX_FEE - FOREIGN_BUY_FEE)

        # 자산 가격 변동 적용 (넘파이 인덱싱으로 속도 최적화)
        day_returns = path_matrix[day] + daily_expecteds
        for idx, asset in enumerate(asset_order):
            portfolio[asset] *= (1 + day_returns[idx])

        portfolio["현금(CMA)"] *= (1 + DAILY_CASH_RETURN)

        # 21영업일 주기 리밸런싱
        if day % 21 == 20:
            portfolio = rebalance_portfolio(portfolio, current_weights, weight_dict)
            current_weights = weight_dict.copy()
            
            # 매월 말 자산 잔고 기록 (월별 차트용)
            monthly_history.append(sum(portfolio.values()))

    # 만약 영업일 배수로 인해 60개가 안 채워졌다면 최종 잔고 추가
    while len(monthly_history) < 60:
        monthly_history.append(sum(portfolio.values()))

    return sum(portfolio.values()), monthly_history[:60]

def monte_carlo(simulations=10000):
    returns_df, base_weights, expected_dict, signal_df = load_data()
    
    circular_assets = ["SSINENG", "SKITERNX"]
    stationary_assets = ["YUANTA", "KOSEF_TIPS", "SOL_AI", "FCX", "PLTR", "QQQM", "VRT"]
    asset_order = circular_assets + stationary_assets

    final_results = []
    # 월별 추적을 위한 2차원 행렬 (1만 행 x 60열)
    monthly_matrix = np.zeros((simulations, 60))

    print("시뮬레이션 시작...")
    for i in range(simulations):
        if (i + 1) % 1000 == 0:
            print(f"{i+1}회 진행 완료")

        final_value, monthly_history = run_dca_simulation(
            returns_df, base_weights, expected_dict, signal_df, 
            asset_order, circular_assets, stationary_assets
        )
        final_results.append(final_value)
        monthly_matrix[i, :] = monthly_history

    final_results = np.array(final_results)
    
    # 지표 계산
    prob_achieve = np.mean(final_results >= TARGET_VALUE)
    median_final = np.median(final_results)
    percentile_5 = np.percentile(final_results, 5)
    percentile_95 = np.percentile(final_results, 95)
    
    return {
        "prob_achieve": prob_achieve,
        "median_final": median_final,
        "percentile_5": percentile_5,
        "percentile_95": percentile_95,
        "monthly_matrix": monthly_matrix
    }

def plot_monthly_chart(monthly_matrix):
    """요청하신 월별 잔고 분포 차트(monthly_chart) 시각화 함수"""
    months = np.arange(1, 61)
    
    # 월별 분위수 계산
    p5 = np.percentile(monthly_matrix, 5, axis=0)
    p50 = np.percentile(monthly_matrix, 50, axis=0)
    p95 = np.percentile(monthly_matrix, 95, axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(months, p50 / 10000, label="Median Scenario (50%)", color="blue", lw=2)
    plt.plot(months, p5 / 10000, label="Worst Scenario (5%)", color="red", linestyle="--")
    plt.plot(months, p95 / 10000, label="Best Scenario (95%)", color="green", linestyle="--")
    
    # 목표선 표시
    plt.axhline(y=TARGET_VALUE / 10000, color="orange", linestyle=":", label="Target (1.5 Billion KRW)")
    
    plt.title("Monthly Portfolio Value Distribution (5 Years)", fontsize=14)
    plt.xlabel("Months", fontsize=11)
    plt.ylabel("Portfolio Value (Ten Thousand KRW)", fontsize=11)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig("monthly_chart.png", dpi=300, bbox_inches="tight")

    plt.show()

if __name__ == "__main__":
    mc_result = monte_carlo(simulations=10000)

    print("\n" + "=" * 60)
    print("최종 몬테카를로 검증 결과")
    print("=" * 60)
    print(f"1억 5천만원 달성 확률 (prob_achieve) : {mc_result['prob_achieve']:.2%}")
    print(f"중앙값 최종 잔고 (median_final)     : {mc_result['median_final']:,.0f}원")
    print(f"최악 시나리오 하위 5% (percentile_5)  : {mc_result['percentile_5']:,.0f}원")
    print(f"최선 시나리오 상위 5% (percentile_95) : {mc_result['percentile_95']:,.0f}원")
    print("=" * 60)

    # 결과 csv 저장
    result_df = pd.DataFrame({
        "지표": [
            "prob_achieve",
            "median_final",
            "percentile_5",
            "percentile_95"
        ],
        "값": [
            mc_result["prob_achieve"],
            mc_result["median_final"],
            mc_result["percentile_5"],
            mc_result["percentile_95"]
        ]
    })

    result_df.to_csv(
        "mc_result.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("mc_result.csv 저장 완료")

    # 차트 생성 및 저장
    plot_monthly_chart(mc_result["monthly_matrix"])

    print("monthly_chart.png 저장 완료")