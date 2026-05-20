# Module B - 포트폴리오 최적화




#1 DCC-GARCH 공분산 추정
import pandas as pd
import numpy as np
import warnings
from arch import arch_model

# 1. 쓸데없는 경고창 숨기기
warnings.filterwarnings('ignore')

# 2. 🚀 [핵심] 판다스 출력 옵션을 소수점 끝까지, 모든 열이 보이도록 완전히 개방
pd.set_option('display.max_columns', None)    # 중간에 ...으로 생략하지 말고 9개 열 다 보여주기
pd.set_option('display.max_rows', None)       # 행 생략 안 함
pd.set_option('display.width', 1000)          # 화면 너비를 넓게 잡아서 줄바꿈 방지
pd.set_option('display.precision', 15)        # 소수점 아래 15자리까지 자르지 말고 전부 표시

print("⏳ 깃허브에서 데이터를 불러오는 중...")
base_url = "https://raw.githubusercontent.com/harryk00/stockportfolio26_erica/main/module_B/"
log_returns = pd.read_csv(base_url + "log_returns.csv", index_col=0, parse_dates=True).sort_index().dropna()

# 3. DCC-GARCH 공분산 계산 함수 정의
def get_dcc_garch_covariance(returns):
    assets = returns.columns
    current_vols = {}
    for asset in assets:
        am = arch_model(returns[asset] * 100, vol='Garch', p=1, q=1, mean='Zero')
        res = am.fit(disp='off')
        pred_var = res.forecast(horizon=1).variance.iloc[-1, 0] / 10000 
        current_vols[asset] = np.sqrt(pred_var)
    
    D = np.diag(list(current_vols.values()))
    R = returns.ewm(span=60).corr().xs(returns.index[-1], level=0).values
    cov_matrix = D @ R @ D
    return pd.DataFrame(cov_matrix, index=assets, columns=assets)

print("⏳ DCC-GARCH 공분산 행렬 계산 중... (약 10초 소요)")
dcc_cov_matrix = get_dcc_garch_covariance(log_returns)

print("\n🎉 --- [Step 1 완료] 소수점 끝까지 펼쳐진 원본 공분산 행렬 ---")
print(dcc_cov_matrix)

-> 현재 공분산이 불안정한 상태이므로 scikit-learn 사용해 핼령 shrinkage 작업 수행






#STEP2. Ledoit-Wolf Shrinkage
from sklearn.covariance import LedoitWolf

print("\n⏳ Step 2: Ledoit-Wolf 행렬 안정화 진행 중...")

# 1. 최적의 축소 강도(Alpha) 계산 
# 원본 데이터가 얼마나 불안정한지 파악해서 0과 1 사이의 적절한 마사지 비율을 찾습니다.
lw = LedoitWolf().fit(log_returns)
alpha = lw.shrinkage_

# 2. 목표 행렬(Target Matrix) 만들기
# 자산들이 서로 얽혀서 움직이는 것(공분산)을 모두 0으로 지워버린 가장 안전한 기본 뼈대입니다.
target_matrix = np.diag(np.diag(dcc_cov_matrix))

# 3. 최종 안정화: 1단계 행렬과 안전한 뼈대를 알파 비율만큼 섞어줍니다.
shrunk_cov_matrix = (1 - alpha) * dcc_cov_matrix + alpha * target_matrix

print(f"✅ 계산된 안정화 강도 (Alpha): {alpha:.4f} (이 비율만큼 깎아냈습니다)")
print("\n🎉 --- [Step 2 완료] 에러 방지용 안정화 공분산 행렬 ---")
print(shrunk_cov_matrix)






#Step3. 리스크 패리티 역최적화 
변동성이 큰 위험 자산일수록 비중이 적게 나오고 상대적으로 안전한 자산일수록 비중이 높게 나와야 정상


#1  백테스팅 오류 발생 코드
from scipy.optimize import minimize

print("\n⏳ Step 3: 리스크 패리티 (동일 위험 기여) 기준 비중 계산 중...")

# 1. 목적함수 정의: 각 자산의 위험 기여도가 1/9로 동일해질 때 가장 값이 작아지는(0에 가까워지는) 함수
def risk_parity_objective(weights, cov_matrix):
    # 포트폴리오 전체 변동성 계산
    port_var = weights.T @ cov_matrix @ weights
    port_vol = np.sqrt(port_var)
    
    # 각 자산별 리스크 기여도 (Risk Contribution) 계산
    mrc = (cov_matrix @ weights) / port_vol
    risk_contribution = weights * mrc
    
    # 목표 리스크 기여도 (전체 위험을 9등분)
    target_rc = port_vol / len(weights) 
    
    # 실제 기여도와 목표 기여도의 차이 제곱합 반환
    return np.sum(np.square(risk_contribution - target_rc))

# 2. 최적화 기본 설정
num_assets = len(shrunk_cov_matrix)
initial_weights = np.repeat(1 / num_assets, num_assets) # 초기 비중은 전부 동일하게(11.1%) 시작
cov_matrix_vals = shrunk_cov_matrix.values

# 제약조건: 비중의 총합은 1(100%)이어야 하며, 숏(공매도) 금지로 각 비중은 0~1 사이
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})
bounds = tuple((0, 1) for _ in range(num_assets))

# 3. 최적화기 실행 (SLSQP 알고리즘 사용)
result = minimize(risk_parity_objective, initial_weights, args=(cov_matrix_vals,), 
                  method='SLSQP', bounds=bounds, constraints=constraints)

# 4. 결과 정리
risk_parity_weights = pd.Series(result.x, index=shrunk_cov_matrix.index)

print("✅ [Step 3 완료] 리스크 패리티 기준 비중:")
print("-" * 40)
print((risk_parity_weights * 100).round(2).astype(str) + ' %')
print("-" * 40)

->공분산 행렬 숫자들이 매우 작은데 0에 수렴한다고 가정해서 판단해서 모든 비율이 1/9 11.11 %로 나옴



#2 재테스팅
from scipy.optimize import minimize

print("\n⏳ Step 3: 리스크 패리티 재계산 중 (오차 스케일링 적용)...")

def risk_parity_objective(weights, cov_matrix):
    port_var = weights.T @ cov_matrix @ weights
    port_vol = np.sqrt(port_var)
    mrc = (cov_matrix @ weights) / port_vol
    risk_contribution = weights * mrc
    target_rc = port_vol / len(weights)
    
    # 🚀 핵심 처방: 컴퓨터가 미세한 차이를 무시하지 못하도록 오차를 엄청나게 키움
    return np.sum(np.square(risk_contribution - target_rc)) * 1_000_000_000

num_assets = len(shrunk_cov_matrix)
initial_weights = np.repeat(1 / num_assets, num_assets)
cov_matrix_vals = shrunk_cov_matrix.values

constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})
bounds = tuple((0.0, 1.0) for _ in range(num_assets))

# 계산기 민감도(ftol)도 극한으로 끌어올림
result = minimize(risk_parity_objective, initial_weights, args=(cov_matrix_vals,), 
                  method='SLSQP', bounds=bounds, constraints=constraints,
                  options={'ftol': 1e-15, 'maxiter': 1000})

risk_parity_weights = pd.Series(result.x, index=shrunk_cov_matrix.index)

print("✅ [Step 3 수정 완료] 진짜 리스크 패리티 비중:")
print("-" * 40)
print((risk_parity_weights * 100).round(2).astype(str) + ' %')
print("-" * 40)






STEP4. 블랙-리터만 + Idzorek 초기 확신도 기반, 투자 확신도 주입

#1 오류 발생코드


from numpy.linalg import inv

print("\n⏳ Step 4: 블랙-리터만 + Idzorek (투자 확신도 주입) 계산 중...")

# 1. 뷰(View) 설정: 계획서에 적어주신 투자 아이디어와 확신도(Confidence)
# P 행렬 (어떤 자산에 투자하는가? 1은 투자, 0은 무관)
# Q 벡터 (해당 뷰의 목표 초과 수익률. 여기서는 보수적으로 5%~10% 수준으로 가정)
# Confidence 벡터 (다솜님의 확신도 % 수치)

views = [
    # 1번 뷰: AI 슈퍼사이클 (SOL_AI, PLTR, VRT, SSINENG) - 75% 확신
    {"assets": ["SOL_AI", "PLTR", "VRT", "SSINENG"], "return": 0.10, "conf": 0.75},
    # 2번 뷰: ESS 에너지 (SKITERNX) - 65% 확신
    {"assets": ["SKITERNX"], "return": 0.08, "conf": 0.65},
    # 3번 뷰: 구리 공급 부족 (FCX) - 60% 확신
    {"assets": ["FCX"], "return": 0.07, "conf": 0.60},
    # 4번 뷰: 인플레 헤지 (KOSEF_TIPS) - 55% 확신
    {"assets": ["KOSEF_TIPS"], "return": 0.05, "conf": 0.55},
    # 5번 뷰: 배당 레버리지 (YUANTA) - 50% 확신
    {"assets": ["YUANTA"], "return": 0.05, "conf": 0.50}
]

assets = list(shrunk_cov_matrix.columns)
num_views = len(views)

P = np.zeros((num_views, num_assets))
Q = np.zeros(num_views)
confidences = np.zeros(num_views)

for i, view in enumerate(views):
    for asset in view["assets"]:
        P[i, assets.index(asset)] = 1.0 / len(view["assets"]) # 동일 비중으로 뷰 배분
    Q[i] = view["return"]
    confidences[i] = view["conf"]

# 2. 시장 내재 수익률 (Pi) 역산
tau = 0.05 # 스케일링 상수 (일반적인 설정값)
risk_aversion = 2.5 # 위험 회피 계수 (일반적인 설정값)
cov_matrix_vals = shrunk_cov_matrix.values
market_weights_vals = risk_parity_weights.values

Pi = risk_aversion * cov_matrix_vals @ market_weights_vals

# 3. Idzorek의 오메가(Omega) 행렬 계산: 확신도(%)를 분산 값으로 변환
Omega = np.zeros((num_views, num_views))
for i in range(num_views):
    alpha = (1 - confidences[i]) / confidences[i]
    Omega[i, i] = tau * alpha * (P[i] @ cov_matrix_vals @ P[i].T)

# 4. 블랙-리터만 최종 수식 결합
M_inv = inv(inv(tau * cov_matrix_vals) + P.T @ inv(Omega) @ P)
expected_returns = M_inv @ (inv(tau * cov_matrix_vals) @ Pi + P.T @ inv(Omega) @ Q)

# 결과 정리
expected_returns_series = pd.Series(expected_returns, index=assets)

print("✅ [Step 4 완료] 다솜님의 뷰가 반영된 자산별 기대수익률:")
print("-" * 50)
print((expected_returns_series * 100).round(2).astype(str) + ' %')
print("-" * 50)

코너해(Corner Solution) 발생




#2 재테스팅
 기본 세팅
 1. 위험도 스케일업: 일간 변동성(공분산)에 252(1년 주식 개장일) 곱해서 연간 변동성으로 단위 맞춰줌
 2. 상한선 강제: 금융공학 실무에서도 사용하며 한 종목에 30%이상 담지않는 것으로 현실적 제약 조건 걸어줌

from scipy.optimize import minimize

print("\n⏳ Step 5: 최종 비중 재계산 중 (단위 보정 및 몰빵 방지 적용)...")

def bl_objective(weights, exp_returns, cov_matrix, risk_av):
    port_return = np.sum(weights * exp_returns)
    # 1: 일간 변동성에 252(1년 거래일)를 곱해 연간 스케일로 맞춤
    port_var = weights.T @ (cov_matrix * 252) @ weights
    
    # 계산기가 소수점 차이를 뚜렷하게 인식하도록 스케일링
    return -(port_return - (risk_av / 2) * port_var) * 1000

# 2: 한 종목에 최대 30%까지만 투자할 수 있도록 상한선(0.3) 설정
max_weight = 0.30 
bounds = tuple((0.0, max_weight) for _ in range(num_assets))
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})

result_bl = minimize(bl_objective, initial_weights, 
                     args=(expected_returns, cov_matrix_vals, risk_aversion),
                     method='SLSQP', bounds=bounds, constraints=constraints,
                     options={'ftol': 1e-15, 'maxiter': 1000})

bl_weights = pd.Series(result_bl.x, index=assets)

# 파트 A 리스크 점수 반영 (기존과 동일)
try:
    current_risk_score = float(output_for_B.iloc[-1, 0])
except:
    current_risk_score = 5.0

max_cash_ratio = 0.5 
cash_ratio = (current_risk_score / 10.0) * max_cash_ratio

final_stock_weights = bl_weights * (1 - cash_ratio)
final_portfolio = final_stock_weights.copy()
final_portfolio['현금(CMA)'] = cash_ratio

print(f"🚨 파트 A 이번 달 시장 리스크 점수: {current_risk_score:.1f} / 10.0")
print("\n🎉 --- [최종 완료] 오류가 수정된 현실적인 최종 투자 비중 ---")
print((final_portfolio * 100).round(2).astype(str) + ' %')
print("-" * 55)
