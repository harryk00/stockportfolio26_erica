import pandas as pd
import numpy as np
from pykrx import stock
import yfinance as yf
import pandas_datareader.data as web
import warnings
warnings.filterwarnings('ignore')


# ================================================
# 기본 설정
# ================================================

START_DATE = "2022-01-01"
END_DATE   = "2024-12-31"

# 국내 일반 주식 (pykrx 주식 함수 사용)
KR_STOCKS = {
    "011930": "SSINENG",   # 신성이엔지
    "003470": "YUANTA",    # 유안타증권
    "475150": "SKITERNX",  # SK이터닉스
}

# 국내 ETF (yfinance .KS 방식으로 변경)
# pykrx ETF 함수가 불안정하여 yfinance로 대체
KR_ETFS_YF = {
    "455850.KS": "SOL_AI",      # SOL AI 반도체소부장
    "430500.KS": "KOSEF_TIPS",  # KOSEF 물가채
}

# 해외 자산
US_TICKERS = ["QQQM", "PLTR", "VRT", "FCX"]

# ------------------------------------------------
# 대리 자산 설정
# QQQM 추가 (2020년 10월 13일 상장)
# ------------------------------------------------
PROXY_CONFIG = {
    "SOL_AI": {
        "proxy_code": "091160.KS",    # KODEX 반도체 ETF
        "proxy_name": "KODEX_반도체",
        "list_date":  "2023-04-01",
        "proxy_type": "kr_yf"         # yfinance KR ETF
    },
    "SKITERNX": {
        "proxy_code": "139250.KS",    # TIGER 200 에너지화학
        "proxy_name": "TIGER_에너지화학",
        "list_date":  "2023-08-01",
        "proxy_type": "kr_yf"
    },
    
}


# ================================================
# Step 1. 국내 주식 수집 (pykrx)
# ================================================

def get_korean_stock_prices(
    tickers: dict, start: str, end: str
) -> pd.DataFrame:
    """
    pykrx로 국내 일반 주식 종가를 가져옵니다.
    """
    start_fmt = start.replace("-", "")
    end_fmt   = end.replace("-", "")
    frames = []

    for code, name in tickers.items():
        print(f"  국내 주식 수집 중: {name} ({code})")
        try:
            df = stock.get_market_ohlcv_by_date(
                fromdate=start_fmt,
                todate=end_fmt,
                ticker=code
            )
            close = df[["종가"]].rename(columns={"종가": name})
            close.index = pd.to_datetime(close.index).tz_localize(None)
            frames.append(close)
            print(f"    ✅ {name}: {len(close)}일 수집 완료")

        except Exception as e:
            print(f"    ⚠️ {name} 수집 실패: {e}")

    return pd.concat(frames, axis=1) if frames else pd.DataFrame()


# ================================================
# Step 2. 국내 ETF + 해외 자산 수집 (yfinance 통합)
# pykrx ETF 함수 불안정 → yfinance .KS 방식으로 변경
# ================================================

def get_yf_prices(
    tickers: dict, start: str, end: str, label: str = ""
) -> pd.DataFrame:
    """
    yfinance로 자산 종가를 가져옵니다.
    국내 ETF (.KS 티커)와 해외 주식 모두 이 함수로 처리합니다.

    tickers: {"티커코드": "컬럼이름"} 형태
    """
    codes = list(tickers.keys())
    names = list(tickers.values())

    print(f"  {label} 수집 중: {names}")
    try:
        raw = yf.download(
            tickers=codes,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False
        )

        # Fix: MultiIndex 컬럼 처리
        if isinstance(raw.columns, pd.MultiIndex):
            raw = raw["Close"]
        else:
            raw = raw[["Close"]]
            raw.columns = codes

        # Fix: 타임존 제거
        raw.index = pd.to_datetime(raw.index).tz_localize(None)

        # 컬럼명을 사람이 읽기 쉬운 이름으로 변경
        raw = raw.rename(columns=tickers)

        print(f"    ✅ 수집 완료: {len(raw)}일")
        return raw

    except Exception as e:
        print(f"    ⚠️ 수집 실패: {e}")
        return pd.DataFrame()


# ================================================
# Step 3. 환율 수집 + 원화 환산
# ================================================

def get_usd_krw(start: str, end: str) -> pd.Series:
    """
    FRED API에서 원달러 환율을 가져옵니다.
    """
    print("  원달러 환율 수집 중...")
    try:
        fx = web.DataReader("DEXKOUS", "fred", start=start, end=end)
        fx.columns = ["USDKRW"]
        fx.index = pd.to_datetime(fx.index).tz_localize(None)
        fx = fx.ffill()
        print(f"    ✅ 환율 수집 완료 (최근값: {fx['USDKRW'].iloc[-1]:.1f}원)")
        return fx["USDKRW"]

    except Exception as e:
        print(f"    ⚠️ 환율 수집 실패: {e}")
        return None


def convert_to_krw(
    prices: pd.DataFrame, usd_krw: pd.Series
) -> pd.DataFrame:
    """
    달러 기준 가격을 원화로 환산합니다.
    """
    fx_aligned = usd_krw.reindex(prices.index).ffill()
    result = prices.multiply(fx_aligned, axis=0)
    print("    ✅ 원화 환산 완료")
    return result


# ================================================
# 대리 자산 처리
# ================================================

def build_proxy_series(
    actual_prices: pd.Series,
    proxy_prices:  pd.Series,
    list_date:     str
) -> pd.Series:
    """
    실제 자산과 대리 자산을 수익률 기준으로 이어 붙입니다.

    상장일 이전: 대리 자산 로그 수익률
    상장일 이후: 실제 자산 로그 수익률
    → 기준값 100으로 가격 재구성
    """
    splice_date = pd.to_datetime(list_date)

    # 실제 자산 수익률 (상장일 이후)
    actual_returns = np.log(
        actual_prices / actual_prices.shift(1)
    ).loc[splice_date:]

    # 대리 자산 수익률 (상장일 이전)
    proxy_returns = np.log(
        proxy_prices / proxy_prices.shift(1)
    ).loc[:splice_date]

    # 이어 붙이기 (중복 날짜 제거)
    combined = pd.concat([
        proxy_returns.iloc[:-1],
        actual_returns
    ])
    combined = combined[~combined.index.duplicated(keep='last')]

    # 로그 수익률 → 가격 재구성 (기준값 100)
    reconstructed = 100 * np.exp(combined.cumsum())
    return reconstructed


def apply_proxy_assets(
    all_prices: pd.DataFrame,
    usd_krw:    pd.Series,
    start:      str,
    end:        str
) -> pd.DataFrame:
    """
    대리 자산이 필요한 종목의 상장 전 기간을 채웁니다.
    """
    print("\n대리 자산 처리 중...")

    for asset_name, config in PROXY_CONFIG.items():
        if asset_name not in all_prices.columns:
            print(f"  ⚠️ {asset_name} 컬럼 없음, 건너뜀")
            continue

        print(f"  {asset_name} → {config['proxy_name']} 로 대체 처리")

        try:
            # 대리 자산 수집
            if config["proxy_type"] == "kr_yf":
                proxy_df = get_yf_prices(
                    {config["proxy_code"]: config["proxy_name"]},
                    start, end,
                    label="국내 ETF 대리자산"
                )
                if proxy_df.empty:
                    print(f"    ⚠️ 수집 실패, 건너뜀")
                    continue
                proxy_prices = proxy_df[config["proxy_name"]]

            elif config["proxy_type"] == "us":
                proxy_df = get_yf_prices(
                    {config["proxy_code"]: config["proxy_name"]},
                    start, end,
                    label="해외 대리자산"
                )
                if proxy_df.empty:
                    print(f"    ⚠️ 수집 실패, 건너뜀")
                    continue
                # 해외 대리자산은 원화 환산
                proxy_krw    = convert_to_krw(proxy_df, usd_krw)
                proxy_prices = proxy_krw[config["proxy_name"]]

            # 이어 붙이기
            actual_prices = all_prices[asset_name]
            spliced = build_proxy_series(
                actual_prices,
                proxy_prices,
                config["list_date"]
            )
            spliced.name = asset_name
            all_prices[asset_name] = spliced

            print(f"    ✅ {config['list_date']} 이전 → "
                  f"{config['proxy_name']} 수익률로 대체 완료")

        except Exception as e:
            print(f"    ⚠️ {asset_name} 처리 중 오류: {e}")
            continue

    return all_prices


# ================================================
# 전체 실행
# ================================================

def get_all_prices(
    start: str = START_DATE,
    end:   str = END_DATE
) -> pd.DataFrame:
    """
    모든 자산의 원화 기준 일별 가격 데이터를 반환합니다.
    """
    print("=" * 50)
    print("전체 데이터 수집 시작")
    print("=" * 50)

    # 1. 국내 주식 (pykrx)
    print("\n[1] 국내 주식")
    kr_stock_prices = get_korean_stock_prices(KR_STOCKS, start, end)

    # 2. 국내 ETF (yfinance .KS)
    print("\n[2] 국내 ETF")
    kr_etf_prices = get_yf_prices(KR_ETFS_YF, start, end, label="국내 ETF")

    # 3. 해외 주식 (yfinance, 달러)
    print("\n[3] 해외 주식")
    us_tickers_dict = {t: t for t in US_TICKERS}
    us_prices_usd = get_yf_prices(us_tickers_dict, start, end, label="해외 주식")

    # 4. 환율 + 원화 환산
    print("\n[4] 환율 및 원화 환산")
    usd_krw = get_usd_krw(start, end)

    # 국내 ETF도 원화이지만 yfinance는 원화로 반환하므로 환산 불필요
    us_prices_krw = convert_to_krw(us_prices_usd, usd_krw)

    # 5. 전체 합치기
    all_prices = pd.concat([
        kr_stock_prices,
        kr_etf_prices,
        us_prices_krw
    ], axis=1)

    # 6. 대리 자산 처리
    all_prices = apply_proxy_assets(all_prices, usd_krw, start, end)

    # 7. 결측치 처리
    all_prices = all_prices.bfill()  # 앞부분 NaN → 뒤 값으로 채우기
    all_prices = all_prices.ffill()  # 중간 NaN → 앞 값으로 채우기
    all_prices = all_prices.dropna(how="all")

    # 8. 결과 확인
    print("\n" + "=" * 50)
    print("✅ 전체 데이터 수집 완료")
    print(f"   기간: {all_prices.index[0].date()} ~ "
          f"{all_prices.index[-1].date()}")
    print(f"   종목 수: {len(all_prices.columns)}개")
    print(f"   종목: {list(all_prices.columns)}")
    print(f"   행 수: {len(all_prices)}일")

    # 9. 결측치 최종 점검
    missing = all_prices.isnull().sum()
    if missing.sum() > 0:
        print("\n⚠️ 결측치 남아있음:")
        print(missing[missing > 0])
    else:
        print("\n✅ 결측치 없음")

    print("=" * 50)
    return all_prices


# ================================================
# 로그 수익률 변환
# ================================================

def calc_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    일별 종가를 로그 수익률로 변환합니다.
    0값은 NaN 처리 후 직전값으로 채웁니다.
    """
    prices = prices.replace(0, np.nan).ffill()
    log_returns = np.log(prices / prices.shift(1))
    log_returns = log_returns.dropna(how="all")
    return log_returns


# ================================================
# 실행
# ================================================

if __name__ == "__main__":

    prices  = get_all_prices()
    returns = calc_log_returns(prices)

    print("\n📊 가격 데이터 미리보기 (최근 3일)")
    print(prices.tail(3).to_string())

    print("\n📊 로그 수익률 미리보기 (최근 3일)")
    print(returns.tail(3).to_string())

    prices.to_csv("price_data.csv")
    returns.to_csv("log_returns.csv")
    print("\n✅ price_data.csv, log_returns.csv 저장 완료")