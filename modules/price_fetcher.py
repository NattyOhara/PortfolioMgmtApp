"""
価格取得モジュール
Yahoo Financeからの株価と為替レート取得機能
"""

import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Optional, Tuple
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


def get_current_prices(tickers: List[str]) -> Dict[str, float]:
    """
    複数銘柄の現在株価を一括取得
    
    Args:
        tickers: ティッカーシンボルのリスト
    
    Returns:
        Dict[str, float]: ティッカーをキーとした現在株価の辞書
    """
    prices = {}
    failed_tickers = []
    
    try:
        # 並列処理で株価を取得
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_ticker = {
                executor.submit(get_single_price, ticker): ticker 
                for ticker in tickers
            }
            
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    price = future.result()
                    if price is not None:
                        prices[ticker] = price
                    else:
                        failed_tickers.append(ticker)
                except Exception as e:
                    logger.error(f"株価取得エラー {ticker}: {str(e)}")
                    failed_tickers.append(ticker)
        
        if failed_tickers:
            st.warning(f"以下の銘柄の株価取得に失敗しました: {failed_tickers}")
        
        logger.info(f"株価取得完了: {len(prices)}/{len(tickers)}銘柄")
        return prices
        
    except Exception as e:
        logger.error(f"株価一括取得エラー: {str(e)}")
        st.error(f"株価取得中にエラーが発生しました: {str(e)}")
        return {}


def get_single_price(ticker: str) -> Optional[float]:
    """
    単一銘柄の現在株価を取得
    
    Args:
        ticker: ティッカーシンボル
    
    Returns:
        float: 現在株価、取得失敗時はNone
    """
    try:
        stock = yf.Ticker(ticker)
        
        # 最新の株価データを取得（1日分）
        data = stock.history(period="1d", interval="1m")
        
        if data.empty:
            # 分足データが取得できない場合は日足で試行
            data = stock.history(period="2d")
            if data.empty:
                logger.warning(f"株価データが取得できません: {ticker}")
                return None
        
        # 最新の終値を取得
        latest_price = data['Close'].iloc[-1]
        
        if pd.isna(latest_price) or latest_price <= 0:
            logger.warning(f"無効な株価データ: {ticker} = {latest_price}")
            return None
        
        return float(latest_price)
        
    except Exception as e:
        logger.error(f"株価取得エラー {ticker}: {str(e)}")
        return None


def get_exchange_rates() -> Dict[str, float]:
    """
    主要通貨ペアの為替レートを取得
    
    Returns:
        Dict[str, float]: 通貨ペアをキーとした為替レートの辞書
    """
    currency_pairs = {
        'USDJPY=X': 'USD/JPY',
        'EURJPY=X': 'EUR/JPY',
        'GBPJPY=X': 'GBP/JPY',
        'AUDJPY=X': 'AUD/JPY',
        'CADJPY=X': 'CAD/JPY',
        'CHFJPY=X': 'CHF/JPY'
    }
    
    rates = {}
    
    try:
        for pair_symbol, pair_name in currency_pairs.items():
            try:
                ticker = yf.Ticker(pair_symbol)
                data = ticker.history(period="1d")
                
                if not data.empty:
                    rate = data['Close'].iloc[-1]
                    if not pd.isna(rate) and rate > 0:
                        rates[pair_symbol] = float(rate)
                        logger.debug(f"為替レート取得成功: {pair_name} = {rate}")
                    else:
                        logger.warning(f"無効な為替レートデータ: {pair_name}")
                else:
                    logger.warning(f"為替レートデータが取得できません: {pair_name}")
                    
            except Exception as e:
                logger.error(f"為替レート取得エラー {pair_name}: {str(e)}")
        
        logger.info(f"為替レート取得完了: {len(rates)}ペア")
        return rates
        
    except Exception as e:
        logger.error(f"為替レート一括取得エラー: {str(e)}")
        return {}


def determine_currency_from_ticker(ticker: str) -> str:
    """
    ティッカーシンボルから上場通貨を判定
    
    Args:
        ticker: ティッカーシンボル
    
    Returns:
        str: 通貨コード（USD, JPY, EUR等）
    """
    ticker = ticker.upper().strip()
    
    # 日本株の判定
    if any(suffix in ticker for suffix in ['.T', '.JP', '.OS']):
        return 'JPY'
    
    # 欧州株の判定
    if any(suffix in ticker for suffix in ['.AS', '.PA', '.DE', '.MI', '.MC']):
        if '.AS' in ticker:  # オランダ
            return 'EUR'
        elif '.PA' in ticker:  # フランス
            return 'EUR'
        elif '.DE' in ticker:  # ドイツ
            return 'EUR'
        elif '.MI' in ticker:  # イタリア
            return 'EUR'
        elif '.MC' in ticker:  # スペイン
            return 'EUR'
    
    # 英国株の判定
    if '.L' in ticker or '.LON' in ticker:
        return 'GBP'
    
    # カナダ株の判定
    if any(suffix in ticker for suffix in ['.TO', '.V']):
        return 'CAD'
    
    # オーストラリア株の判定
    if '.AX' in ticker:
        return 'AUD'
    
    # 香港株の判定
    if '.HK' in ticker:
        return 'HKD'
    
    # その他のアジア株
    if any(suffix in ticker for suffix in ['.SS', '.SZ']):  # 中国
        return 'CNY'
    elif '.KS' in ticker:  # 韓国
        return 'KRW'
    elif '.SI' in ticker:  # シンガポール
        return 'SGD'
    
    # デフォルトは米国株（USD）
    return 'USD'


def convert_to_jpy(price: float, currency: str, exchange_rates: Dict[str, float]) -> float:
    """
    現地通貨価格を日本円に換算
    
    Args:
        price: 現地通貨での価格
        currency: 通貨コード
        exchange_rates: 為替レート辞書
    
    Returns:
        float: 日本円換算価格
    """
    if currency == 'JPY':
        return price
    
    # 為替レートペアのマッピング
    rate_mapping = {
        'USD': 'USDJPY=X',
        'EUR': 'EURJPY=X',
        'GBP': 'GBPJPY=X',
        'AUD': 'AUDJPY=X',
        'CAD': 'CADJPY=X',
        'CHF': 'CHFJPY=X'
    }
    
    rate_symbol = rate_mapping.get(currency)
    
    if rate_symbol and rate_symbol in exchange_rates:
        rate = exchange_rates[rate_symbol]
        jpy_price = price * rate
        logger.debug(f"通貨換算: {price} {currency} × {rate} = {jpy_price} JPY")
        return jpy_price
    else:
        logger.warning(f"為替レートが見つかりません: {currency}")
        # フォールバック：概算レートを使用
        fallback_rates = {
            'USD': 150.0,  # 概算レート
            'EUR': 160.0,
            'GBP': 180.0,
            'AUD': 100.0,
            'CAD': 110.0,
            'CHF': 165.0,
            'HKD': 19.0,
            'SGD': 110.0
        }
        
        if currency in fallback_rates:
            rate = fallback_rates[currency]
            jpy_price = price * rate
            st.warning(f"概算レートを使用: {currency} = {rate} JPY")
            return jpy_price
        else:
            st.error(f"通貨 {currency} の換算レートが不明です")
            return price  # 換算せずに返す


def get_stock_chart_data(ticker: str, period: str = "1mo") -> pd.DataFrame:
    """
    単一銘柄のチャート用データを取得
    
    Args:
        ticker: ティッカーシンボル
        period: 取得期間（1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max）
    
    Returns:
        pd.DataFrame: OHLCV データ
    """
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval="1d")
        
        if data.empty:
            logger.warning(f"チャートデータが取得できませんでした: {ticker}")
            return pd.DataFrame()
        
        # インデックスをリセットして日付カラムを作成
        data.reset_index(inplace=True)
        logger.info(f"チャートデータ取得完了: {ticker} ({len(data)}日分)")
        return data
        
    except Exception as e:
        logger.error(f"チャートデータ取得エラー {ticker}: {str(e)}")
        return pd.DataFrame()


def get_historical_data(tickers: List[str], period: str = "5y") -> pd.DataFrame:
    """
    複数銘柄の過去データを取得
    
    Args:
        tickers: ティッカーシンボルのリスト
        period: 取得期間（1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max）
    
    Returns:
        pd.DataFrame: 各銘柄の調整後終値データ
    """
    try:
        with st.spinner(f"過去データを取得中... (期間: {period})"):
            # yfinanceで一括ダウンロード
            data = yf.download(
                tickers,
                period=period,
                interval="1d",
                group_by='ticker',
                auto_adjust=True,
                prepost=True,
                threads=True
            )
            
            if data.empty:
                logger.warning("過去データが取得できませんでした")
                return pd.DataFrame()
            
            # 単一銘柄の場合の処理
            if len(tickers) == 1:
                if 'Close' in data.columns:
                    result = pd.DataFrame({tickers[0]: data['Close']})
                else:
                    logger.warning(f"終値データが見つかりません: {tickers[0]}")
                    return pd.DataFrame()
            else:
                # 複数銘柄の場合、Closeカラムのみ抽出
                result = pd.DataFrame()
                for ticker in tickers:
                    try:
                        if (ticker, 'Close') in data.columns:
                            result[ticker] = data[(ticker, 'Close')]
                        elif ticker in data.columns and 'Close' in data[ticker].columns:
                            result[ticker] = data[ticker]['Close']
                    except Exception as e:
                        logger.warning(f"データ抽出エラー {ticker}: {str(e)}")
            
            # NaN値の処理
            result = result.dropna(how='all')  # すべてがNaNの行を削除
            
            logger.info(f"過去データ取得完了: {len(result)}日分, {len(result.columns)}銘柄")
            return result
            
    except Exception as e:
        logger.error(f"過去データ取得エラー: {str(e)}")
        st.error(f"過去データの取得中にエラーが発生しました: {str(e)}")
        return pd.DataFrame()


@st.cache_data(ttl=300)  # 5分間キャッシュ
def cached_get_current_prices(tickers_tuple: Tuple[str, ...]) -> Dict[str, float]:
    """
    キャッシュ機能付きの株価取得（Streamlit用）
    
    Args:
        tickers_tuple: ティッカーシンボルのタプル（キャッシュキー用）
    
    Returns:
        Dict[str, float]: 株価辞書
    """
    return get_current_prices(list(tickers_tuple))


@st.cache_data(ttl=900)  # 15分間キャッシュ
def cached_get_exchange_rates() -> Dict[str, float]:
    """
    キャッシュ機能付きの為替レート取得（Streamlit用）
    
    Returns:
        Dict[str, float]: 為替レート辞書
    """
    return get_exchange_rates()


def get_company_names(tickers: List[str]) -> Dict[str, str]:
    """
    複数銘柄の企業名を一括取得
    
    Args:
        tickers: ティッカーシンボルのリスト
    
    Returns:
        Dict[str, str]: ティッカーをキーとした企業名の辞書
    """
    company_names = {}
    failed_tickers = []
    
    try:
        # 並列処理で企業名を取得
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_ticker = {
                executor.submit(get_single_company_name, ticker): ticker 
                for ticker in tickers
            }
            
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    company_name = future.result()
                    if company_name:
                        company_names[ticker] = company_name
                    else:
                        failed_tickers.append(ticker)
                        company_names[ticker] = ticker  # フォールバック
                except Exception as e:
                    logger.error(f"企業名取得エラー {ticker}: {str(e)}")
                    failed_tickers.append(ticker)
                    company_names[ticker] = ticker  # フォールバック
        
        if failed_tickers:
            logger.warning(f"以下の銘柄の企業名取得に失敗しました: {failed_tickers}")
        
        logger.info(f"企業名取得完了: {len(company_names)}/{len(tickers)}銘柄")
        return company_names
        
    except Exception as e:
        logger.error(f"企業名一括取得エラー: {str(e)}")
        return {ticker: ticker for ticker in tickers}  # フォールバック


def get_single_company_name(ticker: str) -> Optional[str]:
    """
    単一銘柄の企業名を取得
    
    Args:
        ticker: ティッカーシンボル
    
    Returns:
        str: 企業名、取得失敗時はNone
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # longNameを最優先で取得
        if 'longName' in info and info['longName']:
            return info['longName']
        
        # shortNameも試行
        if 'shortName' in info and info['shortName']:
            return info['shortName']
        
        # 企業名が取得できない場合
        logger.warning(f"企業名が取得できません: {ticker}")
        return None
        
    except Exception as e:
        logger.error(f"企業名取得エラー {ticker}: {str(e)}")
        return None


@st.cache_data(ttl=3600)  # 1時間キャッシュ（企業名は変わりにくいため長めに設定）
def cached_get_company_names(tickers_tuple: Tuple[str, ...]) -> Dict[str, str]:
    """
    キャッシュ機能付きの企業名取得（Streamlit用）
    
    Args:
        tickers_tuple: ティッカーシンボルのタプル（キャッシュキー用）
    
    Returns:
        Dict[str, str]: 企業名辞書
    """
    return get_company_names(list(tickers_tuple))