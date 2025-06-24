"""
データアダプターモジュール
既存のモジュールとDataManagerの間のブリッジ機能を提供
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class DataAdapter:
    """
    既存のモジュールが期待する形式でDataManagerのデータを提供
    """
    
    def __init__(self, data_bundle: Dict[str, Any]):
        """
        データアダプターを初期化
        
        Args:
            data_bundle: DataManagerから取得したデータバンドル
        """
        self.data_bundle = data_bundle
        logger.debug("データアダプター初期化完了")
    
    
    # price_fetcher.py 互換メソッド
    def get_current_price(self, ticker: str) -> float:
        """
        単一銘柄の現在価格を取得（price_fetcher互換）
        """
        return self.data_bundle['current_prices'].get(ticker, 0.0)
    
    
    def get_multiple_current_prices(self, tickers: List[str]) -> Dict[str, float]:
        """
        複数銘柄の現在価格を取得（price_fetcher互換）
        """
        result = {}
        for ticker in tickers:
            result[ticker] = self.data_bundle['current_prices'].get(ticker, 0.0)
        return result
    
    
    def get_historical_data(self, ticker: str, period: str = "5y") -> pd.DataFrame:
        """
        単一銘柄の過去データを取得（price_fetcher互換）
        """
        return self.data_bundle['historical_prices'].get(ticker, pd.DataFrame())
    
    
    def get_multiple_historical_data(self, tickers: List[str], period: str = "5y") -> Dict[str, pd.DataFrame]:
        """
        複数銘柄の過去データを取得（price_fetcher互換）
        """
        result = {}
        for ticker in tickers:
            result[ticker] = self.data_bundle['historical_prices'].get(ticker, pd.DataFrame())
        return result
    
    
    def get_exchange_rates(self) -> Dict[str, float]:
        """
        為替レートを取得（price_fetcher互換）
        """
        return self.data_bundle['exchange_rates']
    
    
    def get_currency_mapping(self) -> Dict[str, str]:
        """
        通貨マッピングを取得
        """
        return self.data_bundle['currency_mapping']
    
    
    # country_fetcher.py 互換メソッド
    def get_ticker_complete_info(self, ticker: str) -> Dict[str, Any]:
        """
        単一銘柄の完全企業情報を取得（country_fetcher互換）
        """
        return self.data_bundle['company_info'].get(ticker, {})
    
    
    def get_multiple_ticker_complete_info(self, tickers: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        複数銘柄の完全企業情報を取得（country_fetcher互換）
        """
        result = {}
        for ticker in tickers:
            result[ticker] = self.data_bundle['company_info'].get(ticker, {})
        return result
    
    
    def get_ticker_country(self, ticker: str) -> Optional[str]:
        """
        単一銘柄の本社所在国を取得（country_fetcher互換）
        """
        info = self.data_bundle['company_info'].get(ticker, {})
        return info.get('country')
    
    
    def get_ticker_sector(self, ticker: str) -> Optional[str]:
        """
        単一銘柄のセクターを取得（country_fetcher互換）
        """
        info = self.data_bundle['company_info'].get(ticker, {})
        return info.get('sector')
    
    
    def get_multiple_ticker_countries(self, tickers: List[str]) -> Dict[str, Optional[str]]:
        """
        複数銘柄の本社所在国を取得（country_fetcher互換）
        """
        result = {}
        for ticker in tickers:
            info = self.data_bundle['company_info'].get(ticker, {})
            result[ticker] = info.get('country')
        return result
    
    
    def get_ticker_valuation(self, ticker: str) -> Dict[str, Optional[float]]:
        """
        単一銘柄のバリュエーション指標を取得（country_fetcher互換）
        """
        info = self.data_bundle['company_info'].get(ticker, {})
        
        valuation_keys = ['forwardPE', 'priceToBook', 'priceToSalesTrailing12Months', 
                         'enterpriseToEbitda', 'pegRatio', 'marketCap', 'beta', 'dividendYield']
        
        result = {}
        for key in valuation_keys:
            result[key] = info.get(key)
        
        return result
    
    
    # factor_analysis.py 互換メソッド
    def get_fama_french_factors(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Fama-Frenchファクターデータを取得（factor_analysis互換）
        
        Returns:
            pd.DataFrame: ファクターデータ（空の場合はDataFrame()）
        """
        logger.info(f"🔍 キャッシュからファクターデータ取得: {start_date} ～ {end_date}")
        
        factor_data = self.data_bundle.get('factor_data', {})
        
        # データが空またはキーが存在しない場合
        if not factor_data:
            logger.warning("⚠️ ファクターデータがキャッシュにありません")
            return pd.DataFrame()
        
        # 最初のDataFrameを取得（通常は'FF5_Factors'）
        factor_df = None
        for key, df in factor_data.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                factor_df = df
                logger.info(f"📊 ファクターデータ発見: {key} ({len(df)}日分)")
                break
        
        if factor_df is None:
            logger.warning("⚠️ 有効なファクターDataFrameが見つかりません")
            return pd.DataFrame()
        
        # 日付フィルタリング（指定されている場合）
        if start_date is not None and end_date is not None:
            try:
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                
                # インデックスが日付の場合のみフィルタリング
                if hasattr(factor_df.index, 'date') or pd.api.types.is_datetime64_any_dtype(factor_df.index):
                    mask = (factor_df.index >= start_dt) & (factor_df.index <= end_dt)
                    filtered_df = factor_df[mask]
                    
                    logger.info(f"📅 日付フィルタリング結果: {len(filtered_df)}日分 (元: {len(factor_df)}日分)")
                    return filtered_df
                else:
                    logger.info("📅 インデックスが日付形式ではないため、フィルタリングをスキップ")
                    return factor_df
                    
            except Exception as e:
                logger.warning(f"⚠️ 日付フィルタリングエラー: {str(e)}")
                return factor_df
        
        logger.info(f"📊 ファクターデータ取得成功: {len(factor_df)}日分")
        return factor_df
    
    
    # pnl_calculator.py 互換メソッド
    def get_etf_benchmark_data(self) -> Dict[str, Dict[str, Optional[float]]]:
        """
        ETFベンチマークデータを取得（pnl_calculator互換）
        """
        return self.data_bundle['etf_benchmarks']
    
    
    # データ品質情報の提供
    def get_data_quality_report(self) -> Dict[str, Any]:
        """
        データ品質レポートを取得
        """
        return self.data_bundle.get('data_quality', {})
    
    
    def get_data_freshness_summary(self) -> str:
        """
        データの新鮮度サマリーを取得
        """
        quality = self.get_data_quality_report()
        if not quality:
            return "データ品質情報なし"
        
        return (f"価格データ: {quality.get('price_success_rate', 0):.1f}%成功, "
                f"企業情報: {quality.get('company_info_success_rate', 0):.1f}%成功, "
                f"過去データ: {quality.get('historical_data_success_rate', 0):.1f}%成功")
    
    
    def get_missing_data_tickers(self) -> List[str]:
        """
        データが不足している銘柄リストを取得
        """
        quality = self.get_data_quality_report()
        missing_data = quality.get('missing_data', [])
        
        return [item['ticker'] for item in missing_data]
    
    
    # ユーティリティメソッド
    def has_sufficient_data(self, ticker: str) -> bool:
        """
        銘柄が十分なデータを持っているかチェック
        """
        # 最低限、現在価格があれば十分とみなす
        current_price = self.data_bundle['current_prices'].get(ticker, 0)
        return current_price > 0
    
    
    def get_available_tickers(self) -> List[str]:
        """
        データが利用可能な銘柄リストを取得
        """
        return [ticker for ticker in self.data_bundle['current_prices'].keys() 
                if self.has_sufficient_data(ticker)]
    
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        データの概要を取得
        """
        current_prices = self.data_bundle['current_prices']
        company_info = self.data_bundle['company_info']
        historical_prices = self.data_bundle['historical_prices']
        
        return {
            'total_tickers': len(current_prices),
            'tickers_with_prices': len([p for p in current_prices.values() if p > 0]),
            'tickers_with_company_info': len([info for info in company_info.values() if info]),
            'tickers_with_historical_data': len([df for df in historical_prices.values() if not df.empty]),
            'exchange_rates_available': len(self.data_bundle['exchange_rates']),
            'factor_data_available': bool(self.data_bundle['factor_data']),
            'etf_benchmarks_available': bool(self.data_bundle['etf_benchmarks'])
        }


# ヘルパー関数
def create_data_adapter(data_bundle: Dict[str, Any]) -> DataAdapter:
    """
    DataAdapterインスタンスを作成
    """
    return DataAdapter(data_bundle)


def validate_data_bundle(data_bundle: Dict[str, Any]) -> bool:
    """
    データバンドルの妥当性をチェック
    """
    required_keys = ['current_prices', 'exchange_rates', 'company_info', 
                    'historical_prices', 'currency_mapping']
    
    return all(key in data_bundle for key in required_keys)