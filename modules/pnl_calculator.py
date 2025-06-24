"""
損益計算モジュール
ポートフォリオの損益計算とパフォーマンス分析機能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def calculate_pnl(
    ticker: str,
    shares: float,
    avg_cost_jpy: float,
    current_price_local: float,
    exchange_rate: float = 1.0
) -> Dict[str, float]:
    """
    単一銘柄の損益計算（日本円ベース）
    
    Args:
        ticker: ティッカーシンボル
        shares: 保有株数
        avg_cost_jpy: 日本円ベース平均購入単価
        current_price_local: 現在株価（現地通貨）
        exchange_rate: 為替レート（現地通貨→JPY）
    
    Returns:
        dict: 損益情報
    """
    try:
        # 現在価格を日本円に換算
        current_price_jpy = current_price_local * exchange_rate
        
        # 現在評価額
        current_value_jpy = current_price_jpy * shares
        
        # 投資額（簿価）
        cost_basis_jpy = avg_cost_jpy * shares
        
        # 損益額
        pnl_amount = current_value_jpy - cost_basis_jpy
        
        # 損益率
        pnl_percentage = (pnl_amount / cost_basis_jpy) * 100 if cost_basis_jpy > 0 else 0
        
        result = {
            'ticker': ticker,
            'shares': shares,
            'avg_cost_jpy': avg_cost_jpy,
            'current_price_local': current_price_local,
            'current_price_jpy': current_price_jpy,
            'exchange_rate': exchange_rate,
            'current_value_jpy': current_value_jpy,
            'cost_basis_jpy': cost_basis_jpy,
            'pnl_amount': pnl_amount,
            'pnl_percentage': pnl_percentage
        }
        
        logger.debug(f"損益計算完了 {ticker}: {pnl_amount:,.0f}円 ({pnl_percentage:.2f}%)")
        return result
        
    except Exception as e:
        logger.error(f"損益計算エラー {ticker}: {str(e)}")
        return {
            'ticker': ticker,
            'shares': shares,
            'avg_cost_jpy': avg_cost_jpy,
            'current_price_local': 0,
            'current_price_jpy': 0,
            'exchange_rate': exchange_rate,
            'current_value_jpy': 0,
            'cost_basis_jpy': avg_cost_jpy * shares,
            'pnl_amount': -(avg_cost_jpy * shares),
            'pnl_percentage': -100.0
        }


def calculate_portfolio_pnl(
    portfolio_df: pd.DataFrame,
    current_prices: Dict[str, float],
    exchange_rates: Dict[str, float],
    currency_mapping: Dict[str, str]
) -> pd.DataFrame:
    """
    ポートフォリオ全体の損益計算
    
    Args:
        portfolio_df: ポートフォリオデータ
        current_prices: 現在株価辞書
        exchange_rates: 為替レート辞書
        currency_mapping: ティッカーと通貨のマッピング
    
    Returns:
        pd.DataFrame: 損益計算結果
    """
    pnl_results = []
    
    for _, row in portfolio_df.iterrows():
        ticker = row['Ticker']
        shares = row['Shares']
        avg_cost_jpy = row['AvgCostJPY']
        
        # 現在株価を取得
        current_price_local = current_prices.get(ticker, 0)
        
        # 通貨と為替レートを取得
        currency = currency_mapping.get(ticker, 'USD')
        exchange_rate = get_exchange_rate_for_currency(currency, exchange_rates)
        
        # 損益計算
        pnl_data = calculate_pnl(
            ticker=ticker,
            shares=shares,
            avg_cost_jpy=avg_cost_jpy,
            current_price_local=current_price_local,
            exchange_rate=exchange_rate
        )
        
        # 通貨情報を追加
        pnl_data['currency'] = currency
        
        pnl_results.append(pnl_data)
    
    # DataFrameに変換
    pnl_df = pd.DataFrame(pnl_results)
    
    logger.info(f"ポートフォリオ損益計算完了: {len(pnl_df)}銘柄")
    return pnl_df


def get_exchange_rate_for_currency(currency: str, exchange_rates: Dict[str, float]) -> float:
    """
    通貨に対応する為替レートを取得
    
    Args:
        currency: 通貨コード
        exchange_rates: 為替レート辞書
    
    Returns:
        float: 為替レート（JPYに対する）
    """
    if currency == 'JPY':
        return 1.0
    
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
        return exchange_rates[rate_symbol]
    
    # フォールバック：概算レート
    fallback_rates = {
        'USD': 150.0,
        'EUR': 160.0,
        'GBP': 180.0,
        'AUD': 100.0,
        'CAD': 110.0,
        'CHF': 165.0,
        'HKD': 19.0,
        'SGD': 110.0,
        'CNY': 21.0,
        'KRW': 0.11
    }
    
    return fallback_rates.get(currency, 1.0)


def calculate_portfolio_summary(pnl_df: pd.DataFrame) -> Dict[str, float]:
    """
    ポートフォリオサマリーの計算
    
    Args:
        pnl_df: 損益計算結果DataFrame
    
    Returns:
        dict: ポートフォリオサマリー
    """
    try:
        total_cost_basis = pnl_df['cost_basis_jpy'].sum()
        total_current_value = pnl_df['current_value_jpy'].sum()
        total_pnl_amount = pnl_df['pnl_amount'].sum()
        
        overall_pnl_percentage = (total_pnl_amount / total_cost_basis) * 100 if total_cost_basis > 0 else 0
        
        # 勝率計算
        profitable_positions = (pnl_df['pnl_amount'] > 0).sum()
        total_positions = len(pnl_df)
        win_rate = (profitable_positions / total_positions) * 100 if total_positions > 0 else 0
        
        # 最大・最小損益
        max_gain = pnl_df['pnl_amount'].max()
        max_loss = pnl_df['pnl_amount'].min()
        max_gain_pct = pnl_df['pnl_percentage'].max()
        max_loss_pct = pnl_df['pnl_percentage'].min()
        
        # 最大・最小損益銘柄
        max_gain_ticker = pnl_df.loc[pnl_df['pnl_amount'].idxmax(), 'ticker'] if not pnl_df.empty else ''
        max_loss_ticker = pnl_df.loc[pnl_df['pnl_amount'].idxmin(), 'ticker'] if not pnl_df.empty else ''
        
        summary = {
            'total_positions': total_positions,
            'total_cost_basis_jpy': total_cost_basis,
            'total_current_value_jpy': total_current_value,
            'total_pnl_amount_jpy': total_pnl_amount,
            'overall_pnl_percentage': overall_pnl_percentage,
            'win_rate': win_rate,
            'profitable_positions': profitable_positions,
            'losing_positions': total_positions - profitable_positions,
            'max_gain_amount': max_gain,
            'max_loss_amount': max_loss,
            'max_gain_percentage': max_gain_pct,
            'max_loss_percentage': max_loss_pct,
            'max_gain_ticker': max_gain_ticker,
            'max_loss_ticker': max_loss_ticker,
            'average_position_size': total_cost_basis / total_positions if total_positions > 0 else 0
        }
        
        logger.info(f"ポートフォリオサマリー計算完了: 総損益 {total_pnl_amount:,.0f}円 ({overall_pnl_percentage:.2f}%)")
        return summary
        
    except Exception as e:
        logger.error(f"ポートフォリオサマリー計算エラー: {str(e)}")
        return {}


def calculate_sector_allocation_by_region(pnl_df: pd.DataFrame, ticker_countries: dict = None) -> pd.DataFrame:
    """
    地域別配分の計算（Yahoo Finance country情報ベース）
    
    Args:
        pnl_df: 損益計算結果DataFrame
        ticker_countries: ティッカー別本社所在国辞書
    
    Returns:
        pd.DataFrame: 地域別配分データ
    """
    try:
        if pnl_df.empty:
            logger.warning("損益データが空です")
            return pd.DataFrame()
            
        from modules.country_fetcher import classify_region_by_country
        
        # 地域分類関数
        def get_region_for_ticker(ticker):
            if ticker_countries and ticker in ticker_countries:
                country = ticker_countries[ticker]
                if country and country.strip():  # 空文字もチェック
                    return classify_region_by_country(country)
            
            # フォールバック：ティッカーサフィックスベース
            ticker_str = str(ticker)
            if '.T' in ticker_str or '.JP' in ticker_str:
                return '日本'
            elif '.AS' in ticker_str or '.PA' in ticker_str or '.DE' in ticker_str or '.MI' in ticker_str or '.L' in ticker_str or '.SW' in ticker_str:
                return '欧州'
            elif '.TO' in ticker_str or '.V' in ticker_str:
                return '北米（その他）'
            elif '.AX' in ticker_str:
                return 'アジア太平洋'
            elif '.HK' in ticker_str or '.SS' in ticker_str or '.KS' in ticker_str:
                return 'アジア太平洋'
            else:
                # ETFや不明確な商品の検出
                ticker_upper = ticker.upper()
                etf_indicators = ['ETF', 'FUND', 'GOLD', 'GLD', 'SLV', 'GLDM', 'EPI', 'INDEX', 'SPDR', 'ISHARES', 'VANGUARD']
                is_likely_etf = any(indicator in ticker_upper for indicator in etf_indicators)
                
                if is_likely_etf:
                    return 'その他'
                else:
                    # 明確にアメリカ企業と判断できる有名企業のみアメリカに分類
                    well_known_us_companies = [
                        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM',
                        'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'BAC', 'ADBE', 'CRM', 
                        'NFLX', 'KO', 'PEP', 'ORCL', 'CSCO', 'INTC', 'VZ', 'PFE', 'TMO',
                        'NKE', 'MRK', 'ABT', 'CVX', 'WMT', 'XOM', 'LLY', 'COST', 'SPGI', 
                        'ZTS', 'CAT', 'MSTR', 'IONQ'
                    ]
                    
                    if ticker_upper in well_known_us_companies:
                        return '米国'
                    else:
                        return 'その他'
        
        # 地域分類を適用
        pnl_df_copy = pnl_df.copy()
        pnl_df_copy['country'] = pnl_df_copy['ticker'].apply(get_region_for_ticker)
        
        logger.info(f"地域分類結果: {pnl_df_copy['country'].value_counts().to_dict()}")
        
        # 地域別集計
        sector_allocation = pnl_df_copy.groupby('country').agg({
            'current_value_jpy': 'sum',
            'cost_basis_jpy': 'sum',
            'pnl_amount': 'sum',
            'ticker': 'count'
        }).rename(columns={'ticker': 'position_count'})
        
        if sector_allocation.empty:
            logger.warning("地域集計結果が空です")
            return pd.DataFrame()
        
        # 配分比率を計算
        total_value = sector_allocation['current_value_jpy'].sum()
        if total_value > 0:
            sector_allocation['allocation_percentage'] = (
                sector_allocation['current_value_jpy'] / total_value * 100
            )
        else:
            sector_allocation['allocation_percentage'] = 0
        
        # 損益率を計算
        sector_allocation['pnl_percentage'] = sector_allocation.apply(
            lambda row: (row['pnl_amount'] / row['cost_basis_jpy'] * 100) if row['cost_basis_jpy'] > 0 else 0, 
            axis=1
        )
        
        result_df = sector_allocation.reset_index()
        logger.info(f"地域配分計算完了: {len(result_df)}地域")
        return result_df
        
    except Exception as e:
        logger.error(f"地域配分計算エラー: {str(e)}")
        return pd.DataFrame()


def calculate_sector_allocation(pnl_df: pd.DataFrame, ticker_info: dict = None) -> pd.DataFrame:
    """
    セクター別配分の計算（Yahoo Finance sector情報ベース）
    
    Args:
        pnl_df: 損益計算結果DataFrame
        ticker_info: ティッカー別企業情報辞書（country、sectorを含む）
    
    Returns:
        pd.DataFrame: セクター別配分データ
    """
    try:
        if pnl_df.empty:
            logger.warning("損益データが空です")
            return pd.DataFrame()
        
        # セクター分類関数
        def get_sector_for_ticker(ticker):
            if ticker_info and ticker in ticker_info and ticker_info[ticker]:
                sector = ticker_info[ticker].get('sector')
                if sector and sector.strip():  # 空文字もチェック
                    return sector.strip()
            
            # フォールバック：日本株かどうかで分類
            if '.T' in str(ticker) or '.JP' in str(ticker):
                return "その他（日本）"
            else:
                return "その他"
        
        # セクター分類を適用
        pnl_df_copy = pnl_df.copy()
        pnl_df_copy['sector'] = pnl_df_copy['ticker'].apply(get_sector_for_ticker)
        
        logger.info(f"セクター分類結果: {pnl_df_copy['sector'].value_counts().to_dict()}")
        
        # セクター別集計
        sector_allocation = pnl_df_copy.groupby('sector').agg({
            'current_value_jpy': 'sum',
            'cost_basis_jpy': 'sum',
            'pnl_amount': 'sum',
            'ticker': 'count'
        }).rename(columns={'ticker': 'position_count'})
        
        if sector_allocation.empty:
            logger.warning("セクター集計結果が空です")
            return pd.DataFrame()
        
        # 配分比率を計算
        total_value = sector_allocation['current_value_jpy'].sum()
        if total_value > 0:
            sector_allocation['allocation_percentage'] = (
                sector_allocation['current_value_jpy'] / total_value * 100
            )
        else:
            sector_allocation['allocation_percentage'] = 0
        
        # 損益率を計算
        sector_allocation['pnl_percentage'] = sector_allocation.apply(
            lambda row: (row['pnl_amount'] / row['cost_basis_jpy'] * 100) if row['cost_basis_jpy'] > 0 else 0, 
            axis=1
        )
        
        result_df = sector_allocation.reset_index()
        logger.info(f"セクター配分計算完了: {len(result_df)}セクター")
        return result_df
        
    except Exception as e:
        logger.error(f"セクター配分計算エラー: {str(e)}")
        return pd.DataFrame()


def calculate_performance_metrics(pnl_df: pd.DataFrame, risk_free_rate: float = 0.1) -> Dict[str, float]:
    """
    パフォーマンス指標の計算
    
    Args:
        pnl_df: 損益計算結果DataFrame
        risk_free_rate: リスクフリーレート（年率%）
    
    Returns:
        dict: パフォーマンス指標
    """
    try:
        if pnl_df.empty:
            return {}
        
        # 重み（時価総額比率）を計算
        total_value = pnl_df['current_value_jpy'].sum()
        weights = pnl_df['current_value_jpy'] / total_value if total_value > 0 else np.ones(len(pnl_df)) / len(pnl_df)
        
        # 重み付き平均リターン
        weighted_return = (pnl_df['pnl_percentage'] * weights).sum()
        
        # 銘柄別リターンの標準偏差（簡易版）
        returns_std = pnl_df['pnl_percentage'].std()
        
        # シャープレシオ（簡易計算）
        excess_return = weighted_return - risk_free_rate
        sharpe_ratio = excess_return / returns_std if returns_std > 0 else 0
        
        # 最大ドローダウン（単純版：最大損失銘柄）
        max_drawdown = pnl_df['pnl_percentage'].min()
        
        metrics = {
            'weighted_return': weighted_return,
            'returns_std': returns_std,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'risk_free_rate': risk_free_rate
        }
        
        logger.info(f"パフォーマンス指標計算完了: シャープレシオ {sharpe_ratio:.3f}")
        return metrics
        
    except Exception as e:
        logger.error(f"パフォーマンス指標計算エラー: {str(e)}")
        return {}


def calculate_position_sizing_analysis(pnl_df: pd.DataFrame) -> Dict[str, any]:
    """
    ポジションサイジング分析
    
    Args:
        pnl_df: 損益計算結果DataFrame
    
    Returns:
        dict: ポジションサイジング分析結果
    """
    try:
        if pnl_df.empty:
            return {}
        
        total_value = pnl_df['current_value_jpy'].sum()
        
        # 各ポジションの比率
        pnl_df['position_weight'] = pnl_df['current_value_jpy'] / total_value * 100
        
        # 集中度分析
        top_5_weight = pnl_df.nlargest(5, 'position_weight')['position_weight'].sum()
        top_10_weight = pnl_df.nlargest(10, 'position_weight')['position_weight'].sum()
        
        # ハーフィンダール指数（集中度指標）
        hhi = (pnl_df['position_weight'] ** 2).sum()
        
        # 等分散からの偏差
        equal_weight = 100 / len(pnl_df)
        weight_variance = pnl_df['position_weight'].var()
        
        analysis = {
            'total_positions': len(pnl_df),
            'top_5_concentration': top_5_weight,
            'top_10_concentration': top_10_weight,
            'herfindahl_index': hhi,
            'equal_weight_benchmark': equal_weight,
            'weight_variance': weight_variance,
            'max_position_weight': pnl_df['position_weight'].max(),
            'min_position_weight': pnl_df['position_weight'].min(),
            'largest_position': pnl_df.loc[pnl_df['position_weight'].idxmax(), 'ticker'],
            'smallest_position': pnl_df.loc[pnl_df['position_weight'].idxmin(), 'ticker']
        }
        
        logger.info(f"ポジションサイジング分析完了: 上位5銘柄集中度 {top_5_weight:.1f}%")
        return analysis
        
    except Exception as e:
        logger.error(f"ポジションサイジング分析エラー: {str(e)}")
        return {}


def get_etf_benchmark_data() -> Dict[str, Dict[str, Optional[float]]]:
    """
    ベンチマークETFのバリュエーション指標を取得
    
    Returns:
        Dict[str, Dict[str, Optional[float]]]: ETF別バリュエーション指標辞書
    """
    try:
        # ベンチマークETFとその対応する市場指標名
        benchmark_etfs = {
            'ACWI': 'MSCI ACWI',
            'QQQ': 'NASDAQ-100', 
            'SPY': 'S&P 500',
            'EWJ': 'TOPIX (Japan)'
        }
        
        # ETFスクレイパーの結果をシミュレート（実際の実装では etf_scraper モジュールを使用）
        # 現在はサンプルデータを返す（主要指標のみ）
        etf_data = {
            'MSCI ACWI': {
                'forwardPE': 15.8,
                'priceToBook': 2.2,
                'returnOnEquity': 0.14,
                'dividendYield': 2.1,
                'beta': 1.0,
                # 他の指標はNoneまたは不明
                'priceToSalesTrailing12Months': None,
                'enterpriseToEbitda': None,
                'pegRatio': None,
                'marketCap': None,
                'returnOnAssets': None,
                'operatingMargins': None,
                'profitMargins': None
            },
            'NASDAQ-100': {
                'forwardPE': 26.5,
                'priceToBook': 5.1,
                'returnOnEquity': 0.22,
                'dividendYield': 0.8,
                'beta': 1.15,
                # 他の指標はNoneまたは不明
                'priceToSalesTrailing12Months': None,
                'enterpriseToEbitda': None,
                'pegRatio': None,
                'marketCap': None,
                'returnOnAssets': None,
                'operatingMargins': None,
                'profitMargins': None
            },
            'S&P 500': {
                'forwardPE': 19.2,
                'priceToBook': 3.8,
                'returnOnEquity': 0.18,
                'dividendYield': 1.6,
                'beta': 1.0,
                # 他の指標はNoneまたは不明
                'priceToSalesTrailing12Months': None,
                'enterpriseToEbitda': None,
                'pegRatio': None,
                'marketCap': None,
                'returnOnAssets': None,
                'operatingMargins': None,
                'profitMargins': None
            },
            'TOPIX (Japan)': {
                'forwardPE': 13.5,
                'priceToBook': 1.1,
                'returnOnEquity': 0.08,
                'dividendYield': 2.8,
                'beta': 0.85,
                # 他の指標はNoneまたは不明
                'priceToSalesTrailing12Months': None,
                'enterpriseToEbitda': None,
                'pegRatio': None,
                'marketCap': None,
                'returnOnAssets': None,
                'operatingMargins': None,
                'profitMargins': None
            }
        }
        
        logger.info(f"ベンチマークETFデータ取得完了: {len(etf_data)}指標")
        return etf_data
        
    except Exception as e:
        logger.error(f"ベンチマークETFデータ取得エラー: {str(e)}")
        return {}


def calculate_portfolio_valuation_metrics(pnl_df: pd.DataFrame, ticker_complete_info: dict, include_etf_benchmarks: bool = True) -> pd.DataFrame:
    """
    ポートフォリオのバリュエーション指標統計を計算（ベンチマークETF比較含む）
    
    Args:
        pnl_df: 損益計算結果DataFrame
        ticker_complete_info: ティッカー別完全企業情報辞書
        include_etf_benchmarks: ベンチマークETFデータを含めるかどうか
    
    Returns:
        pd.DataFrame: バリュエーション指標統計
    """
    try:
        if pnl_df.empty or not ticker_complete_info:
            logger.warning("バリュエーション計算用データが不足しています")
            return pd.DataFrame()
        
        # ベンチマークETFデータを取得
        etf_benchmark_data = get_etf_benchmark_data() if include_etf_benchmarks else {}
        
        # バリュエーション指標のキー（新しい財務指標を追加）
        valuation_keys = ['forwardPE', 'priceToBook', 'priceToSalesTrailing12Months', 
                         'enterpriseToEbitda', 'pegRatio', 'marketCap', 'beta', 'dividendYield',
                         'returnOnEquity', 'returnOnAssets', 'operatingMargins', 'profitMargins']
        
        # 日本語ラベル
        japanese_labels = {
            'forwardPE': '予想PER',
            'priceToBook': 'PBR',
            'priceToSalesTrailing12Months': 'PSR',
            'enterpriseToEbitda': 'EV/EBITDA',
            'pegRatio': 'PEGレシオ',
            'marketCap': '時価総額（円）',
            'beta': 'ベータ',
            'dividendYield': '配当利回り',
            'returnOnEquity': 'ROE',
            'returnOnAssets': 'ROA', 
            'operatingMargins': '営業利益率',
            'profitMargins': '純利益率'
        }
        
        results = []
        
        for metric in valuation_keys:
            # 各銘柄のバリュエーション指標と重みを取得
            metric_data = []
            weights = []
            
            for _, row in pnl_df.iterrows():
                ticker = row['ticker']
                portfolio_weight = row['current_value_jpy']
                
                if ticker in ticker_complete_info:
                    metric_value = ticker_complete_info[ticker].get(metric)
                    if metric_value is not None and pd.notna(metric_value):
                        metric_data.append(metric_value)
                        weights.append(portfolio_weight)
            
            # ベンチマークETFデータを取得（該当する指標のみ）
            etf_values = {}
            for etf_name, etf_data in etf_benchmark_data.items():
                etf_values[etf_name] = etf_data.get(metric)
            
            if not metric_data:
                # データがない場合でもETFデータは表示
                result_row = {
                    '指標': japanese_labels[metric]
                }
                # ETF列を指標の右に追加
                result_row.update(etf_values)
                # ポートフォリオ統計を追加
                result_row.update({
                    '加重平均': None,
                    '中央値': None,
                    '25%タイル': None,
                    '75%タイル': None,
                    '最小値': None,
                    '最大値': None,
                    '有効銘柄数': 0
                })
                results.append(result_row)
                continue
            
            # numpy配列に変換
            metric_array = np.array(metric_data)
            weights_array = np.array(weights)
            
            # 統計計算
            # 加重平均
            total_weight = weights_array.sum()
            weighted_avg = np.sum(metric_array * weights_array) / total_weight if total_weight > 0 else None
            
            # パーセンタイル
            q25 = np.percentile(metric_array, 25)
            median = np.percentile(metric_array, 50)
            q75 = np.percentile(metric_array, 75)
            min_val = np.min(metric_array)
            max_val = np.max(metric_array)
            
            result_row = {
                '指標': japanese_labels[metric]
            }
            # ETF列を指標の右に追加
            result_row.update(etf_values)
            # ポートフォリオ統計を追加
            result_row.update({
                '加重平均': weighted_avg,
                '中央値': median,
                '25%タイル': q25,
                '75%タイル': q75,
                '最小値': min_val,
                '最大値': max_val,
                '有効銘柄数': len(metric_data)
            })
            results.append(result_row)
        
        result_df = pd.DataFrame(results)
        logger.info(f"バリュエーション統計計算完了: {len(result_df)}指標")
        return result_df
        
    except Exception as e:
        logger.error(f"バリュエーション統計計算エラー: {str(e)}")
        return pd.DataFrame()