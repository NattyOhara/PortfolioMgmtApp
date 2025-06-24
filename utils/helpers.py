"""
汎用ヘルパー関数
共通的に使用される機能
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Any, Dict, List, Optional, Union
import logging
from datetime import datetime, timedelta
import time

logger = logging.getLogger(__name__)


def format_currency(amount: float, currency: str = 'JPY') -> str:
    """
    通貨フォーマット
    
    Args:
        amount: 金額
        currency: 通貨コード
    
    Returns:
        str: フォーマット済み通貨文字列
    """
    try:
        if currency == 'JPY':
            return f"¥{amount:,.0f}"
        elif currency == 'USD':
            return f"${amount:,.2f}"
        elif currency == 'EUR':
            return f"€{amount:,.2f}"
        elif currency == 'GBP':
            return f"£{amount:,.2f}"
        else:
            return f"{amount:,.2f} {currency}"
    except:
        return f"{amount} {currency}"


def format_percentage(value: float, decimal_places: int = 2) -> str:
    """
    パーセンテージフォーマット
    
    Args:
        value: パーセンテージ値
        decimal_places: 小数点以下桁数
    
    Returns:
        str: フォーマット済みパーセンテージ文字列
    """
    try:
        if value > 0:
            return f"+{value:.{decimal_places}f}%"
        else:
            return f"{value:.{decimal_places}f}%"
    except:
        return f"{value}%"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    安全な除算（ゼロ除算回避）
    
    Args:
        numerator: 分子
        denominator: 分母
        default: デフォルト値
    
    Returns:
        float: 除算結果またはデフォルト値
    """
    try:
        if denominator == 0 or pd.isna(denominator):
            return default
        return numerator / denominator
    except:
        return default


def calculate_returns(prices: pd.Series) -> pd.Series:
    """
    価格系列からリターン系列を計算
    
    Args:
        prices: 価格系列
    
    Returns:
        pd.Series: リターン系列
    """
    try:
        return prices.pct_change().dropna()
    except Exception as e:
        logger.error(f"リターン計算エラー: {str(e)}")
        return pd.Series()


def calculate_cumulative_returns(returns: pd.Series) -> pd.Series:
    """
    累積リターンを計算
    
    Args:
        returns: リターン系列
    
    Returns:
        pd.Series: 累積リターン系列
    """
    try:
        return (1 + returns).cumprod() - 1
    except Exception as e:
        logger.error(f"累積リターン計算エラー: {str(e)}")
        return pd.Series()


def handle_missing_data(df: pd.DataFrame, method: str = 'drop') -> pd.DataFrame:
    """
    欠損データの処理
    
    Args:
        df: DataFrame
        method: 処理方法 ('drop', 'forward_fill', 'backward_fill')
    
    Returns:
        pd.DataFrame: 処理済みDataFrame
    """
    try:
        if method == 'drop':
            return df.dropna()
        elif method == 'forward_fill':
            return df.fillna(method='ffill')
        elif method == 'backward_fill':
            return df.fillna(method='bfill')
        else:
            return df
    except Exception as e:
        logger.error(f"欠損データ処理エラー: {str(e)}")
        return df


def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    DataFrameの検証
    
    Args:
        df: 検証するDataFrame
        required_columns: 必須列のリスト
    
    Returns:
        bool: 検証結果
    """
    try:
        if df.empty:
            return False
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.warning(f"不足している列: {missing_columns}")
            return False
        
        return True
    except Exception as e:
        logger.error(f"DataFrame検証エラー: {str(e)}")
        return False


def create_date_range(start_date: str, end_date: str, freq: str = 'D') -> pd.DatetimeIndex:
    """
    日付範囲を作成
    
    Args:
        start_date: 開始日（YYYY-MM-DD形式）
        end_date: 終了日（YYYY-MM-DD形式）
        freq: 頻度（'D', 'B', 'M'等）
    
    Returns:
        pd.DatetimeIndex: 日付インデックス
    """
    try:
        return pd.date_range(start=start_date, end=end_date, freq=freq)
    except Exception as e:
        logger.error(f"日付範囲作成エラー: {str(e)}")
        return pd.DatetimeIndex([])


def get_business_days_between(start_date: datetime, end_date: datetime) -> int:
    """
    営業日数を計算
    
    Args:
        start_date: 開始日
        end_date: 終了日
    
    Returns:
        int: 営業日数
    """
    try:
        business_days = pd.bdate_range(start=start_date, end=end_date)
        return len(business_days)
    except Exception as e:
        logger.error(f"営業日数計算エラー: {str(e)}")
        return 0


def retry_operation(func, max_retries: int = 3, delay: float = 1.0):
    """
    リトライ機能付き操作実行
    
    Args:
        func: 実行する関数
        max_retries: 最大リトライ回数
        delay: リトライ間隔（秒）
    
    Returns:
        Any: 関数の実行結果
    """
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"最大リトライ回数到達: {str(e)}")
                raise e
            else:
                logger.warning(f"リトライ {attempt + 1}/{max_retries}: {str(e)}")
                time.sleep(delay)


def log_performance(func):
    """
    関数実行時間をログ出力するデコレータ
    
    Args:
        func: 対象関数
    
    Returns:
        関数: ラップした関数
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            logger.info(f"{func.__name__} 実行時間: {end_time - start_time:.2f}秒")
            return result
        except Exception as e:
            end_time = time.time()
            logger.error(f"{func.__name__} エラー（実行時間: {end_time - start_time:.2f}秒）: {str(e)}")
            raise e
    return wrapper


def display_dataframe_info(df: pd.DataFrame, title: str = "DataFrame情報"):
    """
    DataFrameの情報を表示
    
    Args:
        df: 表示するDataFrame
        title: タイトル
    """
    try:
        with st.expander(f"📊 {title}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("行数", len(df))
            
            with col2:
                st.metric("列数", len(df.columns))
            
            with col3:
                memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024
                st.metric("メモリ使用量", f"{memory_usage:.2f} MB")
            
            if not df.empty:
                st.subheader("データ型")
                dtype_df = pd.DataFrame({
                    '列名': df.columns,
                    'データ型': df.dtypes.values,
                    '欠損値数': df.isnull().sum().values
                })
                st.dataframe(dtype_df, use_container_width=True)
    except Exception as e:
        logger.error(f"DataFrame情報表示エラー: {str(e)}")


def create_download_link(df: pd.DataFrame, filename: str, link_text: str = "ダウンロード"):
    """
    DataFrameのダウンロードリンクを作成
    
    Args:
        df: ダウンロードするDataFrame
        filename: ファイル名
        link_text: リンクテキスト
    """
    try:
        csv = df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label=f"📥 {link_text}",
            data=csv,
            file_name=filename,
            mime="text/csv"
        )
    except Exception as e:
        logger.error(f"ダウンロードリンク作成エラー: {str(e)}")


def show_loading_spinner(text: str = "処理中..."):
    """
    ローディングスピナーを表示
    
    Args:
        text: 表示テキスト
    """
    return st.spinner(text)


def display_error_message(error: Exception, context: str = ""):
    """
    エラーメッセージを表示
    
    Args:
        error: エラーオブジェクト
        context: エラーのコンテキスト
    """
    error_msg = f"{context}: {str(error)}" if context else str(error)
    st.error(f"⚠️ {error_msg}")
    logger.error(error_msg)


def display_success_message(message: str):
    """
    成功メッセージを表示
    
    Args:
        message: 表示メッセージ
    """
    st.success(f"✅ {message}")
    logger.info(message)


def display_warning_message(message: str):
    """
    警告メッセージを表示
    
    Args:
        message: 表示メッセージ
    """
    st.warning(f"⚠️ {message}")
    logger.warning(message)


def display_info_message(message: str):
    """
    情報メッセージを表示
    
    Args:
        message: 表示メッセージ
    """
    st.info(f"ℹ️ {message}")
    logger.info(message)


def get_color_palette(n_colors: int) -> List[str]:
    """
    カラーパレットを取得
    
    Args:
        n_colors: 必要な色数
    
    Returns:
        List[str]: 色のリスト
    """
    import plotly.colors as pc
    
    if n_colors <= 10:
        return pc.qualitative.Plotly[:n_colors]
    else:
        # 色数が多い場合は繰り返し
        base_colors = pc.qualitative.Plotly
        return (base_colors * ((n_colors // len(base_colors)) + 1))[:n_colors]


def calculate_correlation_significance(corr_matrix: pd.DataFrame, n_observations: int) -> pd.DataFrame:
    """
    相関係数の有意性を計算
    
    Args:
        corr_matrix: 相関行列
        n_observations: 観測数
    
    Returns:
        pd.DataFrame: 有意性マトリックス（True/False）
    """
    try:
        from scipy import stats
        
        # t統計量を計算
        t_stat = corr_matrix * np.sqrt((n_observations - 2) / (1 - corr_matrix**2))
        
        # p値を計算
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stat), n_observations - 2))
        
        # 5%水準で有意かどうか
        significant = p_values < 0.05
        
        return significant
    except Exception as e:
        logger.error(f"相関有意性計算エラー: {str(e)}")
        return pd.DataFrame()


def clean_numeric_data(series: pd.Series) -> pd.Series:
    """
    数値データのクリーニング
    
    Args:
        series: 数値系列
    
    Returns:
        pd.Series: クリーニング済み系列
    """
    try:
        # 数値以外を除去
        numeric_series = pd.to_numeric(series, errors='coerce')
        
        # 無限大値を除去
        numeric_series = numeric_series.replace([np.inf, -np.inf], np.nan)
        
        # 異常値検出（3シグマルール）
        if len(numeric_series.dropna()) > 0:
            mean_val = numeric_series.mean()
            std_val = numeric_series.std()
            
            if std_val > 0:
                outlier_mask = (np.abs(numeric_series - mean_val) > 3 * std_val)
                numeric_series[outlier_mask] = np.nan
        
        return numeric_series
    except Exception as e:
        logger.error(f"数値データクリーニングエラー: {str(e)}")
        return series