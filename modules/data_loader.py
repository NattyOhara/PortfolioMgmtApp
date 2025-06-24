"""
データローダーモジュール
CSVファイルの読み込みとバリデーション機能
"""

import pandas as pd
import streamlit as st
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


def load_portfolio_data(uploaded_file) -> Optional[pd.DataFrame]:
    """
    CSVファイルからポートフォリオデータを読み込む
    
    Args:
        uploaded_file: Streamlitのアップロードファイルオブジェクト
    
    Returns:
        pd.DataFrame: バリデーション済みのポートフォリオデータ
        None: エラーが発生した場合
    """
    try:
        # CSVファイルの読み込み
        df = pd.read_csv(uploaded_file)
        
        # データバリデーション
        validation_result, error_messages = validate_portfolio_data(df)
        
        if not validation_result:
            for error in error_messages:
                st.error(error)
            return None
        
        # データクリーニング
        df = clean_portfolio_data(df)
        
        logger.info(f"ポートフォリオデータを正常に読み込みました: {len(df)}銘柄")
        return df
        
    except Exception as e:
        logger.error(f"ファイル読み込みエラー: {str(e)}")
        st.error(f"ファイル読み込みエラー: {str(e)}")
        return None


def validate_portfolio_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    ポートフォリオデータのバリデーション
    
    Args:
        df: 検証するDataFrame
    
    Returns:
        Tuple[bool, List[str]]: (検証結果, エラーメッセージリスト)
    """
    errors = []
    
    # 必須列の存在チェック
    required_columns = ['Ticker', 'Shares', 'AvgCostJPY']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        errors.append(f"必要な列が不足しています: {missing_columns}")
    
    # データが空でないかチェック
    if df.empty:
        errors.append("データが空です。")
        return False, errors
    
    # 列が存在する場合のデータ型チェック
    if 'Shares' in df.columns:
        if not pd.api.types.is_numeric_dtype(df['Shares']):
            errors.append("Shares列は数値である必要があります。")
        elif (df['Shares'] <= 0).any():
            errors.append("Shares列は正の値である必要があります。")
    
    if 'AvgCostJPY' in df.columns:
        if not pd.api.types.is_numeric_dtype(df['AvgCostJPY']):
            errors.append("AvgCostJPY列は数値である必要があります。")
        elif (df['AvgCostJPY'] <= 0).any():
            errors.append("AvgCostJPY列は正の値である必要があります。")
    
    # ティッカーシンボルの重複チェック
    if 'Ticker' in df.columns:
        duplicates = df['Ticker'].duplicated()
        if duplicates.any():
            duplicate_tickers = df.loc[duplicates, 'Ticker'].tolist()
            errors.append(f"重複するティッカーシンボルがあります: {duplicate_tickers}")
    
    # NaN値のチェック
    if df.isnull().any().any():
        null_columns = df.columns[df.isnull().any()].tolist()
        errors.append(f"以下の列にNaN値が含まれています: {null_columns}")
    
    return len(errors) == 0, errors


def clean_portfolio_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    ポートフォリオデータのクリーニング
    
    Args:
        df: クリーニング前のDataFrame
    
    Returns:
        pd.DataFrame: クリーニング後のDataFrame
    """
    # データフレームのコピーを作成
    cleaned_df = df.copy()
    
    # ティッカーシンボルの大文字変換と空白除去
    cleaned_df['Ticker'] = cleaned_df['Ticker'].astype(str).str.strip().str.upper()
    
    # 数値列の型変換
    cleaned_df['Shares'] = pd.to_numeric(cleaned_df['Shares'], errors='coerce')
    cleaned_df['AvgCostJPY'] = pd.to_numeric(cleaned_df['AvgCostJPY'], errors='coerce')
    
    # NaN値を含む行を除去
    cleaned_df = cleaned_df.dropna()
    
    # 重複行の除去（ティッカーベース）
    cleaned_df = cleaned_df.drop_duplicates(subset=['Ticker'], keep='first')
    
    # インデックスをリセット
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    logger.info(f"データクリーニング完了: {len(cleaned_df)}銘柄")
    return cleaned_df


def get_sample_data() -> pd.DataFrame:
    """
    サンプルポートフォリオデータを取得
    
    Returns:
        pd.DataFrame: サンプルデータ
    """
    sample_data = {
        'Ticker': ['AAPL', 'MSFT', '7203.T', 'ASML', 'TSLA'],
        'Shares': [100, 50, 1000, 20, 30],
        'AvgCostJPY': [15000, 25000, 800, 60000, 20000]
    }
    return pd.DataFrame(sample_data)


def export_portfolio_data(df: pd.DataFrame, filename: str = "portfolio_export.csv") -> bytes:
    """
    ポートフォリオデータをCSVフォーマットでエクスポート
    
    Args:
        df: エクスポートするDataFrame
        filename: ファイル名
    
    Returns:
        bytes: CSV形式のバイトデータ
    """
    return df.to_csv(index=False).encode('utf-8')


def display_data_summary(df: pd.DataFrame):
    """
    データサマリーの表示
    
    Args:
        df: 表示するDataFrame
    """
    st.subheader("📊 データサマリー")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("銘柄数", len(df))
    
    with col2:
        total_shares = df['Shares'].sum()
        st.metric("総保有株数", f"{total_shares:,.0f}")
    
    with col3:
        total_investment = (df['Shares'] * df['AvgCostJPY']).sum()
        st.metric("総投資額", f"¥{total_investment:,.0f}")
    
    with col4:
        avg_cost = (df['Shares'] * df['AvgCostJPY']).sum() / df['Shares'].sum()
        st.metric("平均取得単価", f"¥{avg_cost:,.0f}")
    
    # 詳細統計
    with st.expander("📈 詳細統計"):
        st.dataframe(df.describe(), use_container_width=True)