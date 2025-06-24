"""
株式ポートフォリオ管理Webアプリ
メインアプリケーション
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Dict, Any, List
import logging
from datetime import datetime, timezone, timedelta
import os
import sys
from dotenv import load_dotenv

# 文字エンコーディングの設定（エラー回避）
try:
    if sys.platform == "win32":
        import locale
        # Windowsでのロケール設定
        try:
            locale.setlocale(locale.LC_ALL, 'Japanese_Japan.UTF-8')
        except:
            try:
                locale.setlocale(locale.LC_ALL, 'ja_JP.UTF-8')
            except:
                pass  # ロケール設定に失敗しても続行
except Exception:
    pass

# 環境変数の読み込み
load_dotenv()

# API設定 - 警告メッセージは後で表示
# Gemini API設定
GEMINI_AVAILABLE = False
GEMINI_ERROR_MSG = None
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
    
    # APIキーの確認
    if not os.getenv('GEMINI_API_KEY'):
        GEMINI_AVAILABLE = False
        GEMINI_ERROR_MSG = "Gemini APIキーが設定されていません。.envファイルにGEMINI_API_KEYを設定してください。"
except ImportError as e:
    GEMINI_AVAILABLE = False
    GEMINI_ERROR_MSG = f"Google Generative AIライブラリがインストールされていません。以下のコマンドを実行してください:\n仮想環境内で: pip install google-generativeai\nエラー詳細: {str(e)}"
except Exception as e:
    GEMINI_AVAILABLE = False
    GEMINI_ERROR_MSG = f"Gemini API設定エラー: {str(e)}"

# Google Search API設定
GOOGLE_SEARCH_AVAILABLE = False
GOOGLE_SEARCH_ERROR_MSG = None
try:
    from googleapiclient.discovery import build
    GOOGLE_SEARCH_AVAILABLE = True
    
    # APIキーの確認
    if not os.getenv('GOOGLE_API_KEY') or not os.getenv('GOOGLE_SEARCH_ENGINE_ID'):
        GOOGLE_SEARCH_AVAILABLE = False
        GOOGLE_SEARCH_ERROR_MSG = "Google Search APIの設定が不完全です。.envファイルにGOOGLE_API_KEYとGOOGLE_SEARCH_ENGINE_IDを設定してください。"
except ImportError as e:
    GOOGLE_SEARCH_AVAILABLE = False
    GOOGLE_SEARCH_ERROR_MSG = f"Google APIクライアントライブラリがインストールされていません。以下のコマンドを実行してください:\n仮想環境内で: pip install google-api-python-client\nエラー詳細: {str(e)}"

# BeautifulSoup設定
SCRAPING_AVAILABLE = False
SCRAPING_ERROR_MSG = None
try:
    from bs4 import BeautifulSoup
    import requests
    SCRAPING_AVAILABLE = True
except ImportError as e:
    SCRAPING_AVAILABLE = False
    SCRAPING_ERROR_MSG = f"スクレイピングライブラリがインストールされていません。以下のコマンドを実行してください:\n仮想環境内で: pip install beautifulsoup4 requests\nエラー詳細: {str(e)}"

# 全機能の可用性チェック
REPORT_GENERATION_AVAILABLE = GEMINI_AVAILABLE and GOOGLE_SEARCH_AVAILABLE and SCRAPING_AVAILABLE

# ローカルモジュールのインポート
from modules.data_loader import load_portfolio_data, validate_portfolio_data, display_data_summary
from modules.data_manager import get_data_manager
from modules.data_adapter import create_data_adapter, validate_data_bundle
from modules.price_fetcher import (
    cached_get_current_prices, cached_get_exchange_rates, 
    determine_currency_from_ticker, convert_to_jpy, get_historical_data, get_stock_chart_data
)
from modules.pnl_calculator import (
    calculate_portfolio_pnl, calculate_portfolio_summary,
    calculate_sector_allocation_by_region, calculate_sector_allocation, calculate_performance_metrics,
    calculate_portfolio_valuation_metrics, get_etf_benchmark_data
)
from modules.country_fetcher import cached_get_multiple_ticker_countries, cached_get_multiple_ticker_info, cached_get_multiple_ticker_complete_info
from modules.risk_calculator import (
    calculate_portfolio_risk, calculate_var_cvar, stress_test_scenario
)
from modules.visualizer import (
    create_pnl_chart, create_allocation_pie, create_correlation_heatmap,
    create_var_distribution, create_performance_summary_chart, create_sector_allocation_chart,
    create_price_history_chart, create_stock_candlestick_chart, create_stock_line_chart,
    create_factor_beta_chart, create_rolling_beta_chart, create_factor_contribution_chart
)
from utils.currency_mapper import get_currency_mapping, get_market_info
from utils.helpers import (
    format_currency, format_percentage, display_error_message,
    display_success_message, display_warning_message, show_loading_spinner,
    calculate_returns
)
from modules.factor_analysis import (
    get_fama_french_factors, calculate_portfolio_returns_robust, perform_factor_regression,
    calculate_rolling_betas, calculate_factor_contributions, get_factor_interpretation
)

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main_dashboard():
    """メインダッシュボード"""
    st.set_page_config(
        page_title="ポートフォリオ管理",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # API設定の警告メッセージを表示
    if GEMINI_ERROR_MSG or GOOGLE_SEARCH_ERROR_MSG or SCRAPING_ERROR_MSG:
        st.error("🚨 運用レポート機能（ニュース分析付き）を使用するために、以下の設定が必要です：")
        
        if GEMINI_ERROR_MSG:
            st.error(f"**Gemini API**: {GEMINI_ERROR_MSG}")
        if GOOGLE_SEARCH_ERROR_MSG:
            st.error(f"**Google Search API**: {GOOGLE_SEARCH_ERROR_MSG}")
        if SCRAPING_ERROR_MSG:
            st.error(f"**スクレイピングライブラリ**: {SCRAPING_ERROR_MSG}")
            
        with st.expander("📋 詳細なインストール手順を表示"):
            st.markdown("""
            ### 仮想環境でのライブラリインストール手順
            
            **1. 仮想環境をアクティベート:**
            ```bash
            # Windows
            venv\\Scripts\\activate
            
            # Linux/Mac
            source venv/bin/activate
            ```
            
            **2. 必要なライブラリをインストール:**
            ```bash
            pip install --upgrade pip
            pip install google-generativeai
            pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
            pip install beautifulsoup4 requests
            pip install -r requirements.txt --upgrade
            ```
            
            **3. インストール確認:**
            ```bash
            python -c "import google.generativeai; print('✓ Gemini API ready')"
            python -c "from googleapiclient.discovery import build; print('✓ Google Search API ready')"
            python -c "from bs4 import BeautifulSoup; print('✓ Scraping libraries ready')"
            ```
            
            **4. .envファイルにAPIキーを設定:**
            ```env
            GOOGLE_API_KEY=your-google-cloud-api-key
            GOOGLE_SEARCH_ENGINE_ID=your-search-engine-id
            GEMINI_API_KEY=your-gemini-api-key
            ```
            
            詳細は `INSTALL_GUIDE.md` を参照してください。
            """)
        
        st.info("💡 上記の設定が完了するまで、従来のChatGPT機能（設定されている場合）または基本的なパフォーマンス分析のみが利用可能です。")
    
    # セッションステートの初期化
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = 0
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None
    if 'portfolio_df' not in st.session_state:
        st.session_state.portfolio_df = None
    
    st.title("📊 株式ポートフォリオ管理ダッシュボード")
    st.markdown("---")
    
    # サイドバー：ファイルアップロード
    with st.sidebar:
        st.header("📁 データインポート")
        
        # portfolio_filesフォルダ内のCSVファイルを自動検出
        portfolio_files_dir = "portfolio_files"
        detected_files = []
        selected_file = None
        
        if os.path.exists(portfolio_files_dir):
            detected_files = [f for f in os.listdir(portfolio_files_dir) if f.endswith('.csv')]
            
        if detected_files:
            st.subheader("📂 検出されたファイル")
            selected_file_name = st.selectbox(
                "ポートフォリオファイルを選択:",
                ["選択してください"] + detected_files,
                help="portfolio_filesフォルダ内のCSVファイル"
            )
            
            if selected_file_name != "選択してください":
                selected_file_path = os.path.join(portfolio_files_dir, selected_file_name)
                try:
                    with open(selected_file_path, 'rb') as f:
                        selected_file = f.read()
                    st.success(f"ファイル '{selected_file_name}' が選択されました！")
                    
                    # プレビュー表示
                    try:
                        preview_df = pd.read_csv(selected_file_path)
                        st.write("**データプレビュー:**")
                        st.dataframe(preview_df.head(), use_container_width=True)
                    except:
                        pass
                except Exception as e:
                    st.error(f"ファイル読み込みエラー: {str(e)}")
            
            st.markdown("---")
        
        uploaded_file = st.file_uploader(
            "または新しいCSVファイルをアップロード",
            type=['csv'],
            help="ファイル形式: Ticker, Shares, AvgCostJPY"
        )
        
        if uploaded_file:
            st.success("ファイルが正常にアップロードされました！")
            
            # セッションステートにファイルデータを保存
            if st.session_state.uploaded_data != uploaded_file.getvalue():
                st.session_state.uploaded_data = uploaded_file.getvalue()
                st.session_state.portfolio_df = None  # データが変更されたらリセット
            
            # 簡易プレビュー
            try:
                preview_df = pd.read_csv(uploaded_file)
                uploaded_file.seek(0)  # ファイルポインタをリセット
                st.write("**データプレビュー:**")
                st.dataframe(preview_df.head(), use_container_width=True)
            except:
                pass
        
        # selected_fileの場合もセッションステートを管理
        elif selected_file:
            if st.session_state.uploaded_data != selected_file:
                st.session_state.uploaded_data = selected_file
                st.session_state.portfolio_df = None  # データが変更されたらリセット
            
        st.markdown("---")
        st.subheader("📋 CSVファイル形式")
        st.code("""
Ticker,Shares,AvgCostJPY
AAPL,100,15000
MSFT,50,25000
7203.T,1000,800
        """)
    
    # メインコンテンツ
    current_file = uploaded_file or selected_file
    
    if current_file is not None:
        # セッションステートからデータを取得するか新規処理
        if st.session_state.portfolio_df is None:
            if uploaded_file:
                portfolio_df = validate_and_load_portfolio_data(uploaded_file)
            else:
                # selected_fileの場合、BytesIOオブジェクトを作成
                import io
                file_like = io.BytesIO(selected_file)
                portfolio_df = validate_and_load_portfolio_data(file_like)
            
            if portfolio_df is not None:
                st.session_state.portfolio_df = portfolio_df
        else:
            portfolio_df = st.session_state.portfolio_df
        
        if portfolio_df is not None:
            display_portfolio_dashboard(portfolio_df)
    else:
        display_welcome_page()


def validate_and_load_portfolio_data(uploaded_file) -> Optional[pd.DataFrame]:
    """ポートフォリオデータの検証と読み込み"""
    try:
        # CSVファイルの読み込み
        portfolio_df = load_portfolio_data(uploaded_file)
        
        if portfolio_df is not None:
            display_success_message(f"ポートフォリオデータを正常に読み込みました（{len(portfolio_df)}銘柄）")
            return portfolio_df
        else:
            return None
            
    except Exception as e:
        display_error_message(e, "ファイル読み込み中にエラーが発生しました")
        return None


def display_portfolio_dashboard(portfolio_df: pd.DataFrame):
    """ポートフォリオダッシュボードの表示"""
    
    try:
        # データサマリーの表示
        display_data_summary(portfolio_df)
        
        # セッションステートでデータバンドルをチェック
        tickers = portfolio_df['Ticker'].tolist()
        tickers_key = tuple(sorted(tickers))
        
        if ('data_bundle' not in st.session_state or 
            st.session_state.get('data_tickers') != tickers_key):
            
            with show_loading_spinner("全データを一括取得中（過去5年分のファクターデータを含む）..."):
                # 新しいデータマネージャーを使用して全データを一括取得
                data_manager = get_data_manager()
                data_bundle = data_manager.load_portfolio_data(portfolio_df)
                
                # セッションステートに保存
                st.session_state.data_bundle = data_bundle
                st.session_state.data_tickers = tickers_key
                
                # データアダプターを作成
                st.session_state.data_adapter = create_data_adapter(data_bundle)
                
                # ファクターデータ取得成功の確認
                factor_data = data_bundle.get('factor_data', {})
                if factor_data:
                    for key, df in factor_data.items():
                        if isinstance(df, pd.DataFrame) and not df.empty:
                            factor_start = df.index.min().strftime('%Y-%m-%d') if hasattr(df.index, 'strftime') else str(df.index.min())
                            factor_end = df.index.max().strftime('%Y-%m-%d') if hasattr(df.index, 'strftime') else str(df.index.max())
                            st.toast(f"🎯 過去5年分Fama-Frenchデータ取得完了！({factor_start}～{factor_end}, {len(df):,}日分)", icon="✅")
                            break
        else:
            # 既存のデータを使用
            data_bundle = st.session_state.data_bundle
            data_adapter = st.session_state.data_adapter
        
        # データ品質の表示
        if 'data_adapter' in st.session_state:
            data_adapter = st.session_state.data_adapter
            quality_summary = data_adapter.get_data_freshness_summary()
            st.info(f"📊 データ品質: {quality_summary}")
        
        # 損益計算（アダプターを使用）
        if 'data_adapter' not in st.session_state:
            display_warning_message("データ取得に失敗しました。ページを再読み込みしてください。")
            return
        
        data_adapter = st.session_state.data_adapter
        
        # アダプターから必要なデータを取得
        current_prices = data_adapter.get_multiple_current_prices(tickers)
        exchange_rates = data_adapter.get_exchange_rates()
        currency_mapping = data_adapter.get_currency_mapping()
        
        # 損益計算
        pnl_df = calculate_portfolio_pnl(
            portfolio_df, current_prices, exchange_rates, currency_mapping
        )
        
        if pnl_df.empty:
            display_warning_message("損益計算ができませんでした。しばらく後に再試行してください。")
            return
        
        # ポートフォリオサマリーを計算
        portfolio_summary = calculate_portfolio_summary(pnl_df)
        
        # 基本メトリクス表示
        display_portfolio_metrics(portfolio_summary)
        
        st.markdown("---")
        
        # タブによる詳細表示
        tab_names = [
            "📈 パフォーマンス", "⚠️ リスク分析", "🌍 配分分析", 
            "💰 バリュエーション", "📰 運用報告", "📊 株価チャート", "🔍 詳細データ"
        ]
        
        # ユニークキーでタブを管理
        selected_tab = st.radio(
            "表示するタブを選択:",
            options=tab_names,
            index=st.session_state.current_tab,
            horizontal=True,
            key="tab_selector"
        )
        
        # 現在のタブインデックスを更新
        if selected_tab:
            st.session_state.current_tab = tab_names.index(selected_tab)
        
        st.markdown("---")
        
        # 選択されたタブの内容を表示
        if selected_tab == "📈 パフォーマンス":
            display_performance_analysis(pnl_df, portfolio_summary)
        elif selected_tab == "⚠️ リスク分析":
            display_risk_analysis(pnl_df, tickers, portfolio_df)
        elif selected_tab == "🌍 配分分析":
            display_allocation_analysis(pnl_df, tickers)
        elif selected_tab == "💰 バリュエーション":
            display_valuation_analysis(pnl_df, tickers)
        elif selected_tab == "📰 運用報告":
            display_investment_report(pnl_df, tickers)
        elif selected_tab == "📊 株価チャート":
            display_stock_charts(tickers)
        elif selected_tab == "🔍 詳細データ":
            display_detailed_data(pnl_df, portfolio_df, tickers)
            
    except Exception as e:
        display_error_message(e, "ダッシュボード表示中にエラーが発生しました")


def display_portfolio_metrics(summary: Dict[str, float]):
    """ポートフォリオメトリクスの表示"""
    if not summary:
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="総評価額",
            value=format_currency(summary.get('total_current_value_jpy', 0)),
            delta=format_currency(summary.get('total_pnl_amount_jpy', 0))
        )
    
    with col2:
        st.metric(
            label="総損益率",
            value=format_percentage(summary.get('overall_pnl_percentage', 0)),
            delta=f"{summary.get('profitable_positions', 0)}勝/{summary.get('losing_positions', 0)}敗"
        )
    
    with col3:
        st.metric(
            label="勝率",
            value=format_percentage(summary.get('win_rate', 0)),
            delta=f"平均ポジション: {format_currency(summary.get('average_position_size', 0))}"
        )
    
    with col4:
        best_ticker = summary.get('max_gain_ticker', '')
        worst_ticker = summary.get('max_loss_ticker', '')
        st.metric(
            label="最高/最低パフォーマンス",
            value=f"{best_ticker}: {format_percentage(summary.get('max_gain_percentage', 0))}",
            delta=f"{worst_ticker}: {format_percentage(summary.get('max_loss_percentage', 0))}"
        )


def display_performance_analysis(pnl_df: pd.DataFrame, summary: Dict[str, float]):
    """パフォーマンス分析の表示"""
    st.subheader("📈 パフォーマンス分析")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 損益チャート
        pnl_chart = create_pnl_chart(pnl_df)
        st.plotly_chart(pnl_chart, use_container_width=True)
    
    with col2:
        # 資産配分チャート
        allocation_chart = create_allocation_pie(pnl_df)
        st.plotly_chart(allocation_chart, use_container_width=True)
    
    # パフォーマンスサマリー
    if summary:
        performance_chart = create_performance_summary_chart(summary)
        st.plotly_chart(performance_chart, use_container_width=True)


def display_risk_analysis(pnl_df: pd.DataFrame, tickers: list, portfolio_df: pd.DataFrame):
    """リスク分析の表示"""
    st.subheader("⚠️ リスク分析")
    
    # セッションステートでリスク分析設定を管理
    if 'risk_analysis_period' not in st.session_state:
        st.session_state.risk_analysis_period = "1y"
    if 'risk_time_scale' not in st.session_state:
        st.session_state.risk_time_scale = "日次"
    
    # 設定UI
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.write("分析期間を選択してください：")
    with col2:
        analysis_period = st.selectbox(
            "データ期間",
            options=["1mo", "3mo", "6mo", "ytd", "1y", "2y", "5y"],
            index=["1mo", "3mo", "6mo", "ytd", "1y", "2y", "5y"].index(st.session_state.risk_analysis_period),
            help="相関分析・リスク指標計算に使用する過去データの期間",
            key="risk_analysis_period_selector"
        )
        st.session_state.risk_analysis_period = analysis_period
    
    with col3:
        time_scale = st.selectbox(
            "リスク時間軸",
            options=["日次", "月次", "年次"],
            index=["日次", "月次", "年次"].index(st.session_state.risk_time_scale),
            help="VaR/CVaRとストレステストの表示時間スケール",
            key="risk_time_scale_selector"
        )
        st.session_state.risk_time_scale = time_scale
    
    # 時間スケール変換係数を事前に計算
    def get_time_scale_factor(scale):
        if scale == "日次":
            return 1, "日"
        elif scale == "月次":
            return np.sqrt(20), "月"  # 20営業日
        elif scale == "年次":
            return np.sqrt(252), "年"  # 252営業日
        return 1, "日"
    
    scale_factor, scale_label = get_time_scale_factor(time_scale)
    
    try:
        # データアダプターからキャッシュされた過去データを取得
        data_adapter = st.session_state.get('data_adapter')
        if not data_adapter:
            st.error("データが初期化されていません。ページを再読み込みしてください。")
            return
        
        with show_loading_spinner(f"過去データを処理中..."):
            # キャッシュされた過去データから必要な期間を抽出
            historical_data_dict = data_adapter.get_multiple_historical_data(tickers, period="5y")
            
            # 指定期間にフィルタリング
            from datetime import datetime, timedelta
            if analysis_period == "1mo":
                start_date = datetime.now() - timedelta(days=30)
            elif analysis_period == "3mo":
                start_date = datetime.now() - timedelta(days=90)
            elif analysis_period == "6mo":
                start_date = datetime.now() - timedelta(days=180)
            elif analysis_period == "ytd":
                start_date = datetime(datetime.now().year, 1, 1)
            elif analysis_period == "1y":
                start_date = datetime.now() - timedelta(days=365)
            elif analysis_period == "2y":
                start_date = datetime.now() - timedelta(days=730)
            else:  # 5y
                start_date = datetime.now() - timedelta(days=1825)
            
            # データフレームを結合
            historical_data = pd.DataFrame()
            for ticker in tickers:
                ticker_data = historical_data_dict.get(ticker, pd.DataFrame())
                if not ticker_data.empty:
                    # 指定期間でフィルタリング
                    ticker_data = ticker_data[ticker_data.index >= start_date]
                    if not ticker_data.empty:
                        historical_data[ticker] = ticker_data['Close']
            
            if historical_data.empty:
                st.warning("指定期間の過去データがありません。")
                return
            
            # データが少なすぎる場合の警告
            if len(historical_data) < 20:
                st.warning(f"データ期間が短すぎます（{len(historical_data)}日）。より長い期間を選択することをお勧めします。")
            
            # 日次リターンを計算
            returns_df = pd.DataFrame()
            for ticker in tickers:
                if ticker in historical_data.columns:
                    returns = calculate_returns(historical_data[ticker])
                    if not returns.empty:
                        returns_df[ticker] = returns
            
            if returns_df.empty:
                st.error("リターンデータの計算に失敗しました。")
                return
            
            st.info(f"📊 分析期間: {analysis_period} ({len(returns_df)}営業日のデータ)")
            
            # ポートフォリオ重みを計算
            total_value = pnl_df['current_value_jpy'].sum()
            weights = (pnl_df['current_value_jpy'] / total_value).values
            
            # データが揃っている銘柄のみでウェイトを再計算
            valid_tickers = [ticker for ticker in tickers if ticker in returns_df.columns]
            valid_pnl = pnl_df[pnl_df['ticker'].isin(valid_tickers)]
            
            if len(valid_tickers) != len(tickers):
                missing_tickers = set(tickers) - set(valid_tickers)
                st.warning(f"以下の銘柄のデータが不足しているため、分析から除外されます: {', '.join(missing_tickers)}")
            
            if len(valid_tickers) < 2:
                st.error("相関分析には少なくとも2銘柄のデータが必要です。")
                return
            
            # 有効な銘柄のウェイトを再計算
            valid_total_value = valid_pnl['current_value_jpy'].sum()
            valid_weights = (valid_pnl['current_value_jpy'] / valid_total_value).values
            
            # リスク指標計算
            risk_metrics = calculate_portfolio_risk(returns_df[valid_tickers], valid_weights)
            
            if risk_metrics:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 リスク指標")
                    portfolio_vol_scaled = risk_metrics.get('portfolio_volatility', 0) * scale_factor
                    st.metric(f"ポートフォリオボラティリティ（{scale_label}次）", 
                             format_percentage(portfolio_vol_scaled * 100))
                    st.metric("平均相関", 
                             f"{risk_metrics.get('average_correlation', 0):.3f}")
                    st.metric("分散効果", 
                             f"{risk_metrics.get('diversification_ratio', 1):.2f}x")
                    
                    # 個別銘柄ボラティリティの表示
                    with st.expander(f"個別銘柄ボラティリティ（{scale_label}次）"):
                        individual_vols = risk_metrics.get('individual_volatilities', pd.Series())
                        for ticker, vol in individual_vols.items():
                            vol_scaled = vol * scale_factor
                            st.write(f"**{ticker}**: {format_percentage(vol_scaled * 100)}")
                
                with col2:
                    # 相関ヒートマップ
                    if 'correlation_matrix' in risk_metrics:
                        corr_chart = create_correlation_heatmap(risk_metrics['correlation_matrix'])
                        st.plotly_chart(corr_chart, use_container_width=True)
            
            # ポートフォリオリターンを計算
            portfolio_returns = (returns_df[valid_tickers] * valid_weights).sum(axis=1)
            
            # VaR/CVaR計算
            var_metrics = calculate_var_cvar(pd.Series(portfolio_returns))
            
            if var_metrics:
                st.subheader(f"📉 VaR/CVaR分析（{scale_label}次）")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    var_95_scaled = var_metrics.get('VaR_95', 0) * scale_factor
                    st.metric(f"VaR (95%)", format_percentage(var_95_scaled * 100))
                
                with col2:
                    cvar_95_scaled = var_metrics.get('CVaR_95', 0) * scale_factor
                    st.metric(f"CVaR (95%)", format_percentage(cvar_95_scaled * 100))
                
                with col3:
                    var_99_scaled = var_metrics.get('VaR_99', 0) * scale_factor
                    st.metric(f"VaR (99%)", format_percentage(var_99_scaled * 100))
                
                with col4:
                    daily_vol = portfolio_returns.std()
                    scaled_vol = daily_vol * scale_factor
                    st.metric(f"{scale_label}率ボラティリティ", format_percentage(scaled_vol * 100))
                
                # VaR分布チャート（時間軸に応じてスケール）
                var_chart = create_var_distribution(pd.Series(portfolio_returns), var_metrics, scale_factor, scale_label)
                st.plotly_chart(var_chart, use_container_width=True)
                
                # ストレステスト
                st.subheader("🚨 ストレステスト")
                stress_results = stress_test_scenario(returns_df[valid_tickers], valid_weights, 
                                                     stress_factor=1.5, correlation_shock=0.8)
                
                if stress_results:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        normal_vol = stress_results.get('normal_portfolio_vol', 0)
                        normal_vol_scaled = normal_vol * scale_factor
                        st.metric(f"通常時ボラティリティ（{scale_label}次）", format_percentage(normal_vol_scaled * 100))
                    
                    with col2:
                        stressed_vol = stress_results.get('stressed_portfolio_vol', 0)
                        stressed_vol_scaled = stressed_vol * scale_factor
                        st.metric(f"ストレス時ボラティリティ（{scale_label}次）", format_percentage(stressed_vol_scaled * 100))
                    
                    with col3:
                        stress_multiplier = stress_results.get('stress_multiplier', 1)
                        st.metric("ストレス倍率", f"{stress_multiplier:.2f}x")
                    
                    with col4:
                        # ストレス時の想定損失（95%信頼区間、約2標準偏差）
                        stress_loss_95 = -stressed_vol_scaled * 1.96  # 負の値として表示
                        st.metric(f"想定最大損失 95%（{scale_label}次）", format_percentage(stress_loss_95 * 100))
                    
                    # ストレステスト詳細
                    with st.expander("🔍 ストレステスト詳細"):
                        st.write("**ストレス条件:**")
                        st.write(f"- ボラティリティ増加倍率: {stress_results.get('stress_factor', 1.5):.1f}倍")
                        st.write(f"- ストレス時相関係数: {stress_results.get('correlation_shock', 0.8):.1f}")
                        st.write(f"- 通常時ポートフォリオボラティリティ（年率）: {format_percentage(normal_vol * 100)}")
                        st.write(f"- ストレス時ポートフォリオボラティリティ（年率）: {format_percentage(stressed_vol * 100)}")
                        
                        st.write("**想定損失シナリオ（ストレス時）:**")
                        scenarios = [
                            ("68%信頼区間（1σ）", -stressed_vol_scaled * 1.0, "約68%の確率で損失がこの範囲内"),
                            ("95%信頼区間（1.96σ）", -stressed_vol_scaled * 1.96, "約95%の確率で損失がこの範囲内"),
                            ("99%信頼区間（2.58σ）", -stressed_vol_scaled * 2.58, "約99%の確率で損失がこの範囲内"),
                            ("99.7%信頼区間（3σ）", -stressed_vol_scaled * 3.0, "約99.7%の確率で損失がこの範囲内")
                        ]
                        
                        for scenario_name, loss_pct, description in scenarios:
                            st.write(f"- **{scenario_name}**: {format_percentage(loss_pct * 100)} ({description})")
                
                # ファクターエクスポージャー分析
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.subheader("📊 ファクターエクスポージャー分析")
                with col2:
                    if st.button("❓ ファクター分析について", key="factor_help_button"):
                        st.session_state.show_factor_help = not st.session_state.get('show_factor_help', False)
                
                # ヘルプ表示
                if st.session_state.get('show_factor_help', False):
                    with st.expander("📚 ファクター分析の詳細解説", expanded=True):
                        st.markdown("""
                        ## 🎯 **Fama-French 5ファクター + Momentumモデルとは**
                        
                        ノーベル経済学賞受賞のユージン・ファーマ教授らが開発した、株式リターンを説明する代表的なモデルです。
                        あなたのポートフォリオが「どんな投資スタイル」なのかを数値で明確に示します。
                        
                        ---
                        
                        ## 📊 **各ファクターの意味**
                        
                        ### 🔵 **市場プレミアム (Mkt-RF)**
                        - **ベータ > 1.0**: 📈 **攻撃的** - 市場より大きく動く（ハイリスク・ハイリターン）
                        - **ベータ < 1.0**: 🛡️ **守備的** - 市場より穏やかに動く（ローリスク・ローリターン）
                        - **ベータ ≈ 1.0**: ⚖️ **市場並み** - 市場全体と同じリスク水準
                        
                        ### 🟠 **小型株プレミアム (SMB: Small Minus Big)**
                        - **ベータ > 0.2**: 🏢 **小型株重視** - 成長余地の大きい小さな会社に投資
                        - **ベータ < -0.2**: 🏦 **大型株重視** - 安定した大企業に投資
                        - **例**: 新興企業 vs GAFAM
                        
                        ### 🟣 **バリュープレミアム (HML: High Minus Low)**
                        - **ベータ > 0.2**: 💎 **割安株投資** - PERやPBRが低い「掘り出し物」狙い
                        - **ベータ < -0.2**: 🚀 **成長株投資** - 高成長期待のグロース株狙い
                        - **例**: バフェット流バリュー投資 vs テスラ等の成長株投資
                        
                        ### 🟢 **収益性プレミアム (RMW: Robust Minus Weak)**
                        - **ベータ > 0.2**: 💰 **優良企業重視** - ROEが高く利益を安定的に出す会社
                        - **ベータ < -0.2**: 🎯 **成長投資** - 現在は利益が少なくても将来性重視
                        - **例**: 配当貴族銘柄 vs スタートアップ
                        
                        ### 🔴 **投資プレミアム (CMA: Conservative Minus Aggressive)**
                        - **ベータ > 0.2**: 🏛️ **堅実経営重視** - 設備投資を抑えて利益重視の会社
                        - **ベータ < -0.2**: 🚁 **積極投資重視** - 将来のために大胆に投資する会社
                        - **例**: 成熟企業 vs R&D集約企業
                        
                        ### ⚡ **モメンタムプレミアム (Mom)**
                        - **ベータ > 0.2**: 📈 **トレンド追随** - 上昇している株はまだ上がる
                        - **ベータ < -0.2**: ↩️ **逆張り投資** - 下落した株の反発を狙う
                        - **例**: 勢いのある銘柄追随 vs 割安放置銘柄狙い
                        
                        ---
                        
                        ## 📡 **データソースと計算方法**
                        
                        ### **データ提供元**
                        - **Fama-French Data Library** (Kenneth R. French教授のWebサイト)
                        - ダートマス大学タック・スクール・オブ・ビジネス
                        - **URL**: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
                        
                        ### **計算ユニバース**
                        - **対象市場**: 🇺🇸 **米国株式市場**
                        - **対象銘柄**: NYSE、NASDAQ、AMEX上場の普通株
                        - **除外銘柄**: REIT、ADR、クローズドエンド・ファンド等
                        - **更新頻度**: 日次更新
                        - **歴史**: 1926年から現在まで（約100年の実績）
                        
                        ### **ファクター構築方法**
                        1. **SMB**: 時価総額で小型/大型に分類 → 小型株リターン - 大型株リターン
                        2. **HML**: PBRで割安/割高に分類 → 割安株リターン - 割高株リターン  
                        3. **RMW**: ROEで収益性高/低に分類 → 高収益株リターン - 低収益株リターン
                        4. **CMA**: 投資率で保守/積極に分類 → 保守的企業リターン - 積極企業リターン
                        5. **Mom**: 過去12ヶ月リターンで分類 → 上昇株リターン - 下落株リターン
                        
                        ---
                        
                        ## ⚠️ **使用上の重要な留意点**
                        
                        ### 🌍 **地域的制約**
                        - ファクターデータは **米国市場ベース** で計算
                        - あなたのポートフォリオに **日本株・欧州株・新興国株** が含まれる場合：
                          - ベータ値は「米国市場のファクターに対する感応度」として解釈
                          - 地域固有のファクター（例：日本株の「品質プレミアム」）は反映されない
                          - **推奨**: 地域別にポートフォリオを分けて分析
                        
                        ### 📅 **時期的制約**  
                        - ファクターの効果は **時期によって変動**
                        - 過去のパフォーマンスは将来を保証しない
                        - 金融危機時などは通常と異なるファクター関係になる可能性
                        
                        ### 📊 **統計的制約**
                        - **決定係数(R²)が低い場合**: ファクターで説明できない部分が大きい
                        - **有意でないベータ**: そのファクターへのエクスポージャーは不明確
                        - **推奨**: 複数期間での分析、他の分析手法との併用
                        
                        ### 💼 **投資判断での活用法**
                        - ファクター分析は **「現在のポートフォリオ特性の把握」** が主目的
                        - 意図しないリスクの発見（例：思った以上に小型株に偏っている）
                        - リバランスの参考（例：バリュー偏重を是正したい）
                        - **注意**: ファクター分析だけで投資判断をしないこと
                        
                        ---
                        
                        ## 🎯 **実践的な活用例**
                        
                        ### **Case 1: バランス型投資家**
                        - 全ファクターのベータが -0.2 ～ 0.2 の範囲
                        - → 特定のスタイルに偏らないバランス投資
                        
                        ### **Case 2: グロース投資家** 
                        - SMB > 0 (小型株)、HML < 0 (成長株)、Mom > 0 (モメンタム)
                        - → 小型成長株の上昇トレンド狙い
                        
                        ### **Case 3: バリュー投資家**
                        - HML > 0 (割安株)、RMW > 0 (高収益)、CMA > 0 (堅実経営)  
                        - → 堅実な割安株投資
                        
                        💡 **あなたのポートフォリオはどのタイプに近いでしょうか？**
                        """)
                        
                        st.info("💡 **ヒント**: この分析結果を参考に、意図したポートフォリオになっているかチェックしてみましょう！")
                with show_loading_spinner("Fama-French ファクターデータを処理中..."):
                    try:
                        # データアダプターからファクターデータを取得（選択された期間に応じて）
                        factor_start_date = start_date.strftime('%Y-%m-%d')
                        factor_end_date = datetime.now().strftime('%Y-%m-%d')
                        
                        factor_data = data_adapter.get_fama_french_factors(
                            start_date=factor_start_date, 
                            end_date=factor_end_date
                        )
                        
                        if isinstance(factor_data, pd.DataFrame) and not factor_data.empty:
                            # ファクターデータの期間情報を表示
                            actual_start = factor_data.index.min().strftime('%Y-%m-%d') if hasattr(factor_data.index, 'strftime') else str(factor_data.index.min())
                            actual_end = factor_data.index.max().strftime('%Y-%m-%d') if hasattr(factor_data.index, 'strftime') else str(factor_data.index.max())
                            
                            st.success(f"🎯 **Fama-French ファクターデータ使用中 ({analysis_period}期間)**\n\n"
                                     f"- 📊 選択期間: {factor_start_date} ～ {factor_end_date}\n"
                                     f"- 📈 実際データ: {actual_start} ～ {actual_end}\n"
                                     f"- 📅 データ数: {len(factor_data):,}営業日分\n"
                                     f"- 🔍 ファクター: {', '.join(factor_data.columns)}")
                        
                        if factor_data is not None and not factor_data.empty:
                            # ロバストなポートフォリオリターン計算を実行
                            st.info(f"🔄 ポートフォリオリターンを計算中... 対象銘柄: {', '.join(tickers)}, 期間: {analysis_period}")
                            
                            # デバッグ情報を表示
                            with st.expander("🔍 ファクター分析データ情報"):
                                st.write("**分析設定:**")
                                st.write(f"- データ期間: {analysis_period}")
                                st.write(f"- 対象銘柄数: {len(tickers)}")
                                st.write("**PnLデータ構造:**")
                                st.write(f"- Shape: {pnl_df.shape}")
                                st.write(f"- Columns: {pnl_df.columns.tolist()}")
                                st.write("**PnLデータサンプル:**")
                                st.dataframe(pnl_df[['ticker', 'shares', 'current_value_jpy']].head())
                            
                            # ロバストなポートフォリオリターン計算
                            portfolio_returns = calculate_portfolio_returns_robust(pnl_df, period=analysis_period)
                            
                            # 結果の確認
                            if portfolio_returns.empty:
                                st.error("❌ ポートフォリオリターンの計算に失敗しました")
                                st.info("💡 以下を確認してください：")
                                st.write("- ティッカーシンボルが正しいか")
                                st.write("- 選択したデータ期間に株価データが存在するか")
                                st.write("- ネットワーク接続が正常か")
                            else:
                                st.success(f"✅ ポートフォリオリターン計算完了: {len(portfolio_returns)}日分")
                                with st.expander("📊 ポートフォリオリターン統計"):
                                    st.write(f"**基本統計 (日次):**")
                                    st.write(f"- 平均リターン: {portfolio_returns.mean():.6f} ({portfolio_returns.mean()*252:.3%} 年率)")
                                    st.write(f"- ボラティリティ: {portfolio_returns.std():.6f} ({portfolio_returns.std()*np.sqrt(252):.3%} 年率)")
                                    st.write(f"- 最大: {portfolio_returns.max():.6f}")
                                    st.write(f"- 最小: {portfolio_returns.min():.6f}")
                                    st.write(f"- データ期間: {portfolio_returns.index[0].strftime('%Y-%m-%d')} ~ {portfolio_returns.index[-1].strftime('%Y-%m-%d')}")
                            
                            if not portfolio_returns.empty:
                                # ファクター回帰分析
                                factor_results = perform_factor_regression(portfolio_returns, factor_data)
                                
                                if factor_results:
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        # ファクターベータのチャート
                                        beta_chart = create_factor_beta_chart(factor_results)
                                        st.plotly_chart(beta_chart, use_container_width=True)
                                        
                                        # 回帰統計
                                        st.subheader("📈 回帰統計")
                                        alpha = factor_results.get('alpha', 0)
                                        alpha_pval = factor_results.get('alpha_pvalue', 1)
                                        r_squared = factor_results.get('r_squared', 0)
                                        
                                        col_a, col_b = st.columns(2)
                                        with col_a:
                                            alpha_significance = "有意" if alpha_pval < 0.05 else "非有意"
                                            st.metric(
                                                "アルファ（年率）",
                                                f"{alpha * 252:.2%}",
                                                f"p値: {alpha_pval:.3f} ({alpha_significance})"
                                            )
                                        with col_b:
                                            st.metric("決定係数 (R²)", f"{r_squared:.3f}", f"説明力: {r_squared*100:.1f}%")
                                    
                                    with col2:
                                        # ファクター解釈
                                        st.subheader("🔍 ファクター解釈")
                                        betas = factor_results.get('betas', {})
                                        pvalues = factor_results.get('factor_pvalues', {})
                                        
                                        for factor, beta in betas.items():
                                            pval = pvalues.get(factor, 1.0)
                                            significance = ""
                                            if pval < 0.01:
                                                significance = "🟢 高度に有意"
                                            elif pval < 0.05:
                                                significance = "🟡 有意"
                                            elif pval < 0.1:
                                                significance = "🟠 やや有意"
                                            else:
                                                significance = "⚪ 非有意"
                                            
                                            interpretation = get_factor_interpretation(factor, beta)
                                            st.write(f"**{factor}**: {beta:.3f} ({significance})")
                                            st.write(f"　→ {interpretation}")
                                            st.write("")
                                    
                                    # ローリングベータ分析
                                    with st.expander("📈 ローリングベータ分析（時系列）"):
                                        rolling_betas = calculate_rolling_betas(portfolio_returns, factor_data)
                                        if not rolling_betas.empty:
                                            rolling_chart = create_rolling_beta_chart(rolling_betas, analysis_period)
                                            st.plotly_chart(rolling_chart, use_container_width=True)
                                            
                                            st.info("💡 ローリングベータはファクターエクスポージャーの時間変化を示します（1ヶ月窓）。急激な変化は投資スタイルの変更やリバランスを示唆する可能性があります。")
                                        else:
                                            st.warning("ローリングベータの計算に必要なデータが不足しています（最低1ヶ月分必要）")
                                    
                                    # ファクター寄与度分析
                                    with st.expander("📊 ファクター寄与度分析"):
                                        contributions = calculate_factor_contributions(factor_data, betas)
                                        if not contributions.empty:
                                            contribution_chart = create_factor_contribution_chart(contributions, analysis_period)
                                            st.plotly_chart(contribution_chart, use_container_width=True)
                                            
                                            # 総寄与度サマリー
                                            total_contributions = contributions.sum()
                                            st.subheader("📋 累積寄与度サマリー")
                                            for factor, contrib in total_contributions.items():
                                                contrib_pct = contrib * 100
                                                if contrib_pct > 0:
                                                    st.write(f"✅ **{factor}**: +{contrib_pct:.2f}% （プラス寄与）")
                                                else:
                                                    st.write(f"❌ **{factor}**: {contrib_pct:.2f}% （マイナス寄与）")
                                        else:
                                            st.warning("ファクター寄与度の計算に失敗しました")
                                
                                else:
                                    st.warning("ファクター回帰分析に失敗しました。データ期間やポートフォリオ構成を確認してください。")
                            else:
                                st.warning("ポートフォリオリターンの計算に失敗しました。株価データの取得状況を確認してください。")
                        else:
                            st.warning("🚫 **Fama-French 5年分ファクターデータが利用できません**\n\n"
                                     "考えられる原因:\n"
                                     "1. ネットワーク接続の問題\n"
                                     "2. Kenneth Frenchサイトの一時的な問題\n"
                                     "3. データキャッシュの問題\n\n"
                                     "**対処方法:**\n"
                                     "- ページを再読み込みしてください\n"
                                     "- しばらく時間をおいて再試行してください")
                            
                            if isinstance(factor_data, pd.DataFrame):
                                st.info(f"📊 取得されたデータ: {len(factor_data)}行 (空のDataFrame)")
                            else:
                                st.error(f"❌ データ形式エラー: {type(factor_data)}")
                    
                    except Exception as e:
                        st.error(f"ファクター分析中にエラーが発生しました: {str(e)}")
                        logger.error(f"ファクター分析エラー: {str(e)}")
                
                # 統計情報の詳細表示
                with st.expander(f"📈 詳細統計（{scale_label}次ベース）"):
                    stats_col1, stats_col2 = st.columns(2)
                    
                    with stats_col1:
                        st.write(f"**リターン統計（{scale_label}次）:**")
                        
                        # 時間軸に応じた統計表示
                        if time_scale == "日次":
                            avg_return_scaled = portfolio_returns.mean()
                            max_return_scaled = portfolio_returns.max()
                            min_return_scaled = portfolio_returns.min()
                            st.write(f"平均日次リターン: {format_percentage(avg_return_scaled * 100)}")
                            st.write(f"最大日次リターン: {format_percentage(max_return_scaled * 100)}")
                            st.write(f"最小日次リターン: {format_percentage(min_return_scaled * 100)}")
                            st.write(f"年率リターン（参考）: {format_percentage(avg_return_scaled * 252 * 100)}")
                        
                        elif time_scale == "月次":
                            avg_return_scaled = portfolio_returns.mean() * 20  # 20営業日
                            max_return_scaled = portfolio_returns.max() * np.sqrt(20)
                            min_return_scaled = portfolio_returns.min() * np.sqrt(20)
                            st.write(f"平均月次リターン: {format_percentage(avg_return_scaled * 100)}")
                            st.write(f"想定最大月次リターン: {format_percentage(max_return_scaled * 100)}")
                            st.write(f"想定最小月次リターン: {format_percentage(min_return_scaled * 100)}")
                            st.write(f"年率リターン（参考）: {format_percentage(avg_return_scaled * 12 * 100)}")
                        
                        elif time_scale == "年次":
                            avg_return_scaled = portfolio_returns.mean() * 252  # 252営業日
                            max_return_scaled = portfolio_returns.max() * np.sqrt(252)
                            min_return_scaled = portfolio_returns.min() * np.sqrt(252)
                            st.write(f"平均年次リターン: {format_percentage(avg_return_scaled * 100)}")
                            st.write(f"想定最大年次リターン: {format_percentage(max_return_scaled * 100)}")
                            st.write(f"想定最小年次リターン: {format_percentage(min_return_scaled * 100)}")
                    
                    with stats_col2:
                        st.write("**リスク統計:**")
                        skewness = portfolio_returns.skew()
                        kurtosis = portfolio_returns.kurtosis()
                        daily_vol = portfolio_returns.std()
                        scaled_vol = daily_vol * scale_factor
                        
                        st.write(f"歪度: {skewness:.3f}")
                        st.write(f"尖度: {kurtosis:.3f}")
                        st.write(f"{scale_label}次ボラティリティ: {format_percentage(scaled_vol * 100)}")
                        st.write(f"データ期間: {len(portfolio_returns)}営業日")
                        st.write(f"欠損データ: {portfolio_returns.isna().sum()}日")
    
    except Exception as e:
        display_error_message(e, "リスク分析中にエラーが発生しました")


def display_allocation_analysis(pnl_df: pd.DataFrame, tickers: List[str]):
    """配分分析の表示"""
    st.subheader("🌍 配分分析")
    
    try:
        # データアダプターからキャッシュされた企業情報を取得
        data_adapter = st.session_state.get('data_adapter')
        if not data_adapter:
            st.error("データが初期化されていません。ページを再読み込みしてください。")
            return
        
        with show_loading_spinner("企業情報を処理中..."):
            # キャッシュされた完全な企業情報を取得
            ticker_complete_info = data_adapter.get_multiple_ticker_complete_info(tickers)
            
            # 配分分析用に基本情報のみを抽出
            ticker_info = {}
            for ticker, info in ticker_complete_info.items():
                ticker_info[ticker] = {
                    'country': info.get('country'),
                    'sector': info.get('sector')
                }
            
            # 取得状況の確認
            country_success = sum(1 for info in ticker_info.values() if info.get('country'))
            sector_success = sum(1 for info in ticker_info.values() if info.get('sector'))
            
            st.info(f"企業情報（キャッシュより）: 国情報 {country_success}/{len(tickers)}銘柄, セクター情報 {sector_success}/{len(tickers)}銘柄")
        
        # 分析タイプの選択
        analysis_type = st.radio(
            "分析タイプを選択:",
            options=["地域別", "セクター別"],
            horizontal=True,
            key="allocation_analysis_type"
        )
        
        if analysis_type == "地域別":
            # 地域別配分分析
            ticker_countries = {ticker: info['country'] for ticker, info in ticker_info.items()}
            
            # デバッグ情報表示
            with st.expander("🔍 本社所在国情報の詳細"):
                st.write("**取得された本社所在国情報:**")
                for ticker, country in ticker_countries.items():
                    status = "✅" if country else "❌"
                    country_display = country if country else "取得失敗"
                    st.write(f"{status} **{ticker}**: {country_display}")
            
            # 地域別配分を計算
            allocation_df = calculate_sector_allocation_by_region(pnl_df, ticker_countries)
            category_label = "地域"
            
        else:  # セクター別
            # デバッグ情報表示
            with st.expander("🔍 セクター情報の詳細"):
                st.write("**取得されたセクター情報:**")
                for ticker, info in ticker_info.items():
                    sector = info.get('sector')
                    status = "✅" if sector else "❌"
                    sector_display = sector if sector else "取得失敗"
                    st.write(f"{status} **{ticker}**: {sector_display}")
            
            # セクター別配分を計算
            allocation_df = calculate_sector_allocation(pnl_df, ticker_info)
            category_label = "セクター"
        
        if not allocation_df.empty:
            # 配分チャート
            try:
                allocation_chart = create_sector_allocation_chart(allocation_df)
                allocation_chart.update_layout(title=f'{category_label}別配分')
                st.plotly_chart(allocation_chart, use_container_width=True)
            except Exception as chart_error:
                st.error(f"チャート作成中にエラーが発生しました: {str(chart_error)}")
                # デバッグ情報を表示
                st.write("**配分データの内容:**")
                st.dataframe(allocation_df)
            
            # 配分テーブル
            st.subheader(f"📋 {category_label}別配分詳細")
            display_df = allocation_df.copy()
            
            # 列名を日本語に変更
            column_mapping = {
                'country': category_label,
                'sector': category_label,
                'current_value_jpy': '現在価値（円）',
                'cost_basis_jpy': '取得原価（円）',
                'pnl_amount': '損益金額（円）',
                'position_count': '銘柄数',
                'allocation_percentage': '配分比率（%）',
                'pnl_percentage': '損益率（%）'
            }
            display_df = display_df.rename(columns=column_mapping)
            
            # 数値フォーマット
            display_df['現在価値（円）'] = display_df['現在価値（円）'].apply(lambda x: format_currency(x))
            display_df['取得原価（円）'] = display_df['取得原価（円）'].apply(lambda x: format_currency(x))
            display_df['損益金額（円）'] = display_df['損益金額（円）'].apply(lambda x: format_currency(x))
            display_df['配分比率（%）'] = display_df['配分比率（%）'].apply(lambda x: format_percentage(x))
            display_df['損益率（%）'] = display_df['損益率（%）'].apply(lambda x: format_percentage(x))
            
            st.dataframe(display_df, use_container_width=True)
            
            # サマリー情報
            st.subheader(f"📊 {category_label}別サマリー")
            col1, col2, col3 = st.columns(3)
            
            category_col = 'country' if analysis_type == "地域別" else 'sector'
            
            with col1:
                top_category = allocation_df.loc[allocation_df['allocation_percentage'].idxmax(), category_col]
                top_allocation = allocation_df['allocation_percentage'].max()
                st.metric(f"最大配分{category_label}", f"{top_category}", f"{top_allocation:.1f}%")
            
            with col2:
                best_category = allocation_df.loc[allocation_df['pnl_percentage'].idxmax(), category_col]
                best_performance = allocation_df['pnl_percentage'].max()
                st.metric(f"最高パフォーマンス{category_label}", f"{best_category}", f"{best_performance:+.1f}%")
            
            with col3:
                total_categories = len(allocation_df)
                profitable_categories = len(allocation_df[allocation_df['pnl_percentage'] > 0])
                st.metric("分散状況", f"{total_categories}{category_label}", f"利益{category_label}: {profitable_categories}")
        else:
            st.warning(f"{category_label}別配分データの計算に失敗しました。企業情報の取得状況を確認してください。")
            
            # デバッグ情報
            st.write("**デバッグ情報:**")
            if analysis_type == "セクター別":
                sector_count = sum(1 for info in ticker_info.values() if info and info.get('sector'))
                st.write(f"セクター情報取得成功: {sector_count}/{len(tickers)}銘柄")
            else:
                country_count = sum(1 for info in ticker_info.values() if info and info.get('country'))
                st.write(f"国情報取得成功: {country_count}/{len(tickers)}銘柄")
    
    except Exception as e:
        display_error_message(e, "配分分析中にエラーが発生しました")


def display_valuation_analysis(pnl_df: pd.DataFrame, tickers: List[str]):
    """バリュエーション分析の表示"""
    st.subheader("💰 バリュエーション分析")
    
    try:
        # データアダプターからキャッシュされたバリュエーション情報を取得
        data_adapter = st.session_state.get('data_adapter')
        if not data_adapter:
            st.error("データが初期化されていません。ページを再読み込みしてください。")
            return
        
        with show_loading_spinner("バリュエーション情報を処理中..."):
            try:
                # キャッシュされた完全な企業情報を取得
                ticker_complete_info = data_adapter.get_multiple_ticker_complete_info(tickers)
                
                # データ取得結果の検証
                if not ticker_complete_info:
                    st.error("企業情報が利用できません。")
                    return
                
                # 成功した銘柄数をカウント
                successful_tickers = [ticker for ticker, info in ticker_complete_info.items() 
                                    if info and (info.get('country') or info.get('sector'))]
                
                if len(successful_tickers) == 0:
                    st.error("すべての銘柄で企業情報が不足しています。")
                    return
                elif len(successful_tickers) < len(tickers):
                    failed_tickers = [ticker for ticker in tickers if ticker not in successful_tickers]
                    st.warning(f"一部の銘柄で情報が不足しています: {', '.join(failed_tickers)}")
                    st.info(f"利用可能: {len(successful_tickers)}/{len(tickers)}銘柄")
                else:
                    st.success(f"すべての銘柄の企業情報を利用できます: {len(successful_tickers)}銘柄")
                    
            except Exception as e:
                st.error(f"企業情報処理エラー: {str(e)}")
                return
        
        # バリュエーション統計を計算
        valuation_stats_df = calculate_portfolio_valuation_metrics(pnl_df, ticker_complete_info)
        
        if not valuation_stats_df.empty:
            st.subheader("📊 ポートフォリオバリュエーション統計")
            
            # ベンチマーク比較の説明
            st.info("💡 **ベンチマーク比較**: 左側の4列（MSCI ACWI、NASDAQ-100、S&P 500、TOPIX）は主要市場指数のETF指標値です。ポートフォリオの加重平均と比較できます。")
            
            # 統計テーブルの表示
            display_stats_df = valuation_stats_df.copy()
            
            # 数値フォーマット関数
            def format_valuation_value(value, metric_name):
                if value is None or pd.isna(value):
                    return "-"
                try:
                    if metric_name in ['ROE', 'ROA', '営業利益率', '純利益率']:
                        # 財務指標は小数形式なので100倍してパーセント表示
                        return f"{value * 100:.2f}%"
                    elif metric_name in ['配当利回り']:
                        # 配当利回りは既にパーセント値の場合が多い
                        return f"{value:.2f}%"
                    elif metric_name in ['時価総額（円）']:
                        if value >= 1e12:
                            return f"{value/1e12:.2f}兆円"
                        elif value >= 1e9:
                            return f"{value/1e9:.2f}億円"
                        elif value >= 1e6:
                            return f"{value/1e6:.2f}百万円"
                        else:
                            return f"{value:,.0f}円"
                    else:
                        return f"{value:.2f}"
                except:
                    return "-"
            
            # ETFベンチマーク列を特定
            etf_columns = ['MSCI ACWI', 'NASDAQ-100', 'S&P 500', 'TOPIX (Japan)']
            
            # 数値列をフォーマット（ポートフォリオ統計とETFベンチマーク列の両方）
            numeric_cols = ['加重平均', '中央値', '25%タイル', '75%タイル', '最小値', '最大値'] + etf_columns
            for col in numeric_cols:
                if col in display_stats_df.columns:
                    display_stats_df[col] = display_stats_df.apply(
                        lambda row: format_valuation_value(row[col], row['指標']), axis=1
                    )
            
            st.dataframe(display_stats_df, use_container_width=True)
            
            # サマリー情報
            st.subheader("📈 主要指標サマリー")
            
            # 重要な指標を抜き出して表示
            key_metrics = ['予想PER', 'PBR', 'ROE', '配当利回り']
            
            cols = st.columns(len(key_metrics))
            for i, metric in enumerate(key_metrics):
                metric_row = valuation_stats_df[valuation_stats_df['指標'] == metric]
                if not metric_row.empty:
                    weighted_avg = metric_row.iloc[0]['加重平均']
                    valid_count = metric_row.iloc[0]['有効銘柄数']
                    
                    with cols[i]:
                        if weighted_avg is not None and not pd.isna(weighted_avg):
                            if metric == 'ROE':
                                # ROEは小数形式なので100倍してパーセント表示
                                value_display = f"{weighted_avg * 100:.2f}%"
                            elif metric == '配当利回り':
                                # 配当利回りは既にパーセント値
                                value_display = f"{weighted_avg:.2f}%"
                            else:
                                value_display = f"{weighted_avg:.2f}"
                            st.metric(
                                label=f"{metric}（加重平均）",
                                value=value_display,
                                delta=f"有効銘柄: {valid_count}/{len(tickers)}"
                            )
                        else:
                            st.metric(
                                label=f"{metric}（加重平均）",
                                value="データなし",
                                delta=f"有効銘柄: 0/{len(tickers)}"
                            )
            
            # データ取得状況と診断情報
            with st.expander("🔍 データ取得状況の詳細"):
                st.write("**📊 データ取得統計:**")
                
                # 基本情報の取得状況
                country_success = sum(1 for info in ticker_complete_info.values() if info and info.get('country'))
                sector_success = sum(1 for info in ticker_complete_info.values() if info and info.get('sector'))
                
                st.write(f"- 本社所在国: {country_success}/{len(tickers)}銘柄 ({country_success/len(tickers)*100:.1f}%)")
                st.write(f"- セクター情報: {sector_success}/{len(tickers)}銘柄 ({sector_success/len(tickers)*100:.1f}%)")
                
                st.write("**💰 バリュエーション指標の取得状況:**")
                for _, row in valuation_stats_df.iterrows():
                    metric_name = row['指標']
                    valid_count = row['有効銘柄数']
                    success_rate = (valid_count / len(tickers)) * 100
                    
                    if success_rate >= 80:
                        status = "🟢"
                    elif success_rate >= 50:
                        status = "🟡" 
                    else:
                        status = "🔴"
                    
                    st.write(f"{status} **{metric_name}**: {valid_count}/{len(tickers)}銘柄 ({success_rate:.1f}%)")
                
                # 診断とトラブルシューティング
                st.write("**🔧 トラブルシューティング:**")
                low_success_metrics = [row['指標'] for _, row in valuation_stats_df.iterrows() 
                                     if (row['有効銘柄数'] / len(tickers)) < 0.5]
                
                if low_success_metrics:
                    st.warning(f"以下の指標の取得率が低いです: {', '.join(low_success_metrics)}")
                    st.write("**改善提案:**")
                    st.write("- ネットワーク接続を確認してください")
                    st.write("- ティッカーシンボルが正しいか確認してください") 
                    st.write("- しばらく時間をおいてから再試行してください（API制限の可能性）")
                    st.write("- ETFや個別株式で取得可能な指標が異なる場合があります")
                else:
                    st.success("すべての指標が良好に取得されています！")
        else:
            st.warning("バリュエーション統計の計算に失敗しました。企業情報の取得状況を確認してください。")
            
            # デバッグ情報
            st.write("**デバッグ情報:**")
            valuation_keys = ['forwardPE', 'priceToBook', 'priceToSalesTrailing12Months', 
                            'enterpriseToEbitda', 'pegRatio', 'marketCap', 'beta', 'dividendYield',
                            'returnOnEquity', 'returnOnAssets', 'operatingMargins', 'profitMargins']
            
            for key in valuation_keys:
                count = sum(1 for info in ticker_complete_info.values() 
                          if info and info.get(key) is not None)
                st.write(f"{key}: {count}/{len(tickers)}銘柄")
    
    except Exception as e:
        display_error_message(e, "バリュエーション分析中にエラーが発生しました")


def display_detailed_data(pnl_df: pd.DataFrame, original_df: pd.DataFrame, tickers: List[str]):
    """詳細データの表示"""
    st.subheader("🔍 詳細データ")
    
    # 損益詳細テーブル
    st.subheader("💰 損益詳細")
    
    try:
        # データアダプターからキャッシュされた企業情報を取得
        data_adapter = st.session_state.get('data_adapter')
        if not data_adapter:
            st.error("データが初期化されていません。ページを再読み込みしてください。")
            return
        
        with show_loading_spinner("詳細データを処理中..."):
            # キャッシュされた完全な企業情報を取得
            ticker_complete_info = data_adapter.get_multiple_ticker_complete_info(tickers)
            
            # データ取得結果の検証
            if not ticker_complete_info:
                st.error("企業情報が利用できません。")
                return
            
            # 成功した銘柄数をカウント
            successful_count = sum(1 for info in ticker_complete_info.values() 
                                 if info and (info.get('country') or info.get('sector')))
            
            if successful_count == 0:
                st.error("すべての銘柄で企業情報が不足しています。")
                return
            elif successful_count < len(tickers):
                st.info(f"企業情報（キャッシュより）: {successful_count}/{len(tickers)}銘柄で利用可能")
            else:
                st.success(f"企業情報（キャッシュより）: {successful_count}/{len(tickers)}銘柄で利用可能")
        
        # 企業名を取得
        if 'company_names_cache' not in st.session_state:
            from modules.price_fetcher import cached_get_company_names
            with show_loading_spinner("企業名を取得中..."):
                st.session_state.company_names_cache = cached_get_company_names(tuple(tickers))
        
        company_names = st.session_state.company_names_cache
        
        # 表示用にフォーマット
        display_pnl_df = pnl_df.copy()
        
        # 本社所在国、地域、セクター情報を追加
        from modules.country_fetcher import classify_region_by_country
        
        def get_company_info(ticker):
            info = ticker_complete_info.get(ticker, {})
            if not info:
                info = {}
            
            country = info.get('country')
            sector = info.get('sector')
            region = classify_region_by_country(country)
            
            # セクターの処理：取得失敗時は「その他」とする
            if not sector or (isinstance(sector, str) and sector.strip() == ""):
                if '.T' in str(ticker) or '.JP' in str(ticker):
                    sector_display = "その他（日本）"
                else:
                    sector_display = "その他"
            else:
                sector_display = sector.strip()
            
            # バリュエーション指標を取得
            def safe_format_value(value, format_type='number', metric_key=None):
                if value is None or pd.isna(value):
                    return ""
                
                # 営業利益率・純利益率の外れ値チェック
                if metric_key in ['operatingMargins', 'profitMargins'] and value is not None:
                    if value < -1.0 or value > 1.0:
                        return ""  # 外れ値はブランク表示
                
                try:
                    if format_type == 'percentage':
                        # Yahoo Financeの財務指標は既に小数形式（0.10 = 10%）なので100倍してパーセント表示
                        return f"{value * 100:.2f}%"
                    elif format_type == 'dividend_percentage':
                        # 配当利回りは既にパーセント値として提供される場合が多い
                        return f"{value:.2f}%"
                    elif format_type == 'large_number':
                        # 時価総額などの大きな数値
                        if value >= 1e12:
                            return f"{value/1e12:.2f}T"
                        elif value >= 1e9:
                            return f"{value/1e9:.2f}B"
                        elif value >= 1e6:
                            return f"{value/1e6:.2f}M"
                        else:
                            return f"{value:,.0f}"
                    else:
                        return f"{value:.2f}"
                except:
                    return ""
            
            valuation_data = {
                'forwardPE': safe_format_value(info.get('forwardPE')),
                'priceToBook': safe_format_value(info.get('priceToBook')),
                'priceToSalesTrailing12Months': safe_format_value(info.get('priceToSalesTrailing12Months')),
                'enterpriseToEbitda': safe_format_value(info.get('enterpriseToEbitda')),
                'pegRatio': safe_format_value(info.get('pegRatio')),
                'marketCap': safe_format_value(info.get('marketCap'), 'large_number'),
                'beta': safe_format_value(info.get('beta')),
                'dividendYield': safe_format_value(info.get('dividendYield'), 'dividend_percentage'),
                'returnOnEquity': safe_format_value(info.get('returnOnEquity'), 'percentage'),
                'returnOnAssets': safe_format_value(info.get('returnOnAssets'), 'percentage'),
                'operatingMargins': safe_format_value(info.get('operatingMargins'), 'percentage', 'operatingMargins'),
                'profitMargins': safe_format_value(info.get('profitMargins'), 'percentage', 'profitMargins')
            }
            
            return (
                country if country else "取得失敗",
                region,
                sector_display,
                valuation_data
            )
        
        # 企業情報カラムを追加
        company_data = [get_company_info(ticker) for ticker in display_pnl_df['ticker']]
        display_pnl_df['企業名'] = [company_names.get(ticker, ticker) for ticker in display_pnl_df['ticker']]
        display_pnl_df['本社所在国'] = [data[0] for data in company_data]
        display_pnl_df['地域'] = [data[1] for data in company_data]
        display_pnl_df['セクター'] = [data[2] for data in company_data]
        
        # バリュエーション指標を追加
        display_pnl_df['予想PER'] = [data[3]['forwardPE'] for data in company_data]
        display_pnl_df['PBR'] = [data[3]['priceToBook'] for data in company_data]
        display_pnl_df['PSR'] = [data[3]['priceToSalesTrailing12Months'] for data in company_data]
        display_pnl_df['EV/EBITDA'] = [data[3]['enterpriseToEbitda'] for data in company_data]
        display_pnl_df['PEGレシオ'] = [data[3]['pegRatio'] for data in company_data]
        display_pnl_df['時価総額'] = [data[3]['marketCap'] for data in company_data]
        display_pnl_df['ベータ'] = [data[3]['beta'] for data in company_data]
        display_pnl_df['配当利回り'] = [data[3]['dividendYield'] for data in company_data]
        # 新しい財務指標を追加
        display_pnl_df['ROE'] = [data[3]['returnOnEquity'] for data in company_data]
        display_pnl_df['ROA'] = [data[3]['returnOnAssets'] for data in company_data]
        display_pnl_df['営業利益率'] = [data[3]['operatingMargins'] for data in company_data]
        display_pnl_df['純利益率'] = [data[3]['profitMargins'] for data in company_data]
        
        # 数値カラムをフォーマット
        numeric_columns = ['avg_cost_jpy', 'current_price_jpy', 'current_value_jpy', 
                          'cost_basis_jpy', 'pnl_amount']
        
        for col in numeric_columns:
            if col in display_pnl_df.columns:
                display_pnl_df[col] = display_pnl_df[col].apply(lambda x: format_currency(x))
        
        if 'pnl_percentage' in display_pnl_df.columns:
            display_pnl_df['pnl_percentage'] = display_pnl_df['pnl_percentage'].apply(
                lambda x: format_percentage(x)
            )
        
        # カラム順序を調整（基本情報、損益情報、バリュエーション指標の順）
        basic_columns = ['ticker', '企業名', '本社所在国', '地域', 'セクター']
        pnl_columns = ['shares', 'avg_cost_jpy', 'current_price_jpy', 'current_value_jpy', 
                      'cost_basis_jpy', 'pnl_amount', 'pnl_percentage']
        valuation_columns = ['予想PER', 'PBR', 'PSR', 'EV/EBITDA', 'PEGレシオ', 
                           '時価総額', 'ベータ', '配当利回り', 'ROE', 'ROA', '営業利益率', '純利益率']
        
        # 存在するカラムのみを含める
        columns_order = []
        for col_list in [basic_columns, pnl_columns, valuation_columns]:
            columns_order.extend([col for col in col_list if col in display_pnl_df.columns])
        
        # 残りのカラムも追加
        other_columns = [col for col in display_pnl_df.columns if col not in columns_order]
        display_pnl_df = display_pnl_df[columns_order + other_columns]
        
        st.dataframe(display_pnl_df, use_container_width=True)
        
    except Exception as e:
        display_error_message(e, "詳細データ表示中にエラーが発生しました")
        # エラー時は元の表示にフォールバック
        display_pnl_df = pnl_df.copy()
        numeric_columns = ['avg_cost_jpy', 'current_price_jpy', 'current_value_jpy', 
                          'cost_basis_jpy', 'pnl_amount']
        
        for col in numeric_columns:
            if col in display_pnl_df.columns:
                display_pnl_df[col] = display_pnl_df[col].apply(lambda x: format_currency(x))
        
        if 'pnl_percentage' in display_pnl_df.columns:
            display_pnl_df['pnl_percentage'] = display_pnl_df['pnl_percentage'].apply(
                lambda x: format_percentage(x)
            )
        
        st.dataframe(display_pnl_df, use_container_width=True)
    
    # オリジナルデータ
    with st.expander("📄 オリジナルデータ"):
        st.dataframe(original_df, use_container_width=True)
    
    # データダウンロード
    col1, col2 = st.columns(2)
    
    with col1:
        pnl_csv = pnl_df.to_csv(index=False)
        st.download_button(
            label="📥 損益データをダウンロード",
            data=pnl_csv,
            file_name="portfolio_pnl.csv",
            mime="text/csv"
        )
    
    with col2:
        original_csv = original_df.to_csv(index=False)
        st.download_button(
            label="📥 オリジナルデータをダウンロード",
            data=original_csv,
            file_name="portfolio_original.csv",
            mime="text/csv"
        )


def display_stock_charts(tickers: List[str]):
    """株価チャート（Geminiニュース分析付き）の表示"""
    st.subheader("📊 株価チャート")
    
    if not tickers:
        st.warning("表示する銘柄がありません。")
        return
    
    # セッションステートでチャート設定を管理
    if 'chart_ticker' not in st.session_state:
        st.session_state.chart_ticker = tickers[0] if tickers else ""
    if 'chart_from_date' not in st.session_state:
        st.session_state.chart_from_date = datetime.now() - timedelta(days=30)
    if 'chart_to_date' not in st.session_state:
        st.session_state.chart_to_date = datetime.now()
    if 'chart_model' not in st.session_state:
        st.session_state.chart_model = "gemini-1.5-pro"
    
    # 5年前の日付制限
    max_past_date = datetime.now() - timedelta(days=5*365)
    
    st.markdown("### ⚙️ 設定")
    
    # 銘柄選択と期間設定
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        selected_ticker = st.selectbox(
            "表示する銘柄を選択",
            options=tickers,
            index=tickers.index(st.session_state.chart_ticker) if st.session_state.chart_ticker in tickers else 0,
            help="チャートを表示する銘柄を選択してください",
            key="chart_ticker_selector"
        )
        st.session_state.chart_ticker = selected_ticker
    
    with col2:
        from_date = st.date_input(
            "開始日",
            value=st.session_state.chart_from_date.date(),
            min_value=max_past_date.date(),
            max_value=datetime.now().date(),
            help="チャート表示開始日（最大過去5年まで）",
            key="chart_from_date_input"
        )
        # 日付制限チェック
        from_date_dt = datetime.combine(from_date, datetime.min.time())
        if from_date_dt < max_past_date:
            st.error(f"⚠️ 開始日は過去5年間（{max_past_date.strftime('%Y-%m-%d')}）以降を選択してください。")
            from_date_dt = max_past_date
            st.info(f"開始日を {max_past_date.strftime('%Y-%m-%d')} に設定しました。")
        st.session_state.chart_from_date = from_date_dt
    
    with col3:
        to_date = st.date_input(
            "終了日",
            value=st.session_state.chart_to_date.date(),
            min_value=from_date,
            max_value=datetime.now().date(),
            help="チャート表示終了日",
            key="chart_to_date_input"
        )
        to_date_dt = datetime.combine(to_date, datetime.min.time())
        
        # 期間妥当性チェック
        if to_date_dt <= from_date_dt:
            st.error("⚠️ 終了日は開始日より後の日付を選択してください。")
            return
        
        # 期間が長すぎないかチェック
        days_diff = (to_date_dt - from_date_dt).days
        if days_diff > 5*365:
            st.error("⚠️ 選択期間が5年を超えています。期間を短縮してください。")
            return
            
        st.session_state.chart_to_date = to_date_dt
    
    with col4:
        model_options = ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro"]
        selected_model = st.selectbox(
            "Geminiモデル",
            options=model_options,
            index=model_options.index(st.session_state.chart_model) if st.session_state.chart_model in model_options else 0,
            help="ニュース分析に使用するGeminiモデルを選択",
            key="chart_model_selector"
        )
        st.session_state.chart_model = selected_model
    
    if selected_ticker and from_date_dt and to_date_dt:
        try:
            # データアダプターからキャッシュされた過去データを取得
            data_adapter = st.session_state.get('data_adapter')
            if not data_adapter:
                st.error("データが初期化されていません。ページを再読み込みしてください。")
                return
            
            with show_loading_spinner(f"{selected_ticker}のチャートデータを処理中..."):
                # 5年間のデータを取得して期間でフィルタリング
                full_data = data_adapter.get_historical_data(selected_ticker, period="5y")
                
                if not full_data.empty:
                    # 指定期間でフィルタリング
                    chart_data = full_data[
                        (full_data.index >= from_date_dt) & 
                        (full_data.index <= to_date_dt)
                    ]
                else:
                    chart_data = pd.DataFrame()
                
                if not chart_data.empty:
                    # 終値ラインチャート
                    period_str = f"{from_date_dt.strftime('%Y-%m-%d')} to {to_date_dt.strftime('%Y-%m-%d')}"
                    line_chart = create_stock_line_chart(chart_data, selected_ticker, period_str)
                    st.plotly_chart(line_chart, use_container_width=True)
                    
                    # 基本統計情報
                    with st.expander("📈 期間統計"):
                        col1, col2, col3, col4 = st.columns(4)
                        
                        period_return = ((chart_data['Close'].iloc[-1] / chart_data['Close'].iloc[0]) - 1) * 100
                        start_price = chart_data['Close'].iloc[0]
                        end_price = chart_data['Close'].iloc[-1]
                        max_price = chart_data['Close'].max()
                        min_price = chart_data['Close'].min()
                        
                        with col1:
                            st.metric("期間リターン", f"{period_return:+.2f}%")
                        with col2:
                            st.metric("開始価格", f"{start_price:.2f}")
                        with col3:
                            st.metric("最新価格", f"{end_price:.2f}")
                        with col4:
                            st.metric("期間高値/安値", f"{max_price:.2f} / {min_price:.2f}")
                    
                    # Geminiニュース分析機能（チャートの下に配置）
                    st.markdown("---")
                    st.markdown("### 📰 銘柄ニュース分析（Gemini AI）")
                    
                    if REPORT_GENERATION_AVAILABLE:
                        # ニュース分析設定
                        col1, col2, col3 = st.columns([1, 1, 2])
                        
                        with col1:
                            # ニュース記事数の選択を追加
                            if 'stock_news_count' not in st.session_state:
                                st.session_state.stock_news_count = 15
                            
                            stock_news_count = st.slider(
                                "取得記事数",
                                min_value=0,
                                max_value=100,
                                value=st.session_state.stock_news_count,
                                step=5,
                                help="取得するニュース記事の最大数（0-100）",
                                key=f"stock_news_count_slider_{selected_ticker}"
                            )
                            st.session_state.stock_news_count = stock_news_count
                        
                        with col2:
                            if st.button(f"🔍 ニュース分析を実行", type="secondary", key=f"news_analysis_btn_{selected_ticker}"):
                                generate_stock_news_analysis(selected_ticker, from_date_dt, to_date_dt, selected_model, st.session_state.stock_news_count)
                        
                        with col3:
                            st.caption("選択した期間の銘柄関連ニュースをAIが分析します")
                        
                        # 前回の分析結果があれば自動で表示
                        analysis_key = f'stock_news_analysis_{selected_ticker}'
                        if analysis_key in st.session_state:
                            st.markdown("#### 📊 分析結果")
                            display_stock_news_analysis_result(
                                st.session_state[analysis_key],
                                selected_ticker
                            )
                        else:
                            st.info(f"💡 「ニュース分析を実行」ボタンをクリックして、{selected_ticker}の期間ニュース分析を開始できます。")
                    
                    else:
                        missing_components = []
                        if not GEMINI_AVAILABLE:
                            missing_components.append("Gemini API")
                        if not GOOGLE_SEARCH_AVAILABLE:
                            missing_components.append("Google Search API")
                        if not SCRAPING_AVAILABLE:
                            missing_components.append("スクレイピングライブラリ")
                        st.warning(f"ニュース分析機能に必要なコンポーネントが不足しています: {', '.join(missing_components)}")
                else:
                    st.error(f"{selected_ticker}の指定期間のチャートデータを取得できませんでした。")
                    
        except Exception as e:
            display_error_message(e, f"{selected_ticker}のチャート表示中にエラーが発生しました")


def generate_stock_news_analysis(ticker: str, from_date: datetime, to_date: datetime, model_name: str = "gemini-1.5-pro", news_count: int = 15):
    """個別銘柄のニュース分析を生成"""
    try:
        # モジュールをインポート
        from modules.google_search import get_financial_news_urls
        from modules.news_scraper import scrape_news_articles
        from modules.gemini_api import GeminiClient, safe_text_processing
        
        # ステップ1: 銘柄固有のニュースを検索
        with st.spinner(f"{ticker}関連のニュースを検索中..."):
            # 企業名を取得
            try:
                from modules.price_fetcher import cached_get_company_names
                company_names = cached_get_company_names((ticker,))
                company_name = company_names.get(ticker, ticker)
            except:
                company_name = ticker
                
            # 検索クエリを銘柄固有に設定
            search_topics = [
                f"{ticker} {company_name} 株価",
                f"{ticker} {company_name} 決算",
                f"{ticker} {company_name} ニュース",
                f"{ticker} 業績 発表",
                f"{ticker} 株式 分析",
                f"{company_name} 企業 動向"
            ]
            
            news_items = get_financial_news_urls(
                start_date=from_date,
                end_date=to_date,
                search_topics=search_topics
            )
            
            if not news_items:
                st.warning(f"{ticker}({company_name})に関連するニュースが見つかりませんでした。期間を調整してお試しください。")
                return
        
        # ステップ2: ニュース記事をスクレイピング
        with st.spinner(f"{min(len(news_items), news_count)}件のニュース記事を取得中（最大{news_count}件）..."):
            articles_text = scrape_news_articles(
                news_items=news_items,
                max_articles=news_count,  # ユーザー指定の記事数
                delay=0.5
            )
            
            if not articles_text or len(articles_text) < 50:
                st.warning("ニュース記事の取得に失敗しました。時間をおいてもう一度お試しください。")
                return
        
        # ステップ3: Gemini APIで銘柄分析を生成
        with st.spinner("AI分析レポートを生成中..."):
            gemini_client = GeminiClient(model_name=model_name)
            
            # 銘柄固有の分析プロンプト
            prompt = create_stock_analysis_prompt(
                ticker=ticker,
                company_name=company_name,
                articles_text=articles_text,
                from_date=from_date,
                to_date=to_date
            )
            
            try:
                safe_prompt = safe_text_processing(prompt)
                response = gemini_client.model.generate_content(
                    safe_prompt,
                    generation_config=gemini_client.generation_config
                )
                
                if response.text:
                    analysis_result = {
                        "success": True,
                        "ticker": ticker,
                        "company_name": company_name,
                        "analysis": safe_text_processing(response.text),
                        "period": f"{from_date.strftime('%Y-%m-%d')} ~ {to_date.strftime('%Y-%m-%d')}",
                        "news_count": len(news_items),
                        "model_used": model_name,
                        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    # セッションステートに保存
                    st.session_state[f'stock_news_analysis_{ticker}'] = analysis_result
                    st.success(f"✅ {ticker}のニュース分析が完了しました！")
                    
                else:
                    st.error("AIレスポンスが空でした。もう一度お試しください。")
                    
            except Exception as e:
                st.error(f"AI分析中にエラーが発生しました: {str(e)}")
                
    except Exception as e:
        st.error(f"ニュース分析中にエラーが発生しました: {str(e)}")


def create_stock_analysis_prompt(ticker: str, company_name: str, articles_text: str, 
                                from_date: datetime, to_date: datetime) -> str:
    """銘柄分析用のプロンプトを作成"""
    from modules.gemini_api import safe_text_processing
    
    from_date_str = f"{from_date.year}年{from_date.month}月{from_date.day}日"
    to_date_str = f"{to_date.year}年{to_date.month}月{to_date.day}日"
    
    # テキストを安全に処理
    safe_articles_text = safe_text_processing(articles_text[:12000])
    
    prompt = f"""以下のニュース記事を基に、{ticker}（{company_name}）の
{from_date_str}から{to_date_str}までの期間における企業分析レポートを作成してください。

【対象企業】{ticker} - {company_name}
【分析期間】{from_date_str} ～ {to_date_str}

【収集したニュース記事】
{safe_articles_text}

【分析レポート要件】

## 1. 期間中の主要なニュース・イベント（300-400字）
- 決算発表や業績予想の内容
- 新製品・サービス・事業発表
- 経営陣の発言や戦略発表
- M&A、提携、投資活動
- その他重要な企業イベント

## 2. 株価に影響を与えた要因分析（400-500字）
- ポジティブ要因（株価押し上げ要因）
- ネガティブ要因（株価押し下げ要因）
- 市場の反応と投資家センチメント
- 業界全体の動向との関係
- 競合他社との比較における位置付け

## 3. 企業の財務・業績分析（300-400字）
- 売上高、利益の動向
- 成長性の評価
- 収益性・効率性の変化
- バランスシートの健全性
- キャッシュフロー状況

## 4. 今後の展望と注目ポイント（300-400字）
- 短期的（3-6ヶ月）な注目要因
- 中長期的な成長ドライバー
- 潜在的なリスク要因
- 業界トレンドとの関係
- 投資判断における考慮事項

【出力要件】
- 合計1200-1600字程度
- 客観的で分析的な文体
- ニュース記事から得られた具体的な情報を積極的に引用
- 投資推奨は避け、情報提供と分析に徹する
- 見出しや段落を適切に使用して読みやすく構成

【重要事項】
- 具体的な売買推奨は一切行わない
- 分析は参考情報の提供に留める
- 最後に「本分析は情報提供のみを目的としており、投資判断は自己責任で行ってください」という免責事項を記載
"""
    
    return prompt


def display_stock_news_analysis_result(analysis_result: Dict[str, Any], ticker: str):
    """銘柄ニュース分析結果の表示"""
    try:
        if not analysis_result.get("success", False):
            st.error(f"分析結果の取得に失敗しました: {analysis_result.get('error', 'Unknown error')}")
            return
        
        st.markdown(f"### 📋 {ticker} AI分析レポート")
        
        # レポート概要
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("対象銘柄", f"{analysis_result.get('ticker', 'N/A')}")
        with col2:
            st.metric("分析期間", analysis_result.get('period', 'N/A'))
        with col3:
            st.metric("参照ニュース数", f"{analysis_result.get('news_count', 0)}件")
        with col4:
            st.metric("使用モデル", analysis_result.get('model_used', 'N/A'))
        
        st.markdown("---")
        
        # AI分析内容
        analysis_content = analysis_result.get("analysis", "分析内容なし")
        st.markdown(analysis_content)
        
        st.markdown("---")
        
        # 生成情報
        st.caption(f"🤖 生成時刻: {analysis_result.get('timestamp', 'N/A')} | 企業名: {analysis_result.get('company_name', 'N/A')}")
        
        # 免責事項
        st.markdown("### ⚠️ 免責事項")
        st.warning("""
        **重要:** この分析レポートは情報提供のみを目的としており、投資推奨ではありません。
        - AI分析は収集したニュース記事に基づく参考情報です
        - 投資判断は必ずご自身の責任で行ってください
        - 過去の情報やパフォーマンスは将来の結果を保証するものではありません
        - 投資にはリスクが伴います。専門家への相談を推奨します
        """)
    
    except Exception as e:
        st.error(f"分析結果表示エラー: {str(e)}")



def display_investment_report(pnl_df: pd.DataFrame, tickers: List[str]):
    """運用報告の表示"""
    st.subheader("📋 運用報告")
    
    try:
        # セッションステートで設定を管理
        if 'report_from_date' not in st.session_state:
            st.session_state.report_from_date = datetime.now() - timedelta(days=30)
        if 'report_to_date' not in st.session_state:
            st.session_state.report_to_date = datetime.now()
        if 'report_model' not in st.session_state:
            st.session_state.report_model = "gemini-1.5-pro"
        
        # 企業名を取得
        if 'company_names_cache' not in st.session_state:
            from modules.price_fetcher import cached_get_company_names
            with show_loading_spinner("企業名を取得中..."):
                st.session_state.company_names_cache = cached_get_company_names(tuple(tickers))
        
        company_names = st.session_state.company_names_cache
        
        # 設定UI
        st.markdown("### ⚙️ 分析設定")
        
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        
        with col1:
            from_date = st.date_input(
                "開始日",
                value=st.session_state.report_from_date.date(),
                help="分析期間の開始日",
                key="report_from_date_input"
            )
            st.session_state.report_from_date = datetime.combine(from_date, datetime.min.time())
        
        with col2:
            to_date = st.date_input(
                "終了日",
                value=st.session_state.report_to_date.date(),
                help="分析期間の終了日",
                key="report_to_date_input"
            )
            st.session_state.report_to_date = datetime.combine(to_date, datetime.min.time())
        
        with col3:
            model_options = ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro"]
            selected_model = st.selectbox(
                "Geminiモデル",
                options=model_options,
                index=model_options.index(st.session_state.report_model) if st.session_state.report_model in model_options else 0,
                help="使用するGeminiモデルを選択",
                key="report_model_selector"
            )
            st.session_state.report_model = selected_model
        
        with col4:
            # ニュース記事数の選択を追加
            if 'report_news_count' not in st.session_state:
                st.session_state.report_news_count = 20
            
            news_count = st.slider(
                "取得記事数",
                min_value=0,
                max_value=100,
                value=st.session_state.report_news_count,
                step=5,
                help="取得するニュース記事の最大数（0-100）",
                key="report_news_count_slider"
            )
            st.session_state.report_news_count = news_count
        
        # 期間の妥当性チェック
        if from_date >= to_date:
            st.error("⚠️ 開始日は終了日より前に設定してください")
            return
        
        # 相対パフォーマンス分析実行ボタン
        if st.button("📈 パフォーマンス分析を実行", type="primary"):
            with show_loading_spinner("相対パフォーマンスを分析中..."):
                performance_result = analyze_relative_performance(
                    pnl_df, tickers, company_names,
                    st.session_state.report_from_date,
                    st.session_state.report_to_date
                )
                st.session_state.performance_result = performance_result
        
        # パフォーマンス結果があれば表示
        if 'performance_result' in st.session_state and st.session_state.performance_result:
            display_relative_performance_analysis(st.session_state.performance_result)
            
            # レポート生成機能の可用性チェック
            if REPORT_GENERATION_AVAILABLE:
                # 運用レポート生成ボタン
                if st.button("📋 運用レポートを生成（ニュース分析付き）", type="secondary"):
                    with show_loading_spinner("運用レポートを生成中..."):
                        report_result = generate_investment_report(
                            st.session_state.performance_result,
                            st.session_state.report_from_date,
                            st.session_state.report_to_date,
                            selected_model,
                            st.session_state.report_news_count
                        )
                        st.session_state.report_result = report_result
                
                # レポート結果があれば表示
                if 'report_result' in st.session_state and st.session_state.report_result:
                    display_investment_report_result(st.session_state.report_result)
            else:
                missing_components = []
                if not GEMINI_AVAILABLE:
                    missing_components.append("Gemini API")
                if not GOOGLE_SEARCH_AVAILABLE:
                    missing_components.append("Google Search API")
                if not SCRAPING_AVAILABLE:
                    missing_components.append("スクレイピングライブラリ")
                st.warning(f"運用レポート生成機能に必要なコンポーネントが不足しています: {', '.join(missing_components)}")
        else:
            st.info("「パフォーマンス分析を実行」ボタンをクリックして、まず相対パフォーマンス分析を開始してください。")
            
            # 分析内容の説明
            with st.expander("📋 運用報告の内容"):
                st.markdown("""
                **相対パフォーマンス分析：**
                
                - 📈 **個別銘柄パフォーマンス**: 期間始点を100とした相対推移
                - 📊 **ポートフォリオ全体**: 保有株数加重での総合パフォーマンス
                - 🏆 **ベンチマーク比較**: MSCI ACWI、NASDAQ100、Topix ETFとの比較
                - 📈 **パフォーマンステーブル**: 各銘柄の終値と期間リターン
                
                **AI運用レポート：**
                
                - 🌍 **経済・政治ニュース**: 期間内の主要な市場環境
                - 📊 **ポートフォリオ評価**: ベンチマーク対比での勝敗分析
                - ⭐ **優良銘柄分析**: 特にパフォーマンスの良かった銘柄
                - ⚠️ **劣後銘柄分析**: パフォーマンスの劣った銘柄とその要因
                
                **対象銘柄：**
                """)
                
                for ticker in tickers:
                    company_name = company_names.get(ticker, ticker)
                    st.markdown(f"- **{ticker}**: {company_name}")
    
    except Exception as e:
        display_error_message(e, "運用報告中にエラーが発生しました")


def analyze_relative_performance(pnl_df: pd.DataFrame, tickers: List[str], company_names: Dict[str, str],
                               from_date: datetime, to_date: datetime) -> Dict[str, Any]:
    """相対パフォーマンス分析を実行"""
    try:
        from modules.price_fetcher import get_historical_data
        
        # 期間を計算してyfinanceの期間パラメータを決定（余裕を持たせて取得）
        period_days = (to_date - from_date).days
        if period_days <= 30:
            period = "3mo"  # 1ヶ月分でも3ヶ月分取得して確実にデータを取得
        elif period_days <= 90:
            period = "6mo"
        elif period_days <= 180:
            period = "1y"
        elif period_days <= 365:
            period = "2y"
        else:
            period = "5y"
        
        # ベンチマークETFのティッカー
        benchmark_tickers = ["ACWI", "QQQ", "1348.T"]  # MSCI ACWI, NASDAQ100, Topix ETF
        benchmark_names = {
            "ACWI": "MSCI ACWI ETF",
            "QQQ": "NASDAQ100 ETF", 
            "1348.T": "Topix ETF"
        }
        
        # 全ティッカー（ポートフォリオ + ベンチマーク）
        all_tickers = tickers + benchmark_tickers
        
        # 過去データを取得
        historical_data = get_historical_data(all_tickers, period)
        
        if historical_data.empty:
            return {"error": "株価データの取得に失敗しました"}
        
        # 欠損値を前日の値で埋める（ffill）
        historical_data = historical_data.fillna(method='ffill')
        
        # タイムゾーンを統一
        if historical_data.index.tz is not None:
            historical_data.index = historical_data.index.tz_localize(None)
        
        # from_dateとto_dateのタイムゾーンを削除
        from_date_naive = from_date.replace(tzinfo=None)
        to_date_naive = to_date.replace(tzinfo=None)
        
        # 指定期間内のデータにフィルタリング
        mask = (historical_data.index >= from_date_naive) & (historical_data.index <= to_date_naive)
        period_data = historical_data.loc[mask]
        
        if period_data.empty:
            # 期間が見つからない場合は最も近い日付から取得を試行
            available_dates = historical_data.index
            closest_start = available_dates[available_dates >= from_date_naive]
            
            if len(closest_start) == 0:
                return {"error": f"指定開始日 {from_date_naive.strftime('%Y-%m-%d')} 以降のデータが見つかりません"}
            
            start_date = closest_start[0]
            end_date = min(to_date_naive, available_dates[-1])
            
            mask = (historical_data.index >= start_date) & (historical_data.index <= end_date)
            period_data = historical_data.loc[mask]
            
            if period_data.empty:
                return {"error": "指定期間のデータが見つかりません"}
        
        # 実際の分析期間を更新
        actual_start = period_data.index[0]
        actual_end = period_data.index[-1]
        
        # 個別銘柄のパフォーマンス計算
        ticker_performance = {}
        for ticker in tickers:
            if ticker in period_data.columns:
                prices = period_data[ticker].fillna(method='ffill')  # 個別にもffill適用
                prices = prices.dropna()
                if len(prices) > 0:
                    # 始点価格を取得
                    start_price = prices.iloc[0]
                    
                    # 相対パフォーマンス = その日の株価 / 始点での株価 * 100
                    normalized = (prices / start_price) * 100
                    
                    # 終値とパフォーマンス
                    end_price = prices.iloc[-1] if len(prices) > 1 else start_price
                    performance_pct = ((end_price / start_price) - 1) * 100
                    
                    ticker_performance[ticker] = {
                        "company_name": company_names.get(ticker, ticker),
                        "normalized_prices": normalized,
                        "end_price": end_price,
                        "performance_pct": performance_pct,
                        "dates": normalized.index,
                        "start_price": start_price,
                        "currency": "検証中"  # determine_currency_from_ticker関数は後で呼び出し
                    }
        
        # ポートフォリオ全体のパフォーマンス計算
        portfolio_performance = calculate_portfolio_performance(pnl_df, period_data, from_date_naive, to_date_naive)
        
        # ベンチマークのパフォーマンス計算
        benchmark_performance = {}
        for benchmark in benchmark_tickers:
            if benchmark in period_data.columns:
                prices = period_data[benchmark].fillna(method='ffill')  # 個別にもffill適用
                prices = prices.dropna()
                if len(prices) > 0:
                    # 始点価格を取得
                    start_price = prices.iloc[0]
                    
                    # 相対パフォーマンス = その日の価格 / 始点での価格 * 100
                    normalized = (prices / start_price) * 100
                    
                    # 終値とパフォーマンス
                    end_price = prices.iloc[-1] if len(prices) > 1 else start_price
                    performance_pct = ((end_price / start_price) - 1) * 100
                    
                    benchmark_performance[benchmark] = {
                        "name": benchmark_names[benchmark],
                        "normalized_prices": normalized,
                        "performance_pct": performance_pct,
                        "dates": normalized.index,
                        "start_price": start_price
                    }
        
        return {
            "success": True,
            "period": f"{actual_start.strftime('%Y-%m-%d')} - {actual_end.strftime('%Y-%m-%d')}",
            "requested_period": f"{from_date_naive.strftime('%Y-%m-%d')} - {to_date_naive.strftime('%Y-%m-%d')}",
            "ticker_performance": ticker_performance,
            "portfolio_performance": portfolio_performance,
            "benchmark_performance": benchmark_performance,
            "period_data": period_data
        }
    
    except Exception as e:
        return {"error": f"パフォーマンス分析エラー: {str(e)}"}


def calculate_portfolio_performance(pnl_df: pd.DataFrame, period_data: pd.DataFrame, 
                                  from_date: datetime, to_date: datetime) -> Dict[str, Any]:
    """ポートフォリオ全体のパフォーマンスを計算（為替換算含む）"""
    try:
        from modules.price_fetcher import cached_get_exchange_rates, determine_currency_from_ticker, convert_to_jpy
        
        # 為替レートを取得
        exchange_rates = cached_get_exchange_rates()
        
        # 株数データを取得
        shares_data = {}
        for _, row in pnl_df.iterrows():
            ticker = row['ticker']
            shares = row['shares']
            shares_data[ticker] = shares
        
        # ffillを適用したperiod_dataを使用
        period_data_filled = period_data.fillna(method='ffill')
        
        # 各日付でのポートフォリオ価値を計算（円換算）
        portfolio_values_jpy = []
        valid_dates = []
        debug_info = []
        
        for date, row in period_data_filled.iterrows():
            total_value_jpy = 0
            valid_tickers = 0
            daily_debug = {"date": date, "tickers": {}}
            
            for ticker, shares in shares_data.items():
                if ticker in period_data_filled.columns and not pd.isna(row[ticker]):
                    # 現地通貨での株価
                    price_local = row[ticker]
                    
                    # 通貨を判定
                    currency = determine_currency_from_ticker(ticker)
                    
                    # 円換算
                    price_jpy = convert_to_jpy(price_local, currency, exchange_rates)
                    
                    # ポートフォリオ価値に追加
                    value_jpy = price_jpy * shares
                    total_value_jpy += value_jpy
                    valid_tickers += 1
                    
                    daily_debug["tickers"][ticker] = {
                        "price_local": price_local,
                        "currency": currency,
                        "price_jpy": price_jpy,
                        "shares": shares,
                        "value_jpy": value_jpy
                    }
            
            # 少なくとも1銘柄のデータがある場合のみ追加
            if valid_tickers > 0:
                portfolio_values_jpy.append(total_value_jpy)
                valid_dates.append(date)
                daily_debug["total_value_jpy"] = total_value_jpy
                debug_info.append(daily_debug)
        
        if len(portfolio_values_jpy) > 0:
            # 始点価値を取得
            start_value = portfolio_values_jpy[0]
            
            # 相対パフォーマンス = その日のポートフォリオ円換算額 / 始点でのポートフォリオ円換算額 * 100
            normalized_values = [(value / start_value) * 100 for value in portfolio_values_jpy]
            
            # パフォーマンス計算
            end_value = portfolio_values_jpy[-1] if len(portfolio_values_jpy) > 1 else start_value
            performance_pct = ((end_value / start_value) - 1) * 100
            
            return {
                "normalized_values": normalized_values,
                "performance_pct": performance_pct,
                "dates": valid_dates,
                "raw_values": portfolio_values_jpy,
                "start_value": start_value,
                "end_value": end_value,
                "debug_info": debug_info[:5]  # 最初の5日分のデバッグ情報
            }
        else:
            return {"error": "ポートフォリオの株価データが不足しています"}
    
    except Exception as e:
        return {"error": f"ポートフォリオ計算エラー: {str(e)}"}


def display_relative_performance_analysis(performance_result: Dict[str, Any]):
    """相対パフォーマンス分析結果の表示"""
    try:
        if not performance_result.get("success", False):
            st.error(f"分析エラー: {performance_result.get('error', 'Unknown error')}")
            return
        
        st.markdown("### 📈 相対パフォーマンス分析結果")
        
        # 期間情報を表示
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**実際の分析期間:** {performance_result['period']}")
        with col2:
            if 'requested_period' in performance_result:
                st.markdown(f"**要求期間:** {performance_result['requested_period']}")
        
        # パフォーマンスグラフの作成
        fig = go.Figure()
        
        # Y軸の範囲を計算するためのすべての値を収集
        all_values = []
        
        # 個別銘柄のグラフ
        ticker_performance = performance_result["ticker_performance"]
        for ticker, data in ticker_performance.items():
            normalized_values = data["normalized_prices"].values
            all_values.extend(normalized_values)
            
            fig.add_trace(go.Scatter(
                x=data["dates"],
                y=normalized_values,
                mode='lines',
                name=f"{ticker} ({data['company_name']})",
                line=dict(width=2),
                hovertemplate=f'<b>{ticker}</b><br>日付: %{{x}}<br>相対パフォーマンス: %{{y:.2f}}<extra></extra>'
            ))
        
        # ポートフォリオ全体のグラフ
        portfolio_data = performance_result["portfolio_performance"]
        if "normalized_values" in portfolio_data:
            portfolio_values = portfolio_data["normalized_values"]
            all_values.extend(portfolio_values)
            
            fig.add_trace(go.Scatter(
                x=portfolio_data["dates"],
                y=portfolio_values,
                mode='lines',
                name="ポートフォリオ全体",
                line=dict(width=4, color='red'),
                hovertemplate='<b>ポートフォリオ全体</b><br>日付: %{x}<br>相対パフォーマンス: %{y:.2f}<extra></extra>'
            ))
        
        # ベンチマークのグラフ
        benchmark_performance = performance_result["benchmark_performance"]
        colors = ["orange", "green", "purple"]
        for i, (benchmark, data) in enumerate(benchmark_performance.items()):
            benchmark_values = data["normalized_prices"].values
            all_values.extend(benchmark_values)
            
            fig.add_trace(go.Scatter(
                x=data["dates"],
                y=benchmark_values,
                mode='lines',
                name=data["name"],
                line=dict(width=3, dash='dash', color=colors[i % len(colors)]),
                hovertemplate=f'<b>{data["name"]}</b><br>日付: %{{x}}<br>相対パフォーマンス: %{{y:.2f}}<extra></extra>'
            ))
        
        # Y軸の範囲を計算
        if all_values:
            min_val = min(all_values)
            max_val = max(all_values)
            y_range = max_val - min_val
            y_padding = y_range * 0.05  # 5%のパディング
            y_min = max(0, min_val - y_padding)  # 0以下にはしない
            y_max = max_val + y_padding
        else:
            y_min, y_max = 95, 105  # デフォルト範囲
        
        fig.update_layout(
            title="相対パフォーマンス推移（期間始点=100）",
            xaxis_title="日付",
            yaxis_title="相対パフォーマンス",
            yaxis=dict(
                range=[y_min, y_max],
                tickformat=".1f"
            ),
            height=600,
            hovermode='x unified',
            showlegend=True
        )
        
        # 100のベースラインを追加
        fig.add_hline(y=100, line_dash="dot", line_color="gray", annotation_text="ベースライン (100)")
        
        # グラフデバッグ情報
        with st.expander("📊 グラフデバッグ情報"):
            st.write(f"**Y軸範囲:** {y_min:.2f} - {y_max:.2f}")
            st.write(f"**全データ点数:** {len(all_values)}")
            if all_values:
                st.write(f"**最小値:** {min(all_values):.2f}")
                st.write(f"**最大値:** {max(all_values):.2f}")
                st.write(f"**100周辺の値の確認:**")
                around_100 = [v for v in all_values if 95 <= v <= 105]
                st.write(f"- 95-105の範囲の値数: {len(around_100)}")
                
                # 個別銘柄の正規化確認
                st.write("**個別銘柄の正規化確認:**")
                for ticker, data in list(ticker_performance.items())[:3]:
                    prices = data["normalized_prices"]
                    st.write(f"- {ticker}: 開始値={prices.iloc[0]:.2f}, 終了値={prices.iloc[-1]:.2f}, データ点数={len(prices)}")
                
                # ポートフォリオの正規化確認
                if "normalized_values" in portfolio_data:
                    pf_values = portfolio_data["normalized_values"]
                    st.write(f"**ポートフォリオ:** 開始値={pf_values[0]:.2f}, 終了値={pf_values[-1]:.2f}, データ点数={len(pf_values)}")
                
                # ベンチマークの正規化確認
                st.write("**ベンチマーク正規化確認:**")
                for benchmark, data in benchmark_performance.items():
                    prices = data["normalized_prices"]
                    st.write(f"- {data['name']}: 開始値={prices.iloc[0]:.2f}, 終了値={prices.iloc[-1]:.2f}")
                    
                # 実際の株価データサンプル
                st.write("**元データサンプル（period_data）:**")
                if not performance_result["period_data"].empty:
                    sample_data = performance_result["period_data"].head(3)
                    st.dataframe(sample_data)
        
        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # パフォーマンステーブルの作成
        st.markdown("### 📊 パフォーマンステーブル")
        
        table_data = []
        
        # 個別銘柄
        for ticker, data in ticker_performance.items():
            table_data.append({
                "種別": "個別銘柄",
                "銘柄/ベンチマーク": f"{ticker} ({data['company_name']})",
                "終値": f"{data['end_price']:.2f}",
                "期間リターン(%)": data['performance_pct'],  # 数値として保存
                "期間リターン表示": f"{data['performance_pct']:+.2f}%"
            })
        
        # ポートフォリオ全体
        if "performance_pct" in portfolio_data:
            table_data.append({
                "種別": "ポートフォリオ",
                "銘柄/ベンチマーク": "ポートフォリオ全体",
                "終値": "-",
                "期間リターン(%)": portfolio_data['performance_pct'],  # 数値として保存
                "期間リターン表示": f"{portfolio_data['performance_pct']:+.2f}%"
            })
        
        # ベンチマーク
        for benchmark, data in benchmark_performance.items():
            table_data.append({
                "種別": "ベンチマーク",
                "銘柄/ベンチマーク": data["name"],
                "終値": "-",
                "期間リターン(%)": data['performance_pct'],  # 数値として保存
                "期間リターン表示": f"{data['performance_pct']:+.2f}%"
            })
        
        table_df = pd.DataFrame(table_data)
        
        # 表示用にソートされたテーブルを作成（数値でソート）
        display_df = table_df.copy()
        display_df = display_df.sort_values("期間リターン(%)", ascending=False)  # 降順でソート
        
        # 表示用列のみを選択
        display_df_formatted = display_df[["種別", "銘柄/ベンチマーク", "終値", "期間リターン表示"]].copy()
        display_df_formatted.columns = ["種別", "銘柄/ベンチマーク", "終値", "期間リターン"]
        
        # 数値ソート用のテーブルを作成
        sortable_df = table_df[["種別", "銘柄/ベンチマーク", "終値", "期間リターン(%)"]].copy()
        sortable_df.columns = ["種別", "銘柄/ベンチマーク", "終値", "期間リターン(%)"]
        
        # ソート可能なテーブルとして表示
        st.dataframe(
            sortable_df, 
            use_container_width=True,
            column_config={
                "期間リターン(%)": st.column_config.NumberColumn(
                    "期間リターン(%)",
                    help="パフォーマンス（数値ソート可能）",
                    format="%.2f"
                )
            }
        )
        
        # デバッグ情報を表示
        with st.expander("🔍 詳細デバッグ情報"):
            st.write("**テーブルソート用データ:**")
            debug_df = table_df[["銘柄/ベンチマーク", "期間リターン(%)", "期間リターン表示"]].copy()
            debug_df = debug_df.sort_values("期間リターン(%)", ascending=False)
            st.dataframe(debug_df)
            
            st.write("**個別銘柄計算詳細:**")
            for ticker, data in ticker_performance.items():
                st.write(f"**{ticker}**:")
                st.write(f"- 開始価格: {data.get('start_price', 'N/A'):.2f}")
                st.write(f"- 終了価格: {data.get('end_price', 'N/A'):.2f}")
                st.write(f"- パフォーマンス: {data.get('performance_pct', 'N/A'):.2f}%")
                if 'normalized_prices' in data:
                    first_5 = data['normalized_prices'].head(5).values
                    st.write(f"- 最初の5つの正規化値: {first_5}")
            
            st.write("**ポートフォリオ計算詳細:**")
            if "debug_info" in portfolio_data:
                for i, debug_day in enumerate(portfolio_data["debug_info"]):
                    st.write(f"**日付 {i+1}: {debug_day['date'].strftime('%Y-%m-%d')}**")
                    st.write(f"- 総価値: ¥{debug_day['total_value_jpy']:,.0f}")
                    for ticker, info in debug_day["tickers"].items():
                        st.write(f"  - {ticker}: {info['price_local']:.2f} {info['currency']} → ¥{info['price_jpy']:.2f} × {info['shares']} = ¥{info['value_jpy']:,.0f}")
            
            st.write("**ベンチマーク計算詳細:**")
            for benchmark, data in benchmark_performance.items():
                st.write(f"**{data['name']}**:")
                st.write(f"- 開始価格: {data.get('start_price', 'N/A'):.2f}")
                st.write(f"- パフォーマンス: {data.get('performance_pct', 'N/A'):.2f}%")
        
        # パフォーマンスサマリー
        st.markdown("### 🏆 パフォーマンスサマリー")
        
        if "performance_pct" in portfolio_data:
            portfolio_return = portfolio_data["performance_pct"]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("ポートフォリオリターン", f"{portfolio_return:+.2f}%")
            
            # ベンチマーク比較（MSCI ACWIをデフォルト）
            if benchmark_performance:
                # デフォルトはMSCI ACWI
                default_benchmark = "ACWI" if "ACWI" in benchmark_performance else list(benchmark_performance.keys())[0]
                
                with col2:
                    acwi_data = benchmark_performance.get(default_benchmark)
                    if acwi_data:
                        acwi_return = acwi_data["performance_pct"]
                        outperformance = portfolio_return - acwi_return
                        st.metric(
                            f"vs {acwi_data['name']}",
                            f"{outperformance:+.2f}%",
                            delta=f"{'勝' if outperformance > 0 else '負'}"
                        )
                
                with col3:
                    avg_benchmark_return = sum(data["performance_pct"] for data in benchmark_performance.values()) / len(benchmark_performance)
                    vs_avg = portfolio_return - avg_benchmark_return
                    st.metric(
                        "vs ベンチマーク平均",
                        f"{vs_avg:+.2f}%",
                        delta=f"{'勝' if vs_avg > 0 else '負'}"
                    )
    
    except Exception as e:
        st.error(f"パフォーマンス表示エラー: {str(e)}")


def generate_investment_report(performance_result: Dict[str, Any], from_date: datetime, 
                             to_date: datetime, model: str = "gemini-1.5-pro", news_count: int = 20) -> Dict[str, Any]:
    """Gemini APIとGoogle Search APIを使用して運用レポートを生成"""
    try:
        if not REPORT_GENERATION_AVAILABLE:
            missing_components = []
            if not GEMINI_AVAILABLE:
                missing_components.append("Gemini API")
            if not GOOGLE_SEARCH_AVAILABLE:
                missing_components.append("Google Search API")
            if not SCRAPING_AVAILABLE:
                missing_components.append("スクレイピングライブラリ")
            
            return {
                "success": False,
                "error": f"必要なコンポーネントが利用できません: {', '.join(missing_components)}",
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        
        # モジュールをインポート
        from modules.google_search import get_financial_news_urls
        from modules.news_scraper import scrape_news_articles
        from modules.gemini_api import generate_gemini_investment_report
        
        # ステップ1: ニュース記事URLを検索
        with st.spinner("金融ニュースを検索中..."):
            news_items = get_financial_news_urls(
                start_date=from_date,
                end_date=to_date,
                search_topics=[
                    "グローバル金融市場 動向",
                    "株式市場 日経平均 ダウ ナスダック",
                    "為替市場 ドル円 ユーロドル",
                    "中央銀行 金融政策 FRB ECB 日銀",
                    "経済指標 インフレ率 雇用統計 GDP",
                    "債券市場 金利 イールドカーブ",
                    "コモディティ市場 原油 金 商品",
                    "地政学リスク 国際情勢"
                ]
            )
            
            if not news_items:
                return {
                    "success": False,
                    "error": "ニュース記事が見つかりませんでした。期間を調整してもう一度お試しください。",
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
        
        # ステップ2: ニュース記事をスクレイピング
        with st.spinner(f"{min(len(news_items), news_count)}件のニュース記事を取得中（最大{news_count}件）..."):
            articles_text = scrape_news_articles(
                news_items=news_items,
                max_articles=news_count,  # ユーザー指定の記事数
                delay=0.5  # サーバー負荷軽減のため0.5秒待機
            )
            
            if not articles_text or len(articles_text) < 100:
                return {
                    "success": False,
                    "error": "ニュース記事の取得に失敗しました。時間をおいてもう一度お試しください。",
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
        
        # ステップ3: Gemini APIで要約を生成
        with st.spinner("AI分析レポートを生成中..."):
            report_result = generate_gemini_investment_report(
                performance_result=performance_result,
                from_date=from_date,
                to_date=to_date,
                news_articles_text=articles_text,
                model_name=model
            )
        
        return report_result
    
    except Exception as e:
        logger.error(f"レポート生成エラー: {e}")
        return {
            "success": False,
            "error": f"運用レポート生成中にエラーが発生しました: {str(e)}",
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }


def create_performance_summary(portfolio_performance: Dict[str, Any], benchmark_performance: Dict[str, Any], 
                             ticker_performance: Dict[str, Any], from_date: datetime, to_date: datetime) -> str:
    """詳細なパフォーマンスサマリーを作成"""
    summary_parts = []
    
    # 期間情報
    summary_parts.append(f"【分析期間】{from_date.strftime('%Y-%m-%d')} - {to_date.strftime('%Y-%m-%d')} ({(to_date - from_date).days}日間)")
    
    # ポートフォリオパフォーマンス詳細
    if "performance_pct" in portfolio_performance:
        portfolio_return = portfolio_performance["performance_pct"]
        summary_parts.append(f"\n【ポートフォリオ全体】")
        summary_parts.append(f"総合リターン: {portfolio_return:+.2f}%")
        
        if "start_value" in portfolio_performance and "end_value" in portfolio_performance:
            start_val = portfolio_performance["start_value"]
            end_val = portfolio_performance["end_value"]
            summary_parts.append(f"期間開始時価値: ¥{start_val:,.0f}")
            summary_parts.append(f"期間終了時価値: ¥{end_val:,.0f}")
            summary_parts.append(f"価値変動額: ¥{end_val - start_val:+,.0f}")
    
    # ベンチマーク比較詳細
    if benchmark_performance:
        summary_parts.append(f"\n【ベンチマーク比較】")
        for benchmark, data in benchmark_performance.items():
            bench_return = data['performance_pct']
            vs_portfolio = portfolio_return - bench_return if "performance_pct" in portfolio_performance else 0
            summary_parts.append(f"- {data['name']}: {bench_return:+.2f}% (vs ポートフォリオ: {vs_portfolio:+.2f}%)")
    
    # 個別銘柄パフォーマンス詳細
    if ticker_performance:
        summary_parts.append(f"\n【個別銘柄パフォーマンス】")
        summary_parts.append(f"総銘柄数: {len(ticker_performance)}銘柄")
        
        # パフォーマンス順にソート
        sorted_tickers = sorted(ticker_performance.items(), key=lambda x: x[1]['performance_pct'], reverse=True)
        
        # 勝率計算
        positive_count = sum(1 for _, data in sorted_tickers if data['performance_pct'] > 0)
        win_rate = (positive_count / len(sorted_tickers)) * 100
        summary_parts.append(f"勝率: {win_rate:.1f}% ({positive_count}/{len(sorted_tickers)}銘柄がプラス)")
        
        # 全銘柄リスト
        summary_parts.append(f"\n【全銘柄リターン一覧】")
        for ticker, data in sorted_tickers:
            summary_parts.append(f"- {ticker} ({data['company_name']}): {data['performance_pct']:+.2f}%")
        
        # 上位5銘柄詳細
        top_5 = sorted_tickers[:5]
        summary_parts.append(f"\n【上位5銘柄詳細】")
        for i, (ticker, data) in enumerate(top_5, 1):
            summary_parts.append(f"{i}位. {ticker} ({data['company_name']})")
            summary_parts.append(f"   リターン: {data['performance_pct']:+.2f}%")
            summary_parts.append(f"   開始価格: {data['start_price']:.2f} {data.get('currency', 'USD')}")
            summary_parts.append(f"   終了価格: {data['end_price']:.2f} {data.get('currency', 'USD')}")
        
        # 下位5銘柄詳細
        bottom_5 = sorted_tickers[-5:] if len(sorted_tickers) >= 5 else sorted_tickers[-len(sorted_tickers):]
        bottom_5.reverse()  # 下位から順に表示
        summary_parts.append(f"\n【下位5銘柄詳細】")
        for i, (ticker, data) in enumerate(bottom_5, 1):
            summary_parts.append(f"{i}位. {ticker} ({data['company_name']})")
            summary_parts.append(f"   リターン: {data['performance_pct']:+.2f}%")
            summary_parts.append(f"   開始価格: {data['start_price']:.2f} {data.get('currency', 'USD')}")
            summary_parts.append(f"   終了価格: {data['end_price']:.2f} {data.get('currency', 'USD')}")
        
        # 統計サマリー
        returns = [data['performance_pct'] for data in ticker_performance.values()]
        if returns:
            import statistics
            summary_parts.append(f"\n【銘柄リターン統計】")
            summary_parts.append(f"平均リターン: {statistics.mean(returns):+.2f}%")
            summary_parts.append(f"中央値リターン: {statistics.median(returns):+.2f}%")
            summary_parts.append(f"最大リターン: {max(returns):+.2f}%")
            summary_parts.append(f"最小リターン: {min(returns):+.2f}%")
            summary_parts.append(f"リターン標準偏差: {statistics.stdev(returns):.2f}%")
    
    return "\n".join(summary_parts)


def create_investment_report_prompt(performance_summary: str, from_date: datetime, to_date: datetime) -> str:
    """運用レポート生成用のプロンプトを作成"""
    
    prompt = f"""以下のポートフォリオパフォーマンスデータを基に、包括的な運用レポートを作成してください。

【分析対象データ】
{performance_summary}

【レポート構成と詳細要件】

## 1. 市場環境分析（800-1000字）
{from_date.strftime('%Y-%m-%d')}から{to_date.strftime('%Y-%m-%d')}の期間における包括的な市場環境分析：

### 経済・金融政策
- 主要中央銀行（FRB、ECB、日銀等）の金融政策動向
- インフレ率、雇用統計、GDP成長率等の主要経済指標
- 金利環境の変化とその市場への影響

### 政治・地政学リスク
- 主要国の政治情勢（米国、欧州、日本、中国等）
- 国際的な政治・外交問題
- 地政学的緊張とその市場インパクト

### 市場テーマとセンチメント
- 期間中の主要な市場テーマ（AI、エネルギー転換、インフレ等）
- 投資家センチメントの変化
- セクターローテーションの動向

## 2. ポートフォリオパフォーマンス詳細評価（600-800字）

### ベンチマーク比較分析
- MSCI ACWI ETF、NASDAQ100 ETF、Topix ETFとの詳細比較
- 各ベンチマークとの勝敗要因分析
- リスク調整後リターン（シャープレシオ的観点）の評価
- ポートフォリオの特徴（成長株vs価値株、地域配分等）がパフォーマンスに与えた影響

### パフォーマンス要因分析
- 市場環境変化がポートフォリオに与えた影響
- セクター配分効果、銘柄選択効果の分析
- 通貨要因の影響（該当する場合）

## 3. 個別銘柄詳細分析（800-1000字）

### 上位パフォーマンス銘柄（上位5銘柄）
各銘柄について以下を分析：
- 期間中の主要な企業発表・ニュース
- 決算内容、業績予想の変化
- セクター・業界動向との関係
- 株価上昇の具体的要因

### 下位パフォーマンス銘柄（下位5銘柄）
各銘柄について以下を分析：
- 期間中の懸念材料・ネガティブニュース
- 決算ミス、業績下方修正等の要因
- 業界逆風、競合環境の変化
- 株価下落の具体的要因

## 4. 今後の投資戦略への示唆（400-600字）

### 市場展望
- 現在の市場環境の持続性
- 注意すべきリスク要因
- 新たな投資機会の可能性

### ポートフォリオ運営への示唆
- 現在のポートフォリオの強み・弱み
- 市場環境変化への対応策（一般論として）
- リスク管理の観点からの留意点

【出力要件】
- 合計2000-3000字程度の詳細な分析
- 客観的で専門的な文体
- データに基づいた具体的な分析
- 投資推奨は避け、情報提供と分析に徹する
- 見出しや段落を適切に使用して読みやすく構成

注意事項: 
- 具体的な売買推奨は一切行わない
- 一般的な市場分析と情報提供に留める
- 免責事項として「過去の実績は将来を保証しない」旨を最後に記載"""
    
    return prompt


def display_investment_report_result(report_result: Dict[str, Any]):
    """運用レポート結果の表示"""
    try:
        if not report_result.get("success", False):
            st.error(f"レポート生成エラー: {report_result.get('error', 'Unknown error')}")
            return
        
        st.markdown("### 📋 AI運用レポート")
        
        # レポート概要
        col1, col2 = st.columns(2)
        with col1:
            st.metric("使用モデル", report_result.get("model_used", "N/A"))
        with col2:
            st.metric("生成時刻", report_result.get("timestamp", "N/A"))
        
        st.markdown("---")
        
        # AIレポート内容
        report_content = report_result.get("report", "レポート内容なし")
        st.markdown(report_content)
        
        st.markdown("---")
        
        # パフォーマンスサマリー
        with st.expander("📊 詳細パフォーマンスデータ"):
            performance_summary = report_result.get("performance_summary", "データなし")
            st.text(performance_summary)
        
        # 免責事項
        st.markdown("### ⚠️ 免責事項")
        st.warning("""
        **重要:** この運用レポートは情報提供のみを目的としており、投資推奨ではありません。
        - AI分析は一般的な情報と市場データに基づく参考情報です
        - 投資判断は必ずご自身の責任で行ってください
        - 過去のパフォーマンスは将来の結果を保証するものではありません
        - 専門家への相談を推奨します
        """)
    
    except Exception as e:
        st.error(f"レポート表示エラー: {str(e)}")


# 古い関数は削除されました（新しい運用レポート機能に置き換え）


def fetch_portfolio_news(tickers: List[str], days: int, max_per_ticker: int) -> List[Dict[str, Any]]:
    """ポートフォリオ銘柄のニュースを取得"""
    try:
        import requests
        import feedparser
        import time
    except ImportError as e:
        st.error(f"必要なライブラリがインストールされていません: {e}")
        st.info("以下のコマンドでインストールしてください:")
        st.code("pip install feedparser requests")
        return []
    
    all_articles = []
    
    for ticker in tickers:
        try:
            # 株価関連キーワードを含む高品質なクエリを構築
            stock_keywords = ["stock", "shares", "earnings", "revenue", "profit", "quarterly", "financial", "results", "guidance", "outlook", "analyst", "rating", "price target", "upgrade", "downgrade"]
            
            # 複数のクエリパターンを試行
            queries = [
                f'"{ticker}" AND (earnings OR revenue OR "quarterly results")',
                f'"{ticker}" AND (stock OR shares OR "price target")',
                f'"{ticker}" AND (analyst OR rating OR upgrade OR downgrade)'
            ]
            
            import urllib.parse
            # 最初のクエリを使用（最も関連性が高い）
            encoded_query = urllib.parse.quote(f'{queries[0]} when:{days}d')
            # 英語ニュースソース優先（US market focus）
            url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en&gl=US&ceid=US:en"
            
            # RSS取得
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                feed = feedparser.parse(response.content)
                
                articles_count = 0
                for entry in feed.entries:
                    if articles_count >= max_per_ticker:
                        break
                    
                    # 記事情報を抽出
                    title = getattr(entry, 'title', 'No Title')
                    summary = getattr(entry, 'summary', '')
                    
                    # 株価関連性をチェック
                    if not is_stock_relevant(title, summary, ticker):
                        continue
                    
                    # 信頼できる金融ニュースソースかチェック
                    source = getattr(entry.source, 'title', 'Unknown') if hasattr(entry, 'source') else 'Google News'
                    if not is_credible_financial_source(source):
                        continue
                    
                    article = {
                        'ticker': ticker,
                        'title': title,
                        'link': getattr(entry, 'link', ''),
                        'published': getattr(entry, 'published', ''),
                        'summary': summary,
                        'source': source
                    }
                    
                    # 公開日時を解析
                    if article['published']:
                        try:
                            from email.utils import parsedate_to_datetime
                            article['published_dt'] = parsedate_to_datetime(article['published'])
                        except:
                            article['published_dt'] = datetime.now(timezone.utc)
                    else:
                        article['published_dt'] = datetime.now(timezone.utc)
                    
                    all_articles.append(article)
                    articles_count += 1
            
            # レート制限（1秒間隔）
            time.sleep(1)
            
        except Exception as e:
            st.warning(f"{ticker}のニュース取得中にエラーが発生しました: {str(e)}")
            continue
    
    return all_articles


def is_stock_relevant(title: str, summary: str, ticker: str) -> bool:
    """記事が株価に関連しているかチェック"""
    
    # 株価関連キーワード
    stock_keywords = [
        # 業績・決算関連
        'earnings', 'revenue', 'profit', 'sales', 'quarterly', 'annual', 'results', 
        'guidance', 'outlook', 'forecast', 'beat', 'miss', 'consensus',
        
        # 株価・評価関連
        'stock', 'shares', 'price', 'target', 'rating', 'upgrade', 'downgrade',
        'buy', 'sell', 'hold', 'analyst', 'analysts', 'recommendation',
        
        # 企業活動関連
        'merger', 'acquisition', 'partnership', 'deal', 'agreement', 'contract',
        'product', 'launch', 'announcement', 'ceo', 'executive', 'management',
        
        # 財務関連
        'dividend', 'split', 'buyback', 'debt', 'cash', 'investment',
        'valuation', 'market cap', 'financial', 'balance sheet',
        
        # 規制・業界関連
        'fda', 'approval', 'regulation', 'lawsuit', 'settlement', 'compliance'
    ]
    
    # 除外キーワード（株価に関係ないニュース）
    exclude_keywords = [
        'sports', 'entertainment', 'celebrity', 'movie', 'music', 'fashion',
        'weather', 'politics', 'election', 'social media', 'personal life',
        'charity', 'donation', 'award', 'festival', 'event', 'party'
    ]
    
    content = (title + ' ' + summary).lower()
    
    # 除外キーワードがある場合は無関係
    for keyword in exclude_keywords:
        if keyword in content:
            return False
    
    # ティッカーシンボルが含まれているかチェック
    ticker_mentioned = ticker.lower() in content or ticker.replace('.T', '').lower() in content
    
    # 株価関連キーワードがあるかチェック
    stock_related = any(keyword in content for keyword in stock_keywords)
    
    # ティッカーと株価関連キーワード両方が必要
    return ticker_mentioned and stock_related


def is_credible_financial_source(source: str) -> bool:
    """信頼できる金融ニュースソースかチェック"""
    
    # 信頼できる金融・ビジネスニュースソース
    credible_sources = [
        # Tier 1: 最高品質の金融ニュース
        'Reuters', 'Bloomberg', 'Financial Times', 'Wall Street Journal', 'WSJ',
        'Associated Press', 'AP News',
        
        # Tier 2: 主要ビジネスメディア
        'CNBC', 'MarketWatch', 'Yahoo Finance', 'Barron\'s', 'Fortune', 'Forbes',
        'Business Insider', 'Benzinga', 'TheStreet', 'Seeking Alpha',
        
        # Tier 3: 専門金融メディア
        'Morningstar', 'Zacks', 'InvestorPlace', 'Motley Fool', 'TipRanks',
        'Finviz', 'GuruFocus', 'Simply Wall St',
        
        # Tier 4: 一般メディアのビジネス部門
        'CNN Business', 'BBC Business', 'CNBC', 'Fox Business',
        'The Guardian Business', 'New York Times Business'
    ]
    
    if not source or source == 'Unknown':
        return False
    
    source_lower = source.lower()
    
    # 完全一致または部分一致チェック
    for credible in credible_sources:
        if credible.lower() in source_lower or source_lower in credible.lower():
            return True
    
    # ドメインベースの追加チェック
    financial_domains = [
        'reuters.com', 'bloomberg.com', 'ft.com', 'wsj.com', 'cnbc.com',
        'marketwatch.com', 'finance.yahoo.com', 'barrons.com',
        'fortune.com', 'forbes.com', 'businessinsider.com'
    ]
    
    for domain in financial_domains:
        if domain in source_lower:
            return True
    
    return False


def get_sample_news_data(tickers: List[str]) -> List[Dict[str, Any]]:
    """サンプルニュースデータを生成"""
    
    sample_articles = []
    
    sample_news_templates = [
        {
            'title_template': '{ticker} Reports Q3 Earnings Beat, Revenue Up 15% YoY',
            'summary': 'The company exceeded analyst expectations with strong quarterly results, driving investor optimism for future growth prospects.',
            'source': 'Reuters'
        },
        {
            'title_template': '{ticker} Stock Surges on New Product Launch Announcement',
            'summary': 'Shares jumped in after-hours trading following the unveiling of innovative products expected to capture significant market share.',
            'source': 'Bloomberg'
        },
        {
            'title_template': 'Analysts Raise {ticker} Price Target, Maintain Buy Rating',
            'summary': 'Multiple investment firms upgraded their outlook citing improved fundamentals and strong competitive positioning in key markets.',
            'source': 'CNBC'
        },
        {
            'title_template': '{ticker} Announces Strategic Partnership with Industry Leader',
            'summary': 'The collaboration is expected to accelerate growth initiatives and enhance the company\'s technological capabilities.',
            'source': 'MarketWatch'
        },
        {
            'title_template': '{ticker} CEO Provides Upbeat Guidance for Next Quarter',
            'summary': 'Management expressed confidence in maintaining growth momentum, citing strong demand trends and operational efficiency improvements.',
            'source': 'Yahoo Finance'
        },
        {
            'title_template': '{ticker} Dividend Increase Signals Management Confidence',
            'summary': 'The board approved a 12% dividend hike, reflecting strong cash generation and optimistic outlook for sustainable returns.',
            'source': 'Barron\'s'
        },
        {
            'title_template': 'FDA Approval Boosts {ticker} Stock in Premarket Trading',
            'summary': 'Regulatory clearance for the company\'s key product removes a major overhang and opens new revenue opportunities.',
            'source': 'Seeking Alpha'
        },
        {
            'title_template': '{ticker} Stock Upgraded to Overweight by Goldman Sachs',
            'summary': 'The investment bank cited improving market conditions and the company\'s strong execution on strategic initiatives.',
            'source': 'Financial Times'
        }
    ]
    
    for i, ticker in enumerate(tickers):
        for j, template in enumerate(sample_news_templates[:3]):  # 各銘柄3記事まで
            article = {
                'ticker': ticker,
                'title': template['title_template'].format(ticker=ticker),
                'summary': template['summary'],
                'source': template['source'],
                'link': f'https://example.com/news/{ticker.lower()}-{j+1}',
                'published': f'{(datetime.now() - timedelta(hours=j*6+i)).strftime("%a, %d %b %Y %H:%M:%S")} GMT',
                'published_dt': datetime.now(timezone.utc) - timedelta(hours=j*6+i)
            }
            sample_articles.append(article)
    
    return sample_articles


def render_news_articles(articles: List[Dict[str, Any]], pnl_df: pd.DataFrame):
    """ニュース記事を表示"""
    if not articles:
        st.warning("ニュース記事が見つかりませんでした。")
        return
    
    st.subheader(f"📰 ニュース一覧 ({len(articles)}件)")
    
    # 銘柄別にグループ化
    ticker_articles = {}
    for article in articles:
        ticker = article['ticker']
        if ticker not in ticker_articles:
            ticker_articles[ticker] = []
        ticker_articles[ticker].append(article)
    
    # 重要度順でソート（時間順）
    for ticker in ticker_articles:
        ticker_articles[ticker].sort(key=lambda x: x.get('published_dt', datetime.now(timezone.utc)), reverse=True)
    
    # 表示オプション
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write("**表示オプション:**")
    with col2:
        show_all = st.checkbox("全銘柄を展開表示", value=False)
    
    # 銘柄別にニュースを表示
    for ticker, ticker_news in ticker_articles.items():
        # 銘柄の時価総額を取得（重要度の参考）
        ticker_value = 0
        if not pnl_df.empty:
            ticker_row = pnl_df[pnl_df['ticker'] == ticker]
            if not ticker_row.empty:
                ticker_value = ticker_row.iloc[0]['current_value_jpy']
        
        value_display = format_currency(ticker_value) if ticker_value > 0 else "データなし"
        
        if show_all:
            # 展開表示
            st.subheader(f"📈 {ticker} ({len(ticker_news)}件) - 評価額: {value_display}")
            render_ticker_news_expanded(ticker_news)
        else:
            # 折りたたみ表示
            with st.expander(f"📈 {ticker} ({len(ticker_news)}件) - 評価額: {value_display}"):
                render_ticker_news_expanded(ticker_news)


def render_ticker_news_expanded(articles: List[Dict[str, Any]]):
    """銘柄のニュースを展開表示"""
    for i, article in enumerate(articles):
        # 記事の年齢を計算
        age = "不明"
        if article.get('published_dt'):
            time_diff = datetime.now(timezone.utc) - article['published_dt']
            if time_diff.days > 0:
                age = f"{time_diff.days}日前"
            elif time_diff.seconds >= 3600:
                hours = time_diff.seconds // 3600
                age = f"{hours}時間前"
            elif time_diff.seconds >= 60:
                minutes = time_diff.seconds // 60
                age = f"{minutes}分前"
            else:
                age = "たった今"
        
        # 記事カード
        with st.container():
            col1, col2, col3 = st.columns([6, 2, 1])
            
            with col1:
                st.write(f"**{article['title']}**")
                if article.get('summary'):
                    st.write(article['summary'])
                if article.get('link'):
                    st.write(f"[記事を読む]({article['link']})")
            
            with col2:
                st.write(f"**出典:** {article.get('source', 'Unknown')}")
                st.write(f"**投稿:** {age}")
            
            with col3:
                # 重要度インジケーター（英語キーワード対応）
                title_lower = article['title'].lower()
                summary_lower = article.get('summary', '').lower()
                content = title_lower + ' ' + summary_lower
                
                importance = "📰"  # デフォルト
                
                # 最高重要度: 決算・業績・株価急変
                high_impact_keywords = ['earnings', 'revenue', 'beat', 'miss', 'guidance', 'upgrade', 'downgrade', 'surge', 'plunge', 'target', 'rating']
                if any(keyword in content for keyword in high_impact_keywords):
                    importance = "🔥"
                # 高重要度: 製品・提携・M&A・規制
                elif any(keyword in content for keyword in ['launch', 'partnership', 'merger', 'acquisition', 'fda', 'approval', 'deal', 'agreement', 'ceo']):
                    importance = "⭐"
                # 中重要度: アナリスト・投資家関連
                elif any(keyword in content for keyword in ['analyst', 'buy', 'sell', 'hold', 'recommendation', 'dividend', 'split']):
                    importance = "📊"
                
                st.write(f"**重要度**")
                st.write(importance)
            
            st.markdown("---")


def display_welcome_page():
    """ウェルカムページの表示"""
    st.markdown("""
    ## 👋 ようこそ！
    
    このアプリケーションは株式ポートフォリオの管理と分析を行うためのツールです。
    
    ### 🚀 機能
    - **CSVインポート**: ポートフォリオデータの簡単アップロード
    - **リアルタイム株価**: Yahoo Financeからの最新データ取得
    - **損益計算**: 多通貨対応の精密な損益計算
    - **リスク分析**: VaR、CVaR、ボラティリティ等の計算
    - **可視化**: インタラクティブなチャートとグラフ
    
    ### 📋 使用方法
    1. 左のサイドバーからCSVファイルをアップロード
    2. データが自動的に分析され、ダッシュボードが表示されます
    3. 各タブで詳細な分析結果を確認できます
    
    ### 📁 CSVファイル形式
    ```
    Ticker,Shares,AvgCostJPY
    AAPL,100,15000
    MSFT,50,25000
    7203.T,1000,800
    ```
    
    左のサイドバーからサンプルファイルをダウンロードして試してみてください！
    """)
    
    # サンプルファイルダウンロード
    sample_data = {
        'Ticker': ['AAPL', 'MSFT', '7203.T', 'ASML', 'TSLA'],
        'Shares': [100, 50, 1000, 20, 30],
        'AvgCostJPY': [15000, 25000, 800, 60000, 20000]
    }
    sample_df = pd.DataFrame(sample_data)
    sample_csv = sample_df.to_csv(index=False)
    
    st.download_button(
        label="📥 サンプルCSVファイルをダウンロード",
        data=sample_csv,
        file_name="sample_portfolio.csv",
        mime="text/csv"
    )


if __name__ == "__main__":
    main_dashboard()