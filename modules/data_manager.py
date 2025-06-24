"""
中央集約型データマネージャー
ポートフォリオ分析に必要な全データを一括取得・キャッシュ管理
"""

import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from typing import Dict, List, Optional, Tuple, Any
import logging
import time
import os
from datetime import datetime, timedelta
import json
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


class DataManager:
    """
    ポートフォリオ分析用の中央集約型データマネージャー
    全ての外部API呼び出しを管理し、データキャッシュを提供
    """
    
    def __init__(self, cache_dir: str = "data_cache"):
        """
        データマネージャーを初期化
        
        Args:
            cache_dir: キャッシュディレクトリのパス
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # メモリキャッシュ
        self.memory_cache = {}
        self.cache_timestamps = {}
        
        # データの有効期限（秒）
        self.cache_expiry = {
            'current_prices': 300,      # 5分
            'exchange_rates': 900,      # 15分
            'company_info': 3600,       # 1時間
            'historical_prices': 3600,  # 1時間
            'factor_data': 86400,       # 24時間
            'etf_benchmarks': 3600      # 1時間
        }
        
        logger.info("データマネージャー初期化完了")
    
    
    def load_portfolio_data(self, portfolio_df: pd.DataFrame) -> Dict[str, Any]:
        """
        ポートフォリオに必要な全データを一括取得
        
        Args:
            portfolio_df: ポートフォリオデータフレーム
            
        Returns:
            Dict: 全ての必要なデータを含む辞書
        """
        logger.info(f"ポートフォリオデータ一括取得開始: {len(portfolio_df)}銘柄")
        
        # ティッカーリストを取得
        tickers = portfolio_df['Ticker'].tolist()
        
        # プログレスバー設定
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
            show_progress = True
        except:
            show_progress = False
        
        data_bundle = {}
        
        # 1. 現在株価取得 (20%)
        if show_progress:
            progress_bar.progress(0.1)
            status_text.text("現在株価を取得中...")
        
        data_bundle['current_prices'] = self.get_current_prices(tickers)
        
        # 2. 為替レート取得 (30%)
        if show_progress:
            progress_bar.progress(0.2)
            status_text.text("為替レートを取得中...")
        
        data_bundle['exchange_rates'] = self.get_exchange_rates()
        
        # 3. 企業情報取得 (50%)
        if show_progress:
            progress_bar.progress(0.3)
            status_text.text("企業情報を取得中...")
        
        data_bundle['company_info'] = self.get_company_info_batch(tickers)
        
        # 4. 過去5年分の株価データ取得 (70%)
        if show_progress:
            progress_bar.progress(0.5)
            status_text.text("過去5年分の株価データを取得中...")
        
        data_bundle['historical_prices'] = self.get_historical_prices_batch(tickers, period="5y")
        
        # 5. Fama-Frenchファクターデータ取得 (85%)
        if show_progress:
            progress_bar.progress(0.7)
            status_text.text("Fama-French 5年分ファクターデータを取得中...")
        
        # ポートフォリオアップロード時は必ず最新の過去5年データを取得
        data_bundle['factor_data'] = self.get_factor_data(force_refresh=True)
        
        # 6. ETFベンチマークデータ取得 (95%)
        if show_progress:
            progress_bar.progress(0.85)
            status_text.text("ベンチマークデータを取得中...")
        
        data_bundle['etf_benchmarks'] = self.get_etf_benchmark_data()
        
        # 7. 通貨マッピング生成 (100%)
        if show_progress:
            progress_bar.progress(0.95)
            status_text.text("データ処理を完了中...")
        
        data_bundle['currency_mapping'] = self.create_currency_mapping(tickers)
        
        # データ品質チェック
        data_bundle['data_quality'] = self.assess_data_quality(data_bundle, tickers)
        
        # キャッシュに保存
        self.save_data_bundle(data_bundle, tickers)
        
        if show_progress:
            progress_bar.progress(1.0)
            status_text.text(f"データ取得完了: {len(tickers)}銘柄")
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
        
        logger.info("ポートフォリオデータ一括取得完了")
        return data_bundle
    
    
    def get_current_prices(self, tickers: List[str]) -> Dict[str, float]:
        """
        現在株価を一括取得
        """
        cache_key = f"current_prices_{hash(tuple(sorted(tickers)))}"
        
        if self.is_cache_valid(cache_key, 'current_prices'):
            logger.info("現在株価：キャッシュから取得")
            return self.memory_cache[cache_key]
        
        logger.info(f"現在株価取得開始: {len(tickers)}銘柄")
        
        prices = {}
        
        # バッチ取得を試行
        try:
            # 10銘柄ずつバッチ処理
            batch_size = 10
            for i in range(0, len(tickers), batch_size):
                batch = tickers[i:i + batch_size]
                logger.info(f"価格取得バッチ {i//batch_size + 1}: {batch}")
                
                try:
                    # yfinanceのバッチダウンロード
                    batch_data = yf.download(batch, period="2d", interval="1d", 
                                           group_by='ticker', auto_adjust=True, 
                                           prepost=True, threads=True)
                    
                    if batch_data.empty:
                        logger.warning(f"バッチ {i//batch_size + 1} でデータが取得できませんでした")
                        continue
                    
                    # 個別銘柄の価格を抽出
                    for ticker in batch:
                        try:
                            if len(batch) == 1:
                                # 単一銘柄の場合
                                ticker_data = batch_data
                            else:
                                # 複数銘柄の場合
                                ticker_data = batch_data[ticker] if ticker in batch_data.columns.levels[0] else None
                            
                            if ticker_data is not None and not ticker_data.empty:
                                # 最新の終値を取得
                                latest_price = ticker_data['Close'].dropna().iloc[-1]
                                prices[ticker] = float(latest_price)
                                logger.debug(f"価格取得成功: {ticker} = {latest_price}")
                            else:
                                logger.warning(f"価格データなし: {ticker}")
                                prices[ticker] = 0.0
                                
                        except Exception as e:
                            logger.error(f"個別価格取得エラー {ticker}: {str(e)}")
                            prices[ticker] = 0.0
                    
                    # レート制限対策
                    if i + batch_size < len(tickers):
                        time.sleep(0.5)
                        
                except Exception as e:
                    logger.error(f"バッチ取得エラー {i//batch_size + 1}: {str(e)}")
                    # 個別取得にフォールバック
                    for ticker in batch:
                        try:
                            stock = yf.Ticker(ticker)
                            hist = stock.history(period="2d")
                            if not hist.empty:
                                prices[ticker] = float(hist['Close'].iloc[-1])
                            else:
                                prices[ticker] = 0.0
                        except:
                            prices[ticker] = 0.0
                        time.sleep(0.2)
        
        except Exception as e:
            logger.error(f"現在株価取得エラー: {str(e)}")
        
        # キャッシュに保存
        self.memory_cache[cache_key] = prices
        self.cache_timestamps[cache_key] = time.time()
        
        success_count = len([p for p in prices.values() if p > 0])
        logger.info(f"現在株価取得完了: {success_count}/{len(tickers)}銘柄成功")
        
        return prices
    
    
    def get_exchange_rates(self) -> Dict[str, float]:
        """
        為替レートを取得
        """
        cache_key = "exchange_rates"
        
        if self.is_cache_valid(cache_key, 'exchange_rates'):
            logger.info("為替レート：キャッシュから取得")
            return self.memory_cache[cache_key]
        
        logger.info("為替レート取得開始")
        
        currency_pairs = ['USDJPY=X', 'EURJPY=X', 'GBPJPY=X', 'AUDJPY=X', 'CADJPY=X', 'CHFJPY=X']
        rates = {}
        
        try:
            # バッチで為替レートを取得
            fx_data = yf.download(currency_pairs, period="5d", interval="1d", 
                                group_by='ticker', auto_adjust=True, threads=True)
            
            if not fx_data.empty:
                for pair in currency_pairs:
                    try:
                        if len(currency_pairs) == 1:
                            pair_data = fx_data
                        else:
                            pair_data = fx_data[pair] if pair in fx_data.columns.levels[0] else None
                        
                        if pair_data is not None and not pair_data.empty:
                            latest_rate = pair_data['Close'].dropna().iloc[-1]
                            rates[pair] = float(latest_rate)
                        else:
                            logger.warning(f"為替データなし: {pair}")
                            
                    except Exception as e:
                        logger.error(f"為替レート取得エラー {pair}: {str(e)}")
            
        except Exception as e:
            logger.error(f"為替レートバッチ取得エラー: {str(e)}")
        
        # フォールバック値を設定
        fallback_rates = {
            'USDJPY=X': 150.0,
            'EURJPY=X': 160.0,
            'GBPJPY=X': 180.0,
            'AUDJPY=X': 100.0,
            'CADJPY=X': 110.0,
            'CHFJPY=X': 165.0
        }
        
        for pair in currency_pairs:
            if pair not in rates:
                rates[pair] = fallback_rates[pair]
                logger.warning(f"フォールバック為替レート使用: {pair} = {fallback_rates[pair]}")
        
        # キャッシュに保存
        self.memory_cache[cache_key] = rates
        self.cache_timestamps[cache_key] = time.time()
        
        logger.info(f"為替レート取得完了: {len(rates)}通貨ペア")
        return rates
    
    
    def get_company_info_batch(self, tickers: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        企業情報を一括取得
        """
        cache_key = f"company_info_{hash(tuple(sorted(tickers)))}"
        
        if self.is_cache_valid(cache_key, 'company_info'):
            logger.info("企業情報：キャッシュから取得")
            return self.memory_cache[cache_key]
        
        logger.info(f"企業情報一括取得開始: {len(tickers)}銘柄")
        
        # 既存の country_fetcher モジュールを使用
        from modules.country_fetcher import get_multiple_ticker_complete_info
        
        # 為替レートも渡す
        exchange_rates = self.get_exchange_rates()
        
        company_info = get_multiple_ticker_complete_info(tickers, exchange_rates)
        
        # キャッシュに保存
        self.memory_cache[cache_key] = company_info
        self.cache_timestamps[cache_key] = time.time()
        
        logger.info(f"企業情報一括取得完了: {len(company_info)}銘柄")
        return company_info
    
    
    def get_historical_prices_batch(self, tickers: List[str], period: str = "5y") -> Dict[str, pd.DataFrame]:
        """
        過去の株価データを一括取得（最大5年分）
        """
        cache_key = f"historical_prices_{period}_{hash(tuple(sorted(tickers)))}"
        
        if self.is_cache_valid(cache_key, 'historical_prices'):
            logger.info(f"過去株価データ（{period}）：キャッシュから取得")
            return self.memory_cache[cache_key]
        
        logger.info(f"過去株価データ取得開始: {len(tickers)}銘柄, 期間: {period}")
        
        historical_data = {}
        
        # バッチサイズを制限（メモリとAPI制限を考慮）
        batch_size = 20
        
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            logger.info(f"過去データ取得バッチ {i//batch_size + 1}: {len(batch)}銘柄")
            
            try:
                # yfinanceでバッチダウンロード
                batch_data = yf.download(batch, period=period, interval="1d", 
                                       group_by='ticker', auto_adjust=True, 
                                       prepost=False, threads=True)
                
                if batch_data.empty:
                    logger.warning(f"過去データバッチ {i//batch_size + 1} で데이터なし")
                    continue
                
                # 個別銘柄のデータを抽出
                for ticker in batch:
                    try:
                        if len(batch) == 1:
                            ticker_data = batch_data
                        else:
                            ticker_data = batch_data[ticker] if ticker in batch_data.columns.levels[0] else pd.DataFrame()
                        
                        if not ticker_data.empty:
                            # データクリーニング
                            ticker_data = ticker_data.dropna()
                            if len(ticker_data) > 0:
                                historical_data[ticker] = ticker_data
                                logger.debug(f"過去データ取得成功: {ticker} ({len(ticker_data)}日分)")
                            else:
                                logger.warning(f"過去データが空: {ticker}")
                                historical_data[ticker] = pd.DataFrame()
                        else:
                            logger.warning(f"過去データなし: {ticker}")
                            historical_data[ticker] = pd.DataFrame()
                            
                    except Exception as e:
                        logger.error(f"個別過去データ取得エラー {ticker}: {str(e)}")
                        historical_data[ticker] = pd.DataFrame()
                
                # レート制限対策
                if i + batch_size < len(tickers):
                    time.sleep(1.0)
                    
            except Exception as e:
                logger.error(f"過去データバッチ取得エラー {i//batch_size + 1}: {str(e)}")
                # 空のデータフレームで埋める
                for ticker in batch:
                    historical_data[ticker] = pd.DataFrame()
        
        # キャッシュに保存
        self.memory_cache[cache_key] = historical_data
        self.cache_timestamps[cache_key] = time.time()
        
        success_count = len([df for df in historical_data.values() if not df.empty])
        logger.info(f"過去株価データ取得完了: {success_count}/{len(tickers)}銘柄成功")
        
        return historical_data
    
    
    def get_factor_data(self, force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Fama-Frenchファクターデータを取得（過去5年分）
        
        Args:
            force_refresh: 強制的に新しいデータを取得する
        """
        cache_key = "factor_data_5y"
        
        if not force_refresh and self.is_cache_valid(cache_key, 'factor_data'):
            logger.info("📦 ファクターデータ：キャッシュから取得")
            return self.memory_cache[cache_key]
        
        logger.info("🎯 Fama-French 5年分ファクターデータ取得開始")
        
        # 堅牢なdirect downloadを使用
        from modules.factor_analysis import download_fama_french_direct, get_fama_french_factors
        
        try:
            # 過去5年分の期間を計算（少し余裕を持たせる）
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=5*365 + 30)).strftime('%Y-%m-%d')  # 30日余裕
            
            logger.info(f"📅 取得期間: {start_date} ～ {end_date}")
            
            # 1. まず堅牢な直接ダウンロードを試行
            try:
                logger.info("🎯 Kenneth French公式サイトから直接ダウンロード試行...")
                factor_dataframe = download_fama_french_direct(start_date, end_date)
                
                if isinstance(factor_dataframe, pd.DataFrame) and not factor_dataframe.empty and len(factor_dataframe) > 500:
                    factor_data = {'FF5_Factors': factor_dataframe}
                    
                    # Streamlit通知
                    try:
                        import streamlit as st
                        st.success(f"✅ **Kenneth French公式データ取得成功！**\n\n"
                                f"- 📊 データ期間: {factor_dataframe.index.min().strftime('%Y-%m-%d')} ～ {factor_dataframe.index.max().strftime('%Y-%m-%d')}\n"
                                f"- 📈 データ数: {len(factor_dataframe):,}営業日分\n"
                                f"- 🔍 ファクター: {', '.join(factor_dataframe.columns)}\n"
                                f"- 🎯 実際の学術研究用データを使用中")
                    except:
                        pass
                    
                    logger.info(f"✅ 直接ダウンロード成功: {len(factor_dataframe)}日分")
                    
                    # キャッシュに保存
                    self.memory_cache[cache_key] = factor_data
                    self.cache_timestamps[cache_key] = time.time()
                    
                    # ファイルキャッシュにも保存
                    self.save_factor_data_to_file(factor_data, start_date, end_date)
                    
                    return factor_data
                else:
                    raise ValueError("直接ダウンロードで十分なデータが取得できませんでした")
                    
            except Exception as e:
                logger.warning(f"⚠️ 直接ダウンロード失敗: {str(e)}")
                logger.info("📡 pandas-datareaderでの取得を試行...")
            
            # 2. pandas-datareaderでの取得を試行
            try:
                factor_dataframe = get_fama_french_factors(start_date, end_date)
                
                if isinstance(factor_dataframe, pd.DataFrame) and not factor_dataframe.empty and len(factor_dataframe) > 500:
                    factor_data = {'FF5_Factors': factor_dataframe}
                    
                    # Streamlit通知
                    try:
                        import streamlit as st
                        st.info(f"✅ **Fama-Frenchファクターデータ取得成功**\n\n"
                               f"- 📊 データ期間: {factor_dataframe.index.min().strftime('%Y-%m-%d')} ～ {factor_dataframe.index.max().strftime('%Y-%m-%d')}\n"
                               f"- 📈 データ数: {len(factor_dataframe):,}営業日分\n"
                               f"- 📡 pandas-datareader経由で取得")
                    except:
                        pass
                    
                    logger.info(f"✅ pandas-datareader成功: {len(factor_dataframe)}日分")
                    
                    # キャッシュに保存
                    self.memory_cache[cache_key] = factor_data
                    self.cache_timestamps[cache_key] = time.time()
                    
                    # ファイルキャッシュにも保存
                    self.save_factor_data_to_file(factor_data, start_date, end_date)
                    
                    return factor_data
                else:
                    raise ValueError("pandas-datareaderで十分なデータが取得できませんでした")
                    
            except Exception as e:
                logger.warning(f"⚠️ pandas-datareader失敗: {str(e)}")
                logger.info("📁 保存済みファクターデータの読み込みを試行...")
            
            # 3. 保存済みデータの読み込みを試行
            try:
                cached_factor_data = self.load_factor_data_from_file()
                if cached_factor_data:
                    logger.info("✅ 保存済みファクターデータを読み込み")
                    
                    try:
                        import streamlit as st
                        st.warning("📁 **保存済みファクターデータを使用中**\n\n"
                                 "ネットワーク接続の問題により、最新データ取得に失敗しました。\n"
                                 "過去に保存されたファクターデータを使用しています。")
                    except:
                        pass
                    
                    self.memory_cache[cache_key] = cached_factor_data
                    self.cache_timestamps[cache_key] = time.time()
                    
                    return cached_factor_data
            except Exception as e:
                logger.warning(f"⚠️ 保存済みデータ読み込み失敗: {str(e)}")
            
            # 4. 最終手段：統計的サンプルデータ
            logger.warning("🔄 実際のデータ取得に失敗、統計的サンプルデータを生成")
            sample_data = self.create_sample_factor_data(start_date, end_date)
            
            try:
                import streamlit as st
                st.error("⚠️ **ファクターデータ取得失敗**\n\n"
                        "実際のFama-Frenchデータの取得に失敗しました。\n"
                        "統計的に現実的なサンプルデータを使用します。\n\n"
                        "**対処方法:**\n"
                        "1. インターネット接続を確認\n"
                        "2. `pip install pandas-datareader`を実行\n"
                        "3. ページを再読み込み")
            except:
                pass
            
            # サンプルデータもキャッシュに保存
            self.memory_cache[cache_key] = sample_data
            self.cache_timestamps[cache_key] = time.time()
            
            return sample_data
            
        except Exception as e:
            logger.error(f"❌ ファクターデータ取得で予期しないエラー: {str(e)}")
            
            # 最終的なフォールバック
            try:
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
                minimal_data = self.create_sample_factor_data(start_date, end_date)
                
                self.memory_cache[cache_key] = minimal_data
                self.cache_timestamps[cache_key] = time.time()
                
                return minimal_data
            except:
                return {}
    
    
    def get_etf_benchmark_data(self) -> Dict[str, Dict[str, Optional[float]]]:
        """
        ETFベンチマークデータを取得
        """
        cache_key = "etf_benchmarks"
        
        if self.is_cache_valid(cache_key, 'etf_benchmarks'):
            logger.info("ETFベンチマークデータ：キャッシュから取得")
            return self.memory_cache[cache_key]
        
        logger.info("ETFベンチマークデータ取得開始")
        
        # 既存の pnl_calculator モジュールのベンチマークデータを使用
        from modules.pnl_calculator import get_etf_benchmark_data
        
        try:
            benchmark_data = get_etf_benchmark_data()
            
            # キャッシュに保存
            self.memory_cache[cache_key] = benchmark_data
            self.cache_timestamps[cache_key] = time.time()
            
            logger.info("ETFベンチマークデータ取得完了")
            return benchmark_data
            
        except Exception as e:
            logger.error(f"ETFベンチマークデータ取得エラー: {str(e)}")
            return {}
    
    
    def create_currency_mapping(self, tickers: List[str]) -> Dict[str, str]:
        """
        ティッカーシンボルから通貨マッピングを作成
        """
        currency_mapping = {}
        
        for ticker in tickers:
            if '.T' in ticker or '.JP' in ticker:
                currency_mapping[ticker] = 'JPY'
            elif '.L' in ticker:
                currency_mapping[ticker] = 'GBP'
            elif '.PA' in ticker or '.DE' in ticker or '.MI' in ticker or '.AS' in ticker:
                currency_mapping[ticker] = 'EUR'
            elif '.TO' in ticker or '.V' in ticker:
                currency_mapping[ticker] = 'CAD'
            elif '.AX' in ticker:
                currency_mapping[ticker] = 'AUD'
            elif '.SW' in ticker:
                currency_mapping[ticker] = 'CHF'
            else:
                currency_mapping[ticker] = 'USD'  # デフォルト
        
        logger.info(f"通貨マッピング作成完了: {len(currency_mapping)}銘柄")
        return currency_mapping
    
    
    def assess_data_quality(self, data_bundle: Dict[str, Any], tickers: List[str]) -> Dict[str, Any]:
        """
        データ品質を評価
        """
        quality_report = {
            'total_tickers': len(tickers),
            'price_success_rate': 0,
            'company_info_success_rate': 0,
            'historical_data_success_rate': 0,
            'missing_data': [],
            'data_freshness': datetime.now().isoformat()
        }
        
        try:
            # 現在価格の成功率
            price_success = len([p for p in data_bundle['current_prices'].values() if p > 0])
            quality_report['price_success_rate'] = price_success / len(tickers) * 100
            
            # 企業情報の成功率
            company_success = len([info for info in data_bundle['company_info'].values() 
                                 if info and (info.get('country') or info.get('sector'))])
            quality_report['company_info_success_rate'] = company_success / len(tickers) * 100
            
            # 過去データの成功率
            historical_success = len([df for df in data_bundle['historical_prices'].values() if not df.empty])
            quality_report['historical_data_success_rate'] = historical_success / len(tickers) * 100
            
            # 不足データの特定
            for ticker in tickers:
                issues = []
                if data_bundle['current_prices'].get(ticker, 0) <= 0:
                    issues.append('price')
                if not data_bundle['company_info'].get(ticker):
                    issues.append('company_info')
                if data_bundle['historical_prices'].get(ticker, pd.DataFrame()).empty:
                    issues.append('historical_data')
                
                if issues:
                    quality_report['missing_data'].append({
                        'ticker': ticker,
                        'missing': issues
                    })
            
        except Exception as e:
            logger.error(f"データ品質評価エラー: {str(e)}")
        
        logger.info(f"データ品質評価完了: 価格 {quality_report['price_success_rate']:.1f}%, "
                   f"企業情報 {quality_report['company_info_success_rate']:.1f}%, "
                   f"過去データ {quality_report['historical_data_success_rate']:.1f}%")
        
        return quality_report
    
    
    def is_cache_valid(self, cache_key: str, data_type: str) -> bool:
        """
        キャッシュの有効性をチェック
        """
        if cache_key not in self.memory_cache or cache_key not in self.cache_timestamps:
            return False
        
        elapsed = time.time() - self.cache_timestamps[cache_key]
        expiry = self.cache_expiry.get(data_type, 3600)
        
        return elapsed < expiry
    
    
    def save_data_bundle(self, data_bundle: Dict[str, Any], tickers: List[str]):
        """
        データバンドルをファイルに保存（PickleとCSV両方）
        """
        try:
            # ファイル名に日付とティッカーハッシュを含める
            ticker_hash = hash(tuple(sorted(tickers)))
            date_str = datetime.now().strftime('%Y%m%d')
            base_filename = f"data_bundle_{date_str}_{abs(ticker_hash)}"
            
            # Pickleファイルとして保存（完全データ）
            pickle_filepath = self.cache_dir / f"{base_filename}.pkl"
            with open(pickle_filepath, 'wb') as f:
                pickle.dump(data_bundle, f)
            
            logger.info(f"データバンドル保存完了: {pickle_filepath}")
            
            # CSV形式でも保存（オフライン利用・分析用）
            self.save_data_bundle_as_csv(data_bundle, tickers, date_str, abs(ticker_hash))
            
            # 古いキャッシュファイルをクリーンアップ（7日以上古い）
            self.cleanup_old_cache_files(days=7)
            
        except Exception as e:
            logger.error(f"データバンドル保存エラー: {str(e)}")
    
    
    def save_data_bundle_as_csv(self, data_bundle: Dict[str, Any], tickers: List[str], date_str: str, ticker_hash: int):
        """
        データバンドルをCSV形式で保存
        """
        try:
            csv_dir = self.cache_dir / "csv_exports" / f"{date_str}_{ticker_hash}"
            csv_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. 現在株価
            if 'current_prices' in data_bundle:
                prices_df = pd.DataFrame([
                    {'Ticker': ticker, 'CurrentPrice': price}
                    for ticker, price in data_bundle['current_prices'].items()
                ])
                prices_df.to_csv(csv_dir / "current_prices.csv", index=False)
            
            # 2. 為替レート
            if 'exchange_rates' in data_bundle:
                fx_df = pd.DataFrame([
                    {'CurrencyPair': pair, 'Rate': rate}
                    for pair, rate in data_bundle['exchange_rates'].items()
                ])
                fx_df.to_csv(csv_dir / "exchange_rates.csv", index=False)
            
            # 3. 企業情報
            if 'company_info' in data_bundle:
                company_data = []
                for ticker, info in data_bundle['company_info'].items():
                    if info:
                        row = {'Ticker': ticker}
                        row.update(info)
                        company_data.append(row)
                
                if company_data:
                    company_df = pd.DataFrame(company_data)
                    company_df.to_csv(csv_dir / "company_info.csv", index=False)
            
            # 4. 過去株価データ（各銘柄別ファイル）
            if 'historical_prices' in data_bundle:
                hist_dir = csv_dir / "historical_prices"
                hist_dir.mkdir(exist_ok=True)
                
                for ticker, df in data_bundle['historical_prices'].items():
                    if not df.empty:
                        # ティッカー名をファイル名に安全な形式に変換
                        safe_ticker = ticker.replace('.', '_').replace('/', '_')
                        df.to_csv(hist_dir / f"{safe_ticker}_historical.csv")
            
            # 5. ファクターデータ
            if 'factor_data' in data_bundle:
                factor_dir = csv_dir / "factor_data"
                factor_dir.mkdir(exist_ok=True)
                
                for factor_name, df in data_bundle['factor_data'].items():
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        df.to_csv(factor_dir / f"{factor_name}.csv")
            
            # 6. データ品質レポート
            if 'data_quality' in data_bundle:
                quality_df = pd.DataFrame([data_bundle['data_quality']])
                quality_df.to_csv(csv_dir / "data_quality_report.csv", index=False)
            
            # 7. メタデータファイル
            metadata = {
                'export_date': datetime.now().isoformat(),
                'tickers': tickers,
                'ticker_count': len(tickers),
                'data_types': list(data_bundle.keys())
            }
            
            with open(csv_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"CSV形式でもデータ保存完了: {csv_dir}")
            
        except Exception as e:
            logger.error(f"CSV保存エラー: {str(e)}")
    
    
    def load_data_bundle(self, tickers: List[str]) -> Optional[Dict[str, Any]]:
        """
        保存されたデータバンドルを読み込み
        """
        try:
            ticker_hash = hash(tuple(sorted(tickers)))
            date_str = datetime.now().strftime('%Y%m%d')
            filename = f"data_bundle_{date_str}_{abs(ticker_hash)}.pkl"
            filepath = self.cache_dir / filename
            
            if filepath.exists():
                with open(filepath, 'rb') as f:
                    data_bundle = pickle.load(f)
                logger.info(f"データバンドル読み込み完了: {filepath}")
                return data_bundle
            
        except Exception as e:
            logger.error(f"データバンドル読み込みエラー: {str(e)}")
        
        return None
    
    
    def cleanup_old_cache_files(self, days: int = 7):
        """
        古いキャッシュファイルを削除
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            for filepath in self.cache_dir.glob("data_bundle_*.pkl"):
                if filepath.stat().st_mtime < cutoff_date.timestamp():
                    filepath.unlink()
                    logger.debug(f"古いキャッシュファイル削除: {filepath}")
                    
        except Exception as e:
            logger.error(f"キャッシュクリーンアップエラー: {str(e)}")
    
    
    def get_data_freshness_info(self) -> Dict[str, str]:
        """
        データの新鮮度情報を取得
        """
        freshness_info = {}
        
        for data_type, cache_key in [
            ('current_prices', 'current_prices'),
            ('exchange_rates', 'exchange_rates'),
            ('company_info', 'company_info'),
            ('factor_data', 'factor_data')
        ]:
            if cache_key in self.cache_timestamps:
                timestamp = self.cache_timestamps[cache_key]
                age_seconds = time.time() - timestamp
                
                if age_seconds < 60:
                    freshness_info[data_type] = f"{int(age_seconds)}秒前"
                elif age_seconds < 3600:
                    freshness_info[data_type] = f"{int(age_seconds/60)}分前"
                else:
                    freshness_info[data_type] = f"{int(age_seconds/3600)}時間前"
            else:
                freshness_info[data_type] = "未取得"
        
        return freshness_info
    
    
    def save_factor_data_to_file(self, factor_data: Dict[str, pd.DataFrame], start_date: str, end_date: str):
        """
        ファクターデータをファイルに保存
        """
        try:
            factor_cache_dir = self.cache_dir / "factor_data"
            factor_cache_dir.mkdir(exist_ok=True)
            
            # 日付ベースのファイル名
            date_str = datetime.now().strftime('%Y%m%d')
            
            for factor_name, df in factor_data.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    # CSVとして保存
                    csv_path = factor_cache_dir / f"{factor_name}_{date_str}.csv"
                    df.to_csv(csv_path)
                    
                    # Pickleとしても保存（より完全なデータ保持）
                    pickle_path = factor_cache_dir / f"{factor_name}_{date_str}.pkl"
                    df.to_pickle(pickle_path)
                    
                    logger.info(f"📁 ファクターデータ保存: {csv_path}")
            
            # メタデータも保存
            metadata = {
                'save_date': datetime.now().isoformat(),
                'start_date': start_date,
                'end_date': end_date,
                'data_types': list(factor_data.keys()),
                'total_days': sum(len(df) for df in factor_data.values() if isinstance(df, pd.DataFrame))
            }
            
            metadata_path = factor_cache_dir / f"metadata_{date_str}.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"❌ ファクターデータ保存エラー: {str(e)}")
    
    
    def load_factor_data_from_file(self) -> Optional[Dict[str, pd.DataFrame]]:
        """
        保存済みファクターデータを読み込み
        """
        try:
            factor_cache_dir = self.cache_dir / "factor_data"
            if not factor_cache_dir.exists():
                return None
            
            # 最新のファイルを探す
            pickle_files = list(factor_cache_dir.glob("FF5_Factors_*.pkl"))
            if not pickle_files:
                return None
            
            # ファイル名から日付を抽出してソート
            latest_file = max(pickle_files, key=lambda x: x.stem.split('_')[-1])
            
            # ファイルの新しさをチェック（7日以内）
            file_age = time.time() - latest_file.stat().st_mtime
            if file_age > 7 * 24 * 3600:  # 7日以上古い
                logger.warning("📁 保存済みファクターデータが古すぎます（7日以上）")
                return None
            
            # データを読み込み
            factor_df = pd.read_pickle(latest_file)
            
            if isinstance(factor_df, pd.DataFrame) and not factor_df.empty:
                logger.info(f"📁 保存済みファクターデータ読み込み: {len(factor_df)}日分")
                return {'FF5_Factors': factor_df}
            
        except Exception as e:
            logger.error(f"❌ 保存済みファクターデータ読み込みエラー: {str(e)}")
        
        return None
    
    
    def create_sample_factor_data(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        サンプルFama-Frenchファクターデータを生成
        """
        try:
            import pandas as pd
            import numpy as np
            from datetime import datetime, timedelta
            
            # 日付範囲を生成
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            # 営業日のみ生成（土日を除く）
            date_range = pd.bdate_range(start=start_dt, end=end_dt)
            
            # サンプルファクターデータを生成
            np.random.seed(42)  # 再現性のため
            n_days = len(date_range)
            
            # Fama-French 5ファクター + Momentum
            factor_data = {
                'Mkt-RF': np.random.normal(0.0005, 0.012, n_days),  # 市場プレミアム
                'SMB': np.random.normal(0.0001, 0.008, n_days),     # 小型株プレミアム
                'HML': np.random.normal(0.0002, 0.009, n_days),     # バリュープレミアム
                'RMW': np.random.normal(0.0001, 0.007, n_days),     # 収益性プレミアム
                'CMA': np.random.normal(-0.0001, 0.006, n_days),    # 投資プレミアム
                'Mom': np.random.normal(0.0003, 0.011, n_days),     # モメンタム
                'RF': np.full(n_days, 0.00008)                      # リスクフリーレート（約2%年率）
            }
            
            # DataFrameに変換
            ff_factors = pd.DataFrame(factor_data, index=date_range)
            
            logger.info(f"サンプルFama-Frenchファクターデータ生成: {len(ff_factors)}日分")
            
            return {'FF5_Factors': ff_factors}
            
        except Exception as e:
            logger.error(f"サンプルファクターデータ生成エラー: {str(e)}")
            return {}


# グローバルデータマネージャーインスタンス
@st.cache_resource
def get_data_manager() -> DataManager:
    """
    グローバルデータマネージャーインスタンスを取得
    """
    return DataManager()