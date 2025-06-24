"""
ETF指標スクレイピングモジュール
etfdb.comから主要ETFのバリュエーション指標を自動取得
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import logging
from typing import Dict, List, Optional
import re

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ETFScraper:
    """ETFデータスクレイピングクラス"""
    
    def __init__(self):
        """初期化"""
        self.base_url = "https://etfdb.com/etf/"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        # 取得対象指標
        self.target_metrics = {
            'PER': 'Price-to-Earnings Ratio',
            'PBR': 'Price-to-Book Ratio', 
            'ROE': 'Return on Equity',
            'Dividend_Yield': 'Dividend Yield',
            'Beta': 'Beta'
        }
        
        # 対象ETF
        self.target_etfs = {
            'ACWI': 'iShares MSCI ACWI ETF',
            'QQQ': 'Invesco QQQ Trust',
            'SPY': 'SPDR S&P 500 ETF',
            'EWJ': 'iShares MSCI Japan ETF'
        }
    
    def get_etf_data(self, ticker: str, retry_count: int = 3) -> Dict[str, Optional[str]]:
        """
        単一ETFの指標取得
        
        Args:
            ticker: ETFティッカーシンボル
            retry_count: リトライ回数
            
        Returns:
            Dict[str, Optional[str]]: 指標データ辞書
        """
        url = f"{self.base_url}{ticker}/"
        result = {metric: None for metric in self.target_metrics.keys()}
        result['ticker'] = ticker
        
        for attempt in range(retry_count):
            try:
                logger.info(f"取得開始: {ticker} (試行 {attempt + 1}/{retry_count})")
                
                # リクエスト送信
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                
                # HTML解析
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # 各指標を検索・取得
                result.update(self._extract_metrics(soup, ticker))
                
                logger.info(f"取得完了: {ticker}")
                return result
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"リクエストエラー {ticker} (試行 {attempt + 1}): {str(e)}")
                if attempt < retry_count - 1:
                    time.sleep(5)  # リトライ前の待機
                else:
                    logger.error(f"最終的に失敗: {ticker}")
                    
            except Exception as e:
                logger.error(f"予期しないエラー {ticker}: {str(e)}")
                break
        
        return result
    
    def _extract_metrics(self, soup: BeautifulSoup, ticker: str) -> Dict[str, Optional[str]]:
        """
        HTMLから指標を抽出
        
        Args:
            soup: BeautifulSoupオブジェクト
            ticker: ETFティッカー
            
        Returns:
            Dict[str, Optional[str]]: 抽出された指標
        """
        metrics = {}
        
        try:
            # Phase 1: 基本的な指標取得（PER, PBR）
            metrics['PER'] = self._extract_pe_ratio(soup)
            metrics['PBR'] = self._extract_pb_ratio(soup)
            
            # Phase 2: 追加指標
            metrics['ROE'] = self._extract_roe(soup)
            metrics['Dividend_Yield'] = self._extract_dividend_yield(soup)
            metrics['Beta'] = self._extract_beta(soup)
            
        except Exception as e:
            logger.error(f"指標抽出エラー {ticker}: {str(e)}")
        
        return metrics
    
    def _extract_pe_ratio(self, soup: BeautifulSoup) -> Optional[str]:
        """PER（株価収益率）を抽出"""
        try:
            # 複数のセレクターを試行
            selectors = [
                'td[data-th="P/E Ratio"]',
                '.metric-value[data-metric="pe-ratio"]',
                'span:contains("P/E")',
                'td:contains("P/E")'
            ]
            
            for selector in selectors:
                elements = soup.select(selector)
                for element in elements:
                    text = element.get_text(strip=True)
                    value = self._extract_numeric_value(text)
                    if value and self._validate_pe_ratio(value):
                        logger.debug(f"PER取得成功: {value}")
                        return value
            
            # テキスト検索による取得
            pe_text = soup.find(text=re.compile(r'P/E.*Ratio', re.IGNORECASE))
            if pe_text:
                parent = pe_text.parent
                for sibling in parent.find_next_siblings():
                    value = self._extract_numeric_value(sibling.get_text())
                    if value and self._validate_pe_ratio(value):
                        return value
                        
        except Exception as e:
            logger.debug(f"PER抽出エラー: {str(e)}")
        
        return None
    
    def _extract_pb_ratio(self, soup: BeautifulSoup) -> Optional[str]:
        """PBR（株価純資産倍率）を抽出"""
        try:
            # 複数のセレクターを試行
            selectors = [
                'td[data-th="P/B Ratio"]',
                '.metric-value[data-metric="pb-ratio"]',
                'span:contains("P/B")',
                'td:contains("P/B")'
            ]
            
            for selector in selectors:
                elements = soup.select(selector)
                for element in elements:
                    text = element.get_text(strip=True)
                    value = self._extract_numeric_value(text)
                    if value and self._validate_pb_ratio(value):
                        logger.debug(f"PBR取得成功: {value}")
                        return value
            
            # テキスト検索による取得
            pb_text = soup.find(text=re.compile(r'P/B.*Ratio', re.IGNORECASE))
            if pb_text:
                parent = pb_text.parent
                for sibling in parent.find_next_siblings():
                    value = self._extract_numeric_value(sibling.get_text())
                    if value and self._validate_pb_ratio(value):
                        return value
                        
        except Exception as e:
            logger.debug(f"PBR抽出エラー: {str(e)}")
        
        return None
    
    def _extract_roe(self, soup: BeautifulSoup) -> Optional[str]:
        """ROE（自己資本利益率）を抽出"""
        try:
            # ROE検索パターン
            selectors = [
                'td[data-th="ROE"]',
                '.metric-value[data-metric="roe"]',
                'span:contains("ROE")',
                'td:contains("Return on Equity")'
            ]
            
            for selector in selectors:
                elements = soup.select(selector)
                for element in elements:
                    text = element.get_text(strip=True)
                    value = self._extract_percentage_value(text)
                    if value:
                        logger.debug(f"ROE取得成功: {value}")
                        return value
                        
        except Exception as e:
            logger.debug(f"ROE抽出エラー: {str(e)}")
        
        return None
    
    def _extract_dividend_yield(self, soup: BeautifulSoup) -> Optional[str]:
        """配当利回りを抽出"""
        try:
            # 配当利回り検索パターン
            selectors = [
                'td[data-th="Dividend Yield"]',
                '.metric-value[data-metric="dividend-yield"]',
                'span:contains("Dividend Yield")',
                'td:contains("Yield")'
            ]
            
            for selector in selectors:
                elements = soup.select(selector)
                for element in elements:
                    text = element.get_text(strip=True)
                    value = self._extract_percentage_value(text)
                    if value:
                        logger.debug(f"配当利回り取得成功: {value}")
                        return value
            
            # "Dividend Yield"テキスト検索
            div_text = soup.find(text=re.compile(r'Dividend.*Yield', re.IGNORECASE))
            if div_text:
                # 周辺要素から数値を検索
                parent = div_text.parent
                for sibling in parent.find_next_siblings():
                    value = self._extract_percentage_value(sibling.get_text())
                    if value:
                        return value
                        
        except Exception as e:
            logger.debug(f"配当利回り抽出エラー: {str(e)}")
        
        return None
    
    def _extract_beta(self, soup: BeautifulSoup) -> Optional[str]:
        """ベータを抽出"""
        try:
            # ベータ検索パターン
            selectors = [
                'td[data-th="Beta"]',
                '.metric-value[data-metric="beta"]',
                'span:contains("Beta")',
                'td:contains("Beta")'
            ]
            
            for selector in selectors:
                elements = soup.select(selector)
                for element in elements:
                    text = element.get_text(strip=True)
                    value = self._extract_numeric_value(text)
                    if value and self._validate_beta(value):
                        logger.debug(f"ベータ取得成功: {value}")
                        return value
                        
        except Exception as e:
            logger.debug(f"ベータ抽出エラー: {str(e)}")
        
        return None
    
    def _extract_numeric_value(self, text: str) -> Optional[str]:
        """テキストから数値を抽出"""
        if not text:
            return None
            
        # 数値パターンの検索
        pattern = r'(\d+\.?\d*)'
        match = re.search(pattern, text.replace(',', ''))
        if match:
            return match.group(1)
        return None
    
    def _extract_percentage_value(self, text: str) -> Optional[str]:
        """テキストからパーセンテージ値を抽出"""
        if not text:
            return None
            
        # パーセンテージパターンの検索
        pattern = r'(\d+\.?\d*)%?'
        match = re.search(pattern, text.replace(',', ''))
        if match:
            value = match.group(1)
            return f"{value}%" if '%' not in text else value
        return None
    
    def _validate_pe_ratio(self, value: str) -> bool:
        """PER値の妥当性チェック"""
        try:
            num = float(value)
            return 0 < num <= 100  # 妥当な範囲
        except:
            return False
    
    def _validate_pb_ratio(self, value: str) -> bool:
        """PBR値の妥当性チェック"""
        try:
            num = float(value)
            return 0 < num <= 10  # 妥当な範囲
        except:
            return False
    
    def _validate_beta(self, value: str) -> bool:
        """ベータ値の妥当性チェック"""
        try:
            num = float(value)
            return -2 <= num <= 3  # 妥当な範囲
        except:
            return False
    
    def scrape_all_etfs(self, tickers_list: List[str] = None) -> pd.DataFrame:
        """
        複数ETFの一括取得
        
        Args:
            tickers_list: 取得するETFティッカーのリスト
            
        Returns:
            pd.DataFrame: 取得結果のDataFrame
        """
        if tickers_list is None:
            tickers_list = list(self.target_etfs.keys())
        
        logger.info(f"ETF一括取得開始: {tickers_list}")
        
        all_results = []
        
        for ticker in tickers_list:
            try:
                # ETFデータ取得
                etf_data = self.get_etf_data(ticker)
                
                # データフレーム形式に変換
                for metric, value in etf_data.items():
                    if metric != 'ticker':
                        all_results.append({
                            'Ticker': ticker,
                            'Metric': metric,
                            'Value': value if value is not None else 'N/A'
                        })
                
                # リクエスト間隔（サイト負荷軽減）
                if ticker != tickers_list[-1]:  # 最後以外
                    logger.info("リクエスト間隔待機中...")
                    time.sleep(2.5)  # 2.5秒間隔
                    
            except Exception as e:
                logger.error(f"ETF取得エラー {ticker}: {str(e)}")
                # エラー時もN/Aで記録
                for metric in self.target_metrics.keys():
                    all_results.append({
                        'Ticker': ticker,
                        'Metric': metric,
                        'Value': 'N/A'
                    })
        
        df = pd.DataFrame(all_results)
        logger.info(f"ETF一括取得完了: {len(df)}レコード")
        
        return df
    
    def save_to_csv(self, data: pd.DataFrame, filename: str = None) -> str:
        """
        CSV保存機能
        
        Args:
            data: 保存するDataFrame
            filename: ファイル名（Noneの場合は自動生成）
            
        Returns:
            str: 保存されたファイルパス
        """
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"etf_data_{timestamp}.csv"
        
        try:
            # ファイルパスを適切に設定
            filepath = f"/mnt/c/Users/naoya/OneDrive/Documents/ClaudeTrial/PortfolioManagementApp/{filename}"
            data.to_csv(filepath, index=False, encoding='utf-8')
            logger.info(f"CSV保存完了: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"CSV保存エラー: {str(e)}")
            return ""
    
    def test_single_etf(self, ticker: str = 'ACWI') -> Dict:
        """
        単一ETFでのテスト機能
        
        Args:
            ticker: テスト対象ティッカー
            
        Returns:
            Dict: テスト結果
        """
        logger.info(f"テスト開始: {ticker}")
        
        result = self.get_etf_data(ticker)
        
        # 結果分析
        success_count = sum(1 for v in result.values() if v is not None and v != 'N/A')
        total_metrics = len(self.target_metrics)
        
        test_result = {
            'ticker': ticker,
            'success_rate': f"{success_count}/{total_metrics}",
            'success_percentage': (success_count / total_metrics) * 100,
            'data': result
        }
        
        logger.info(f"テスト完了: {ticker} - 成功率 {test_result['success_rate']}")
        
        return test_result


def main():
    """メイン実行関数"""
    # ETFスクレイパー初期化
    scraper = ETFScraper()
    
    # Phase 1: 単一ETFテスト
    print("=== Phase 1: 単一ETFテスト ===")
    test_result = scraper.test_single_etf('ACWI')
    print(f"テスト結果: {test_result['success_rate']} ({test_result['success_percentage']:.1f}%)")
    print("取得データ:")
    for k, v in test_result['data'].items():
        print(f"  {k}: {v}")
    
    print("\n" + "="*50)
    
    # Phase 2: 全ETF取得
    print("=== Phase 2: 全ETF一括取得 ===")
    etf_list = ['ACWI', 'QQQ', 'SPY', 'EWJ']
    results_df = scraper.scrape_all_etfs(etf_list)
    
    print("\n取得結果:")
    print(results_df)
    
    # Phase 3: CSV保存
    print("\n=== Phase 3: CSV保存 ===")
    csv_path = scraper.save_to_csv(results_df)
    if csv_path:
        print(f"保存完了: {csv_path}")
    else:
        print("保存失敗")


if __name__ == "__main__":
    main()