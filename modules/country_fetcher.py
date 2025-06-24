"""
企業の本社所在国取得モジュール
Yahoo Financeから企業の本社所在国情報を取得する機能
"""

import yfinance as yf
import streamlit as st
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def get_alternative_ticker_info(ticker: str) -> Optional[dict]:
    """
    yfinanceの標準取得が失敗した場合の代替情報取得
    
    Args:
        ticker: ティッカーシンボル
    
    Returns:
        dict: 代替情報、失敗時はNone
    """
    try:
        import requests
        import time
        
        # Yahoo Finance APIの代替エンドポイントを試行
        alternatives = [
            f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{ticker}?modules=summaryProfile,financialData,defaultKeyStatistics",
            f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}?modules=summaryProfile,financialData,defaultKeyStatistics"
        ]
        
        for url in alternatives:
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                response = requests.get(url, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    if 'quoteSummary' in data and data['quoteSummary']['result']:
                        result = data['quoteSummary']['result'][0]
                        
                        # データを yfinance 形式に変換
                        converted_info = {}
                        
                        # summaryProfile から基本情報
                        if 'summaryProfile' in result:
                            profile = result['summaryProfile']
                            # 複数のキーを試行
                            country_candidates = [
                                profile.get('country'),
                                profile.get('domicile'), 
                                profile.get('headquarters'),
                                profile.get('countryOfDomicile')
                            ]
                            sector_candidates = [
                                profile.get('sector'),
                                profile.get('industry'),
                                profile.get('sectorKey')
                            ]
                            
                            # 最初に見つかった有効な値を使用
                            converted_info['country'] = next((c for c in country_candidates if c and isinstance(c, str) and c.strip()), None)
                            converted_info['sector'] = next((s for s in sector_candidates if s and isinstance(s, str) and s.strip()), None)
                        
                        # financialData から財務情報
                        if 'financialData' in result:
                            financial = result['financialData']
                            converted_info['returnOnEquity'] = financial.get('returnOnEquity', {}).get('raw')
                            converted_info['returnOnAssets'] = financial.get('returnOnAssets', {}).get('raw')
                            converted_info['operatingMargins'] = financial.get('operatingMargins', {}).get('raw')
                            converted_info['profitMargins'] = financial.get('profitMargins', {}).get('raw')
                        
                        # defaultKeyStatistics からバリュエーション指標
                        if 'defaultKeyStatistics' in result:
                            stats = result['defaultKeyStatistics']
                            converted_info['forwardPE'] = stats.get('forwardPE', {}).get('raw')
                            converted_info['priceToBook'] = stats.get('priceToBook', {}).get('raw')
                            converted_info['marketCap'] = stats.get('marketCap', {}).get('raw')
                            converted_info['beta'] = stats.get('beta', {}).get('raw')
                        
                        if converted_info:
                            logger.info(f"代替API成功: {ticker}")
                            return converted_info
                        
            except Exception as e:
                logger.debug(f"代替API失敗 {url}: {str(e)}")
                continue
            
            time.sleep(0.5)  # レート制限対策
            
        return None
        
    except Exception as e:
        logger.error(f"代替情報取得エラー {ticker}: {str(e)}")
        return None


def create_estimated_ticker_info(ticker: str) -> dict:
    """
    ティッカーシンボルから推定情報を作成（強化版）
    
    Args:
        ticker: ティッカーシンボル
    
    Returns:
        dict: 推定された企業情報
    """
    try:
        # ティッカーシンボルから国と地域を推定（詳細化）
        estimated_country = None
        estimated_sector = "その他"
        
        # より詳細な市場別推定
        if '.T' in ticker or '.JP' in ticker:
            estimated_country = "Japan"
            # 日本の場合、ティッカー番号から業種を推定
            ticker_num = ''.join(filter(str.isdigit, ticker))
            if ticker_num:
                num = int(ticker_num) if ticker_num.isdigit() else 0
                if 1000 <= num <= 1999:
                    estimated_sector = "Fishery, Agriculture & Forestry"
                elif 2000 <= num <= 2999:
                    estimated_sector = "Foods"
                elif 3000 <= num <= 3999:
                    estimated_sector = "Textiles & Apparels"
                elif 4000 <= num <= 4999:
                    estimated_sector = "Chemicals"
                elif 5000 <= num <= 5999:
                    estimated_sector = "Pharmaceutical"
                elif 6000 <= num <= 6999:
                    estimated_sector = "Glass & Ceramics Products"
                elif 7000 <= num <= 7999:
                    estimated_sector = "Iron & Steel"
                elif 8000 <= num <= 8999:
                    estimated_sector = "Machinery"
                elif 9000 <= num <= 9999:
                    estimated_sector = "Electric Appliances"
                else:
                    estimated_sector = "その他（日本）"
        elif '.L' in ticker:
            estimated_country = "United Kingdom"
        elif '.PA' in ticker:
            estimated_country = "France"
        elif '.SW' in ticker:
            estimated_country = "Switzerland"
        elif '.TO' in ticker or '.V' in ticker:
            estimated_country = "Canada"
        elif '.AX' in ticker:
            estimated_country = "Australia"
        elif '.DE' in ticker:
            estimated_country = "Germany"
        elif '.MI' in ticker:
            estimated_country = "Italy"
        elif '.AS' in ticker:
            estimated_country = "Netherlands"
        elif '.ST' in ticker:
            estimated_country = "Sweden"
        elif '.HK' in ticker:
            estimated_country = "Hong Kong"
        elif '.SS' in ticker:
            estimated_country = "China"
        elif '.KS' in ticker:
            estimated_country = "South Korea"
        else:
            # サフィックスがない場合の分類ロジック
            ticker_upper = ticker.upper()
            
            # ETFや不明確な商品かどうかを判定
            etf_indicators = ['ETF', 'FUND', 'GOLD', 'GLD', 'SLV', 'GLDM', 'EPI', 'INDEX', 'SPDR', 'ISHARES', 'VANGUARD']
            is_likely_etf = any(indicator in ticker_upper for indicator in etf_indicators)
            
            # 明確にアメリカ企業と判断できる有名企業
            well_known_us_companies = {
                'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'AMZN': 'Consumer Cyclical',
                'TSLA': 'Consumer Cyclical', 'META': 'Technology', 'NVDA': 'Technology', 'JPM': 'Financial Services',
                'JNJ': 'Healthcare', 'V': 'Financial Services', 'PG': 'Consumer Defensive', 'UNH': 'Healthcare',
                'HD': 'Consumer Cyclical', 'MA': 'Financial Services', 'DIS': 'Communication Services',
                'BAC': 'Financial Services', 'ADBE': 'Technology', 'CRM': 'Technology', 'NFLX': 'Communication Services',
                'KO': 'Consumer Defensive', 'PEP': 'Consumer Defensive', 'ORCL': 'Technology', 'CSCO': 'Technology',
                'INTC': 'Technology', 'VZ': 'Communication Services', 'PFE': 'Healthcare', 'TMO': 'Healthcare',
                'NKE': 'Consumer Cyclical', 'MRK': 'Healthcare', 'ABT': 'Healthcare', 'CVX': 'Energy',
                'WMT': 'Consumer Defensive', 'XOM': 'Energy', 'LLY': 'Healthcare', 'COST': 'Consumer Defensive',
                'SPGI': 'Financial Services', 'ZTS': 'Healthcare', 'CAT': 'Industrial', 'UL': 'Consumer Defensive',
                'ASML': 'Technology', 'NVO': 'Healthcare', 'MSTR': 'Technology', 'IONQ': 'Technology'
            }
            
            if ticker_upper in well_known_us_companies:
                # 明確にアメリカ企業と判断できる場合
                estimated_country = "United States"
                estimated_sector = well_known_us_companies[ticker_upper]
            elif is_likely_etf:
                # ETFや投資商品の場合は「その他」に分類
                estimated_country = None
                estimated_sector = "投資商品・ETF"
            else:
                # 不明確な場合は「その他」に分類（強引にアメリカに分類しない）
                estimated_country = None
                estimated_sector = "その他"
        
        result = {
            'country': estimated_country,
            'sector': estimated_sector,
            'forwardPE': None,
            'priceToBook': None,
            'priceToSalesTrailing12Months': None,
            'enterpriseToEbitda': None,
            'pegRatio': None,
            'marketCap': None,
            'beta': None,
            'dividendYield': None,
            'returnOnEquity': None,
            'returnOnAssets': None,
            'operatingMargins': None,
            'profitMargins': None
        }
        
        logger.info(f"推定情報作成: {ticker} -> {estimated_country}")
        return result
        
    except Exception as e:
        logger.error(f"推定情報作成エラー {ticker}: {str(e)}")
        # 最低限のデータを返す
        return {
            'country': None,
            'sector': None,
            'forwardPE': None,
            'priceToBook': None,
            'priceToSalesTrailing12Months': None,
            'enterpriseToEbitda': None,
            'pegRatio': None,
            'marketCap': None,
            'beta': None,
            'dividendYield': None,
            'returnOnEquity': None,
            'returnOnAssets': None,
            'operatingMargins': None,
            'profitMargins': None
        }


def get_ticker_country(ticker: str) -> Optional[str]:
    """
    ティッカーシンボルから本社所在国を取得
    
    Args:
        ticker: ティッカーシンボル
    
    Returns:
        str: 本社所在国名、取得失敗時はNone
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # 'country'フィールドから取得
        country = info.get('country')
        if country:
            logger.debug(f"取得成功: {ticker} -> {country}")
            return country
        else:
            logger.warning(f"国情報が見つかりません: {ticker}")
            return None
            
    except Exception as e:
        logger.error(f"国情報取得エラー {ticker}: {str(e)}")
        return None


def get_ticker_sector(ticker: str) -> Optional[str]:
    """
    ティッカーシンボルからGICSセクターを取得
    
    Args:
        ticker: ティッカーシンボル
    
    Returns:
        str: GICSセクター名、取得失敗時はNone
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # 'sector'フィールドから取得
        sector = info.get('sector')
        if sector:
            logger.debug(f"セクター取得成功: {ticker} -> {sector}")
            return sector
        else:
            logger.warning(f"セクター情報が見つかりません: {ticker}")
            return None
            
    except Exception as e:
        logger.error(f"セクター情報取得エラー {ticker}: {str(e)}")
        return None


def get_ticker_info(ticker: str) -> Dict[str, Optional[str]]:
    """
    ティッカーシンボルから本社所在国とセクターを一括取得
    
    Args:
        ticker: ティッカーシンボル
    
    Returns:
        Dict[str, Optional[str]]: country、sectorの辞書
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # infoが空や不正な場合のチェック
        if not info or not isinstance(info, dict):
            logger.warning(f"企業情報が取得できませんでした: {ticker}")
            return {'country': None, 'sector': None}
        
        country = info.get('country')
        sector = info.get('sector')
        
        # 空文字列もNoneとして扱う
        country = country.strip() if country and isinstance(country, str) else None
        sector = sector.strip() if sector and isinstance(sector, str) else None
        
        result = {
            'country': country,
            'sector': sector
        }
        
        logger.debug(f"企業情報取得: {ticker} -> 国: {country}, セクター: {sector}")
        return result
        
    except Exception as e:
        logger.error(f"企業情報取得エラー {ticker}: {str(e)}")
        return {'country': None, 'sector': None}


def get_ticker_valuation(ticker: str) -> Dict[str, Optional[float]]:
    """
    ティッカーシンボルからバリュエーション指標を取得
    
    Args:
        ticker: ティッカーシンボル
    
    Returns:
        Dict[str, Optional[float]]: バリュエーション指標の辞書
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # infoが空や不正な場合のチェック
        if not info or not isinstance(info, dict):
            logger.warning(f"バリュエーション情報が取得できませんでした: {ticker}")
            return {key: None for key in ['forwardPE', 'priceToBook', 'priceToSalesTrailing12Months', 
                                        'enterpriseToEbitda', 'pegRatio', 'marketCap', 'beta', 'dividendYield']}
        
        # バリュエーション指標を取得
        def safe_get_float(key):
            value = info.get(key)
            if value is None or value == 'N/A' or (isinstance(value, str) and value.strip() == ''):
                return None
            try:
                # 文字列の場合は数値に変換を試行
                if isinstance(value, str):
                    value = value.replace(',', '').replace('%', '')
                    return float(value)
                return float(value) if value is not None else None
            except (ValueError, TypeError):
                return None
        
        result = {
            'forwardPE': safe_get_float('forwardPE'),
            'priceToBook': safe_get_float('priceToBook'),
            'priceToSalesTrailing12Months': safe_get_float('priceToSalesTrailing12Months'),
            'enterpriseToEbitda': safe_get_float('enterpriseToEbitda'),
            'pegRatio': safe_get_float('pegRatio'),
            'marketCap': safe_get_float('marketCap'),
            'beta': safe_get_float('beta'),
            'dividendYield': safe_get_float('dividendYield')
        }
        
        logger.debug(f"バリュエーション取得: {ticker} -> PE: {result['forwardPE']}, PB: {result['priceToBook']}")
        return result
        
    except Exception as e:
        logger.error(f"バリュエーション取得エラー {ticker}: {str(e)}")
        return {key: None for key in ['forwardPE', 'priceToBook', 'priceToSalesTrailing12Months', 
                                    'enterpriseToEbitda', 'pegRatio', 'marketCap', 'beta', 'dividendYield']}


def get_multiple_ticker_countries(tickers: List[str]) -> Dict[str, Optional[str]]:
    """
    複数銘柄の本社所在国を一括取得
    
    Args:
        tickers: ティッカーシンボルのリスト
    
    Returns:
        Dict[str, Optional[str]]: ティッカーをキーとした本社所在国の辞書
    """
    countries = {}
    
    for ticker in tickers:
        try:
            country = get_ticker_country(ticker)
            countries[ticker] = country
        except Exception as e:
            logger.error(f"国情報取得エラー {ticker}: {str(e)}")
            countries[ticker] = None
    
    logger.info(f"本社所在国取得完了: {len([c for c in countries.values() if c])}/{len(tickers)}銘柄")
    return countries


def get_ticker_financial_metrics(ticker: str) -> Dict[str, Optional[float]]:
    """
    ティッカーシンボルから過去5年間の財務指標平均を取得
    
    Args:
        ticker: ティッカーシンボル
    
    Returns:
        Dict[str, Optional[float]]: 財務指標の平均値辞書
    """
    try:
        stock = yf.Ticker(ticker)
        
        # 財務データを取得
        financials = stock.financials
        info = stock.info
        
        result = {
            'returnOnEquity': None,
            'returnOnAssets': None,
            'operatingMargins': None,
            'profitMargins': None
        }
        
        # infoから直接取得できる場合（最新値）
        if info and isinstance(info, dict):
            def safe_get_float(key):
                value = info.get(key)
                if value is None or value == 'N/A' or (isinstance(value, str) and value.strip() == ''):
                    return None
                try:
                    if isinstance(value, str):
                        # %記号を削除するが、値を100で割らない（既に小数形式のため）
                        value = value.replace(',', '').replace('%', '')
                    return float(value) if value is not None else None
                except (ValueError, TypeError):
                    return None
            
            # infoから取得可能な指標
            def safe_get_margin(key):
                """営業利益率・純利益率用の外れ値チェック付き取得"""
                value = safe_get_float(key)
                if value is not None and key in ['operatingMargins', 'profitMargins']:
                    # -100%より小さい、または+100%より大きい場合は外れ値として除外
                    if value < -1.0 or value > 1.0:
                        logger.warning(f"{key}が外れ値のため除外: {value} ({ticker})")
                        return None
                return value
            
            result['returnOnEquity'] = safe_get_float('returnOnEquity')
            result['returnOnAssets'] = safe_get_float('returnOnAssets')
            result['operatingMargins'] = safe_get_margin('operatingMargins')
            result['profitMargins'] = safe_get_margin('profitMargins')
        
        # 財務諸表からの取得は複雑なので、infoからの取得を優先
        # 将来的に詳細な実装が必要な場合はここで財務諸表データを解析
        
        logger.debug(f"財務指標取得: {ticker} -> ROE: {result['returnOnEquity']}, ROA: {result['returnOnAssets']}")
        return result
        
    except Exception as e:
        logger.error(f"財務指標取得エラー {ticker}: {str(e)}")
        return {
            'returnOnEquity': None,
            'returnOnAssets': None,
            'operatingMargins': None,
            'profitMargins': None
        }


def get_ticker_complete_info(ticker: str, exchange_rates: Dict[str, float] = None) -> Dict[str, any]:
    """
    ティッカーシンボルから本社所在国、セクター、バリュエーション指標、財務指標を一括取得
    
    Args:
        ticker: ティッカーシンボル
        exchange_rates: 為替レート辞書（円換算用）
    
    Returns:
        Dict[str, any]: 企業の全情報を含む辞書
    """
    try:
        logger.info(f"企業情報取得開始: {ticker}")
        stock = yf.Ticker(ticker)
        
        # 強化されたタイムアウトとリトライ機能付きでinfo取得
        info = None
        last_error = None
        
        for attempt in range(5):  # 最大5回リトライ
            try:
                logger.info(f"企業情報取得試行 {attempt + 1}/5: {ticker}")
                
                # yfinanceオブジェクトを再作成（キャッシュクリア）
                stock = yf.Ticker(ticker)
                
                # info取得（タイムアウト対策）
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError("Info取得がタイムアウトしました")
                
                try:
                    # Linuxの場合のみalarmを使用、Windowsでは別の方法
                    if hasattr(signal, 'SIGALRM'):
                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(15)  # 15秒タイムアウト
                    
                    info = stock.info
                    
                    if hasattr(signal, 'SIGALRM'):
                        signal.alarm(0)  # アラームをクリア
                        
                except TimeoutError:
                    logger.warning(f"Info取得タイムアウト {ticker}")
                    info = None
                except:
                    if hasattr(signal, 'SIGALRM'):
                        signal.alarm(0)
                    info = None
                
                # データ品質チェック
                if info and isinstance(info, dict):
                    # 最低限のキーが存在するかチェック
                    essential_keys = ['symbol', 'quoteType', 'shortName', 'longName']
                    has_essential = any(key in info for key in essential_keys)
                    
                    if has_essential or len(info) > 5:
                        logger.info(f"企業情報取得成功 {ticker}: {len(info)}項目")
                        
                        # デバッグ用：利用可能なキーを確認
                        available_keys = list(info.keys())
                        country_related = [k for k in available_keys if 'country' in k.lower() or 'domicile' in k.lower()]
                        sector_related = [k for k in available_keys if 'sector' in k.lower() or 'industry' in k.lower()]
                        
                        if country_related:
                            logger.info(f"国関連キー {ticker}: {country_related}")
                        if sector_related:
                            logger.info(f"セクター関連キー {ticker}: {sector_related}")
                        
                        # 重要なキーの値も確認
                        key_sample = ['symbol', 'shortName', 'country', 'sector', 'industry', 'quoteType']
                        for key in key_sample:
                            if key in info:
                                logger.info(f"{ticker}.{key}: {info[key]}")
                        
                        break
                    else:
                        logger.warning(f"企業情報が不十分 {ticker}: {list(info.keys())[:10]}")
                else:
                    logger.warning(f"企業情報が無効 {ticker}: {type(info)}")
                
                last_error = f"データ品質不足（試行 {attempt + 1}）"
                
            except Exception as e:
                last_error = str(e)
                logger.warning(f"企業情報取得エラー（試行 {attempt + 1}/5) {ticker}: {last_error}")
            
            # リトライ前の待機時間（段階的に増加）
            if attempt < 4:
                import time
                wait_time = 1 + (attempt * 0.5)  # 1, 1.5, 2, 2.5秒
                logger.info(f"待機時間 {wait_time}秒")
                time.sleep(wait_time)
        
        # 最終チェック：データが取得できない場合でも代替情報を提供
        if not info or not isinstance(info, dict):
            logger.error(f"企業情報が完全に取得できませんでした: {ticker}, 最後のエラー: {last_error}")
            
            # 代替情報の取得を試行
            alternative_info = get_alternative_ticker_info(ticker)
            if alternative_info:
                logger.info(f"代替情報を取得: {ticker}")
                info = alternative_info
            else:
                # 最後の手段：推定情報を提供
                logger.warning(f"代替情報も取得できません。推定情報を使用: {ticker}")
                result = create_estimated_ticker_info(ticker)
                return result
        
        # 基本情報（複数のキー名で試行）
        def safe_get_string(possible_keys, field_name):
            """複数のキー名を試して文字列データを取得"""
            for key in possible_keys:
                value = info.get(key)
                if value and isinstance(value, str) and value.strip():
                    logger.info(f"✅ 取得成功 {ticker}.{field_name}: {key} = {value}")
                    return value.strip()
            logger.warning(f"❌ 取得失敗 {ticker}.{field_name}: 試行キー = {possible_keys}")
            return None
        
        # 国情報の取得（複数のキー名で試行）
        country_keys = ['country', 'domicile', 'countryOfDomicile', 'headquarters', 'location']
        country = safe_get_string(country_keys, 'country')
        
        # セクター情報の取得（複数のキー名で試行）
        sector_keys = ['sector', 'sectorKey', 'gicsLevel1', 'industryKey', 'industry']
        sector = safe_get_string(sector_keys, 'sector')
        
        # 取得できなかった場合は推定情報を併用
        if not country or not sector:
            logger.info(f"基本情報が不完全 {ticker}: country={country}, sector={sector}")
            estimated_info = create_estimated_ticker_info(ticker)
            
            if not country:
                country = estimated_info.get('country')
                logger.info(f"推定国情報を使用 {ticker}: {country}")
            
            if not sector:
                sector = estimated_info.get('sector')
                logger.info(f"推定セクター情報を使用 {ticker}: {sector}")
        
        # バリュエーション指標の取得関数（改善版）
        def safe_get_float(key, allow_negative=True):
            # 複数のキー名を試行（yfinanceのAPIキー名変更に対応）
            possible_keys = [key]
            
            # 代替キー名を追加
            key_alternatives = {
                'forwardPE': ['forwardPE', 'forwardEps', 'trailingPE'],
                'priceToBook': ['priceToBook', 'bookValue', 'pbRatio'],
                'priceToSalesTrailing12Months': ['priceToSalesTrailing12Months', 'priceToSales', 'psRatio'],
                'enterpriseToEbitda': ['enterpriseToEbitda', 'evToEbitda', 'ev_ebitda'],
                'pegRatio': ['pegRatio', 'pegRatio12Months'],
                'marketCap': ['marketCap', 'sharesOutstanding'],
                'beta': ['beta', 'beta3Year'],
                'dividendYield': ['dividendYield', 'trailingAnnualDividendYield'],
                'returnOnEquity': ['returnOnEquity', 'roe'],
                'returnOnAssets': ['returnOnAssets', 'roa'],
                'operatingMargins': ['operatingMargins', 'operatingMargin'],
                'profitMargins': ['profitMargins', 'profitMargin', 'netProfitMargins']
            }
            
            if key in key_alternatives:
                possible_keys = key_alternatives[key]
            
            value = None
            for try_key in possible_keys:
                value = info.get(try_key)
                if value is not None:
                    break
            
            # 値の検証と変換
            if value is None or value == 'N/A' or value == 'NaN':
                return None
            
            if isinstance(value, str) and value.strip() == '':
                return None
            
            try:
                # 文字列の場合は数値に変換
                if isinstance(value, str):
                    # %記号、カンマを削除
                    value = value.replace(',', '').replace('%', '').replace('$', '').replace('¥', '')
                    # 空文字列チェック
                    if value.strip() == '':
                        return None
                
                # 数値に変換
                float_value = float(value) if value is not None else None
                
                if float_value is None:
                    return None
                
                # 無限大や非数をチェック
                if not np.isfinite(float_value):
                    return None
                
                # キー固有の検証
                if key == 'forwardPE' and float_value is not None and float_value <= 0:
                    return None  # PERは正の値のみ有効
                
                if key in ['operatingMargins', 'profitMargins'] and float_value is not None:
                    # -200%～+200%の範囲外は外れ値として除外
                    if float_value < -2.0 or float_value > 2.0:
                        logger.warning(f"{key}が外れ値のため除外: {float_value} ({ticker})")
                        return None
                
                if key == 'beta' and float_value is not None:
                    # ベータが-10～+10の範囲外は外れ値として除外
                    if float_value < -10.0 or float_value > 10.0:
                        logger.warning(f"ベータが外れ値のため除外: {float_value} ({ticker})")
                        return None
                
                logger.debug(f"取得成功 {ticker}.{key}: {float_value}")
                return float_value
                
            except (ValueError, TypeError) as e:
                logger.debug(f"数値変換失敗 {ticker}.{key}: {value} -> {str(e)}")
                return None
        
        # 時価総額の円換算
        market_cap_original = safe_get_float('marketCap')
        market_cap_jpy = None
        
        if market_cap_original is not None and exchange_rates:
            # 通貨を推定（ティッカーから）
            currency = 'USD'  # デフォルト
            if '.T' in ticker or '.JP' in ticker:
                currency = 'JPY'
            elif '.L' in ticker:
                currency = 'GBP'
            elif '.PA' in ticker or '.DE' in ticker or '.MI' in ticker:
                currency = 'EUR'
            elif '.TO' in ticker or '.V' in ticker:
                currency = 'CAD'
            elif '.AX' in ticker:
                currency = 'AUD'
            
            # 円換算
            if currency == 'JPY':
                market_cap_jpy = market_cap_original
            else:
                rate_symbol = f"{currency}JPY=X"
                exchange_rate = exchange_rates.get(rate_symbol, 1.0)
                market_cap_jpy = market_cap_original * exchange_rate
        
        # 財務指標を取得（改善版）
        financial_metrics = get_ticker_financial_metrics_improved(ticker, info)
        
        result = {
            'country': country,
            'sector': sector,
            'forwardPE': safe_get_float('forwardPE', allow_negative=False),
            'priceToBook': safe_get_float('priceToBook'),
            'priceToSalesTrailing12Months': safe_get_float('priceToSalesTrailing12Months'),
            'enterpriseToEbitda': safe_get_float('enterpriseToEbitda'),
            'pegRatio': safe_get_float('pegRatio'),
            'marketCap': market_cap_jpy if market_cap_jpy is not None else market_cap_original,
            'beta': safe_get_float('beta'),
            'dividendYield': safe_get_float('dividendYield'),
            'returnOnEquity': financial_metrics['returnOnEquity'],
            'returnOnAssets': financial_metrics['returnOnAssets'],
            'operatingMargins': financial_metrics['operatingMargins'],
            'profitMargins': financial_metrics['profitMargins']
        }
        
        logger.debug(f"完全企業情報取得: {ticker} -> 国: {country}, セクター: {sector}, PE: {result['forwardPE']}")
        return result
        
    except Exception as e:
        logger.error(f"完全企業情報取得エラー {ticker}: {str(e)}")
        result = {'country': None, 'sector': None}
        all_keys = ['forwardPE', 'priceToBook', 'priceToSalesTrailing12Months', 
                   'enterpriseToEbitda', 'pegRatio', 'marketCap', 'beta', 'dividendYield',
                   'returnOnEquity', 'returnOnAssets', 'operatingMargins', 'profitMargins']
        result.update({key: None for key in all_keys})
        return result


def get_multiple_ticker_info(tickers: List[str]) -> Dict[str, Dict[str, Optional[str]]]:
    """
    複数銘柄の本社所在国とセクターを一括取得
    
    Args:
        tickers: ティッカーシンボルのリスト
    
    Returns:
        Dict[str, Dict[str, Optional[str]]]: ティッカーをキーとした企業情報の辞書
    """
    ticker_info = {}
    
    for ticker in tickers:
        try:
            info = get_ticker_info(ticker)
            ticker_info[ticker] = info
        except Exception as e:
            logger.error(f"企業情報取得エラー {ticker}: {str(e)}")
            ticker_info[ticker] = {'country': None, 'sector': None}
    
    success_countries = len([info['country'] for info in ticker_info.values() if info['country']])
    success_sectors = len([info['sector'] for info in ticker_info.values() if info['sector']])
    
    logger.info(f"企業情報取得完了: 国 {success_countries}/{len(tickers)}銘柄, セクター {success_sectors}/{len(tickers)}銘柄")
    return ticker_info


def get_ticker_financial_metrics_improved(ticker: str, info: dict) -> Dict[str, Optional[float]]:
    """
    改善版の財務指標取得関数
    
    Args:
        ticker: ティッカーシンボル
        info: yfinanceから取得済みのinfo辞書
    
    Returns:
        Dict[str, Optional[float]]: 財務指標の辞書
    """
    try:
        result = {
            'returnOnEquity': None,
            'returnOnAssets': None,
            'operatingMargins': None,
            'profitMargins': None
        }
        
        if not info or not isinstance(info, dict):
            return result
        
        def safe_get_metric(metric_keys):
            """複数のキー名を試して財務指標を取得"""
            for key in metric_keys:
                value = info.get(key)
                if value is not None and value != 'N/A' and value != 'NaN':
                    try:
                        if isinstance(value, str):
                            value = value.replace(',', '').replace('%', '').strip()
                            if value == '':
                                continue
                        
                        float_value = float(value)
                        
                        # 無限大や非数をチェック
                        if not np.isfinite(float_value):
                            continue
                        
                        # 範囲チェック（-200%～+200%）
                        if -2.0 <= float_value <= 2.0:
                            logger.debug(f"財務指標取得成功 {ticker}.{key}: {float_value}")
                            return float_value
                        else:
                            logger.warning(f"財務指標が範囲外のため除外 {ticker}.{key}: {float_value}")
                            
                    except (ValueError, TypeError):
                        continue
            return None
        
        # ROE（自己資本利益率）
        roe_keys = ['returnOnEquity', 'roe', 'roeTTM', 'trailingROE']
        result['returnOnEquity'] = safe_get_metric(roe_keys)
        
        # ROA（総資産利益率）
        roa_keys = ['returnOnAssets', 'roa', 'roaTTM', 'trailingROA']
        result['returnOnAssets'] = safe_get_metric(roa_keys)
        
        # 営業利益率
        operating_keys = ['operatingMargins', 'operatingMargin', 'operatingMarginTTM', 'trailingOperatingMargin']
        result['operatingMargins'] = safe_get_metric(operating_keys)
        
        # 純利益率
        profit_keys = ['profitMargins', 'profitMargin', 'netProfitMargin', 'netMarginTTM', 'trailingProfitMargins']
        result['profitMargins'] = safe_get_metric(profit_keys)
        
        success_count = sum(1 for v in result.values() if v is not None)
        logger.info(f"財務指標取得: {ticker} -> {success_count}/4指標取得成功")
        
        return result
        
    except Exception as e:
        logger.error(f"財務指標取得エラー {ticker}: {str(e)}")
        return {
            'returnOnEquity': None,
            'returnOnAssets': None,
            'operatingMargins': None,
            'profitMargins': None
        }


def get_multiple_ticker_complete_info(tickers: List[str], exchange_rates: Dict[str, float] = None) -> Dict[str, Dict[str, any]]:
    """
    複数銘柄の完全な企業情報を一括取得
    
    Args:
        tickers: ティッカーシンボルのリスト
        exchange_rates: 為替レート辞書（円換算用）
    
    Returns:
        Dict[str, Dict[str, any]]: ティッカーをキーとした完全企業情報の辞書
    """
    ticker_info = {}
    
    logger.info(f"完全企業情報取得開始: {len(tickers)}銘柄")
    
    # Streamlit progress bar setup
    try:
        import streamlit as st
        progress_bar = st.progress(0)
        status_text = st.empty()
        show_progress = True
    except:
        show_progress = False
    
    successful_count = 0
    
    for i, ticker in enumerate(tickers):
        try:
            if show_progress:
                progress = (i + 1) / len(tickers)
                progress_bar.progress(progress)
                status_text.text(f"取得中: {ticker} ({i+1}/{len(tickers)})")
            
            logger.info(f"進捗: {i+1}/{len(tickers)} - {ticker}")
            info = get_ticker_complete_info(ticker, exchange_rates)
            ticker_info[ticker] = info
            
            # 成功カウント
            if info and (info.get('country') or info.get('sector') or 
                        any(info.get(key) for key in ['forwardPE', 'priceToBook', 'marketCap'])):
                successful_count += 1
            
            # レート制限対策：各銘柄の取得後に待機（動的調整）
            if i < len(tickers) - 1:  # 最後の銘柄以外
                import time
                # 成功率に基づいて待機時間を調整
                if successful_count / (i + 1) < 0.5:
                    wait_time = 1.0  # 成功率が低い場合は長く待機
                else:
                    wait_time = 0.3  # 成功率が高い場合は短く待機
                time.sleep(wait_time)
                
        except Exception as e:
            logger.error(f"完全企業情報取得エラー {ticker}: {str(e)}")
            result = {'country': None, 'sector': None}
            all_keys = ['forwardPE', 'priceToBook', 'priceToSalesTrailing12Months', 
                       'enterpriseToEbitda', 'pegRatio', 'marketCap', 'beta', 'dividendYield',
                       'returnOnEquity', 'returnOnAssets', 'operatingMargins', 'profitMargins']
            result.update({key: None for key in all_keys})
            ticker_info[ticker] = result
    
    # 成功統計
    success_countries = len([info['country'] for info in ticker_info.values() if info['country']])
    success_sectors = len([info['sector'] for info in ticker_info.values() if info['sector']])
    success_valuations = {}
    valuation_keys = ['forwardPE', 'priceToBook', 'priceToSalesTrailing12Months', 'enterpriseToEbitda', 
                      'pegRatio', 'marketCap', 'beta', 'dividendYield', 'returnOnEquity', 'returnOnAssets', 
                      'operatingMargins', 'profitMargins']
    
    for key in valuation_keys:
        success_valuations[key] = len([info[key] for info in ticker_info.values() if info[key] is not None])
    
    logger.info(f"完全企業情報取得完了: 国 {success_countries}/{len(tickers)}銘柄, セクター {success_sectors}/{len(tickers)}銘柄")
    
    # バリュエーション指標の詳細統計
    valuation_success_summary = []
    for key, count in success_valuations.items():
        percentage = (count / len(tickers)) * 100
        valuation_success_summary.append(f"{key}: {count}/{len(tickers)}({percentage:.1f}%)")
    
    logger.info("バリュエーション指標取得状況:")
    for summary in valuation_success_summary:
        logger.info(f"  {summary}")
    
    # プログレスバーをクリーンアップ
    if show_progress:
        try:
            progress_bar.progress(100)
            status_text.text(f"完了: {successful_count}/{len(tickers)}銘柄で情報を取得")
            import time
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
        except:
            pass
    
    # Streamlit用の成功率表示
    try:
        import streamlit as st
        total_metrics = len(valuation_keys)
        successful_metrics = sum(1 for count in success_valuations.values() if count > 0)
        
        if successful_metrics > 0:
            st.success(f"バリュエーション指標取得完了: {successful_metrics}/{total_metrics}種類の指標でデータを取得")
            st.info(f"企業基本情報: {successful_count}/{len(tickers)}銘柄で情報を取得")
        else:
            st.warning("バリュエーション指標のデータが取得できませんでした。ネットワーク接続やティッカーシンボルを確認してください。")
    except:
        pass
    
    return ticker_info


def classify_region_by_country(country: Optional[str]) -> str:
    """
    本社所在国から地域を分類
    
    Args:
        country: 本社所在国名
    
    Returns:
        str: 地域名（日本、米国、欧州、アジア太平洋、その他）
    """
    if not country or country.strip() == '':
        return "その他"
    
    country = country.upper().strip()
    
    # 日本
    if country in ['JAPAN']:
        return "日本"
    
    # 米国
    if country in ['UNITED STATES', 'USA', 'US']:
        return "米国"
    
    # 欧州
    european_countries = [
        'GERMANY', 'FRANCE', 'UNITED KINGDOM', 'UK', 'GREAT BRITAIN',
        'ITALY', 'SPAIN', 'NETHERLANDS', 'SWITZERLAND', 'SWEDEN',
        'NORWAY', 'DENMARK', 'FINLAND', 'BELGIUM', 'AUSTRIA',
        'IRELAND', 'PORTUGAL', 'LUXEMBOURG', 'GREECE', 'POLAND',
        'CZECH REPUBLIC', 'HUNGARY', 'SLOVAKIA', 'SLOVENIA',
        'CROATIA', 'ROMANIA', 'BULGARIA', 'ESTONIA', 'LATVIA',
        'LITHUANIA', 'MALTA', 'CYPRUS'
    ]
    
    if country in european_countries:
        return "欧州"
    
    # アジア太平洋
    asia_pacific_countries = [
        'CHINA', 'SOUTH KOREA', 'KOREA', 'TAIWAN', 'HONG KONG',
        'SINGAPORE', 'MALAYSIA', 'THAILAND', 'INDONESIA',
        'PHILIPPINES', 'VIETNAM', 'INDIA', 'AUSTRALIA',
        'NEW ZEALAND'
    ]
    
    if country in asia_pacific_countries:
        return "アジア太平洋"
    
    # カナダ
    if country in ['CANADA']:
        return "北米（その他）"
    
    # その他
    return "その他"


@st.cache_data(ttl=3600)  # 1時間キャッシュ
def cached_get_multiple_ticker_countries(tickers_tuple: tuple) -> Dict[str, Optional[str]]:
    """
    キャッシュ機能付きの複数銘柄本社所在国取得
    
    Args:
        tickers_tuple: ティッカーシンボルのタプル（キャッシュキー用）
    
    Returns:
        Dict[str, Optional[str]]: 本社所在国辞書
    """
    return get_multiple_ticker_countries(list(tickers_tuple))


@st.cache_data(ttl=3600)  # 1時間キャッシュ
def cached_get_multiple_ticker_info(tickers_tuple: tuple) -> Dict[str, Dict[str, Optional[str]]]:
    """
    キャッシュ機能付きの複数銘柄企業情報取得
    
    Args:
        tickers_tuple: ティッカーシンボルのタプル（キャッシュキー用）
    
    Returns:
        Dict[str, Dict[str, Optional[str]]]: 企業情報辞書
    """
    return get_multiple_ticker_info(list(tickers_tuple))


@st.cache_data(ttl=3600)  # 1時間キャッシュ
def cached_get_multiple_ticker_complete_info(tickers_tuple: tuple, exchange_rates_tuple: tuple = None) -> Dict[str, Dict[str, any]]:
    """
    キャッシュ機能付きの複数銘柄完全企業情報取得
    
    Args:
        tickers_tuple: ティッカーシンボルのタプル（キャッシュキー用）
        exchange_rates_tuple: 為替レートのタプル（キャッシュキー用）
    
    Returns:
        Dict[str, Dict[str, any]]: 完全企業情報辞書
    """
    exchange_rates = dict(exchange_rates_tuple) if exchange_rates_tuple else None
    return get_multiple_ticker_complete_info(list(tickers_tuple), exchange_rates)