"""
通貨判定ユーティリティ
ティッカーシンボルから通貨を判定する機能
"""

from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


def get_currency_from_ticker(ticker: str) -> str:
    """
    ティッカーシンボルから上場通貨を判定
    
    Args:
        ticker: ティッカーシンボル
    
    Returns:
        str: 通貨コード（USD, JPY, EUR等）
    """
    ticker = ticker.upper().strip()
    
    # 日本株の判定
    japanese_suffixes = ['.T', '.JP', '.OS', '.TS']
    if any(suffix in ticker for suffix in japanese_suffixes):
        return 'JPY'
    
    # 欧州株の判定
    european_mappings = {
        '.AS': 'EUR',  # オランダ（アムステルダム）
        '.PA': 'EUR',  # フランス（パリ）
        '.DE': 'EUR',  # ドイツ（フランクフルト）
        '.MI': 'EUR',  # イタリア（ミラノ）
        '.MC': 'EUR',  # スペイン（マドリード）
        '.VI': 'EUR',  # オーストリア（ウィーン）
        '.BR': 'EUR',  # ベルギー（ブリュッセル）
        '.LS': 'EUR',  # ポルトガル（リスボン）
        '.HE': 'EUR',  # フィンランド（ヘルシンキ）
        '.IC': 'EUR',  # アイスランド
    }
    
    for suffix, currency in european_mappings.items():
        if suffix in ticker:
            return currency
    
    # 英国株の判定
    uk_suffixes = ['.L', '.LON']
    if any(suffix in ticker for suffix in uk_suffixes):
        return 'GBP'
    
    # スイス株の判定
    if '.SW' in ticker or '.VX' in ticker:
        return 'CHF'
    
    # カナダ株の判定
    canadian_suffixes = ['.TO', '.V', '.CN']
    if any(suffix in ticker for suffix in canadian_suffixes):
        return 'CAD'
    
    # オーストラリア株の判定
    if '.AX' in ticker:
        return 'AUD'
    
    # 香港株の判定
    if '.HK' in ticker:
        return 'HKD'
    
    # シンガポール株の判定
    if '.SI' in ticker:
        return 'SGD'
    
    # 中国株の判定
    chinese_suffixes = ['.SS', '.SZ']
    if any(suffix in ticker for suffix in chinese_suffixes):
        return 'CNY'
    
    # 韓国株の判定
    if '.KS' in ticker or '.KQ' in ticker:
        return 'KRW'
    
    # インド株の判定
    indian_suffixes = ['.NS', '.BO']
    if any(suffix in ticker for suffix in indian_suffixes):
        return 'INR'
    
    # ブラジル株の判定
    if '.SA' in ticker:
        return 'BRL'
    
    # メキシコ株の判定
    if '.MX' in ticker:
        return 'MXN'
    
    # 南アフリカ株の判定
    if '.JO' in ticker:
        return 'ZAR'
    
    # ロシア株の判定
    if '.ME' in ticker:
        return 'RUB'
    
    # トルコ株の判定
    if '.IS' in ticker:
        return 'TRY'
    
    # デフォルトは米国株（USD）
    return 'USD'


def get_currency_mapping(tickers: List[str]) -> Dict[str, str]:
    """
    ティッカーリストから通貨マッピング辞書を作成
    
    Args:
        tickers: ティッカーシンボルのリスト
    
    Returns:
        Dict[str, str]: ティッカーをキーとした通貨辞書
    """
    currency_mapping = {}
    
    for ticker in tickers:
        currency = get_currency_from_ticker(ticker)
        currency_mapping[ticker] = currency
        logger.debug(f"通貨判定: {ticker} -> {currency}")
    
    logger.info(f"通貨マッピング作成完了: {len(currency_mapping)}銘柄")
    return currency_mapping


def get_supported_currencies() -> List[str]:
    """
    サポートされている通貨のリストを取得
    
    Returns:
        List[str]: サポート通貨リスト
    """
    return [
        'USD',  # 米ドル
        'JPY',  # 日本円
        'EUR',  # ユーロ
        'GBP',  # 英ポンド
        'CHF',  # スイスフラン
        'CAD',  # カナダドル
        'AUD',  # オーストラリアドル
        'HKD',  # 香港ドル
        'SGD',  # シンガポールドル
        'CNY',  # 中国元
        'KRW',  # 韓国ウォン
        'INR',  # インドルピー
        'BRL',  # ブラジルレアル
        'MXN',  # メキシコペソ
        'ZAR',  # 南アフリカランド
        'RUB',  # ロシアルーブル
        'TRY'   # トルコリラ
    ]


def get_currency_exchange_pairs() -> Dict[str, str]:
    """
    為替レート取得用のペア定義
    
    Returns:
        Dict[str, str]: 通貨ペア辞書
    """
    return {
        'USD': 'USDJPY=X',
        'EUR': 'EURJPY=X',
        'GBP': 'GBPJPY=X',
        'CHF': 'CHFJPY=X',
        'CAD': 'CADJPY=X',
        'AUD': 'AUDJPY=X',
        'HKD': 'HKDJPY=X',
        'SGD': 'SGDJPY=X',
        'CNY': 'CNYJPY=X',
        'KRW': 'KRWJPY=X'
    }


def get_fallback_exchange_rates() -> Dict[str, float]:
    """
    フォールバック用の概算為替レート
    
    Returns:
        Dict[str, float]: 概算レート辞書（対JPY）
    """
    return {
        'USD': 150.0,   # 1 USD = 150 JPY
        'EUR': 160.0,   # 1 EUR = 160 JPY
        'GBP': 180.0,   # 1 GBP = 180 JPY
        'CHF': 165.0,   # 1 CHF = 165 JPY
        'CAD': 110.0,   # 1 CAD = 110 JPY
        'AUD': 100.0,   # 1 AUD = 100 JPY
        'HKD': 19.0,    # 1 HKD = 19 JPY
        'SGD': 110.0,   # 1 SGD = 110 JPY
        'CNY': 21.0,    # 1 CNY = 21 JPY
        'KRW': 0.11,    # 1 KRW = 0.11 JPY
        'INR': 1.8,     # 1 INR = 1.8 JPY
        'BRL': 30.0,    # 1 BRL = 30 JPY
        'MXN': 8.5,     # 1 MXN = 8.5 JPY
        'ZAR': 8.0,     # 1 ZAR = 8.0 JPY
        'RUB': 1.6,     # 1 RUB = 1.6 JPY
        'TRY': 4.5      # 1 TRY = 4.5 JPY
    }


def validate_ticker_format(ticker: str) -> bool:
    """
    ティッカーシンボルの形式を検証
    
    Args:
        ticker: ティッカーシンボル
    
    Returns:
        bool: 有効な形式かどうか
    """
    if not ticker or not isinstance(ticker, str):
        return False
    
    ticker = ticker.strip()
    
    # 空文字チェック
    if not ticker:
        return False
    
    # 長すぎる場合は無効
    if len(ticker) > 20:
        return False
    
    # 基本的な文字チェック（英数字、ドット、ハイフンのみ）
    import re
    if not re.match(r'^[A-Za-z0-9.\-]+$', ticker):
        return False
    
    return True


def get_market_info(ticker: str) -> Dict[str, str]:
    """
    ティッカーから市場情報を取得
    
    Args:
        ticker: ティッカーシンボル
    
    Returns:
        Dict[str, str]: 市場情報
    """
    ticker = ticker.upper().strip()
    currency = get_currency_from_ticker(ticker)
    
    # 市場判定
    market_mappings = {
        '.T': '東京証券取引所',
        '.JP': '日本（JASDAQ）',
        '.OS': '大阪証券取引所',
        '.AS': 'ユーロネクスト・アムステルダム',
        '.PA': 'ユーロネクスト・パリ',
        '.DE': 'フランクフルト証券取引所',
        '.MI': 'ボルサ・イタリアーナ',
        '.MC': 'マドリッド証券取引所',
        '.L': 'ロンドン証券取引所',
        '.SW': 'スイス証券取引所',
        '.TO': 'トロント証券取引所',
        '.V': 'TSXベンチャー取引所',
        '.AX': 'オーストラリア証券取引所',
        '.HK': '香港証券取引所',
        '.SI': 'シンガポール証券取引所',
        '.SS': '上海証券取引所',
        '.SZ': '深圳証券取引所',
        '.KS': '韓国証券取引所',
        '.NS': 'インド国立証券取引所',
        '.BO': 'ボンベイ証券取引所'
    }
    
    market = 'NASDAQ/NYSE'  # デフォルト
    for suffix, market_name in market_mappings.items():
        if suffix in ticker:
            market = market_name
            break
    
    # 国判定
    country_mappings = {
        'JPY': '日本',
        'USD': 'アメリカ',
        'EUR': 'ユーロ圏',
        'GBP': 'イギリス',
        'CHF': 'スイス',
        'CAD': 'カナダ',
        'AUD': 'オーストラリア',
        'HKD': '香港',
        'SGD': 'シンガポール',
        'CNY': '中国',
        'KRW': '韓国',
        'INR': 'インド'
    }
    
    country = country_mappings.get(currency, 'その他')
    
    return {
        'ticker': ticker,
        'currency': currency,
        'market': market,
        'country': country
    }