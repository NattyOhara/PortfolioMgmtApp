"""
Gemini APIを使用した市場動向要約モジュール
"""

import os
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import google.generativeai as genai
import sys
import locale

# 文字エンコーディングの設定
def ensure_utf8_encoding():
    """UTF-8エンコーディングを確実にする"""
    try:
        # Windowsでのエンコーディング問題を回避
        if sys.platform == "win32":
            import codecs
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    except Exception:
        pass  # エラーが発生しても続行

# エンコーディング設定を実行
ensure_utf8_encoding()

logger = logging.getLogger(__name__)


def safe_text_processing(text: str) -> str:
    """テキストを安全に処理する（エンコーディングエラー回避）"""
    if not isinstance(text, str):
        text = str(text)
    
    try:
        # Unicode正規化
        import unicodedata
        text = unicodedata.normalize('NFC', text)
        
        # 問題のある文字を安全な文字に置換
        text = text.encode('utf-8', errors='replace').decode('utf-8')
        
        return text
    except Exception as e:
        logger.warning(f"テキスト処理エラー: {e}")
        # フォールバック：ASCII安全な文字のみ保持
        return ''.join(char for char in text if ord(char) < 128)


class GeminiClient:
    """Gemini APIクライアント"""
    
    def __init__(self, model_name: str = "gemini-1.5-pro"):
        """
        Gemini APIクライアントを初期化
        
        Args:
            model_name: 使用するGeminiモデル名
        """
        self.api_key = os.getenv('GEMINI_API_KEY')
        
        if not self.api_key:
            raise ValueError(
                "Gemini APIキーが設定されていません。"
                ".envファイルにGEMINI_API_KEYを設定してください。"
            )
        
        # APIキーを設定
        genai.configure(api_key=self.api_key)
        
        # モデルを初期化
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)
        
        # 生成設定
        self.generation_config = genai.GenerationConfig(
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            max_output_tokens=8192,
        )
        
        logger.info(f"Gemini APIクライアントを初期化: モデル={model_name}")
    
    def generate_market_summary(
        self,
        articles_text: str,
        start_date: datetime,
        end_date: datetime,
        performance_summary: str
    ) -> str:
        """
        市場動向の要約を生成
        
        Args:
            articles_text: スクレイピングした記事の統合テキスト
            start_date: 分析期間の開始日
            end_date: 分析期間の終了日
            performance_summary: ポートフォリオパフォーマンスサマリー
            
        Returns:
            生成された市場動向要約
        """
        prompt = self._create_market_summary_prompt(
            articles_text, start_date, end_date, performance_summary
        )
        
        try:
            logger.info("Gemini APIで市場動向要約を生成中...")
            
            # プロンプトをさらに安全に処理
            safe_prompt = safe_text_processing(prompt)
            
            response = self.model.generate_content(
                safe_prompt,
                generation_config=self.generation_config
            )
            
            if response.text:
                logger.info("市場動向要約の生成に成功")
                # レスポンステキストも安全に処理
                return safe_text_processing(response.text)
            else:
                logger.error("Gemini APIから空のレスポンス")
                raise Exception("要約の生成に失敗しました")
                
        except UnicodeEncodeError as e:
            logger.error(f"文字エンコーディングエラー: {e}")
            raise Exception(f"文字エンコーディングエラーが発生しました。システムの言語設定を確認してください: {str(e)}")
        except Exception as e:
            logger.error(f"Gemini API エラー: {e}")
            raise Exception(f"市場動向要約の生成中にエラーが発生しました: {str(e)}")
    
    def _create_market_summary_prompt(
        self,
        articles_text: str,
        start_date: datetime,
        end_date: datetime,
        performance_summary: str
    ) -> str:
        """市場動向要約用のプロンプトを作成"""
        
        # 安全な日付フォーマット（エンコーディングエラー回避）
        start_date_str = f"{start_date.year}年{start_date.month}月{start_date.day}日"
        end_date_str = f"{end_date.year}年{end_date.month}月{end_date.day}日"
        
        # テキストを安全に処理
        safe_articles_text = safe_text_processing(articles_text[:15000])
        safe_performance_summary = safe_text_processing(performance_summary)
        
        prompt = f"""以下のニュース記事とポートフォリオパフォーマンスデータを基に、
{start_date_str}から{end_date_str}までの
包括的な運用レポートを作成してください。

【ニュース記事から抽出した市場情報】
{safe_articles_text}

【ポートフォリオパフォーマンスデータ】
{safe_performance_summary}

【レポート構成と詳細要件】

## 1. 市場環境分析（800-1000字）
ニュース記事から読み取れる実際の市場動向を基に、以下の観点で分析してください：

### 経済・金融政策
- 主要中央銀行（FRB、ECB、日銀等）の実際の政策決定や発言
- 期間中に発表された経済指標（インフレ率、雇用統計、GDP等）の具体的な数値と市場への影響
- 金利動向と債券市場の変化

### 政治・地政学リスク
- 期間中に実際に発生した政治的イベントや地政学的な出来事
- これらのイベントが市場に与えた具体的な影響

### 市場テーマとセンチメント
- ニュースから読み取れる期間中の主要な投資テーマ
- 投資家センチメントの変化を示す具体的な事例
- セクター別の資金フローやローテーション

## 2. ポートフォリオパフォーマンス詳細評価（600-800字）

### ベンチマーク比較分析
- 提供されたデータに基づくMSCI ACWI ETF、NASDAQ100 ETF、Topix ETFとの詳細比較
- アウトパフォーム/アンダーパフォームの要因分析
- ニュース記事から読み取れる市場環境との関連性

### パフォーマンス要因分析
- 市場環境の変化（ニュースから把握）がポートフォリオに与えた影響
- セクター配分や地域配分の観点からの分析

## 3. 個別銘柄詳細分析（800-1000字）

### 上位パフォーマンス銘柄（上位5銘柄）
- 各銘柄の上昇要因をニュース記事の情報と関連付けて分析
- 業界動向や企業固有のニュースとの関係性
- 市場テーマとの整合性

### 下位パフォーマンス銘柄（下位5銘柄）
- 各銘柄の下落要因をニュース記事の情報と関連付けて分析
- 業界の逆風や企業固有の課題
- リスク要因の具体的な説明

## 4. 今後の投資戦略への示唆（400-600字）

### 市場展望
- ニュース記事から読み取れる今後の注目ポイント
- 継続する可能性のある市場テーマ
- 警戒すべきリスク要因

### ポートフォリオ運営への示唆
- 現在の市場環境を踏まえた一般的な投資戦略の考察
- リスク管理の観点からの留意点

【出力要件】
- 合計2000-3000字程度
- ニュース記事から得られた具体的な情報を積極的に引用
- 客観的で専門的な文体
- データと事実に基づいた分析
- 投資推奨は避け、情報提供と分析に徹する

【重要な注意事項】
- 具体的な売買推奨は一切行わない
- 分析は一般的な市場観察と情報提供に留める
- 最後に「本レポートは情報提供のみを目的としており、投資判断は自己責任で行ってください」という免責事項を記載
- 「過去の実績は将来の結果を保証するものではありません」という注意書きも追加"""
        
        return prompt


def create_performance_summary_text(portfolio_performance: Dict[str, Any], benchmark_performance: Dict[str, Any], 
                                ticker_performance: Dict[str, Any], from_date: datetime, to_date: datetime) -> str:
    """詳細なパフォーマンスサマリーを作成"""
    summary_parts = []
    
    # 期間情報（安全な日付フォーマット）
    from_date_str = f"{from_date.year}-{from_date.month:02d}-{from_date.day:02d}"
    to_date_str = f"{to_date.year}-{to_date.month:02d}-{to_date.day:02d}"
    days_diff = (to_date - from_date).days
    summary_parts.append(f"【分析期間】{from_date_str} - {to_date_str} ({days_diff}日間)")
    
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


def generate_gemini_investment_report(
    performance_result: Dict[str, Any],
    from_date: datetime,
    to_date: datetime,
    news_articles_text: str,
    model_name: str = "gemini-1.5-pro"
) -> Dict[str, Any]:
    """
    Gemini APIを使用して投資レポートを生成
    
    Args:
        performance_result: パフォーマンス計算結果
        from_date: レポート期間の開始日
        to_date: レポート期間の終了日
        news_articles_text: スクレイピングしたニュース記事のテキスト
        model_name: 使用するGeminiモデル名
        
    Returns:
        生成されたレポート情報
    """
    try:
        # パフォーマンスサマリーを準備
        portfolio_performance = performance_result.get("portfolio_performance", {})
        benchmark_performance = performance_result.get("benchmark_performance", {})
        ticker_performance = performance_result.get("ticker_performance", {})
        
        performance_summary = create_performance_summary_text(
            portfolio_performance, benchmark_performance, ticker_performance, from_date, to_date
        )
        
        # Geminiクライアントを初期化
        client = GeminiClient(model_name=model_name)
        
        # 市場動向要約を生成
        report_content = client.generate_market_summary(
            articles_text=news_articles_text,
            start_date=from_date,
            end_date=to_date,
            performance_summary=performance_summary
        )
        
        return {
            "success": True,
            "report": report_content,
            "performance_summary": performance_summary,
            "model_used": model_name,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "news_based": True  # ニュース記事ベースであることを示すフラグ
        }
        
    except Exception as e:
        logger.error(f"レポート生成エラー: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }