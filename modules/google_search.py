"""
Google Custom Search API を使用した金融ニュース検索モジュール
"""

import os
from typing import List, Dict, Optional
from datetime import datetime
import logging
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

logger = logging.getLogger(__name__)


class GoogleSearchClient:
    """Google Custom Search APIクライアント"""
    
    def __init__(self):
        """APIクライアントを初期化"""
        self.api_key = os.getenv('GOOGLE_API_KEY')
        self.search_engine_id = os.getenv('GOOGLE_SEARCH_ENGINE_ID')
        
        if not self.api_key or not self.search_engine_id:
            raise ValueError(
                "Google Search APIの設定が不完全です。"
                "GOOGLE_API_KEYとGOOGLE_SEARCH_ENGINE_IDが.envファイルに設定されているか確認してください。"
            )
        
        self.service = build("customsearch", "v1", developerKey=self.api_key)
    
    def search_financial_news(
        self, 
        start_date: datetime, 
        end_date: datetime,
        query: str = "グローバル金融市場 ニュース",
        num_results: int = 10
    ) -> List[Dict[str, str]]:
        """
        指定期間の金融ニュースを検索
        
        Args:
            start_date: 検索開始日
            end_date: 検索終了日
            query: 検索クエリ
            num_results: 取得する結果数（最大10件/リクエスト）
            
        Returns:
            検索結果のリスト（タイトル、URL、スニペット含む）
        """
        try:
            # 日付をフォーマット
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            # 検索クエリに期間指定を追加
            full_query = f"{query} after:{start_str} before:{end_str}"
            
            logger.info(f"Google Search APIで検索: {full_query}")
            
            # 検索実行
            results = self.service.cse().list(
                q=full_query,
                cx=self.search_engine_id,
                num=min(num_results, 10),  # 最大10件
                lr="lang_ja",  # 日本語優先
                safe="active"  # セーフサーチ有効
            ).execute()
            
            # 結果を整形
            search_results = []
            if 'items' in results:
                for item in results['items']:
                    search_results.append({
                        'title': item.get('title', ''),
                        'url': item.get('link', ''),
                        'snippet': item.get('snippet', ''),
                        'source': item.get('displayLink', '')
                    })
            
            logger.info(f"{len(search_results)}件の検索結果を取得")
            return search_results
            
        except HttpError as e:
            logger.error(f"Google Search APIエラー: {e}")
            raise Exception(f"ニュース検索中にエラーが発生しました: {str(e)}")
        except Exception as e:
            logger.error(f"予期しないエラー: {e}")
            raise
    
    def search_multiple_queries(
        self,
        start_date: datetime,
        end_date: datetime,
        queries: List[str],
        num_per_query: int = 5
    ) -> List[Dict[str, str]]:
        """
        複数のクエリで検索を実行し、結果を統合
        
        Args:
            start_date: 検索開始日
            end_date: 検索終了日
            queries: 検索クエリのリスト
            num_per_query: 各クエリの取得件数
            
        Returns:
            全検索結果の統合リスト
        """
        all_results = []
        seen_urls = set()  # 重複URL除去用
        
        for query in queries:
            try:
                results = self.search_financial_news(
                    start_date=start_date,
                    end_date=end_date,
                    query=query,
                    num_results=num_per_query
                )
                
                # 重複を除いて追加
                for result in results:
                    if result['url'] not in seen_urls:
                        all_results.append(result)
                        seen_urls.add(result['url'])
                        
            except Exception as e:
                logger.warning(f"クエリ '{query}' の検索でエラー: {e}")
                continue
        
        logger.info(f"合計 {len(all_results)} 件の一意なニュース記事を取得")
        return all_results


def get_financial_news_urls(
    start_date: datetime,
    end_date: datetime,
    search_topics: Optional[List[str]] = None
) -> List[Dict[str, str]]:
    """
    金融ニュースURLを取得する便利関数
    
    Args:
        start_date: 検索開始日
        end_date: 検索終了日
        search_topics: 検索トピックのリスト（省略時はデフォルトを使用）
        
    Returns:
        ニュース記事情報のリスト
    """
    # デフォルトの検索トピック
    if search_topics is None:
        search_topics = [
            "グローバル金融市場 動向",
            "株式市場 相場",
            "為替市場 ドル円 ユーロ",
            "中央銀行 金融政策 FRB ECB 日銀",
            "経済指標 インフレ 雇用統計",
            "債券市場 金利",
            "コモディティ 原油 金",
            "地政学リスク 国際情勢"
        ]
    
    try:
        client = GoogleSearchClient()
        return client.search_multiple_queries(
            start_date=start_date,
            end_date=end_date,
            queries=search_topics,
            num_per_query=5
        )
    except Exception as e:
        logger.error(f"ニュースURL取得エラー: {e}")
        raise