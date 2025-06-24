"""
ニュース記事本文をスクレイピングするモジュール
"""

import requests
from bs4 import BeautifulSoup
import logging
import time
from typing import Dict, List, Optional
from urllib.parse import urlparse
import re

logger = logging.getLogger(__name__)


class NewsScraper:
    """ニュース記事スクレイパー"""
    
    def __init__(self, user_agent: Optional[str] = None, timeout: int = 10):
        """
        スクレイパーを初期化
        
        Args:
            user_agent: ユーザーエージェント文字列
            timeout: リクエストタイムアウト（秒）
        """
        self.timeout = timeout
        self.headers = {
            'User-Agent': user_agent or 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # サイト別のコンテンツセレクター
        self.site_selectors = {
            'bloomberg.co.jp': {
                'content': ['article', '.article-body', '.story-body'],
                'remove': ['.ad', '.advertisement', 'aside', 'nav']
            },
            'jp.reuters.com': {
                'content': ['article', '.article-body', '[data-testid="article-body"]'],
                'remove': ['.ad', '.social-links', '.related-content']
            },
            'nikkei.com': {
                'content': ['article', '.article-body', '.cmn-article_text'],
                'remove': ['.ad', '.subscription-promo', '.related-articles']
            },
            'default': {
                'content': ['article', 'main', '.article-body', '.content', '[role="main"]'],
                'remove': ['script', 'style', '.ad', '.advertisement', 'aside', 'nav', 'header', 'footer']
            }
        }
    
    def _get_site_config(self, url: str) -> Dict[str, List[str]]:
        """URLに基づいてサイト固有の設定を取得"""
        domain = urlparse(url).netloc
        
        for site, config in self.site_selectors.items():
            if site in domain:
                return config
        
        return self.site_selectors['default']
    
    def _clean_text(self, text: str) -> str:
        """テキストをクリーンアップ"""
        # 余分な空白を削除
        text = re.sub(r'\s+', ' ', text)
        # 前後の空白を削除
        text = text.strip()
        # 連続する改行を単一の改行に
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text
    
    def scrape_article(self, url: str) -> Dict[str, str]:
        """
        記事本文をスクレイピング
        
        Args:
            url: 記事のURL
            
        Returns:
            記事情報（タイトル、本文、エラー情報など）
        """
        try:
            logger.info(f"記事をスクレイピング: {url}")
            
            # HTTPリクエスト
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            
            # HTMLパース
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # サイト設定を取得
            config = self._get_site_config(url)
            
            # 不要な要素を削除
            for selector in config['remove']:
                for element in soup.select(selector):
                    element.decompose()
            
            # タイトル取得
            title = None
            title_elements = soup.select('h1, .article-title, [class*="title"]')
            if title_elements:
                title = title_elements[0].get_text(strip=True)
            
            # 本文取得
            content = None
            for selector in config['content']:
                content_elements = soup.select(selector)
                if content_elements:
                    # 複数要素の場合は結合
                    content_texts = []
                    for elem in content_elements:
                        text = elem.get_text(separator='\n', strip=True)
                        if text and len(text) > 100:  # 短すぎるテキストは除外
                            content_texts.append(text)
                    
                    if content_texts:
                        content = '\n\n'.join(content_texts)
                        break
            
            # 本文が見つからない場合、段落タグから取得を試みる
            if not content:
                paragraphs = soup.find_all('p')
                content_texts = []
                for p in paragraphs:
                    text = p.get_text(strip=True)
                    if text and len(text) > 50:  # 短い段落は除外
                        content_texts.append(text)
                
                if content_texts:
                    content = '\n\n'.join(content_texts[:20])  # 最初の20段落まで
            
            # テキストクリーンアップ
            if content:
                content = self._clean_text(content)
            
            return {
                'url': url,
                'title': title or 'タイトル不明',
                'content': content or '本文を取得できませんでした',
                'success': bool(content),
                'error': None
            }
            
        except requests.RequestException as e:
            logger.error(f"記事取得エラー ({url}): {e}")
            return {
                'url': url,
                'title': 'エラー',
                'content': '',
                'success': False,
                'error': f'記事の取得に失敗しました: {str(e)}'
            }
        except Exception as e:
            logger.error(f"予期しないエラー ({url}): {e}")
            return {
                'url': url,
                'title': 'エラー',
                'content': '',
                'success': False,
                'error': f'予期しないエラーが発生しました: {str(e)}'
            }
    
    def scrape_multiple_articles(
        self,
        urls: List[str],
        delay: float = 1.0,
        max_articles: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """
        複数の記事をスクレイピング
        
        Args:
            urls: 記事URLのリスト
            delay: リクエスト間の待機時間（秒）
            max_articles: 最大取得記事数
            
        Returns:
            記事情報のリスト
        """
        results = []
        urls_to_scrape = urls[:max_articles] if max_articles else urls
        
        for i, url in enumerate(urls_to_scrape):
            # スクレイピング実行
            result = self.scrape_article(url)
            results.append(result)
            
            # 最後のURLでない場合は待機
            if i < len(urls_to_scrape) - 1:
                time.sleep(delay)
        
        # 成功した記事の数をログ出力
        successful = sum(1 for r in results if r['success'])
        logger.info(f"スクレイピング完了: {successful}/{len(results)} 件成功")
        
        return results


def scrape_news_articles(
    news_items: List[Dict[str, str]],
    max_articles: int = 20,
    delay: float = 1.0
) -> str:
    """
    ニュース記事をスクレイピングして統合テキストを生成
    
    Args:
        news_items: Google Search APIから取得したニュース情報のリスト
        max_articles: 最大取得記事数
        delay: リクエスト間の待機時間
        
    Returns:
        統合された記事テキスト
    """
    scraper = NewsScraper()
    
    # URLのみを抽出
    urls = [item['url'] for item in news_items if 'url' in item]
    
    # スクレイピング実行
    articles = scraper.scrape_multiple_articles(
        urls=urls,
        delay=delay,
        max_articles=max_articles
    )
    
    # 成功した記事のテキストを統合
    combined_text_parts = []
    
    for article in articles:
        if article['success'] and article['content']:
            # ソース情報を含めて追加
            text_part = f"--- 記事タイトル: {article['title']} ---\n"
            text_part += f"ソース: {article['url']}\n\n"
            text_part += article['content']
            combined_text_parts.append(text_part)
    
    # 全記事を結合
    combined_text = "\n\n" + "="*50 + "\n\n".join(combined_text_parts)
    
    logger.info(f"統合テキスト生成完了: {len(combined_text)} 文字")
    
    return combined_text