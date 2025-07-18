"""
News Crawler Module using LangChain and LangGraph
Crawls today's news from multiple sources and processes them for RAG applications.
"""

import asyncio
import os
from datetime import datetime
from typing import Dict, List, TypedDict, Optional, Any
from urllib.parse import urljoin, urlparse
import json
import logging
from dataclasses import dataclass

# LangChain imports
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

# Web scraping imports
import requests
from bs4 import BeautifulSoup
import feedparser
from pydantic import BaseModel, Field

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data models
@dataclass
class NewsArticle:
    title: str
    content: str
    url: str
    published_date: Optional[str] = None
    author: Optional[str] = None
    source: Optional[str] = None
    category: Optional[str] = None

class NewsState(TypedDict):
    """State for the news crawling workflow"""
    target_sources: List[str]
    discovered_urls: List[str]
    crawled_articles: List[NewsArticle]
    processed_articles: List[Dict[str, Any]]
    errors: List[str]
    current_source: Optional[str]
    max_articles_per_source: int
    date_filter: Optional[str]

class NewsCrawlerConfig:
    """Configuration for the news crawler"""
    def __init__(self):
        # Major news sources with RSS feeds
        self.NEWS_SOURCES = {
            "techcrunch": "https://techcrunch.com/feed/",
            "reuters_tech": "https://www.reuters.com/technology/",
            "hackernews": "https://hnrss.org/newest",
            "bbc_tech": "http://feeds.bbci.co.uk/news/technology/rss.xml",
            "cnn_tech": "http://rss.cnn.com/rss/edition.technology.rss",
            "wired": "https://www.wired.com/feed/rss",
            "verge": "https://www.theverge.com/rss/index.xml",
            "naver_news": "https://news.naver.com/"
        }
        
        # CSS selectors for different news sites
        self.CONTENT_SELECTORS = {
            "default": {
                "title": "h1, .headline, .article-title",
                "content": ".article-content, .story-body, .content, .post-content, article p",
                "author": ".author, .byline, .writer",
                "date": ".date, .timestamp, .published, time"
            },
            "techcrunch.com": {
                "title": "h1.article__title",
                "content": ".article-content",
                "author": ".article__byline a",
                "date": ".article__meta time"
            },
            "reuters.com": {
                "title": "h1[data-testid='Headline']",
                "content": "[data-testid='paragraph']",
                "author": "[data-testid='Author']",
                "date": "[data-testid='ArticleDate']"
            },
            "naver.com": {
                "title": ".media_end_head_headline, .article_headline h1, #title_area span",
                "content": ".go_trans._article_content, #dic_area, .se-main-container, #newsct_article",
                "author": ".media_end_head_journalist .name, .media_end_head_journalist a em, .byline, .reporter, .press_link, .author",
                "date": ".media_end_head_info_datestamp_time, .media_end_head_info_datestamp, .article_info_date"
            }
        }

class NaverNewsDiscoverer:
    """네이버 뉴스 전용 URL 발견기"""
    
    def __init__(self):
        self.base_url = "https://news.naver.com"
    
    async def discover_from_naver_sections(self, max_articles: int = 10) -> List[str]:
        """네이버 뉴스 섹션에서 URL 발견"""
        try:
            sections = [
                "/main/list.naver?mode=LPOD&mid=sec&sid1=001",  # 정치
                "/main/list.naver?mode=LPOD&mid=sec&sid1=102",  # 사회
                "/main/list.naver?mode=LPOD&mid=sec&sid1=103",  # 생활/문화
                "/main/list.naver?mode=LPOD&mid=sec&sid1=105",  # IT/과학
                "/main/list.naver?mode=LPOD&mid=sec&sid1=104",  # 세계
                "/main/list.naver?mode=LPOD&mid=sec&sid1=101",  # 경제
                "/main/list.naver?mode=LPOD&mid=sec&sid1=100",  # 스포츠
            ]
            
            all_urls = []
            
            for section_url in sections:
                try:
                    full_url = self.base_url + section_url
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    }
                    response = requests.get(full_url, headers=headers, timeout=10)
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # 네이버 뉴스 기사 링크 찾기 (더 구체적인 선택자 사용)
                    # 네이버 뉴스의 실제 구조에 맞는 선택자들
                    article_selectors = [
                        'a[href*="/article/"]',
                        '.cjs_news_tw a',
                        '.cjs_t a',
                        '.cjs_d a'
                    ]
                    
                    for selector in article_selectors:
                        links = soup.select(selector)
                        for link in links:
                            href = link.get('href', '')
                            if '/article/' in href:
                                if href.startswith('http'):
                                    all_urls.append(href)
                                elif href.startswith('/'):
                                    full_article_url = self.base_url + href
                                    all_urls.append(full_article_url)
                                
                                if len(all_urls) >= max_articles:
                                    break
                        
                        if len(all_urls) >= max_articles:
                            break
                    
                    if len(all_urls) >= max_articles:
                        break
                        
                except Exception as e:
                    logger.error(f"Error crawling Naver section {section_url}: {e}")
                    continue
            
            # 중복 제거 및 제한
            unique_urls = list(set(all_urls))[:max_articles]
            logger.info(f"Discovered {len(unique_urls)} URLs from Naver News")
            return unique_urls
            
        except Exception as e:
            logger.error(f"Error discovering URLs from Naver News: {e}")
            return []
    
    async def discover_today_news_from_naver(self, max_articles: int = 50) -> List[str]:
        """네이버 뉴스에서 오늘의 뉴스 전체 크롤링"""
        try:
            from datetime import datetime
            today = datetime.now().strftime("%Y%m%d")
            
            # 네이버 뉴스 메인 페이지와 각 섹션의 최신 뉴스 URL들
            urls_to_check = [
                "https://news.naver.com/",
                "https://news.naver.com/main/list.naver?mode=LPOD&mid=sec&sid1=001",  # 정치
                "https://news.naver.com/main/list.naver?mode=LPOD&mid=sec&sid1=102",  # 사회  
                "https://news.naver.com/main/list.naver?mode=LPOD&mid=sec&sid1=103",  # 생활/문화
                "https://news.naver.com/main/list.naver?mode=LPOD&mid=sec&sid1=105",  # IT/과학
                "https://news.naver.com/main/list.naver?mode=LPOD&mid=sec&sid1=104",  # 세계
                "https://news.naver.com/main/list.naver?mode=LPOD&mid=sec&sid1=101",  # 경제
                "https://news.naver.com/main/list.naver?mode=LPOD&mid=sec&sid1=100",  # 스포츠
            ]
            
            all_urls = []
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            for url in urls_to_check:
                try:
                    logger.info(f"Crawling Naver news from: {url}")
                    response = requests.get(url, headers=headers, timeout=10)
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # 다양한 네이버 뉴스 링크 패턴 찾기
                    link_patterns = [
                        'a[href*="news.naver.com/main/read.naver"]',
                        'a[href*="news.naver.com/article"]',
                        'a[href*="/article/"]'
                    ]
                    
                    for pattern in link_patterns:
                        links = soup.select(pattern)
                        for link in links:
                            href = link.get('href', '')
                            if href:
                                # 오늘 날짜가 포함된 링크만 선택 (가능한 경우)
                                if href.startswith('http'):
                                    all_urls.append(href)
                                elif href.startswith('/'):
                                    full_url = "https://news.naver.com" + href
                                    all_urls.append(full_url)
                    
                    # 각 페이지마다 잠시 대기
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error crawling {url}: {e}")
                    continue
            
            # 중복 제거
            unique_urls = list(set(all_urls))
            
            # 오늘 날짜가 포함된 URL 우선 선택
            today_urls = [url for url in unique_urls if today in url]
            if today_urls:
                logger.info(f"Found {len(today_urls)} URLs with today's date")
                return today_urls[:max_articles]
            
            # 오늘 날짜 URL이 없으면 일반 URL 반환
            logger.info(f"Found {len(unique_urls)} total URLs from Naver News")
            return unique_urls[:max_articles]
            
        except Exception as e:
            logger.error(f"Error discovering today's news from Naver: {e}")
            return []
    
    async def discover_from_naver_search(self, query: str, max_articles: int = 10) -> List[str]:
        """네이버 뉴스 검색에서 URL 발견"""
        try:
            search_url = f"https://search.naver.com/search.naver?where=news&query={query}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(search_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 뉴스 검색 결과에서 기사 링크 찾기
            news_links = []
            
            # 네이버 뉴스 검색 결과 CSS 선택자
            link_elements = soup.select('a.info')
            
            for link in link_elements:
                href = link.get('href')
                if href and 'news.naver.com' in href and '/article/' in href:
                    news_links.append(href)
                    
                if len(news_links) >= max_articles:
                    break
            
            logger.info(f"Discovered {len(news_links)} URLs from Naver News search for '{query}'")
            return news_links
            
        except Exception as e:
            logger.error(f"Error searching Naver News for '{query}': {e}")
            return []

class NewsSourceDiscoverer:
    """Discovers news URLs from RSS feeds and sitemaps"""
    
    def __init__(self, config: NewsCrawlerConfig):
        self.config = config
        self.naver_discoverer = NaverNewsDiscoverer()
    
    async def discover_from_rss(self, rss_url: str, max_articles: int = 10) -> List[str]:
        """Discover article URLs from RSS feed"""
        try:
            feed = feedparser.parse(rss_url)
            urls = []
            
            for entry in feed.entries[:max_articles]:
                if hasattr(entry, 'link'):
                    urls.append(entry.link)
                    
            logger.info(f"Discovered {len(urls)} URLs from RSS: {rss_url}")
            return urls
            
        except Exception as e:
            logger.error(f"Error parsing RSS feed {rss_url}: {e}")
            return []
    
    async def discover_from_website(self, base_url: str, max_articles: int = 10) -> List[str]:
        """Discover article URLs by scraping website"""
        try:
            response = requests.get(base_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for article links
            article_links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                
                # Filter for article URLs (basic heuristics)
                if any(keyword in href.lower() for keyword in ['article', 'news', 'story', '2025', '2024']):
                    full_url = urljoin(base_url, href)
                    article_links.append(full_url)
                    
                    if len(article_links) >= max_articles:
                        break
            
            logger.info(f"Discovered {len(article_links)} URLs from website: {base_url}")
            return article_links
            
        except Exception as e:
            logger.error(f"Error discovering URLs from {base_url}: {e}")
            return []

class NewsContentExtractor:
    """Extracts content from news articles"""
    
    def __init__(self, config: NewsCrawlerConfig):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""]
        )
    
    def get_selectors_for_domain(self, url: str) -> Dict[str, str]:
        """Get CSS selectors based on the domain"""
        domain = urlparse(url).netloc.lower()
        
        for site, selectors in self.config.CONTENT_SELECTORS.items():
            if site in domain:
                return selectors
                
        return self.config.CONTENT_SELECTORS["default"]
    
    async def extract_article_content(self, url: str) -> Optional[NewsArticle]:
        """Extract article content from URL"""
        try:
            # Use LangChain's WebBaseLoader
            loader = WebBaseLoader([url])
            documents = loader.load()
            
            if not documents:
                logger.warning(f"No content extracted from {url}")
                return None
            
            doc = documents[0]
            
            # Try to extract structured content using BeautifulSoup for better parsing
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            selectors = self.get_selectors_for_domain(url)
            
            # Extract title
            title_elem = soup.select_one(selectors["title"])
            title = title_elem.get_text(strip=True) if title_elem else doc.metadata.get('title', 'Unknown Title')
            
            # Extract author
            author_elem = soup.select_one(selectors["author"])
            author = author_elem.get_text(strip=True) if author_elem else None
            
            # 네이버 뉴스의 경우 추가적인 작성자 추출 시도
            if not author and 'naver.com' in url:
                # 더 많은 선택자 시도
                additional_author_selectors = [
                    '.media_end_head_journalist_name',
                    '.journalist_name',
                    '.reporter_name', 
                    '.article_info .author',
                    '[class*="journalist"] [class*="name"]',
                    '[class*="reporter"]',
                    '.byline_p',
                    '.media_end_head_info span:contains("기자")',
                ]
                
                for selector in additional_author_selectors:
                    try:
                        if ':contains(' in selector:
                            # BeautifulSoup doesn't support :contains, so skip
                            continue
                        elem = soup.select_one(selector)
                        if elem:
                            author = elem.get_text(strip=True)
                            if author:
                                break
                    except:
                        continue
                
                # 텍스트에서 기자명 패턴 추출 시도
                if not author:
                    import re
                    text_content = soup.get_text()
                    # "기자명 기자" 패턴 찾기
                    reporter_pattern = r'([가-힣]{2,4})\s*기자'
                    match = re.search(reporter_pattern, text_content)
                    if match:
                        author = match.group(1) + ' 기자'
            
            # Extract date
            date_elem = soup.select_one(selectors["date"])
            date = date_elem.get_text(strip=True) if date_elem else None
            
            # Normalize published date
            normalized_date = self._normalize_published_date(date) if date else None
            
            # Extract main content
            content_elems = soup.select(selectors["content"])
            if content_elems:
                content = "\n".join([elem.get_text(strip=True) for elem in content_elems])
            else:
                content = doc.page_content
            
            # Clean up content
            content = self._clean_content(content)
            
            # Extract source (press) - especially for Naver news
            source = self._extract_source_info(url, soup)
            
            # Normalize author name
            normalized_author = self._normalize_author(author) if author else None
            
            # Extract category information
            category = _extract_category_info(url, title, content)
            
            # Create article object
            article = NewsArticle(
                title=title,
                content=content,
                url=url,
                published_date=normalized_date,
                author=normalized_author,
                source=source,
                category=category
            )
            print(f"Extracted article: {article}")
            
            logger.info(f"Extracted article: {title[:50]}...")
            return article
            
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {e}")
            return None
    
    def _extract_source_info(self, url: str, soup: BeautifulSoup) -> str:
        """Extract the actual press/source information from the webpage"""
        
        # For Naver news, try to extract the actual press name
        if 'naver.com' in url:
            # Try multiple selectors for press information
            press_selectors = [
                '.media_end_head_top .press_logo img',  # 상단 언론사 로고의 alt 텍스트
                '.media_end_head_top .press_logo .press',  # 언론사명 텍스트
                '.media_end_head_top .press_logo',  # 언론사 영역
                '.press_logo img',  # 일반적인 언론사 로고
                '.press_logo',  # 언론사 영역
                '.media_end_head_info .press',  # 헤더 정보의 언론사
                '.article_info .press',  # 기사 정보의 언론사
                '.media_end_head_top a img',  # 상단 링크의 이미지
                '.media_end_head_info a',  # 헤더 정보의 링크
                '.press_link',  # 언론사 링크
                '.author_press',  # 작성자와 언론사 정보
            ]
            
            for selector in press_selectors:
                try:
                    elem = soup.select_one(selector)
                    if elem:
                        if elem.name == 'img':
                            # 이미지의 alt 속성에서 언론사명 추출
                            press_name = elem.get('alt', '').strip()
                            if press_name and press_name not in ['로고', 'logo', '이미지']:
                                return press_name
                        else:
                            # 텍스트에서 언론사명 추출
                            press_name = elem.get_text(strip=True)
                            if press_name:
                                return press_name
                except Exception as e:
                    continue
            
            # URL 파라미터에서 언론사 정보 추출 시도
            try:
                from urllib.parse import parse_qs, urlparse
                parsed_url = urlparse(url)
                query_params = parse_qs(parsed_url.query)
                
                # oid 파라미터가 있으면 언론사 코드
                if 'oid' in query_params:
                    oid = query_params['oid'][0]
                    # 주요 언론사 코드 매핑 (일부만)
                    press_codes = {
                        '001': '연합뉴스',
                        '003': '뉴시스',
                        '005': '국민일보',
                        '008': '머니투데이',
                        '009': '매일경제',
                        '011': '서울경제',
                        '014': '파이낸셜뉴스',
                        '015': '한국경제',
                        '016': '헤럴드경제',
                        '018': '이데일리',
                        '020': '동아일보',
                        '021': '문화일보',
                        '022': '세계일보',
                        '023': '조선일보',
                        '025': '중앙일보',
                        '028': '한겨레',
                        '032': '경향신문',
                        '037': '한국일보',
                        '055': 'SBS',
                        '056': 'KBS',
                        '057': 'MBC',
                        '081': '서울신문',
                        '092': '한국경제TV',
                        '119': '데일리안',
                        '138': '디지털타임스',
                        '144': '스포츠서울',
                        '213': '뉴데일리',
                        '214': '뉴스1',
                        '277': '아시아경제',
                        '293': 'YTN',
                        '374': 'SBS Biz',
                        '417': '머니S',
                        '421': '뉴스핌',
                        '448': '코리아헤럴드',
                        '449': '채널A',
                        '469': '한국일보',
                    }
                    
                    if oid in press_codes:
                        return press_codes[oid]
                    else:
                        return f"언론사코드_{oid}"
            except Exception as e:
                pass
            
            # 메타태그에서 언론사 정보 추출 시도
            try:
                meta_selectors = [
                    'meta[property="og:site_name"]',
                    'meta[name="author"]',
                    'meta[property="article:author"]',
                ]
                
                for selector in meta_selectors:
                    meta_elem = soup.select_one(selector)
                    if meta_elem:
                        content_attr = meta_elem.get('content', '').strip()
                        if content_attr and '네이버' not in content_attr:
                            return content_attr
            except Exception as e:
                pass
            
            # 페이지 텍스트에서 언론사명 패턴 찾기
            try:
                import re
                page_text = soup.get_text()
                
                # "XXX 기자" 패턴에서 언론사 추출
                press_patterns = [
                    r'([가-힣\w\s]+)\s+기자',
                    r'(\w+뉴스|\w+일보|\w+신문)',  # 뉴스, 일보, 신문으로 끝나는 패턴
                    r'(연합뉴스|조선일보|중앙일보|동아일보|한국경제|매일경제|머니투데이|이데일리|뉴시스)',  # 주요 언론사명 직접 매칭
                ]
                
                for pattern in press_patterns:
                    matches = re.findall(pattern, page_text)
                    if matches:
                        # 가장 자주 나오는 언론사명 선택
                        from collections import Counter
                        most_common = Counter(matches).most_common(1)
                        if most_common:
                            press_name = most_common[0][0].strip()
                            if len(press_name) > 1:  # 한 글자는 제외
                                return press_name
            except Exception as e:
                pass
            
            # 모든 시도가 실패하면 네이버 뉴스로 반환
            return "네이버뉴스"
        
        # 네이버가 아닌 다른 사이트의 경우 도메인 사용
        return urlparse(url).netloc
    
    def _clean_content(self, content: str) -> str:
        """Clean and normalize article content"""
        # Remove extra whitespace
        content = " ".join(content.split())
        
        # Remove common noise patterns
        noise_patterns = [
            "Click here to",
            "Read more",
            "Subscribe to",
            "Follow us on",
            "Share this article",
            "Advertisement"
        ]
        
        for pattern in noise_patterns:
            content = content.replace(pattern, "")
        
        return content.strip()

    def _normalize_published_date(self, date_text: str) -> Optional[str]:
        """Normalize published date to standard format YYYY-MM-DD HH:MM"""
        if not date_text:
            return None
        
        import re
        from datetime import datetime
        
        try:
            # 네이버 뉴스 날짜 패턴 처리
            # "입력2025.07.17. 오전 11:10수정2025.07.17. 오후 1:50기사원문" 형태
            
            # 1. 첫 번째 날짜/시간 정보만 추출 (입력 시간)
            input_pattern = r'입력(\d{4})\.(\d{1,2})\.(\d{1,2})\.\s*(오전|오후)\s*(\d{1,2}):(\d{2})'
            match = re.search(input_pattern, date_text)
            
            if match:
                year, month, day, ampm, hour, minute = match.groups()
                hour = int(hour)
                
                # 오후이고 12시가 아니면 12시간 추가
                if ampm == '오후' and hour != 12:
                    hour += 12
                # 오전 12시는 0시로 변환
                elif ampm == '오전' and hour == 12:
                    hour = 0
                
                # 표준 형식으로 변환
                normalized = f"{year}-{month.zfill(2)}-{day.zfill(2)} {hour:02d}:{minute}"
                return normalized
            
            # 2. 일반적인 날짜 패턴들 처리
            patterns = [
                r'(\d{4})-(\d{1,2})-(\d{1,2})\s+(\d{1,2}):(\d{2})',  # 2025-07-17 14:30
                r'(\d{4})\.(\d{1,2})\.(\d{1,2})\s+(\d{1,2}):(\d{2})',  # 2025.07.17 14:30
                r'(\d{4})/(\d{1,2})/(\d{1,2})\s+(\d{1,2}):(\d{2})',  # 2025/07/17 14:30
                r'(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일\s*(\d{1,2}):(\d{2})',  # 2025년 7월 17일 14:30
            ]
            
            for pattern in patterns:
                match = re.search(pattern, date_text)
                if match:
                    year, month, day, hour, minute = match.groups()
                    normalized = f"{year}-{month.zfill(2)}-{day.zfill(2)} {hour.zfill(2)}:{minute}"
                    return normalized
            
            # 3. 날짜만 있는 경우 (시간 없음)
            date_only_patterns = [
                r'(\d{4})-(\d{1,2})-(\d{1,2})',
                r'(\d{4})\.(\d{1,2})\.(\d{1,2})',
                r'(\d{4})/(\d{1,2})/(\d{1,2})',
                r'(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일',
            ]
            
            for pattern in date_only_patterns:
                match = re.search(pattern, date_text)
                if match:
                    year, month, day = match.groups()
                    normalized = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                    return normalized
            
            # 4. 상대 시간 표현 처리 ("3시간 전", "어제" 등)
            relative_patterns = {
                r'(\d+)분\s*전': lambda m: self._get_relative_time(minutes=-int(m.group(1))),
                r'(\d+)시간\s*전': lambda m: self._get_relative_time(hours=-int(m.group(1))),
                r'어제': lambda m: self._get_relative_time(days=-1),
                r'오늘': lambda m: self._get_relative_time(),
                r'그저께': lambda m: self._get_relative_time(days=-2),
            }
            
            for pattern, func in relative_patterns.items():
                match = re.search(pattern, date_text)
                if match:
                    return func(match)
            
            # 파싱할 수 없으면 None 반환
            return None
            
        except Exception as e:
            logger.warning(f"Error normalizing date '{date_text}': {e}")
            return None
    
    def _get_relative_time(self, days=0, hours=0, minutes=0) -> str:
        """Get relative time from now"""
        from datetime import datetime, timedelta
        
        target_time = datetime.now() + timedelta(days=days, hours=hours, minutes=minutes)
        return target_time.strftime("%Y-%m-%d %H:%M")
    
    def _normalize_author(self, author_text: str) -> Optional[str]:
        """Normalize author name to first reporter's name only"""
        if not author_text:
            return None
        
        import re
        
        try:
            # 여러 기자가 있을 경우 첫 번째 기자만 추출
            # "서소정 기자 ssj@asiae.co.kr박유진 기자 genie@asiae.co.kr" → "서소정"
            
            # 1. 기자명 패턴 추출
            # 한글 이름 + "기자" 패턴
            reporter_pattern = r'([가-힣]{2,4})\s*기자'
            matches = re.findall(reporter_pattern, author_text)
            
            if matches:
                # 첫 번째 기자명 반환
                return matches[0].strip()
            
            # 2. 기자가 없으면 일반적인 이름 패턴 찾기
            # 한글 이름만 (2-4글자)
            name_pattern = r'^([가-힣]{2,4})'
            match = re.match(name_pattern, author_text.strip())
            
            if match:
                return match.group(1)
            
            # 3. 이메일이 포함된 경우 이메일 앞의 이름 추출
            # "name@domain.com" 형태에서 name 부분 추출
            email_pattern = r'([가-힣a-zA-Z]{2,10})\s*[@]'
            match = re.search(email_pattern, author_text)
            
            if match:
                potential_name = match.group(1)
                # 한글만 있는 경우에만 반환
                if re.match(r'^[가-힣]{2,4}$', potential_name):
                    return potential_name
            
            # 4. 텍스트에서 첫 번째 한글 단어 추출 (최후 수단)
            korean_words = re.findall(r'[가-힣]{2,4}', author_text)
            if korean_words:
                # 흔한 단어들 제외
                exclude_words = {'기자', '기사', '뉴스', '연합', '취재', '편집', '특파원', '앵커', '진행'}
                for word in korean_words:
                    if word not in exclude_words:
                        return word
            
            # 추출할 수 없으면 None 반환
            return None
            
        except Exception as e:
            logger.warning(f"Error normalizing author '{author_text}': {e}")
            return None

# LangGraph Workflow Nodes
async def discover_news_urls(state: NewsState) -> NewsState:
    """Discover URLs from news sources"""
    config = NewsCrawlerConfig()
    discoverer = NewsSourceDiscoverer(config)
    
    all_urls = []
    max_per_source = state.get("max_articles_per_source", 10)
    
    for source_name, source_url in config.NEWS_SOURCES.items():
        logger.info(f"Discovering URLs from {source_name}")
        
        if source_name == "naver_news":
            # 네이버 뉴스 전용 처리 - 오늘의 모든 뉴스 크롤링
            urls = await discoverer.naver_discoverer.discover_today_news_from_naver(max_per_source * 3)  # 더 많은 기사 수집
            logger.info(f"Found {len(urls)} URLs from Naver today's news")
        elif source_url.endswith('.xml') or 'rss' in source_url or 'feed' in source_url:
            urls = await discoverer.discover_from_rss(source_url, max_per_source)
        else:
            urls = await discoverer.discover_from_website(source_url, max_per_source)
        
        all_urls.extend(urls)
    
    # Remove duplicates
    unique_urls = list(set(all_urls))
    
    state["discovered_urls"] = unique_urls
    logger.info(f"Total discovered URLs: {len(unique_urls)}")
    
    return state

async def extract_article_content(state: NewsState) -> NewsState:
    """Extract content from discovered URLs"""
    config = NewsCrawlerConfig()
    extractor = NewsContentExtractor(config)
    
    urls = state["discovered_urls"]
    articles = []
    errors = []
    
    # Process URLs concurrently (but with some rate limiting)
    semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent requests
    
    async def extract_single(url: str) -> Optional[NewsArticle]:
        async with semaphore:
            return await extractor.extract_article_content(url)
    
    tasks = [extract_single(url) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for i, result in enumerate(results):
        if isinstance(result, NewsArticle):
            articles.append(result)
        elif isinstance(result, Exception):
            errors.append(f"Error processing {urls[i]}: {str(result)}")
        elif result is None:
            errors.append(f"No content extracted from {urls[i]}")
    
    state["crawled_articles"] = articles
    state["errors"] = errors
    
    logger.info(f"Successfully extracted {len(articles)} articles")
    return state

async def process_articles(state: NewsState) -> NewsState:
    """Process articles with category extraction (no summary/tags)"""
    articles = state["crawled_articles"]
    processed = []
    
    for article in articles:
        try:
            # Extract category from URL or content (no LLM processing)
            category = _extract_category_info(article.url, article.title, article.content)
            
            # Convert to dict for storage
            article_dict = {
                "title": article.title,
                "content": article.content,
                "url": article.url,
                "published_date": article.published_date,
                "author": article.author,
                "source": article.source,
                "category": category,
                "crawled_at": datetime.now().isoformat()
            }
            
            processed.append(article_dict)
            
        except Exception as e:
            logger.error(f"Error processing article {article.title}: {e}")
    
    state["processed_articles"] = processed
    logger.info(f"Processed {len(processed)} articles")
    
    return state

def _extract_category_info(url: str, title: str, content: str) -> str:
    """Extract category information from URL, title, or content"""
    
    # 1. URL에서 카테고리 추출 (네이버 뉴스의 경우)
    if 'naver.com' in url:
        if 'sid1=001' in url:
            return '정치'
        elif 'sid1=102' in url:
            return '사회'
        elif 'sid1=103' in url:
            return '생활/문화'
        elif 'sid1=105' in url:
            return 'IT/과학'
        elif 'sid1=104' in url:
            return '세계'
        elif 'sid1=101' in url:
            return '경제'
        elif 'sid1=100' in url:
            return '스포츠'
    
    # 2. 제목과 내용에서 키워드 기반 카테고리 추출
    import re
    
    text = (title + " " + content).lower()
    
    # 카테고리 키워드 매핑
    category_keywords = {
        '정치': ['대통령', '국회', '정부', '장관', '의원', '정치', '선거', '국정감사', '법안'],
        '경제': ['주식', '증시', '경제', '금리', '부동산', '투자', '기업', '매출', '수익', '코스피', '달러', '환율'],
        '사회': ['사건', '사고', '범죄', '재판', '경찰', '검찰', '화재', '교통사고', '시민', '주민'],
        'IT/과학': ['AI', '인공지능', '기술', '스마트폰', '컴퓨터', '소프트웨어', '앱', '플랫폼', '디지털', '과학', '연구'],
        '스포츠': ['축구', '야구', '농구', '배구', '올림픽', '월드컵', '선수', '경기', '리그', '팀'],
        '문화': ['영화', '드라마', 'K-POP', '음악', '문화', '예술', '전시', '공연', '방송', '연예'],
        '세계': ['미국', '중국', '일본', '유럽', '국제', '외교', '해외', '글로벌', '국제적'],
        '건강': ['코로나', '백신', '의료', '병원', '질병', '건강', '치료', '약물', '의사']
    }
    
    # 각 카테고리별 키워드 매칭 점수 계산
    category_scores = {}
    for category, keywords in category_keywords.items():
        score = 0
        for keyword in keywords:
            score += len(re.findall(keyword, text))
        category_scores[category] = score
    
    # 가장 높은 점수의 카테고리 반환
    if category_scores:
        best_category = max(category_scores.items(), key=lambda x: x[1])
        if best_category[1] > 0:  # 최소 1개 키워드는 매칭되어야 함
            return best_category[0]
    
    # 기본 카테고리
    return '일반'

async def save_articles(state: NewsState) -> NewsState:
    """Save processed articles to file"""
    articles = state["processed_articles"]
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"/Users/kai/workspace/rag-tutorial/rag/data/news_articles_{timestamp}.json"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Save to JSON file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved {len(articles)} articles to {filename}")
    
    # Also save a summary report
    summary_filename = f"/Users/kai/workspace/rag-tutorial/rag/data/crawl_summary_{timestamp}.json"
    summary = {
        "timestamp": timestamp,
        "total_articles": len(articles),
        "sources": list(set([article["source"] for article in articles])),
        "errors": state.get("errors", []),
        "data_file": filename
    }
    
    with open(summary_filename, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    return state

class NewsCrawler:
    """Main news crawler class using LangGraph workflow"""
    
    def __init__(self):
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(NewsState)
        
        # Add nodes
        workflow.add_node("discover_urls", discover_news_urls)
        workflow.add_node("extract_content", extract_article_content)
        workflow.add_node("process_articles", process_articles)
        workflow.add_node("save_articles", save_articles)
        
        # Define the flow
        workflow.add_edge(START, "discover_urls")
        workflow.add_edge("discover_urls", "extract_content")
        workflow.add_edge("extract_content", "process_articles")
        workflow.add_edge("process_articles", "save_articles")
        workflow.add_edge("save_articles", END)
        
        return workflow.compile()
    
    async def crawl_todays_news(self, max_articles_per_source: int = 5) -> Dict:
        """Crawl today's news from configured sources"""
        
        initial_state: NewsState = {
            "target_sources": [],
            "discovered_urls": [],
            "crawled_articles": [],
            "processed_articles": [],
            "errors": [],
            "current_source": None,
            "max_articles_per_source": max_articles_per_source,
            "date_filter": datetime.now().strftime("%Y-%m-%d")
        }
        
        logger.info("Starting news crawling workflow...")
        
        result = await self.workflow.ainvoke(initial_state)
        
        return {
            "success": True,
            "articles_count": len(result["processed_articles"]),
            "sources_count": len(set([article["source"] for article in result["processed_articles"]])),
            "errors_count": len(result["errors"]),
            "articles": result["processed_articles"],
            "errors": result["errors"]
        }

# Convenience function for direct usage
async def crawl_news(max_articles_per_source: int = 5) -> Dict:
    """Crawl today's news - convenience function"""
    crawler = NewsCrawler()
    return await crawler.crawl_todays_news(max_articles_per_source)

async def crawl_naver_today_news(max_articles: int = 50) -> Dict:
    """네이버에서 오늘의 모든 뉴스를 크롤링하는 함수"""
    try:
        from datetime import datetime
        
        config = NewsCrawlerConfig()
        discoverer = NewsSourceDiscoverer(config)
        extractor = NewsContentExtractor(config)
        
        logger.info("Starting Naver today's news crawling...")
        
        # 1. 오늘의 네이버 뉴스 URL 발견
        urls = await discoverer.naver_discoverer.discover_today_news_from_naver(max_articles)
        logger.info(f"Discovered {len(urls)} URLs from Naver today's news")
        
        if not urls:
            return {
                "success": False,
                "articles_count": 0,
                "sources_count": 0,
                "errors_count": 1,
                "articles": [],
                "errors": ["No URLs found from Naver news"]
            }
        
        # 2. 기사 내용 추출 (비동기 처리)
        semaphore = asyncio.Semaphore(3)  # 동시 요청 제한
        
        async def extract_single(url: str) -> Optional[NewsArticle]:
            async with semaphore:
                await asyncio.sleep(1)  # 요청 간격 조절
                return await extractor.extract_article_content(url)
        
        tasks = [extract_single(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        articles = []
        errors = []
        
        for i, result in enumerate(results):
            if isinstance(result, NewsArticle):
                articles.append(result)
            elif isinstance(result, Exception):
                errors.append(f"Error processing {urls[i]}: {str(result)}")
            elif result is None:
                errors.append(f"No content extracted from {urls[i]}")
        
        logger.info(f"Successfully extracted {len(articles)} articles")
        
        # 3. 카테고리 정보 추출 (LLM 사용 안함)
        processed_articles = []
        for article in articles:
            try:
                # 카테고리 추출
                category = _extract_category_info(article.url, article.title, article.content)
                
                article_dict = {
                    "title": article.title,
                    "content": article.content,
                    "url": article.url,
                    "published_date": article.published_date,
                    "author": article.author,
                    "source": article.source,
                    "category": category,
                    "crawled_at": datetime.now().isoformat()
                }
                
                processed_articles.append(article_dict)
                
            except Exception as e:
                logger.error(f"Error processing article {article.title}: {e}")
        
        # 4. 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"/Users/kai/workspace/rag-tutorial/rag/data/naver_today_news_{timestamp}.json"
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        save_data = {
            'collected_at': datetime.now().isoformat(),
            'source': 'naver_today_news',
            'total_articles': len(processed_articles),
            'articles': processed_articles,
            'errors': errors
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        return {
            "success": True,
            "articles_count": len(processed_articles),
            "sources_count": 1,
            "errors_count": len(errors),
            "articles": processed_articles,
            "errors": errors,
            "filename": filename
        }
        
    except Exception as e:
        logger.error(f"Error in Naver today news crawling: {e}")
        return {
            "success": False,
            "articles_count": 0,
            "sources_count": 0,
            "errors_count": 1,
            "articles": [],
            "errors": [str(e)]
        }

async def crawl_naver_news_with_entities(max_articles: int = 500) -> Dict:
    """네이버 뉴스를 크롤링하되 summary 생성 없이 entity 정보만 추출"""
    try:
        from naver_news_crawler import EntityExtractor
        from datetime import datetime
        
        config = NewsCrawlerConfig()
        discoverer = NewsSourceDiscoverer(config)
        extractor = NewsContentExtractor(config)
        entity_extractor = EntityExtractor()
        
        logger.info(f"Starting Naver news crawling with entity extraction (target: {max_articles})...")
        
        # 1. 오늘의 네이버 뉴스 URL 발견
        urls = await discoverer.naver_discoverer.discover_today_news_from_naver(max_articles)
        logger.info(f"Discovered {len(urls)} URLs from Naver today's news")
        
        if not urls:
            return {
                "success": False,
                "articles_count": 0,
                "sources_count": 0,
                "errors_count": 1,
                "articles": [],
                "errors": ["No URLs found from Naver news"]
            }
        
        # 2. 기사 내용 추출 (비동기 처리)
        semaphore = asyncio.Semaphore(3)  # 동시 요청 제한
        
        async def extract_single(url: str) -> Optional[NewsArticle]:
            async with semaphore:
                await asyncio.sleep(1)  # 요청 간격 조절
                return await extractor.extract_article_content(url)
        
        tasks = [extract_single(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        articles = []
        errors = []
        
        for i, result in enumerate(results):
            if isinstance(result, NewsArticle):
                articles.append(result)
            elif isinstance(result, Exception):
                errors.append(f"Error processing {urls[i]}: {str(result)}")
            elif result is None:
                errors.append(f"No content extracted from {urls[i]}")
        
        logger.info(f"Successfully extracted {len(articles)} articles")
        
        # 3. Entity 정보 추출 및 카테고리 정보 추가
        processed_articles = []
        for article in articles:
            try:
                # 엔티티 정보 추출
                entities = entity_extractor.extract_entities(article.title, article.content)
                relationships = entity_extractor.extract_relationships(article.title, article.content, entities)
                
                # 카테고리 정보 추출
                category = _extract_category_info(article.url, article.title, article.content)
                
                article_dict = {
                    "title": article.title,
                    "content": article.content,
                    "url": article.url,
                    "published_date": article.published_date,
                    "author": article.author,
                    "source": article.source,
                    "category": category,
                    "entities": entities,
                    "relationships": relationships,
                    "content_length": len(article.content),
                    "entity_count": sum(len(v) for v in entities.values()),
                    "relationship_count": len(relationships),
                    "crawled_at": datetime.now().isoformat()
                }
                
                processed_articles.append(article_dict)
                
            except Exception as e:
                logger.error(f"Error processing article {article.title}: {e}")
        
        # 4. 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"/Users/kai/workspace/rag-tutorial/rag/data/naver_news_entities_{timestamp}.json"
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        save_data = {
            'collected_at': datetime.now().isoformat(),
            'source': 'naver_news_with_entities',
            'total_articles': len(processed_articles),
            'articles': processed_articles,
            'errors': errors
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        return {
            "success": True,
            "articles_count": len(processed_articles),
            "sources_count": 1,
            "errors_count": len(errors),
            "articles": processed_articles,
            "errors": errors,
            "filename": filename
        }
        
    except Exception as e:
        logger.error(f"Error in Naver news crawling with entities: {e}")
        return {
            "success": False,
            "articles_count": 0,
            "sources_count": 0,
            "errors_count": 1,
            "articles": [],
            "errors": [str(e)]
        }

async def crawl_naver_news_only(keywords: List[str] = None, max_articles: int = 10) -> Dict:
    """네이버 뉴스만 크롤링하는 편의 함수"""
    from naver_news_crawler import crawl_naver_news_by_keyword, crawl_naver_trending_news
    
    if keywords is None:
        keywords = ["AI", "기술", "경제"]
    
    all_articles = []
    
    try:
        if keywords:
            # 키워드별 뉴스 수집
            articles_per_keyword = max_articles // len(keywords)
            for keyword in keywords:
                articles = await crawl_naver_news_by_keyword(keyword, articles_per_keyword)
                all_articles.extend(articles)
        else:
            # 트렌딩 뉴스 수집
            all_articles = await crawl_naver_trending_news(max_articles)
        
        # 중복 제거
        unique_articles = []
        seen_urls = set()
        
        for article in all_articles:
            url = article.get('url', '')
            if url and url not in seen_urls and 'error' not in article:
                seen_urls.add(url)
                unique_articles.append(article)
        
        # 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"/Users/kai/workspace/rag-tutorial/rag/data/naver_news_only_{timestamp}.json"
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        save_data = {
            'collected_at': datetime.now().isoformat(),
            'source': 'naver_news_only',
            'keywords': keywords,
            'total_articles': len(unique_articles),
            'articles': unique_articles
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        return {
            "success": True,
            "articles_count": len(unique_articles),
            "sources_count": 1,
            "errors_count": len(all_articles) - len(unique_articles),
            "articles": unique_articles,
            "errors": [],
            "filename": filename
        }
        
    except Exception as e:
        logger.error(f"Error in Naver news crawling: {e}")
        return {
            "success": False,
            "articles_count": 0,
            "sources_count": 0,
            "errors_count": 1,
            "articles": [],
            "errors": [str(e)]
        }

async def crawl_naver_top_news_by_category(max_articles_per_category: int = 20) -> Dict:
    """네이버 뉴스의 모든 카테고리에서 상위 뉴스를 수집하는 함수"""
    try:
        from datetime import datetime
        
        config = NewsCrawlerConfig()
        discoverer = NewsSourceDiscoverer(config)
        extractor = NewsContentExtractor(config)
        
        logger.info("Starting Naver top news crawling by category...")
        
        # 네이버 뉴스 카테고리 정의
        categories = {
            '정치': 'https://news.naver.com/main/list.naver?mode=LPOD&mid=sec&sid1=001',
            '경제': 'https://news.naver.com/main/list.naver?mode=LPOD&mid=sec&sid1=101', 
            '사회': 'https://news.naver.com/main/list.naver?mode=LPOD&mid=sec&sid1=102',
            '생활/문화': 'https://news.naver.com/main/list.naver?mode=LPOD&mid=sec&sid1=103',
            '세계': 'https://news.naver.com/main/list.naver?mode=LPOD&mid=sec&sid1=104',
            'IT/과학': 'https://news.naver.com/main/list.naver?mode=LPOD&mid=sec&sid1=105',
            '스포츠': 'https://news.naver.com/main/list.naver?mode=LPOD&mid=sec&sid1=100'
        }
        
        all_articles = []
        category_stats = {}
        errors = []
        
        for category_name, category_url in categories.items():
            try:
                logger.info(f"Crawling {category_name} category...")
                
                # 카테고리별 URL 발견
                urls = await _discover_urls_from_category_page(category_url, max_articles_per_category)
                logger.info(f"Found {len(urls)} URLs in {category_name} category")
                
                if not urls:
                    logger.warning(f"No URLs found for {category_name}")
                    continue
                
                # 기사 내용 추출 (비동기 처리)
                semaphore = asyncio.Semaphore(3)  # 동시 요청 제한
                
                async def extract_single(url: str) -> Optional[NewsArticle]:
                    async with semaphore:
                        await asyncio.sleep(0.5)  # 요청 간격 조절
                        return await extractor.extract_article_content(url)
                
                tasks = [extract_single(url) for url in urls]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                category_articles = []
                category_errors = []
                
                for i, result in enumerate(results):
                    if isinstance(result, NewsArticle):
                        # 카테고리 정보 확실히 설정
                        result.category = category_name
                        category_articles.append(result)
                    elif isinstance(result, Exception):
                        category_errors.append(f"Error processing {urls[i]}: {str(result)}")
                    elif result is None:
                        category_errors.append(f"No content extracted from {urls[i]}")
                
                # 통계 정보 저장
                category_stats[category_name] = {
                    'urls_found': len(urls),
                    'articles_extracted': len(category_articles),
                    'errors': len(category_errors)
                }
                
                all_articles.extend(category_articles)
                errors.extend(category_errors)
                
                logger.info(f"Successfully extracted {len(category_articles)} articles from {category_name}")
                
            except Exception as e:
                error_msg = f"Error processing {category_name} category: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
                category_stats[category_name] = {
                    'urls_found': 0,
                    'articles_extracted': 0,
                    'errors': 1
                }
        
        # 3. 기사 데이터 변환
        processed_articles = []
        for article in all_articles:
            try:
                article_dict = {
                    "title": article.title,
                    "content": article.content,
                    "url": article.url,
                    "published_date": article.published_date,
                    "author": article.author,
                    "source": article.source,
                    "category": article.category,
                    "crawled_at": datetime.now().isoformat()
                }
                
                processed_articles.append(article_dict)
                
            except Exception as e:
                logger.error(f"Error processing article {article.title}: {e}")
        
        # 4. 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"/Users/kai/workspace/rag-tutorial/rag/data/naver_top_news_by_category_{timestamp}.json"
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        save_data = {
            'collected_at': datetime.now().isoformat(),
            'source': 'naver_top_news_by_category',
            'total_articles': len(processed_articles),
            'category_stats': category_stats,
            'articles': processed_articles,
            'errors': errors
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        return {
            "success": True,
            "articles_count": len(processed_articles),
            "categories_count": len([c for c in category_stats.values() if c['articles_extracted'] > 0]),
            "category_stats": category_stats,
            "errors_count": len(errors),
            "articles": processed_articles,
            "errors": errors,
            "filename": filename
        }
        
    except Exception as e:
        logger.error(f"Error in Naver top news by category crawling: {e}")
        return {
            "success": False,
            "articles_count": 0,
            "categories_count": 0,
            "errors_count": 1,
            "articles": [],
            "errors": [str(e)]
        }

async def _discover_urls_from_category_page(category_url: str, max_articles: int = 20) -> List[str]:
    """특정 카테고리 페이지에서 기사 URL들을 발견하는 함수"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(category_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # 네이버 뉴스 카테고리 페이지의 기사 링크 찾기
        article_links = []
        
        # 다양한 선택자로 기사 링크 찾기
        link_selectors = [
            'a[href*="/main/read.naver"]',      # 기본 네이버 뉴스 링크
            'a[href*="/article/"]',             # 새로운 네이버 뉴스 링크  
            '.cluster_text a',                  # 클러스터 텍스트 링크
            '.headline a',                      # 헤드라인 링크
            '.list_body a',                     # 리스트 바디 링크
            '.cluster_head a',                  # 클러스터 헤드 링크
            '.AKS-more-news a',                 # 더보기 뉴스 링크
        ]
        
        for selector in link_selectors:
            links = soup.select(selector)
            for link in links:
                href = link.get('href', '')
                if href:
                    # 상대 경로를 절대 경로로 변환
                    if href.startswith('/'):
                        full_url = 'https://news.naver.com' + href
                    elif href.startswith('http'):
                        full_url = href
                    else:
                        continue
                    
                    # 네이버 뉴스 기사 URL인지 확인
                    if ('news.naver.com' in full_url and 
                        ('/article/' in full_url or '/main/read.naver' in full_url)):
                        article_links.append(full_url)
                        
                        if len(article_links) >= max_articles:
                            break
            
            if len(article_links) >= max_articles:
                break
        
        # 중복 제거
        unique_links = list(set(article_links))[:max_articles]
        
        logger.info(f"Found {len(unique_links)} unique article URLs from category page")
        return unique_links
        
    except Exception as e:
        logger.error(f"Error discovering URLs from category page {category_url}: {e}")
        return []

if __name__ == "__main__":
    # Example usage
    async def main():
        result = await crawl_news(max_articles_per_source=50)
        print(f"Crawled {result['articles_count']} articles from {result['sources_count']} sources")
        
        if result['errors']:
            print(f"Encountered {result['errors_count']} errors")
    
    asyncio.run(main())