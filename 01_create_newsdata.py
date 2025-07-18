"""
네이버 뉴스 카테고리별 상위 뉴스 수집 테스트
"""

import asyncio
from news_crawler import crawl_naver_top_news_by_category

async def news_crawling():
    """카테고리별 상위 뉴스 수집 테스트"""
    print("🚀 네이버 뉴스 카테고리별 상위 뉴스 수집 테스트 시작")
    print("=" * 60)
    
    max_articles = int(input("📝 카테고리별 수집할 기사 개수를 입력하세요 (1-50): "))
    
    print(f"📊 카테고리별 {max_articles}개씩 기사 수집을 시작합니다...")
    print("-" * 60)
    
    # 카테고리별 뉴스 수집
    result = await crawl_naver_top_news_by_category(max_articles_per_category=max_articles)
    
    if result['success']:
        print(f"✅ 성공적으로 총 {result['articles_count']}개 기사 수집")
        print(f"📊 수집된 카테고리: {result['categories_count']}개")
        print(f"❌ 에러 수: {result['errors_count']}개")
        
        print("\n📈 카테고리별 통계:")
        print("-" * 50)
        for category, stats in result['category_stats'].items():
            print(f"  {category}:")
            print(f"    - URL 발견: {stats['urls_found']}개")
            print(f"    - 기사 추출: {stats['articles_extracted']}개")
            print(f"    - 에러: {stats['errors']}개")
        
        print(f"\n📁 저장된 파일: {result['filename']}")
        
        print("\n📰 카테고리별 기사 샘플:")
        print("-" * 50)
        
        # 카테고리별로 기사 정리
        by_category = {}
        for article in result['articles']:
            category = article['category']
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(article)
        
        for category, articles in by_category.items():
            print(f"\n🏷️ {category} ({len(articles)}개):")
            for i, article in enumerate(articles[:3]):  # 각 카테고리에서 3개만 표시
                print(f"  {i+1}. {article['title'][:50]}...")
                print(f"     언론사: {article['source']}")
                print(f"     작성자: {article['author']}")
                print(f"     날짜: {article['published_date']}")
        
        print(f"\n🎯 전체 기사 중 정규화 성공률:")
        print("-" * 50)
        
        # 정규화 성공률 계산
        total_articles = len(result['articles'])
        with_date = len([a for a in result['articles'] if a['published_date']])
        with_author = len([a for a in result['articles'] if a['author']])
        with_source = len([a for a in result['articles'] if a['source']])
        
        print(f"  날짜 정규화: {with_date}/{total_articles} ({with_date/total_articles*100:.1f}%)")
        print(f"  작성자 정규화: {with_author}/{total_articles} ({with_author/total_articles*100:.1f}%)")
        print(f"  언론사 추출: {with_source}/{total_articles} ({with_source/total_articles*100:.1f}%)")
        
    else:
        print("❌ 크롤링 실패")
        print(f"에러: {result['errors']}")

if __name__ == "__main__":
    asyncio.run(news_crawling()) 