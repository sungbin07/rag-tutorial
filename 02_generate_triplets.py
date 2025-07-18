# langfuse
from langfuse.langchain import CallbackHandler
import os
import argparse
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import json
from collections import defaultdict

from dotenv import load_dotenv
load_dotenv()

# Initialize Langfuse handler only if keys are available
langfuse_handler = None
# if os.getenv('LANGFUSE_PUBLIC_KEY') and os.getenv('LANGFUSE_SECRET_KEY'):
    # langfuse_handler = CallbackHandler()

tri_prompt = """
다음 뉴스 기사 전체를 읽고, 이 안에서 발생하는 현상, 원인, 정책, 기업, 개념, 상품, 인물 등의 주요 개체들 간  
인과관계, 설명, 무관 관계 등을 추론해서 아래 형식으로 구조화해줘.
---
🎯 출력 형식 (정확하게 지켜줘야 함):

(주어:노드타입)-[관계]->(목적어:노드타입)
---
📌 허용 노드 타입:
- Phenomenon: 관찰된 현상 또는 결과 (예: 판매 감소, 주가 상승)
- Cause: 원인이 되는 요소 (예: 금리 상승, 소비 위축)
- Concept: 설명을 위한 추상 개념 (예: 구매력, 수요 심리)
- Policy: 정책, 제도 (예: 금리 인상, 대출 규제 완화)
- Product: 상품, 서비스 (예: 테슬라 차량, 전기차)
- Company: 기업, 조직 (예: 테슬라, 롯데)
- Person: 인물 (예: 일론 머스크, 신동빈)
- Time: 특정 시점 또는 시기 (예: 2025년 7월, 최근)
---
📌 허용 관계:
- 원인이다
- 결과이다
- 설명한다
- 무관하다
- 관련 있다
- 소속이다
- 영향을 준다
---
✋ 반드시 지켜야 할 규칙:
1. 뉴스에 명시된 정보와, 일반적 상식에 기반한 추론 가능한 관계만 포함해줘
2. 관계가 불명확하거나 무작위적인 것은 제외해줘
3. 동일 문서에서 발생한 다중 관계는 모두 포함해도 돼 (예: 여러 원인 → 하나의 결과)

---

📰 뉴스 기사 전체 내용:

{document_text}
"""

# Global node normalization prompt
global_normalization_prompt = """
다음은 여러 뉴스 기사에서 추출된 모든 그래프 triplet들입니다. 
전체 데이터셋을 분석하여 동일한 의미를 가지지만 다른 표현으로 나타난 노드들을 정규화해주세요.

정규화 규칙:
1. 의미가 같은 노드들은 하나의 표준 형태로 통일
2. 가장 간결하고 명확한 표현을 선택
3. 기업명, 인명은 정확한 공식 명칭 사용
4. 현상이나 개념은 핵심 의미를 담은 간결한 표현 사용
5. 노드 타입은 변경하지 않음
6. 전체 데이터셋에서 일관성 있게 적용

전체 triplet들:
{all_triplets}

출력 형식:
1. 먼저 정규화 매핑을 제시해주세요:
[정규화 매핑]
원본노드:타입 → 정규화노드:타입

2. 그 다음 정규화된 모든 triplet들을 출력해주세요:
[정규화된 Triplets]
(노드:타입)-[관계]->(노드:타입)
"""

def process_single_article(article, category, article_num, total_articles, chain, langfuse_handler):
    """Process a single article to generate triplets"""
    try:
        print(f"🔄 Processing article {article_num} in {category}...")
        
        if 'content' not in article:
            print(f"❌ No content found in article {article_num}")
            return None
            
        document_text = article.get('content', '')
        if not document_text.strip():
            print(f"❌ Empty content in article {article_num}")
            return None
        
        # Truncate very long articles to speed up processing
        max_chars = 4000  # Increased limit for better quality
        if len(document_text) > max_chars:
            document_text = document_text[:max_chars] + "..."
            print(f"⚠️  Article {article_num} truncated to {max_chars} characters")
            
        # Invoke the chain
        config = {}
        if langfuse_handler:
            config["callbacks"] = [langfuse_handler]
        
        result = chain.invoke({"document_text": document_text}, config=config)
        
        # Extract content from result
        if hasattr(result, 'content'):
            triplet_output = result.content
        else:
            triplet_output = str(result)
        
        article_title = article.get('title', 'No title')
        
        # Count triplets generated
        triplet_count = len(re.findall(r'\([^:]+:[^)]+\)-\[[^\]]+\]->\([^:]+:[^)]+\)', triplet_output))
        
        print(f"✅ Article {article_num}: {article_title[:40]}... ({triplet_count} triplets)")
        
        return {
            'title': article_title,
            'category': category,
            'original': triplet_output,
            'article_id': f"{category}_{article_num}",
            'triplet_count': triplet_count,
            'url': article.get('url', ''),
            'published_date': article.get('published_date', ''),
            'source': article.get('source', '')
        }
        
    except Exception as e:
        print(f"❌ Error processing article {article_num}: {e}")
        return None

def process_articles_parallel(articles_by_category, chain, langfuse_handler, max_workers=4):
    """Process articles in parallel to speed up triplet generation"""
    all_triplets_data = []
    processed_count = 0
    error_count = 0
    
    # Prepare tasks
    tasks = []
    total_articles = sum(len(articles) for articles in articles_by_category.values())
    
    article_counter = 0
    for category, category_articles in articles_by_category.items():
        for i, article in enumerate(category_articles):
            article_counter += 1
            task = (article, category, article_counter, total_articles, chain, langfuse_handler)
            tasks.append(task)
    
    # Process in parallel
    print(f"🚀 Processing {total_articles} articles with {max_workers} parallel workers...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(process_single_article, *task): task 
            for task in tasks
        }
        
        # Process completed tasks
        for future in as_completed(future_to_task):
            try:
                result = future.result()
                if result:
                    all_triplets_data.append(result)
                    processed_count += 1
                else:
                    error_count += 1
                    
                # Progress indicator
                total_completed = processed_count + error_count
                if total_completed % 3 == 0:
                    print(f"📊 Progress: {total_completed}/{total_articles} completed ({(total_completed/total_articles)*100:.1f}%)")
                    
            except Exception as e:
                print(f"❌ Task failed: {e}")
                error_count += 1
    
    return all_triplets_data, processed_count, error_count

def perform_global_normalization(all_triplets_data, llm):
    """Perform global normalization across all articles"""
    print("\n🔄 Global Node Normalization 시작...")
    
    # Combine all triplets
    combined_triplets = ""
    for item in all_triplets_data:
        combined_triplets += item['original'] + "\n"
    
    # Perform global normalization
    global_norm_template = PromptTemplate.from_template(global_normalization_prompt)
    global_norm_chain = global_norm_template | llm
    
    try:
        result = global_norm_chain.invoke({"all_triplets": combined_triplets})
        
        if hasattr(result, 'content'):
            normalized_output = result.content
        else:
            normalized_output = str(result)
        
        return normalized_output
    except Exception as e:
        print(f"Error in global normalization: {e}")
        return combined_triplets

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='News Triplet Generator')
    parser.add_argument('--limit', '-l', type=int, default=None, 
                       help='Limit number of articles to process (for testing). Use -1 or omit for all articles.')
    parser.add_argument('--categories', '-c', nargs='+', default=None,
                       help='Specific categories to process (e.g., --categories 정치 경제)')
    parser.add_argument('--output', '-o', type=str, default='triplets_output.json',
                       help='Output file name')
    parser.add_argument('--parallel-workers', type=int, default=4,
                       help='Number of parallel workers for article processing (default: 4)')
    parser.add_argument('--skip-normalization', action='store_true',
                       help='Skip global normalization step (faster, but less unified)')
    
    args = parser.parse_args()
    
    # Initialize components
    doc_reasoning_prompt = PromptTemplate.from_template(tri_prompt)
    llm = ChatOpenAI(model="gpt-4.1", temperature=0)
    chain = doc_reasoning_prompt | llm 

    # Read news data
    data_file = '/Users/kai/workspace/rag-tutorial/rag/data/naver_top_news_by_category_20250717_230748.json'
    
    if not os.path.exists(data_file):
        print(f"Data file not found: {data_file}")
        return
    
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading data file: {e}")
        return

    # Process articles with optional limitations
    if 'articles' not in data:
        print("No articles found in data")
        return
    
    articles = data['articles']
    total_articles = len(articles)
    
    # Filter by categories if specified
    if args.categories:
        filtered_articles = [article for article in articles if article.get('category', 'Unknown') in args.categories]
        print(f"Filtered to {len(filtered_articles)} articles from categories: {args.categories}")
        articles = filtered_articles
        total_articles = len(articles)
    
    # Apply limit if specified
    if args.limit and args.limit > 0:
        articles = articles[:args.limit]
        total_articles = len(articles)
        print(f"🧪 TEST MODE: Limited to {total_articles} articles")
    elif args.limit == -1:
        print(f"🚀 FULL MODE: Processing all {total_articles} articles")
    else:
        print(f"🚀 FULL MODE: Processing all {total_articles} articles")
    
    print(f"Found {total_articles} articles to process")
    
    # Group articles by category for processing
    articles_by_category = {}
    for article in articles:
        category = article.get('category', 'Unknown')
        if category not in articles_by_category:
            articles_by_category[category] = []
        articles_by_category[category].append(article)
    
    print(f"Categories found: {list(articles_by_category.keys())}")
    print(f"Articles per category: {[(cat, len(arts)) for cat, arts in articles_by_category.items()]}")
    
    processing_mode = "TEST" if args.limit and args.limit > 0 else "FULL"
    print(f"\n📝 Phase 1: {processing_mode} MODE - Triplet 생성 (병렬 처리, {args.parallel_workers}개 워커)")
    print("=" * 80)
    
    # Phase 1: Generate triplets from articles using parallel processing
    start_time = time.time()
    all_triplets_data, processed_count, error_count = process_articles_parallel(
        articles_by_category, chain, langfuse_handler, args.parallel_workers
    )
    end_time = time.time()
    
    processing_time = end_time - start_time
    total_triplets = sum([item['triplet_count'] for item in all_triplets_data])
    
    print(f"\n✅ Phase 1 완료 ({processing_time:.1f}초):")
    print(f"   - 성공적으로 처리된 기사: {processed_count}/{total_articles}")
    print(f"   - 오류 발생: {error_count}개")
    print(f"   - 총 생성된 triplet: {total_triplets}")
    print(f"   - 평균 처리 시간: {processing_time/total_articles:.1f}초/기사")
    print(f"   - 처리 속도: {total_articles/processing_time:.1f}기사/분")
    
    # Save all_triplets_data json list
    
    triplets_output_file = f"triplets_news_data.json"
    with open(triplets_output_file, 'w', encoding='utf-8') as f:
        json.dump(all_triplets_data, f, ensure_ascii=False, indent=2)
    print(f"💾 Triplets data saved to: {triplets_output_file}")
    
    # Phase 2: Global normalization (optional)
    global_normalized_result = None
    if not args.skip_normalization and all_triplets_data:
        print(f"\n📝 Phase 2: 글로벌 노드 정규화")
        print("=" * 80)
        
        normalization_start = time.time()
        
        # Count total triplets for optimization decision
        print(f"📊 총 {total_triplets}개 triplet 정규화 예정")
        
        if total_triplets <= 800:  # Reasonable limit for single LLM call
            print("🔄 Single global normalization processing...")
            global_normalized_result = perform_global_normalization(all_triplets_data, llm)
        else:
            print(f"⚠️  Large dataset ({total_triplets} triplets), using category-based batching...")
            # Category-based processing for large datasets
            category_results = []
            
            categories_triplets = {}
            for item in all_triplets_data:
                cat = item['category']
                if cat not in categories_triplets:
                    categories_triplets[cat] = []
                categories_triplets[cat].append(item)
            
            print(f"🔄 Processing {len(categories_triplets)} categories separately...")
            
            for category, cat_items in categories_triplets.items():
                cat_triplet_count = sum([item['triplet_count'] for item in cat_items])
                print(f"   Processing {category}: {len(cat_items)} articles, {cat_triplet_count} triplets")
                
                cat_result = perform_global_normalization(cat_items, llm)
                category_results.append(f"=== {category} ===\n{cat_result}")
            
            # Combine category results
            global_normalized_result = "\n\n".join(category_results)
        
        normalization_time = time.time() - normalization_start
        print(f"\n✅ Phase 2 완료 ({normalization_time:.1f}초)")
    
    # Save results
    final_results = {
        'processing_summary': {
            'mode': processing_mode,
            'limit_applied': args.limit,
            'categories_filter': args.categories,
            'total_articles_found': len(data['articles']),
            'total_articles_processed': processed_count,
            'total_errors': error_count,
            'success_rate': f"{(processed_count/total_articles)*100:.1f}%",
            'categories': list(set([item['category'] for item in all_triplets_data])),
            'articles_by_category': {cat: len([item for item in all_triplets_data if item['category'] == cat]) 
                                   for cat in set([item['category'] for item in all_triplets_data])},
            'total_triplets_generated': total_triplets,
            'processing_time_seconds': processing_time,
            'average_time_per_article': processing_time/total_articles if total_articles > 0 else 0,
            'articles_per_minute': total_articles/processing_time*60 if processing_time > 0 else 0,
            'normalization_applied': not args.skip_normalization and global_normalized_result is not None,
            'parallel_workers': args.parallel_workers
        },
        'individual_triplets': all_triplets_data,
        'global_normalized_result': global_normalized_result,
        'processing_metadata': {
            'model_used': 'gpt-4.1',
            'max_article_length': 4000,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'command_args': vars(args)
        }
    }
    
    # Save results to file
    output_file = args.output
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Results saved to {output_file}")
        print(f"   File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
    except Exception as e:
        print(f"❌ Error saving results: {e}")
    
    print(f"\n🎉 {processing_mode} 처리 완료!")
    print(f"   📊 총 {processed_count}개 기사 처리 완료")
    print(f"   📈 성공률: {(processed_count/total_articles)*100:.1f}%")
    print(f"   🔗 총 생성된 관계: {total_triplets}개")
    print(f"   ⚡ 처리 속도: {total_articles/processing_time*60:.1f}기사/분")

if __name__ == "__main__":
    main() 