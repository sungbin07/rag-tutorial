from typing import TypedDict, List, Literal, Optional, Dict
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt
from neo4j import GraphDatabase
import re
import os
from dotenv import load_dotenv

load_dotenv()

# Enhanced State 정의 - Memory와 HITL 지원
class EnhancedAgentState(TypedDict):
    question: str
    cypher: str
    result: List[dict]
    summary: str
    # Memory 관련
    conversation_history: List[dict]
    user_preferences: dict
    # Quality control
    quality_score: int  # 1-5 scale
    needs_human_review: bool
    human_feedback: Optional[str]
    # Loop control
    iteration_count: int
    max_iterations: int
    retry_reason: Optional[str]
    # Self-reflection
    confidence_score: float
    result_relevance: str  # "high", "medium", "low"
    needs_refinement: bool

class GraphQAAgent:
    def __init__(self, llm, driver: GraphDatabase.driver):
        self.llm = llm
        self.driver = driver
        self.prompt = PromptTemplate.from_template("""
너는 뉴스 기반 그래프 데이터를 다루는 분석가야.
다음 질문에 답하기 위해 필요한 Cypher 쿼리를 생성하고, 그 결과를 요약해줘.

질문: {question}
1. 관련 Cypher 쿼리 생성
2. 결과를 바탕으로 자연어로 요약

아웃풋 형식:
- Cypher: <쿼리>
- 요약: <결과 요약>
""")

    def run(self, state: EnhancedAgentState) -> EnhancedAgentState:
        prompt = self.prompt.format(question=state["question"])
        response = self.llm.invoke(prompt).content
        
        cypher = extract_cypher(response)
        
        try:
            with self.driver.session() as session:
                result = session.run(cypher).data()
        except Exception as e:
            result = [{"error": str(e)}]
        
        return {
            "cypher": cypher,
            "result": result,
            "summary": response
        }

def extract_cypher(text: str) -> str:
    """텍스트에서 Cypher 쿼리 추출"""
    # ```cypher 코드 블록 내부의 쿼리 추출
    match = re.search(r'```cypher\s*\n?(.*?)\n?```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # ``` 코드 블록 내부의 쿼리 추출 (언어 지정 없음)
    match = re.search(r'```\s*\n?(MATCH.*?)\n?```', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Cypher: 이후의 텍스트 추출 (코드 블록 제외)
    match = re.search(r'Cypher:\s*(?:```cypher\s*)?\n?(MATCH.*?)(?:\n```|\n요약:|$)', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # MATCH로 시작하는 쿼리 찾기
    match = re.search(r'(MATCH.*?RETURN.*?)(?:\n|$)', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    return "MATCH (n) RETURN n LIMIT 5"  # 기본 쿼리

# 기존 hallucination 함수들 제거됨 - 이제 실제 데이터만 사용

# LLM 초기화
def get_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Neo4j 연결
def get_neo4j_driver():
    print("🔗 Connecting to Neo4j...")
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    print(uri, username, password)
    try:
        driver = GraphDatabase.driver(uri, auth=(username, password))
        return driver
    except Exception as e:
        print(f"Neo4j 연결 실패: {e}")
        return None

# 데이터베이스 상태 확인 함수
def check_neo4j_status(driver):
    """Neo4j 데이터베이스의 현재 상태 확인"""
    try:
        with driver.session() as session:
            print("=== Neo4j 데이터베이스 상태 확인 ===")
            
            # 노드 타입별 개수 확인
            node_result = session.run("""
                MATCH (n) 
                RETURN labels(n)[0] as label, count(n) as count
                ORDER BY count DESC
                LIMIT 10
            """)
            
            print("\n📊 노드 타입별 개수:")
            has_data = False
            for record in node_result:
                print(f"   {record['label']}: {record['count']}개")
                has_data = True
            
            if not has_data:
                print("   데이터가 없습니다.")
                return False
            
            # 관계 타입별 개수 확인
            rel_result = session.run("""
                MATCH ()-[r]->() 
                RETURN type(r) as relationship, count(r) as count
                ORDER BY count DESC
                LIMIT 10
            """)
            
            print("\n🔗 관계 타입별 개수:")
            for record in rel_result:
                print(f"   {record['relationship']}: {record['count']}개")
            
            # 샘플 데이터 확인 (수정된 부분)
            sample_result = session.run("""
                MATCH (n) 
                WHERE n.name IS NOT NULL
                RETURN labels(n)[0] as type, n.name as name
                LIMIT 5
            """)
            
            print("\n📋 샘플 데이터:")
            for record in sample_result:
                print(f"   {record['type']}: {record['name']}")
            
            return True
    except Exception as e:
        print(f"데이터베이스 상태 확인 중 오류: {e}")
        return False

def get_schema_info(driver):
    """Neo4j 스키마 정보 수집"""
    try:
        with driver.session() as session:
            # 노드 레이블들
            labels_result = session.run("CALL db.labels()")
            labels = [record["label"] for record in labels_result]
            
            # 관계 타입들
            rel_result = session.run("CALL db.relationshipTypes()")
            relationships = [record["relationshipType"] for record in rel_result]
            
            # 샘플 노드 속성들
            sample_props = {}
            for label in labels[:5]:  # 상위 5개 레이블만 확인
                try:
                    prop_result = session.run(f"MATCH (n:{label}) RETURN keys(n) as props LIMIT 1")
                    props = prop_result.single()
                    if props:
                        sample_props[label] = props["props"]
                except:
                    pass
            
            return {
                "labels": labels,
                "relationships": relationships,
                "sample_properties": sample_props
            }
    except Exception as e:
        print(f"스키마 정보 수집 오류: {e}")
        return {"labels": [], "relationships": [], "sample_properties": {}}

def generate_cypher_with_llm(question: str, schema_info: dict) -> str:
    """LLM을 사용하여 완전 동적 Cypher 쿼리 생성"""
    llm = get_llm()
    
    # 스키마 정보를 더 자세히 포맷팅
    labels_str = ", ".join(schema_info.get("labels", [])[:15])  # 상위 15개
    relationships_str = ", ".join(schema_info.get("relationships", [])[:15])  # 상위 15개
    
    prompt = f"""
당신은 Neo4j Cypher 쿼리 전문가입니다. 
주어진 자연어 질문을 분석하여 가장 적절한 Cypher 쿼리를 생성해주세요.

=== 데이터베이스 스키마 정보 ===
노드 레이블: {labels_str}
관계 타입: {relationships_str}

=== 중요한 가이드라인 ===
1. 질문의 의도를 파악하여 적절한 쿼리 패턴을 선택하세요
2. 키워드는 CONTAINS를 사용하여 부분 매칭하세요
3. 인과관계나 연관성을 찾는 질문이면 multi-hop 패턴을 사용하세요
4. 단순 정보 검색이면 키워드 기반 검색을 사용하세요
5. 모든 결과는 10개 이하로 제한하세요

=== 질문 분석 ===
질문: "{question}"

위 질문을 분석하여 다음 중 가장 적절한 패턴을 선택하고 실제 키워드로 채워서 완성된 Cypher 쿼리를 생성하세요:

**패턴 1: 3-hop 인과관계 탐색 (원인 → 중간결과 → 최종결과)**
```
MATCH (a)-[:원인이다]->(b)-[:원인이다]->(c)
WHERE [키워드 조건들]
RETURN a.name as 원인1, b.name as 중간결과, c.name as 최종결과
LIMIT 10
```

**패턴 2: 2-hop 관계 탐색 (시작 → 결과)**
```
MATCH (a)-[:원인이다|관련_있다|결과이다]->(b)
WHERE [키워드 조건들]
RETURN a.name as 시작, b.name as 결과
LIMIT 10
```

**패턴 3: 키워드 중심 노드 검색**
```
MATCH (n)
WHERE [키워드 조건들]
RETURN labels(n)[0] as 타입, n.name as 이름
LIMIT 10
```

**패턴 4: 특정 엔티티와 연결된 모든 관계 탐색**
```
MATCH (center)-[r]-(connected)
WHERE [중심 노드 조건]
RETURN center.name as 중심, type(r) as 관계, connected.name as 연결된_엔티티
LIMIT 10
```

질문의 의도를 파악하여 가장 적절한 패턴을 선택하고, 질문에서 추출한 키워드들로 WHERE 조건을 구성하세요.
키워드 조건은 n.name CONTAINS '키워드' 형태로 작성하세요.

완성된 Cypher 쿼리:
"""
    
    try:
        response = llm.invoke(prompt).content
        
        # 응답에서 쿼리 추출
        cypher = extract_cypher_from_response(response)
        
        # 기본 검증
        if not cypher or not ("MATCH" in cypher.upper() and "RETURN" in cypher.upper()):
            # LLM 응답이 유효하지 않으면 일반적인 키워드 검색으로 fallback
            keywords = extract_keywords_from_question(question)
            if keywords:
                keyword_conditions = " OR ".join([f"n.name CONTAINS '{kw}'" for kw in keywords[:3]])
                cypher = f"""
MATCH (n)
WHERE {keyword_conditions}
RETURN labels(n)[0] as 타입, n.name as 이름
LIMIT 10
"""
            else:
                cypher = """
MATCH (n)
WHERE n.name IS NOT NULL
RETURN labels(n)[0] as 타입, n.name as 이름
LIMIT 5
"""
        
        return cypher.strip()
        
    except Exception as e:
        print(f"LLM 쿼리 생성 오류: {e}")
        # 최종 fallback
        return """
MATCH (n)
WHERE n.name IS NOT NULL
RETURN labels(n)[0] as 타입, n.name as 이름
LIMIT 5
"""

def extract_cypher_from_response(response: str) -> str:
    """LLM 응답에서 Cypher 쿼리 추출"""
    # 다양한 패턴으로 쿼리 추출 시도
    patterns = [
        r'```cypher\s*(.*?)\s*```',
        r'```\s*(MATCH.*?)\s*```',
        r'완성된 Cypher 쿼리:\s*(.*?)(?:\n\n|\n$|$)',
        r'(MATCH.*?LIMIT\s+\d+)',
        r'(MATCH.*?RETURN.*?)(?:\n|$)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            query = match.group(1).strip()
            # 기본 검증
            if "MATCH" in query.upper() and "RETURN" in query.upper():
                return query
    
    return ""

def get_appropriate_cypher_query(question: str, schema_info: dict) -> str:
    """완전히 LLM 기반 쿼리 생성 - rule 제거"""
    
    # LLM으로만 쿼리 생성
    cypher = generate_cypher_with_llm(question, schema_info)
    
    print(f"🤖 LLM이 생성한 쿼리: {cypher[:100]}...")
    
    return cypher

def extract_keywords_from_question(question: str) -> list:
    """질문에서 핵심 키워드 추출"""
    # 불용어 제거
    stop_words = {'은', '는', '이', '가', '을', '를', '에', '에서', '로', '으로', '와', '과', 
                  '의', '이다', '입니다', '무엇', '어떤', '왜', '어떻게', '언제', '어디서',
                  '때문', '이유', '원인', '결과', '있다', '있는', '하는', '한다', '됩니다'}
    
    # 간단한 키워드 추출
    words = question.replace('?', '').replace('.', '').split()
    keywords = [word for word in words if word not in stop_words and len(word) > 1]
    
    return keywords[:5]  # 최대 5개 키워드만

def create_reasoning_chain(result_data: list) -> str:
    """검색 결과를 바탕으로 추론 체인 생성"""
    if not result_data:
        return "관련 데이터를 찾을 수 없습니다."
    
    chains = []
    for item in result_data:
        # 다양한 필드명 패턴 처리
        if item.get('최종결과'):  # 3-hop
            chain = f"{item.get('원인1', 'Unknown')} → {item.get('중간결과', 'Unknown')} → {item['최종결과']}"
        elif item.get('중간결과'):  # 2-hop with 중간결과
            chain = f"{item.get('원인1', 'Unknown')} → {item['중간결과']}"
        elif item.get('결과'):  # 2-hop with 결과
            chain = f"{item.get('시작', 'Unknown')} → {item['결과']}"
        elif item.get('이름'):  # 단일 노드 결과
            chain = f"{item.get('타입', 'Unknown')}: {item['이름']}"
        else:
            # 기타 필드들을 동적으로 처리
            values = [str(v) for k, v in item.items() if v is not None and str(v) != 'None']
            chain = " → ".join(values) if values else "데이터 없음"
        
        chains.append(chain)
    
    return "\n".join([f"• {chain}" for chain in chains[:5]])  # 최대 5개만

# 노드 함수들
def qa_node(state: EnhancedAgentState) -> EnhancedAgentState:
    """QA 노드 - 동적 쿼리 생성 버전"""
    llm = get_llm()
    driver = get_neo4j_driver()
    
    if driver is None:
        return {
            "cypher": "CONNECTION_ERROR",
            "result": [{"error": "Neo4j 연결 실패"}],
            "summary": "데이터베이스 연결에 실패했습니다."
        }
    
    # 데이터베이스 상태 및 스키마 정보 확인
    has_data = check_neo4j_status(driver)
    
    if not has_data:
        driver.close()
        return {
            "cypher": "NO_DATA",
            "result": [{"error": "데이터베이스에 데이터가 없습니다"}],
            "summary": "Neo4j 데이터베이스가 비어있습니다. 데이터를 먼저 로드해주세요."
        }
    
    # 스키마 정보 수집
    schema_info = get_schema_info(driver)
    print(f"\n🔍 스키마 정보: {len(schema_info['labels'])}개 레이블, {len(schema_info['relationships'])}개 관계")
    
    # 질문에 맞는 동적 쿼리 생성
    cypher = get_appropriate_cypher_query(state["question"], schema_info)
    print(f"\n📝 생성된 쿼리:\n{cypher}")
    
    try:
        with driver.session() as session:
            result = session.run(cypher).data()
            print(f"\n📊 검색 결과: {len(result)}개 항목 발견")
    except Exception as e:
        print(f"\n❌ 쿼리 실행 오류: {e}")
        result = [{"error": str(e)}]
    
    driver.close()
    
    # LLM으로 결과 요약
    if result and len(result) > 0 and "error" not in str(result[0]):
        summary_prompt = f"""
질문: {state["question"]}

Neo4j 검색 결과:
{result[:5]}  # 처음 5개만 요약에 사용

위 Neo4j 그래프 데이터베이스의 검색 결과를 바탕으로 질문에 대한 자세하고 정확한 답변을 생성해주세요.
검색된 실제 데이터의 관계와 패턴을 분석하여 설명해주세요.
"""
        summary = llm.invoke(summary_prompt).content
    else:
        summary = f"질문 '{state['question']}'에 대한 관련 데이터를 찾을 수 없습니다. 다른 키워드로 다시 시도해보세요."
    
    return {
        "cypher": cypher,
        "result": result,
        "summary": summary
    }

def fetch_source_articles(state: EnhancedAgentState, driver) -> List[Dict]:
    """검색 결과와 관련된 실제 뉴스 원본 기사들을 가져오는 함수"""
    
    # 검색 결과에서 관련 엔티티들 추출
    search_results = state.get("result", [])
    if not search_results:
        return []
    
    # 검색 결과에서 엔티티 이름들 추출
    entity_names = set()
    for result in search_results:
        for key, value in result.items():
            if isinstance(value, str) and value.strip():
                entity_names.add(value.strip())
    
    if not entity_names:
        return []
    
    # 관련 Article 노드들을 찾아서 실제 뉴스 원본 가져오기
    entity_list = list(entity_names)[:10]  # 최대 10개만
    
    # Article 노드가 있는지 확인하고 가져오기
    try:
        with driver.session() as session:
            # 1. 직접 Article 타입이 있는지 확인
            article_query = """
            MATCH (a:Article)
            WHERE any(entity IN $entities WHERE a.title CONTAINS entity OR a.content CONTAINS entity)
            RETURN a.title as title, a.content as content, a.published_date as date, a.url as url
            LIMIT 5
            """
            
            articles = session.run(article_query, {"entities": entity_list}).data()
            
            if articles:
                return articles
            
            # 2. Article이 없으면 엔티티와 연결된 기사 찾기 (MENTIONED_IN 관계)
            related_article_query = """
            MATCH (entity)-[:MENTIONED_IN]->(a:Article)
            WHERE entity.name IN $entities
            RETURN DISTINCT a.title as title, a.content as content, a.published_date as date, a.url as url
            LIMIT 5
            """
            
            articles = session.run(related_article_query, {"entities": entity_list}).data()
            
            if articles:
                return articles
            
            # 3. 마지막으로 일반적인 키워드 검색으로 관련 기사 찾기
            keyword_query = """
            MATCH (a)
            WHERE a.title IS NOT NULL AND a.content IS NOT NULL
            AND any(entity IN $entities WHERE a.title CONTAINS entity OR a.content CONTAINS entity)
            RETURN a.title as title, a.content as content, 
                   COALESCE(a.published_date, a.date, 'Unknown') as date, 
                   COALESCE(a.url, 'No URL') as url
            LIMIT 5
            """
            
            articles = session.run(keyword_query, {"entities": entity_list}).data()
            return articles
            
    except Exception as e:
        print(f"❌ 원본 기사 검색 중 오류: {e}")
        return []

def reasoning_node(state: EnhancedAgentState) -> EnhancedAgentState:
    """추론 노드 - 실제 검색 결과와 뉴스 원본 기반 분석만 수행"""
    llm = get_llm()
    driver = get_neo4j_driver()
    
    search_results = state.get("result", [])
    question = state["question"]
    
    # 실제 뉴스 원본 기사들 가져오기
    source_articles = fetch_source_articles(state, driver) if driver else []
    
    if not search_results and not source_articles:
        return {
            "summary": f"{state.get('summary', '')}\n\n❌ 분석할 데이터가 없습니다. 검색 결과를 확인해주세요."
        }
    
    # 검색 결과 요약
    results_summary = []
    for i, result in enumerate(search_results[:5], 1):
        result_str = " | ".join([f"{k}: {v}" for k, v in result.items() if v])
        results_summary.append(f"{i}. {result_str}")
    
    # 원본 기사 내용 요약 (첫 200자만)
    articles_summary = []
    for i, article in enumerate(source_articles[:3], 1):
        title = article.get('title', '제목 없음')[:100]
        content_preview = article.get('content', '')[:200]
        date = article.get('date', '날짜 미상')
        articles_summary.append(f"{i}. [{date}] {title}\n   내용: {content_preview}...")
    
    # 실제 데이터만 기반으로 분석하는 프롬프트
    reasoning_prompt = f"""
다음은 Neo4j 그래프 데이터베이스에서 검색한 실제 결과와 관련 뉴스 원본 기사들입니다.
오직 이 데이터에 기반해서만 분석해주세요. 일반적인 지식이나 추측은 절대 사용하지 마세요.

질문: {question}

=== 그래프 검색 결과 ===
{chr(10).join(results_summary) if results_summary else "검색 결과 없음"}

=== 관련 뉴스 원본 기사 ===
{chr(10).join(articles_summary) if articles_summary else "관련 기사 없음"}

분석 규칙:
1. 위 데이터에서 확인 가능한 사실만 언급하세요
2. "~로 보인다", "~일 것이다" 같은 추측 표현 금지
3. 데이터에 없는 내용은 "제공된 데이터에서 확인할 수 없습니다"라고 명시
4. 검색 결과와 기사 내용 간의 연관성을 중심으로 분석
5. 구체적인 인용과 함께 설명

데이터 기반 분석:
"""
    
    try:
        analysis = llm.invoke(reasoning_prompt).content
        
        # 현재 요약에 분석 결과 추가
        current_summary = state.get("summary", "")
        updated_summary = f"{current_summary}\n\n=== 데이터 기반 분석 ===\n{analysis}"
        
        return {
            "summary": updated_summary
        }
        
    except Exception as e:
        return {
            "summary": f"{state.get('summary', '')}\n\n❌ 분석 중 오류 발생: {e}"
        }
    finally:
        if driver:
            driver.close()

def narration_node(state: EnhancedAgentState) -> EnhancedAgentState:
    """내러티브 노드 - 실제 검색 결과만으로 자연어 답변 생성"""
    llm = get_llm()
    driver = get_neo4j_driver()
    
    search_results = state.get("result", [])
    question = state["question"]
    analysis = state.get("summary", "")
    
    # 실제 뉴스 원본 기사들 가져오기
    source_articles = fetch_source_articles(state, driver) if driver else []
    
    if not search_results and not source_articles:
        return {
            "summary": f"{analysis}\n\n답변: 죄송합니다. '{question}'에 대한 정보를 데이터베이스에서 찾을 수 없습니다."
        }
    
    # 데이터 정리
    facts_from_search = []
    for result in search_results[:5]:
        fact = " | ".join([f"{k}: {v}" for k, v in result.items() if v])
        facts_from_search.append(fact)
    
    # 기사 인용 정보
    article_quotes = []
    for i, article in enumerate(source_articles[:3], 1):
        title = article.get('title', '제목 없음')
        content = article.get('content', '')[:300]  # 첫 300자
        date = article.get('date', '날짜 미상')
        url = article.get('url', '')
        
        quote = f"""
기사 {i}: {title}
발행일: {date}
내용 발췌: "{content}..."
출처: {url if url != 'No URL' else '출처 미상'}
"""
        article_quotes.append(quote)
    
    # 실제 데이터만 사용하는 답변 생성 프롬프트
    narration_prompt = f"""
사용자의 질문에 대해 오직 제공된 실제 데이터만 사용하여 정확하고 자연스러운 답변을 작성하세요.

질문: {question}

=== 사용 가능한 데이터 ===

1. 그래프 검색 결과:
{chr(10).join(facts_from_search) if facts_from_search else "검색 결과 없음"}

2. 관련 뉴스 기사 원문:
{chr(10).join(article_quotes) if article_quotes else "관련 기사 없음"}

3. 분석 결과:
{analysis}

답변 작성 규칙:
1. 반드시 위 데이터에서 확인 가능한 내용만 사용
2. 기사 제목, 날짜, 내용을 구체적으로 인용
3. "데이터에 따르면", "기사에서 확인된 바에 따르면" 등의 표현 사용
4. 데이터에 없는 내용은 절대 추가하지 않음
5. 자연스럽고 읽기 쉬운 문체로 작성
6. 출처를 명확히 표시

답변:
"""
    
    try:
        narrative = llm.invoke(narration_prompt).content
        
        # 최종 답변으로 요약 업데이트
        final_summary = f"{analysis}\n\n=== 최종 답변 ===\n{narrative}"
        
        return {
            "summary": final_summary
        }
        
    except Exception as e:
        return {
            "summary": f"{analysis}\n\n❌ 답변 생성 중 오류 발생: {e}"
        }
    finally:
        if driver:
            driver.close()

# Enhanced 노드 함수들
def memory_node(state: EnhancedAgentState) -> EnhancedAgentState:
    """Memory 관리 노드 - 대화 이력과 사용자 선호도 관리"""
    
    # 현재 질문을 대화 이력에 추가
    new_conversation = {
        "question": state["question"],
        "timestamp": "현재시간",  # 실제로는 datetime.now() 사용
        "cypher": state.get("cypher", ""),
        "result_count": len(state.get("result", []))
    }
    
    conversation_history = state.get("conversation_history", [])
    conversation_history.append(new_conversation)
    
    # 사용자 선호도 업데이트 (간단한 패턴 기반)
    user_preferences = state.get("user_preferences", {})
    
    if "경제" in state["question"] or "주가" in state["question"]:
        user_preferences["interests"] = user_preferences.get("interests", []) + ["경제"]
    if "기술" in state["question"] or "AI" in state["question"]:
        user_preferences["interests"] = user_preferences.get("interests", []) + ["기술"]
    
    # 최근 5개 대화만 유지
    if len(conversation_history) > 5:
        conversation_history = conversation_history[-5:]
    
    return {
        "conversation_history": conversation_history,
        "user_preferences": user_preferences
    }

def self_reflection_node(state: EnhancedAgentState) -> EnhancedAgentState:
    """Self-reflection 노드 - 결과 품질 자체 평가"""
    llm = get_llm()
    
    reflection_prompt = f"""
다음 검색 결과와 요약을 평가해주세요:

질문: {state["question"]}
검색 결과 개수: {len(state.get("result", []))}
생성된 요약: {state.get("summary", "")[:200]}...

다음 기준으로 평가하세요:
1. 결과 관련성 (high/medium/low)
2. 신뢰도 점수 (0.0-1.0)
3. 품질 점수 (1-5)
4. 개선 필요 여부 (true/false)

JSON 형식으로 응답:
{{"relevance": "high", "confidence": 0.8, "quality": 4, "needs_refinement": false}}
"""
    
    try:
        reflection_result = llm.invoke(reflection_prompt).content
        # JSON 파싱 시도
        import json
        scores = json.loads(reflection_result.strip().replace("```json", "").replace("```", ""))
        
        return {
            "result_relevance": scores.get("relevance", "medium"),
            "confidence_score": scores.get("confidence", 0.5),
            "quality_score": scores.get("quality", 3),
            "needs_refinement": scores.get("needs_refinement", False)
        }
    except:
        # 파싱 실패 시 기본값
        return {
            "result_relevance": "medium",
            "confidence_score": 0.5,
            "quality_score": 3,
            "needs_refinement": len(state.get("result", [])) == 0
        }

def quality_gate_node(state: EnhancedAgentState) -> EnhancedAgentState:
    """품질 게이트 노드 - 사람 검토 필요성 판단"""
    
    # 데모를 위해 더 자주 HITL을 트리거하도록 조건 수정
    needs_review = (
        state.get("confidence_score", 0) < 0.9 or  # 신뢰도 임계값 상향 (0.6 -> 0.9)
        state.get("quality_score", 0) < 5 or       # 품질 임계값 상향 (3 -> 5)
        len(state.get("result", [])) < 10 or       # 결과 개수 임계값 상향 (0 -> 10)
        state.get("result_relevance") != "high"    # 관련성이 high가 아니면 검토 필요
    )
    
    print(f"🔍 품질 검사 결과:")
    print(f"   신뢰도: {state.get('confidence_score', 0)} (임계값: 0.9)")
    print(f"   품질: {state.get('quality_score', 0)} (임계값: 5)")
    print(f"   결과 개수: {len(state.get('result', []))} (임계값: 10)")
    print(f"   관련성: {state.get('result_relevance', 'unknown')} (요구값: high)")
    print(f"   → 인간 검토 필요: {needs_review}")
    
    return {
        "needs_human_review": needs_review
    }

def human_review_node(state: EnhancedAgentState) -> EnhancedAgentState:
    """Human-in-the-Loop 노드 - 사용자 피드백 수집"""
    
    # interrupt를 사용하여 사용자 입력 대기
    feedback = interrupt({
        "type": "human_review",
        "question": state["question"],
        "current_result": state.get("summary", ""),
        "quality_issues": {
            "confidence": state.get("confidence_score", 0),
            "quality": state.get("quality_score", 0),
            "relevance": state.get("result_relevance", "unknown")
        },
        "request": "결과를 검토하고 피드백을 제공해주세요. '승인', '거부', 또는 구체적인 개선 요청을 입력하세요."
    })
    
    return {
        "human_feedback": feedback
    }

def refinement_node(state: EnhancedAgentState) -> EnhancedAgentState:
    """개선 노드 - 피드백 기반 재생성"""
    llm = get_llm()
    
    # 반복 횟수 증가
    iteration_count = state.get("iteration_count", 0) + 1
    
    # 피드백이 있으면 이를 반영한 새로운 쿼리 생성
    human_feedback = state.get("human_feedback", "")
    
    refinement_prompt = f"""
이전 결과에 대한 피드백을 반영하여 개선된 접근을 제안하세요.

원래 질문: {state["question"]}
이전 요약: {state.get("summary", "")}
사용자 피드백: {human_feedback}
반복 횟수: {iteration_count}

다음을 제안하세요:
1. 개선된 검색 전략
2. 다른 접근 방식
3. 추가 고려사항

개선 제안:
"""
    
    improvement = llm.invoke(refinement_prompt).content
    
    return {
        "iteration_count": iteration_count,
        "retry_reason": f"피드백 반영: {human_feedback}",
        "summary": f"{state.get('summary', '')}\n\n개선 제안:\n{improvement}"
    }

# 조건부 분기 함수들
def should_continue_routing(state: EnhancedAgentState) -> Literal["memory", "self_reflection"]:
    """라우팅 결정"""
    return "memory"

def quality_routing(state: EnhancedAgentState) -> Literal["human_review", "narration", "refinement"]:
    """품질 기반 라우팅"""
    if state.get("needs_human_review", False):
        return "human_review"
    elif state.get("needs_refinement", False) and state.get("iteration_count", 0) < state.get("max_iterations", 3):
        return "refinement"
    else:
        return "narration"

def feedback_routing(state: EnhancedAgentState) -> Literal["refinement", "narration", "END"]:
    """피드백 기반 라우팅"""
    feedback = state.get("human_feedback", "")
    
    if "거부" in feedback or "개선" in feedback:
        if state.get("iteration_count", 0) < state.get("max_iterations", 3):
            return "refinement"
        else:
            return "END"  # 최대 반복 도달
    else:
        return "narration"

# Enhanced 그래프 구성
def create_enhanced_graph():
    """고도화된 LangGraph - Memory, HITL, Self-RAG, Loops 포함"""
    
    # 체크포인터로 메모리 기능 활성화
    checkpointer = InMemorySaver()
    
    graph = StateGraph(EnhancedAgentState)
    
    # 모든 노드 추가
    graph.add_node("qa", qa_node)
    graph.add_node("memory", memory_node)
    graph.add_node("self_reflection", self_reflection_node)
    graph.add_node("quality_gate", quality_gate_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("refinement", refinement_node)
    graph.add_node("reasoning", reasoning_node)
    graph.add_node("narration", narration_node)

    # 복잡한 엣지 구조 정의
    graph.add_edge(START, "qa")
    
    # QA 후 조건부 분기
    graph.add_conditional_edges(
        "qa",
        should_continue_routing,
        {
            "memory": "memory",
            "self_reflection": "self_reflection"
        }
    )
    
    # Memory에서 Self-reflection으로
    graph.add_edge("memory", "self_reflection")
    
    # Self-reflection에서 Quality Gate로
    graph.add_edge("self_reflection", "quality_gate")
    
    # Quality Gate에서 조건부 분기
    graph.add_conditional_edges(
        "quality_gate",
        quality_routing,
        {
            "human_review": "human_review",
            "narration": "narration", 
            "refinement": "refinement"
        }
    )
    
    # Human Review 후 피드백 기반 분기
    graph.add_conditional_edges(
        "human_review",
        feedback_routing,
        {
            "refinement": "refinement",
            "narration": "narration",
            "END": END
        }
    )
    
    # Refinement에서 QA로 루프백 (재시도)
    graph.add_edge("refinement", "qa")
    
    # Narration에서 Reasoning으로
    graph.add_edge("narration", "reasoning")
    
    # Reasoning에서 최종 종료
    graph.add_edge("reasoning", END)
    
    # 체크포인터 설정으로 메모리와 HITL 활성화
    return graph.compile(checkpointer=checkpointer)

# 기존 단순 그래프는 유지
def create_simple_graph():
    """기존 단순 그래프"""
    graph = StateGraph(EnhancedAgentState)
    
    graph.add_node("qa", qa_node)
    graph.add_node("reasoning", reasoning_node)
    graph.add_node("narration", narration_node)

    graph.add_edge(START, "qa")
    graph.add_edge("qa", "reasoning")
    graph.add_edge("reasoning", "narration")
    graph.add_edge("narration", END)

    return graph.compile()

# 실행 예제 - Enhanced 버전
def main():
    # Enhanced graph 사용
    executable_graph = create_enhanced_graph()
    
    # 테스트 실행 - 고도화된 기능 테스트
    test_question = "삼성전자 위축의 정치적 원인은 무엇인가요?"
    
    # Thread config for memory
    config = {
        "configurable": {
            "thread_id": "user_session_1"
        }
    }
    
    initial_state = {
        "question": test_question,
        "cypher": "",
        "result": [],
        "summary": "",
        "conversation_history": [],
        "user_preferences": {},
        "quality_score": 0,
        "needs_human_review": False,
        "human_feedback": None,
        "iteration_count": 0,
        "max_iterations": 3,
        "retry_reason": None,
        "confidence_score": 0.0,
        "result_relevance": "low",
        "needs_refinement": False
    }
    
    try:
        result = executable_graph.invoke(initial_state, config=config)
        
        print("=== Enhanced Agent 실행 결과 ===")
        print(f"질문: {result['question']}")
        print(f"반복 횟수: {result.get('iteration_count', 0)}")
        print(f"품질 점수: {result.get('quality_score', 0)}")
        print(f"신뢰도: {result.get('confidence_score', 0)}")
        print(f"결과 관련성: {result.get('result_relevance', 'unknown')}")
        print(f"대화 이력: {len(result.get('conversation_history', []))}개")
        print(f"사용자 선호도: {result.get('user_preferences', {})}")
        print(f"최종 요약: {result.get('summary', '')}")
        
    except Exception as e:
        print(f"Enhanced Agent 실행 중 오류: {e}")
        
        # Fallback으로 단순 그래프 실행
        print("\n=== Fallback to Simple Agent ===")
        simple_graph = create_simple_graph()
        result = simple_graph.invoke(initial_state)
        
        print(f"질문: {result['question']}")
        print(f"Cypher: {result['cypher']}")
        print(f"결과: {len(result['result'])}개 항목")
        print(f"요약: {result['summary']}")

def test_hitl_interactive():
    """HITL 기능을 대화형으로 테스트하는 함수"""
    print("=== HITL 대화형 테스트 시작 ===")
    
    executable_graph = create_enhanced_graph()
    
    # Thread config for memory
    config = {
        "configurable": {
            "thread_id": "hitl_test_session"
        }
    }
    
    # 사용자 질문 입력
    question = input("질문을 입력하세요 (기본값: 삼성전자 위축의 정치적 원인은 무엇인가요?): ").strip()
    if not question:
        question = "삼성전자 위축의 정치적 원인은 무엇인가요?"
    
    initial_state = {
        "question": question,
        "cypher": "",
        "result": [],
        "summary": "",
        "conversation_history": [],
        "user_preferences": {},
        "quality_score": 0,
        "needs_human_review": False,
        "human_feedback": None,
        "iteration_count": 0,
        "max_iterations": 3,
        "retry_reason": None,
        "confidence_score": 0.0,
        "result_relevance": "low",
        "needs_refinement": False
    }
    
    try:
        # 스트림 실행으로 중간 결과 확인
        for event in executable_graph.stream(initial_state, config=config):
            for node_name, node_result in event.items():
                print(f"\n--- {node_name.upper()} 노드 실행 완료 ---")
                
                if node_name == "qa":
                    print(f"생성된 Cypher: {node_result.get('cypher', 'N/A')}")
                    print(f"결과 개수: {len(node_result.get('result', []))}")
                
                elif node_name == "self_reflection":
                    print(f"품질 점수: {node_result.get('quality_score', 0)}/5")
                    print(f"신뢰도: {node_result.get('confidence_score', 0):.2f}")
                    print(f"결과 관련성: {node_result.get('result_relevance', 'unknown')}")
                    print(f"인간 검토 필요: {node_result.get('needs_human_review', False)}")
                
                elif node_name == "quality_gate":
                    needs_review = node_result.get('needs_human_review', False)
                    if needs_review:
                        print("⚠️  품질이 낮아 인간 검토가 필요합니다!")
                
                elif node_name == "human_review":
                    print("👤 인간 검토 단계에 도달했습니다.")
                    print(f"현재 요약: {node_result.get('summary', '')[:200]}...")
                    
                    # 실제 사용자 피드백 받기
                    print("\n다음 옵션 중 선택하세요:")
                    print("1. 승인 (현재 결과로 진행)")
                    print("2. 개선 요청 (다시 시도)")
                    print("3. 거부 (종료)")
                    
                    choice = input("선택 (1/2/3): ").strip()
                    
                    if choice == "1":
                        feedback = "승인"
                    elif choice == "2":
                        feedback = "개선 " + input("개선 요청사항을 입력하세요: ")
                    elif choice == "3":
                        feedback = "거부"
                    else:
                        feedback = "승인"  # 기본값
                    
                    print(f"선택된 피드백: {feedback}")
                    
                    # 피드백을 state에 업데이트
                    node_result["human_feedback"] = feedback
                
                elif node_name == "reasoning":
                    print(f"추론 결과 길이: {len(node_result.get('summary', ''))}")
                
                elif node_name == "narration":
                    print("📝 최종 내러티브 생성 완료")
        
        print("\n=== 최종 실행 완료 ===")
        
    except Exception as e:
        print(f"HITL 테스트 중 오류: {e}")
        import traceback
        traceback.print_exc()

def main_interactive():
    """메인 대화형 인터페이스"""
    print("GraphRAG Agent with HITL - 대화형 모드")
    print("1. 기본 실행 (자동)")
    print("2. HITL 대화형 테스트")
    print("3. 종료")
    
    choice = input("선택하세요 (1/2/3): ").strip()
    
    if choice == "1":
        main()
    elif choice == "2":
        test_hitl_interactive()
    elif choice == "3":
        print("종료합니다.")
    else:
        print("잘못된 선택입니다. 기본 실행을 시작합니다.")
        main()

if __name__ == "__main__":
    main_interactive()
