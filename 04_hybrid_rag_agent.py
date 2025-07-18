#!/usr/bin/env python3
"""
Enhanced Hybrid RAG Agent: ChromaDB + Neo4j + LangGraph
실제 뉴스 원본(ChromaDB) + 관계 그래프(Neo4j) 결합 시스템
상세한 과정 설명 및 판단 로직 투명성 개선
"""

from typing import TypedDict, List, Literal, Optional, Dict, Tuple
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from neo4j import GraphDatabase
import chromadb
import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from langfuse.langchain import CallbackHandler
langfuse_handler = CallbackHandler()

# Enhanced State with detailed process tracking
class DetailedHybridRAGState(TypedDict):
    question: str
    # 검색 과정 상세 추적
    search_strategy: Dict  # 검색 전략 및 판단 과정
    chroma_process: Dict   # ChromaDB 검색 상세 과정
    neo4j_process: Dict    # Neo4j 검색 상세 과정
    quality_assessment: Dict  # 품질 평가 과정
    synthesis_process: Dict   # 통합 과정
    # 결과
    chroma_results: List[Dict]
    neo4j_results: List[Dict] 
    final_answer: str
    # 메타 정보
    sources: List[str]
    confidence_score: float
    iteration_count: int
    execution_log: List[Dict]  # 실행 과정 로그

class EnhancedHybridRAGAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # ChromaDB 클라이언트는 나중에 초기화
        self.chroma_client = None
        self.news_collection = None
        
        # Neo4j 드라이버
        self.neo4j_driver = self._get_neo4j_driver()
        
        # 실행 로그
        self.execution_log = []
    
    def _get_chroma_client(self):
        """ChromaDB 클라이언트 지연 초기화"""
        if self.chroma_client is None:
            self.chroma_client = chromadb.PersistentClient(path="chroma_db_news_2")
            self.news_collection = self.chroma_client.get_collection("naver_news")
        return self.news_collection
    
    def _get_neo4j_driver(self):
        """Neo4j 연결"""
        try:
            uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
            username = os.getenv("NEO4J_USERNAME", "neo4j")
            password = os.getenv("NEO4J_PASSWORD", "password")
            driver = GraphDatabase.driver(uri, auth=(username, password))
            return driver
        except Exception as e:
            print(f"Neo4j 연결 실패: {e}")
            return None
    
    def log_step(self, step_name: str, details: Dict):
        """실행 단계 로깅"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step_name,
            "details": details
        }
        self.execution_log.append(log_entry)

def analyze_question_strategy(state: DetailedHybridRAGState) -> DetailedHybridRAGState:
    """질문 분석 및 검색 전략 수립"""
    agent = EnhancedHybridRAGAgent()
    question = state["question"]
    
    print(f"🎯 Step 1: 질문 분석 및 검색 전략 수립")
    print(f"📝 질문: '{question}'")
    
    # LLM을 사용한 질문 분석
    analysis_prompt = f"""
다음 질문을 분석하여 최적의 검색 전략을 수립하세요:

질문: {question}

분석 항목:
1. 질문 유형 (사실 확인, 인과관계, 현황 분석, 예측 등)
2. 핵심 키워드 추출 (3-5개)
3. 시간적 범위 (최신, 특정 기간, 일반적)
4. 검색 우선순위 (뉴스 내용 vs 관계 그래프)
5. 예상 답변 복잡도 (단순/복합)

JSON 형식으로 응답:
{{
    "question_type": "질문 유형",
    "keywords": ["키워드1", "키워드2", "키워드3"],
    "time_scope": "시간적 범위",
    "search_priority": "chroma_first|neo4j_first|parallel",
    "complexity": "simple|complex",
    "reasoning": "전략 선택 이유"
}}
"""
    
    try:
        analysis_result = agent.llm.invoke(analysis_prompt, config={"callbacks": [langfuse_handler]}).content
        # JSON 파싱 시도
        try:
            strategy = json.loads(analysis_result)
        except:
            # JSON 파싱 실패시 기본 전략
            strategy = {
                "question_type": "일반 질문",
                "keywords": question.split()[:3],
                "time_scope": "최신",
                "search_priority": "parallel",
                "complexity": "complex",
                "reasoning": "자동 분석 실패로 기본 전략 적용"
            }
        
        print(f"  📊 질문 유형: {strategy['question_type']}")
        print(f"  🔑 핵심 키워드: {', '.join(strategy['keywords'])}")
        print(f"  ⏰ 시간 범위: {strategy['time_scope']}")
        print(f"  🎯 검색 우선순위: {strategy['search_priority']}")
        print(f"  🧠 복잡도: {strategy['complexity']}")
        print(f"  💡 전략 근거: {strategy['reasoning']}")
        
        agent.log_step("question_analysis", strategy)
        
        return {
            "search_strategy": strategy,
            "execution_log": agent.execution_log
        }
        
    except Exception as e:
        print(f"  ❌ 질문 분석 오류: {e}")
        default_strategy = {
            "question_type": "일반 질문",
            "keywords": question.split()[:3],
            "time_scope": "최신",
            "search_priority": "parallel",
            "complexity": "complex",
            "reasoning": "오류 발생으로 기본 전략 적용"
        }
        return {
            "search_strategy": default_strategy,
            "execution_log": []
        }

def enhanced_chroma_search_node(state: DetailedHybridRAGState) -> DetailedHybridRAGState:
    """강화된 ChromaDB 검색 (상세 과정 추적)"""
    agent = EnhancedHybridRAGAgent()
    question = state["question"]
    strategy = state.get("search_strategy", {})
    keywords = strategy.get("keywords", question.split())
    
    print(f"\n🔍 Step 2: ChromaDB 뉴스 검색 (상세 과정)")
    print(f"📋 검색 전략: {strategy.get('search_priority', 'parallel')}")
    print(f"🔑 추출된 키워드: {', '.join(keywords)}")
    
    chroma_process = {
        "start_time": time.time(),
        "search_method": "semantic_embedding",
        "keywords_used": keywords,
        "steps": []
    }
    
    try:
        # Step 1: 임베딩 생성
        print(f"  🧮 Step 2.1: 질문 임베딩 생성 중...")
        start_embed = time.time()
        query_embedding = agent.embeddings.embed_query(question)
        embed_time = time.time() - start_embed
        
        chroma_process["steps"].append({
            "step": "embedding_generation",
            "time_taken": embed_time,
            "embedding_dim": len(query_embedding),
            "success": True
        })
        
        print(f"    ✅ 임베딩 생성 완료 (차원: {len(query_embedding)}, 소요시간: {embed_time:.2f}초)")
        
        # Step 2: ChromaDB 검색 실행
        print(f"  🔎 Step 2.2: 의미적 유사도 검색 실행...")
        start_search = time.time()
        
        news_collection = agent._get_chroma_client()
        results = news_collection.query(
            query_embeddings=[query_embedding],
            n_results=10,  # 더 많은 결과 요청 후 필터링
            include=["documents", "metadatas", "distances"]
        )
        
        search_time = time.time() - start_search
        
        chroma_process["steps"].append({
            "step": "semantic_search",
            "time_taken": search_time,
            "raw_results_count": len(results["documents"][0]) if results["documents"] else 0,
            "success": True
        })
        
        print(f"    ✅ 검색 완료 (원본 결과: {len(results['documents'][0]) if results['documents'] else 0}개, 소요시간: {search_time:.2f}초)")
        
        # Step 3: 결과 품질 평가 및 필터링
        print(f"  📊 Step 2.3: 결과 품질 평가 및 필터링...")
        
        chroma_results = []
        if results["documents"] and results["documents"][0]:
            for i, (doc, metadata, distance) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0], 
                results["distances"][0]
            )):
                relevance_score = 1 - distance
                
                # 품질 필터링 (관련도 임계값)
                if relevance_score > 0.3:  # 최소 관련도 임계값
                    # 키워드 기반 추가 점수
                    title = metadata.get("title", "").lower()
                    content = doc.lower()
                    keyword_bonus = 0
                    
                    for keyword in keywords:
                        if keyword.lower() in title:
                            keyword_bonus += 0.2
                        elif keyword.lower() in content:
                            keyword_bonus += 0.1
                    
                    final_score = min(relevance_score + keyword_bonus, 1.0)
                    
                    chroma_results.append({
                        "content": doc,
                        "title": metadata.get("title", "제목 없음"),
                        "url": metadata.get("url", ""),
                        "published_date": metadata.get("published_date", ""),
                        "relevance_score": final_score,
                        "semantic_score": relevance_score,
                        "keyword_bonus": keyword_bonus,
                        "chunk_id": metadata.get("chunk_id", i),
                        "quality_tier": "high" if final_score > 0.7 else "medium" if final_score > 0.5 else "low"
                    })
        
        # 품질별 정렬 및 상위 5개 선택
        chroma_results.sort(key=lambda x: x["relevance_score"], reverse=True)
        filtered_results = chroma_results[:5]
        
        chroma_process["steps"].append({
            "step": "quality_filtering",
            "filtered_count": len(filtered_results),
            "quality_distribution": {
                "high": len([r for r in filtered_results if r["quality_tier"] == "high"]),
                "medium": len([r for r in filtered_results if r["quality_tier"] == "medium"]),
                "low": len([r for r in filtered_results if r["quality_tier"] == "low"])
            }
        })
        
        print(f"    ✅ 품질 필터링 완료:")
        print(f"      📈 고품질: {len([r for r in filtered_results if r['quality_tier'] == 'high'])}개")
        print(f"      📊 중품질: {len([r for r in filtered_results if r['quality_tier'] == 'medium'])}개")
        print(f"      📉 저품질: {len([r for r in filtered_results if r['quality_tier'] == 'low'])}개")
        
        # Step 4: 결과 상세 표시
        print(f"  📄 Step 2.4: 최종 뉴스 검색 결과 ({len(filtered_results)}개)")
        for i, result in enumerate(filtered_results, 1):
            print(f"    {i}. [{result['quality_tier'].upper()}] {result['title'][:60]}...")
            print(f"       관련도: {result['relevance_score']:.3f} (의미: {result['semantic_score']:.3f} + 키워드: {result['keyword_bonus']:.3f})")
            print(f"       날짜: {result['published_date']}")
        
        chroma_process["end_time"] = time.time()
        chroma_process["total_time"] = chroma_process["end_time"] - chroma_process["start_time"]
        chroma_process["success"] = True
        
        agent.log_step("chroma_search", chroma_process)
        
        return {
            "chroma_results": filtered_results,
            "chroma_process": chroma_process
        }
        
    except Exception as e:
        print(f"  ❌ ChromaDB 검색 오류: {e}")
        print(f"  🔄 키워드 기반 폴백 검색 시도...")
        
        # 폴백: 키워드 기반 검색
        try:
            news_collection = agent._get_chroma_client()
            all_results = news_collection.get(include=["documents", "metadatas"])
            
            keyword_results = []
            for i, (doc, metadata) in enumerate(zip(all_results["documents"], all_results["metadatas"])):
                title = metadata.get("title", "").lower()
                content = doc.lower()
                
                score = 0
                for keyword in keywords:
                    if keyword.lower() in title:
                        score += 3
                    elif keyword.lower() in content:
                        score += 1
                
                if score > 0:
                    keyword_results.append({
                        "content": doc,
                        "title": metadata.get("title", "제목 없음"),
                        "url": metadata.get("url", ""),
                        "published_date": metadata.get("published_date", ""),
                        "relevance_score": score / len(keywords),
                        "semantic_score": 0,
                        "keyword_bonus": score / len(keywords),
                        "chunk_id": metadata.get("chunk_id", i),
                        "quality_tier": "medium"
                    })
            
            keyword_results.sort(key=lambda x: x["relevance_score"], reverse=True)
            keyword_results = keyword_results[:5]
            
            chroma_process["search_method"] = "keyword_fallback"
            chroma_process["success"] = True
            chroma_process["error"] = str(e)
            
            print(f"    ✅ 폴백 검색 완료: {len(keyword_results)}개 발견")
            
            return {
                "chroma_results": keyword_results,
                "chroma_process": chroma_process
            }
            
        except Exception as fallback_error:
            print(f"  ❌ 폴백 검색도 실패: {fallback_error}")
            chroma_process["success"] = False
            chroma_process["error"] = f"원본: {e}, 폴백: {fallback_error}"
            
            return {
                "chroma_results": [],
                "chroma_process": chroma_process
            }

def enhanced_neo4j_search_node(state: DetailedHybridRAGState) -> DetailedHybridRAGState:
    """강화된 Neo4j 검색 (정교한 필터링 및 관련도 평가)"""
    agent = EnhancedHybridRAGAgent()
    question = state["question"]
    strategy = state.get("search_strategy", {})
    keywords = strategy.get("keywords", question.split())
    
    print(f"\n🔗 Step 3: Neo4j 그래프 검색 (정교한 관련도 평가)")
    print(f"🔑 원본 키워드: {', '.join(keywords)}")
    
    # Step 0: 키워드 정제 및 조합 생성
    print(f"  🧹 Step 3.0: 키워드 정제 및 관련도 가중치 설정...")
    
    # 키워드 정제 (불용어 제거 및 중요도 분류)
    primary_keywords = []  # 핵심 키워드 (예: 코스피, 삼성전자)
    context_keywords = []  # 맥락 키워드 (예: 상승, 원인)
    
    # 금융/경제 관련 핵심 키워드 식별
    financial_entities = ["코스피", "kospi", "삼성전자", "sk하이닉스", "lg", "현대", "주가", "증시", "지수"]
    financial_concepts = ["상승", "하락", "급등", "급락", "원인", "이유", "영향", "요인"]
    
    for keyword in keywords:
        keyword_lower = keyword.lower()
        if any(entity in keyword_lower for entity in financial_entities):
            primary_keywords.append(keyword)
        elif any(concept in keyword_lower for concept in financial_concepts):
            context_keywords.append(keyword)
        elif len(keyword) > 1:  # 기타 키워드
            context_keywords.append(keyword)
    
    # 최소 하나의 핵심 키워드가 필요
    if not primary_keywords and any(kw in question.lower() for kw in ["코스피", "주가", "증시"]):
        primary_keywords = ["코스피"]
    
    print(f"    🎯 핵심 키워드: {', '.join(primary_keywords) if primary_keywords else '없음'}")
    print(f"    📝 맥락 키워드: {', '.join(context_keywords)}")
    
    neo4j_process = {
        "start_time": time.time(),
        "keywords_used": keywords,
        "primary_keywords": primary_keywords,
        "context_keywords": context_keywords,
        "search_patterns": [],
        "steps": []
    }
    
    if not agent.neo4j_driver:
        print(f"  ❌ Neo4j 연결 없음")
        neo4j_process["success"] = False
        neo4j_process["error"] = "Neo4j driver not available"
        return {"neo4j_results": [], "neo4j_process": neo4j_process}
    
    try:
        # Step 1: 정교한 검색 패턴 선택
        print(f"  🧠 Step 3.1: 정교한 검색 패턴 선택...")
        
        # 질문 유형에 따른 패턴 우선순위 결정
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["원인", "이유", "왜"]):
            pattern_priority = ["targeted_causal", "targeted_relationship", "contextual_network"]
            print(f"    📊 정교한 인과관계 검색 패턴 선택")
        elif any(word in question_lower for word in ["관련", "연관", "영향"]):
            pattern_priority = ["contextual_network", "targeted_relationship", "targeted_causal"]
            print(f"    🕸️ 정교한 네트워크 검색 패턴 선택")
        else:
            pattern_priority = ["targeted_relationship", "targeted_causal", "contextual_network"]
            print(f"    🔗 정교한 관계 검색 패턴 선택")
        
        # 정교한 검색 패턴 정의
        search_patterns = {
            "targeted_relationship": {
                "name": "핵심 키워드 중심 관계 검색",
                "query": """
                MATCH (a)-[r]->(b)
                WHERE (
                    any(pk IN $primary_keywords WHERE a.name CONTAINS pk OR b.name CONTAINS pk)
                    AND 
                    any(ck IN $context_keywords WHERE a.name CONTAINS ck OR b.name CONTAINS ck)
                )
                RETURN a.name as source, type(r) as relationship, b.name as target,
                       'targeted' as pattern_type
                LIMIT 8
                """,
                "description": "핵심 키워드와 맥락 키워드가 모두 포함된 관계 탐색"
            },
            "targeted_causal": {
                "name": "핵심 인과관계 체인",
                "query": """
                MATCH (a)-[:원인이다]->(b)-[:결과이다]->(c)
                WHERE (
                    any(pk IN $primary_keywords WHERE a.name CONTAINS pk OR b.name CONTAINS pk OR c.name CONTAINS pk)
                )
                RETURN a.name as cause, b.name as intermediate, c.name as effect,
                       'targeted_causal' as pattern_type
                LIMIT 5
                """,
                "description": "핵심 키워드가 포함된 원인→결과 체인 탐색"
            },
            "contextual_network": {
                "name": "맥락 기반 네트워크",
                "query": """
                MATCH (center)-[r]-(connected)
                WHERE (
                    any(pk IN $primary_keywords WHERE center.name CONTAINS pk)
                    AND 
                    any(ck IN $context_keywords WHERE connected.name CONTAINS ck OR type(r) CONTAINS '상승' OR type(r) CONTAINS '원인')
                )
                RETURN center.name as center_entity, type(r) as relationship, connected.name as related_entity,
                       'contextual' as pattern_type
                LIMIT 6
                """,
                "description": "핵심 엔티티와 맥락적으로 관련된 네트워크 탐색"
            }
        }
        
        # 핵심 키워드가 없으면 폴백 패턴 사용
        if not primary_keywords:
            print(f"    ⚠️ 핵심 키워드 없음 - 폴백 검색 패턴 사용")
            search_patterns["fallback"] = {
                "name": "폴백 키워드 검색",
                "query": """
                MATCH (a)-[r]->(b)
                WHERE any(keyword IN $context_keywords WHERE a.name CONTAINS keyword OR b.name CONTAINS keyword)
                RETURN a.name as source, type(r) as relationship, b.name as target,
                       'fallback' as pattern_type
                LIMIT 5
                """,
                "description": "맥락 키워드만을 사용한 기본 검색"
            }
            pattern_priority = ["fallback"]
        
        neo4j_results = []
        
        # Step 2: 선택된 우선순위에 따라 패턴 실행
        print(f"  🔍 Step 3.2: 정교한 그래프 검색 실행...")
        
        with agent.neo4j_driver.session() as session:
            for i, pattern_key in enumerate(pattern_priority, 1):
                if pattern_key not in search_patterns:
                    continue
                    
                pattern = search_patterns[pattern_key]
                print(f"    🎯 패턴 {i}: {pattern['name']} 실행 중...")
                print(f"       📝 설명: {pattern['description']}")
                
                try:
                    start_pattern = time.time()
                    
                    # 패턴에 따른 파라미터 설정
                    if pattern_key == "fallback":
                        query_params = {"context_keywords": context_keywords}
                    else:
                        query_params = {
                            "primary_keywords": primary_keywords,
                            "context_keywords": context_keywords
                        }
                    
                    results = session.run(pattern["query"], query_params).data()
                    pattern_time = time.time() - start_pattern
                    
                    if results:
                        print(f"       ✅ 성공: {len(results)}개 관계 발견 (소요시간: {pattern_time:.2f}초)")
                        
                        # 결과에 패턴 정보와 관련도 점수 추가
                        for result in results:
                            result["search_pattern"] = pattern_key
                            result["pattern_name"] = pattern["name"]
                            
                            # 관련도 점수 계산 (안전한 처리)
                            try:
                                relevance_score = calculate_graph_relevance(result, primary_keywords, context_keywords, question)
                                result["relevance_score"] = relevance_score
                            except Exception as relevance_error:
                                print(f"         ⚠️ 관련도 계산 오류: {relevance_error}")
                                print(f"         📋 결과 구조: {result}")
                                result["relevance_score"] = 0.5  # 기본값
                        
                        # 관련도 순으로 정렬
                        results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
                        
                        neo4j_results.extend(results)
                        
                        neo4j_process["search_patterns"].append({
                            "pattern": pattern_key,
                            "name": pattern["name"],
                            "results_count": len(results),
                            "avg_relevance": sum(r.get("relevance_score", 0) for r in results) / len(results) if results else 0,
                            "time_taken": pattern_time,
                            "success": True
                        })
                        
                    else:
                        print(f"       ⚠️ 결과 없음 (소요시간: {pattern_time:.2f}초)")
                        neo4j_process["search_patterns"].append({
                            "pattern": pattern_key,
                            "name": pattern["name"],
                            "results_count": 0,
                            "time_taken": pattern_time,
                            "success": True
                        })
                        
                except Exception as pattern_error:
                    print(f"       ❌ 패턴 실행 오류: {pattern_error}")
                    neo4j_process["search_patterns"].append({
                        "pattern": pattern_key,
                        "name": pattern["name"],
                        "error": str(pattern_error),
                        "success": False
                    })
        
        # Step 3: 관련도 기반 필터링
        print(f"  📊 Step 3.3: 관련도 기반 결과 필터링...")
        
        if neo4j_results:
            # 관련도 점수가 없는 결과에 기본값 설정
            for result in neo4j_results:
                if "relevance_score" not in result:
                    result["relevance_score"] = 0.0
            # 관련도 임계값 적용 (0.3 이상만 유지)
            filtered_results = [r for r in neo4j_results if r.get("relevance_score", 0) >= 0.3]
            
            # 중복 제거 (동일한 관계)
            unique_results = []
            seen_relationships = set()
            
            for result in filtered_results:
                try:
                    if "relationship" in result:
                        key = f"{result['source']}-{result['relationship']}-{result['target']}"
                    elif "cause" in result:
                        key = f"{result['cause']}-{result['intermediate']}-{result['effect']}"
                    else:
                        key = str(result)
                    
                    if key not in seen_relationships:
                        seen_relationships.add(key)
                        unique_results.append(result)
                except Exception as key_error:
                    print(f"    ⚠️ 중복 제거 중 오류: {key_error}")
                    print(f"    📋 문제 결과: {result}")
                    # 오류가 있어도 결과는 포함 (단, 기본 키 사용)
                    key = str(hash(str(result)))
                    if key not in seen_relationships:
                        seen_relationships.add(key)
                        unique_results.append(result)
            
            neo4j_results = unique_results[:10]  # 상위 10개로 제한
            
            print(f"    ✅ 필터링 완료: {len(filtered_results)}개 → {len(neo4j_results)}개 (중복 제거)")
            
            # 관련도별 분포
            high_rel = len([r for r in neo4j_results if r.get("relevance_score", 0) >= 0.7])
            med_rel = len([r for r in neo4j_results if 0.5 <= r.get("relevance_score", 0) < 0.7])
            low_rel = len([r for r in neo4j_results if 0.3 <= r.get("relevance_score", 0) < 0.5])
            
            print(f"    📊 관련도 분포: 고관련({high_rel}개) 중관련({med_rel}개) 저관련({low_rel}개)")
            
            # 관계 유형별 분류
            relationship_types = {}
            
            for result in neo4j_results:
                try:
                    if "relationship" in result:
                        rel_type = result["relationship"]
                        relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
                except Exception as rel_error:
                    print(f"    ⚠️ 관계 분류 중 오류: {rel_error}")
                    print(f"    📋 문제 결과: {result}")
            
            print(f"    📈 관계 유형 분포:")
            for rel_type, count in sorted(relationship_types.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"       - {rel_type}: {count}개")
            
            neo4j_process["analysis"] = {
                "relationship_types": relationship_types,
                "relevance_distribution": {"high": high_rel, "medium": med_rel, "low": low_rel},
                "total_relationships": len(neo4j_results)
            }
        else:
            print(f"    ⚠️ Neo4j 결과가 비어있음")
        
        # Step 4: 정제된 결과 상세 표시
        print(f"  🎯 Step 3.4: 정제된 그래프 검색 결과 ({len(neo4j_results)}개)")
        
        for i, result in enumerate(neo4j_results, 1):
            try:
                relevance = result.get("relevance_score", 0)
                pattern = result.get("search_pattern", "unknown").upper()
                
                if "relationship" in result:
                    print(f"    {i}. [{pattern}] {result['source']} -[{result['relationship']}]-> {result['target']}")
                    print(f"       관련도: {relevance:.3f}")
                elif "cause" in result:
                    print(f"    {i}. [CAUSAL] {result['cause']} → {result['intermediate']} → {result['effect']}")
                    print(f"       관련도: {relevance:.3f}")
                elif "center_entity" in result:
                    print(f"    {i}. [NETWORK] {result['center_entity']} -[{result.get('relationship', 'related')}]-> {result['related_entity']}")
                    print(f"       관련도: {relevance:.3f}")
                else:
                    print(f"    {i}. [UNKNOWN] {result}")
                    print(f"       관련도: {relevance:.3f}")
            except Exception as display_error:
                print(f"    {i}. [ERROR] 결과 표시 오류: {display_error}")
                print(f"       원본 데이터: {result}")
        
        neo4j_process["end_time"] = time.time()
        neo4j_process["total_time"] = neo4j_process["end_time"] - neo4j_process["start_time"]
        neo4j_process["success"] = True
        neo4j_process["total_results"] = len(neo4j_results)
        
        agent.log_step("neo4j_search", neo4j_process)
        
        return {
            "neo4j_results": neo4j_results,
            "neo4j_process": neo4j_process
        }
        
    except Exception as e:
        print(f"  ❌ Neo4j 검색 오류: {e}")
        neo4j_process["success"] = False
        neo4j_process["error"] = str(e)
        neo4j_process["end_time"] = time.time()
        
        return {
            "neo4j_results": [],
            "neo4j_process": neo4j_process
        }
    finally:
        if agent.neo4j_driver:
            agent.neo4j_driver.close()

def calculate_graph_relevance(result: Dict, primary_keywords: List[str], context_keywords: List[str], question: str) -> float:
    """그래프 검색 결과의 질문 관련도 계산 (안전한 처리)"""
    try:
        score = 0.0
        question_lower = question.lower()
        
        # 결과에서 텍스트 추출 (안전한 방식)
        text_fields = []
        
        # 다양한 결과 구조 처리
        if "source" in result and "target" in result:
            # 일반 관계: source -[relationship]-> target
            text_fields = [
                str(result.get("source", "")), 
                str(result.get("target", "")), 
                str(result.get("relationship", ""))
            ]
        elif "cause" in result and "intermediate" in result and "effect" in result:
            # 인과관계 체인: cause -> intermediate -> effect
            text_fields = [
                str(result.get("cause", "")),
                str(result.get("intermediate", "")),
                str(result.get("effect", ""))
            ]
        elif "center_entity" in result and "related_entity" in result:
            # 네트워크: center_entity -[relationship]-> related_entity
            text_fields = [
                str(result.get("center_entity", "")),
                str(result.get("related_entity", "")),
                str(result.get("relationship", ""))
            ]
        else:
            # 기타 구조: 모든 값을 텍스트로 변환
            text_fields = [str(v) for v in result.values() if v is not None]
        
        # 빈 필드 제거
        text_fields = [field for field in text_fields if field and field != "None"]
        
        if not text_fields:
            return 0.0
        
        full_text = " ".join(text_fields).lower()
        
        # 1. 핵심 키워드 매칭 (높은 가중치)
        for pk in primary_keywords:
            if pk and pk.lower() in full_text:
                score += 0.4
        
        # 2. 맥락 키워드 매칭 (중간 가중치)
        for ck in context_keywords:
            if ck and ck.lower() in full_text:
                score += 0.2
        
        # 3. 질문 도메인 일치성 (금융/경제 관련)
        financial_terms = ["주가", "코스피", "증시", "상승", "하락", "경제", "금융", "투자", "시장"]
        domain_match = sum(1 for term in financial_terms if term in full_text)
        score += min(domain_match * 0.1, 0.3)
        
        # 4. 관계의 의미적 적합성
        if "relationship" in result:
            rel = str(result.get("relationship", ""))
            if any(word in rel for word in ["원인", "결과", "영향", "상승", "증가"]):
                score += 0.2
        
        # 5. 패널티: 질문과 무관한 주제
        irrelevant_terms = ["에이미", "힐스", "가슴", "축소술", "박보영", "연기쇼"]
        if any(term in full_text for term in irrelevant_terms):
            score -= 0.5
        
        return max(0.0, min(1.0, score))  # 0~1 범위로 정규화
        
    except Exception as e:
        # 모든 예외 상황에서 기본값 반환
        print(f"관련도 계산 중 예외 발생: {e}")
        return 0.5

def quality_assessment_node(state: DetailedHybridRAGState) -> DetailedHybridRAGState:
    """검색 결과 품질 종합 평가"""
    question = state["question"]
    chroma_results = state.get("chroma_results", [])
    neo4j_results = state.get("neo4j_results", [])
    chroma_process = state.get("chroma_process", {})
    neo4j_process = state.get("neo4j_process", {})
    
    print(f"\n📊 Step 4: 검색 결과 품질 종합 평가")
    
    quality_assessment = {
        "chroma_quality": {},
        "neo4j_quality": {},
        "overall_quality": {},
        "recommendations": []
    }
    
    # ChromaDB 결과 품질 평가
    print(f"  📄 ChromaDB 결과 품질 분석:")
    
    if chroma_results:
        avg_relevance = sum(r["relevance_score"] for r in chroma_results) / len(chroma_results)
        high_quality_count = len([r for r in chroma_results if r["quality_tier"] == "high"])
        has_recent_news = any(r.get("published_date", "").startswith("2025-07") for r in chroma_results)
        
        quality_assessment["chroma_quality"] = {
            "result_count": len(chroma_results),
            "avg_relevance": avg_relevance,
            "high_quality_ratio": high_quality_count / len(chroma_results),
            "has_recent_content": has_recent_news,
            "search_success": chroma_process.get("success", False)
        }
        
        print(f"    ✅ 결과 수: {len(chroma_results)}개")
        print(f"    📊 평균 관련도: {avg_relevance:.3f}")
        print(f"    🌟 고품질 비율: {high_quality_count}/{len(chroma_results)} ({high_quality_count/len(chroma_results)*100:.1f}%)")
        print(f"    📅 최신 뉴스: {'있음' if has_recent_news else '없음'}")
        
        if avg_relevance < 0.5:
            quality_assessment["recommendations"].append("ChromaDB 검색 키워드 또는 전략 개선 필요")
            
    else:
        quality_assessment["chroma_quality"] = {
            "result_count": 0,
            "avg_relevance": 0,
            "high_quality_ratio": 0,
            "has_recent_content": False,
            "search_success": False
        }
        print(f"    ❌ 검색 결과 없음")
        quality_assessment["recommendations"].append("ChromaDB 검색 전략 전면 재검토 필요")
    
    # Neo4j 결과 품질 평가
    print(f"  🔗 Neo4j 결과 품질 분석:")
    
    if neo4j_results:
        successful_patterns = len([p for p in neo4j_process.get("search_patterns", []) if p.get("success", False)])
        relationship_diversity = len(set(r.get("relationship", "unknown") for r in neo4j_results if "relationship" in r))
        entity_diversity = len(set(str(r.get("source_type", "")) + str(r.get("target_type", "")) for r in neo4j_results))
        
        quality_assessment["neo4j_quality"] = {
            "result_count": len(neo4j_results),
            "successful_patterns": successful_patterns,
            "relationship_diversity": relationship_diversity,
            "entity_diversity": entity_diversity,
            "search_success": neo4j_process.get("success", False)
        }
        
        print(f"    ✅ 관계 수: {len(neo4j_results)}개")
        print(f"    🎯 성공 패턴: {successful_patterns}개")
        print(f"    🔄 관계 다양성: {relationship_diversity}개 유형")
        print(f"    🏷️ 엔티티 다양성: {entity_diversity}개 조합")
        
        if len(neo4j_results) < 3:
            quality_assessment["recommendations"].append("Neo4j 검색 범위 확장 또는 키워드 조정 필요")
            
    else:
        quality_assessment["neo4j_quality"] = {
            "result_count": 0,
            "successful_patterns": 0,
            "relationship_diversity": 0,
            "entity_diversity": 0,
            "search_success": False
        }
        print(f"    ❌ 그래프 관계 없음")
        quality_assessment["recommendations"].append("Neo4j 데이터베이스 상태 확인 또는 검색 전략 변경 필요")
    
    # 전체 품질 종합 평가
    print(f"  🎯 종합 품질 평가:")
    
    chroma_score = min(quality_assessment["chroma_quality"]["avg_relevance"] * 0.7 + 
                      quality_assessment["chroma_quality"]["high_quality_ratio"] * 0.3, 1.0)
    
    neo4j_score = 0.0
    if neo4j_results and len(neo4j_results) > 0:
        neo4j_score += 0.4
        neo4j_score += min(quality_assessment["neo4j_quality"]["relationship_diversity"] / 10 * 0.3, 0.3)
        neo4j_score += min(quality_assessment["neo4j_quality"]["successful_patterns"] / 3 * 0.3, 0.3)
    neo4j_score = min(neo4j_score, 1.0)
    
    # 데이터 상호 보완성
    complementarity = 0.1 if chroma_results and neo4j_results else 0
    
    overall_score = (chroma_score * 0.6 + neo4j_score * 0.3 + complementarity * 0.1)
    
    quality_assessment["overall_quality"] = {
        "chroma_score": chroma_score,
        "neo4j_score": neo4j_score,
        "complementarity": complementarity,
        "overall_score": overall_score,
        "quality_tier": "high" if overall_score > 0.7 else "medium" if overall_score > 0.4 else "low"
    }
    
    print(f"    📊 ChromaDB 점수: {chroma_score:.3f}")
    print(f"    🔗 Neo4j 점수: {neo4j_score:.3f}")
    print(f"    🤝 상호보완성: {complementarity:.3f}")
    print(f"    🎯 종합 점수: {overall_score:.3f} ({quality_assessment['overall_quality']['quality_tier'].upper()})")
    
    if quality_assessment["recommendations"]:
        print(f"  💡 개선 권장사항:")
        for i, rec in enumerate(quality_assessment["recommendations"], 1):
            print(f"    {i}. {rec}")
    
    return {"quality_assessment": quality_assessment}

def enhanced_synthesis_node(state: DetailedHybridRAGState) -> DetailedHybridRAGState:
    """강화된 통합 답변 생성 (상세 과정 추적)"""
    agent = EnhancedHybridRAGAgent()
    question = state["question"]
    chroma_results = state.get("chroma_results", [])
    neo4j_results = state.get("neo4j_results", [])
    quality_assessment = state.get("quality_assessment", {})
    
    print(f"\n🧠 Step 5: 통합 답변 생성 (상세 과정)")
    
    synthesis_process = {
        "start_time": time.time(),
        "steps": [],
        "data_integration": {},
        "answer_strategy": {}
    }
    
    # Step 1: 데이터 통합 전략 결정
    print(f"  🎯 Step 5.1: 데이터 통합 전략 결정...")
    
    overall_quality = quality_assessment.get("overall_quality", {})
    chroma_quality = quality_assessment.get("chroma_quality", {})
    neo4j_quality = quality_assessment.get("neo4j_quality", {})
    
    # 통합 전략 결정
    if chroma_results and neo4j_results:
        integration_strategy = "hybrid_synthesis"
        print(f"    🤝 하이브리드 통합: 뉴스 + 그래프 결합")
    elif chroma_results:
        integration_strategy = "news_focused"
        print(f"    📰 뉴스 중심: ChromaDB 결과 기반")
    elif neo4j_results:
        integration_strategy = "graph_focused"  
        print(f"    🔗 그래프 중심: Neo4j 관계 기반")
    else:
        integration_strategy = "knowledge_fallback"
        print(f"    🧠 지식 폴백: 일반 지식 기반 (주의: 환각 위험)")
    
    synthesis_process["answer_strategy"]["integration_strategy"] = integration_strategy
    
    # Step 2: 핵심 데이터 추출 및 구조화
    print(f"  📊 Step 5.2: 핵심 데이터 추출 및 구조화...")
    
    # 뉴스 데이터 구조화
    structured_news = []
    if chroma_results:
        for i, result in enumerate(chroma_results[:3], 1):
            structured_news.append({
                "rank": i,
                "title": result["title"],
                "content_snippet": result["content"][:200] + "..." if len(result["content"]) > 200 else result["content"],
                "date": result["published_date"],
                "url": result["url"],
                "relevance": result["relevance_score"],
                "quality_tier": result["quality_tier"]
            })
        
        print(f"    📄 구조화된 뉴스: {len(structured_news)}개")
        for news in structured_news:
            print(f"       #{news['rank']} [{news['quality_tier'].upper()}] {news['title'][:50]}... (관련도: {news['relevance']:.3f})")
    
    # 그래프 관계 구조화
    structured_relationships = []
    if neo4j_results:
        for i, result in enumerate(neo4j_results[:5], 1):
            try:
                if "source" in result and "target" in result:
                    # 일반 관계 구조
                    structured_relationships.append({
                        "rank": i,
                        "type": "direct_relationship",
                        "source": result["source"],
                        "relationship": result["relationship"],
                        "target": result["target"],
                        "pattern": result.get("search_pattern", "unknown"),
                        "relevance": result.get("relevance_score", 0)
                    })
                elif "cause" in result and "intermediate" in result and "effect" in result:
                    # 인과관계 체인 구조
                    structured_relationships.append({
                        "rank": i,
                        "type": "causal_chain",
                        "cause": result["cause"],
                        "intermediate": result["intermediate"],
                        "effect": result["effect"],
                        "pattern": "causal",
                        "relevance": result.get("relevance_score", 0)
                    })
                elif "center_entity" in result and "related_entity" in result:
                    # 네트워크 관계 구조
                    structured_relationships.append({
                        "rank": i,
                        "type": "network_relationship",
                        "source": result["center_entity"],  # center_entity를 source로 매핑
                        "relationship": result["relationship"],
                        "target": result["related_entity"],  # related_entity를 target으로 매핑
                        "pattern": result.get("search_pattern", "network"),
                        "relevance": result.get("relevance_score", 0)
                    })
                else:
                    # 기타 구조 - 가능한 한 정보 추출
                    structured_relationships.append({
                        "rank": i,
                        "type": "unknown",
                        "source": str(result.get("source", result.get("center_entity", "알수없음"))),
                        "relationship": str(result.get("relationship", "관련")),
                        "target": str(result.get("target", result.get("related_entity", "알수없음"))),
                        "pattern": result.get("search_pattern", "unknown"),
                        "relevance": result.get("relevance_score", 0)
                    })
            except Exception as struct_error:
                print(f"    ⚠️ 관계 구조화 오류: {struct_error}")
                print(f"    📋 문제 결과: {result}")
                # 오류가 있어도 기본 구조로 포함
                structured_relationships.append({
                    "rank": i,
                    "type": "error",
                    "source": "오류",
                    "relationship": "알수없음",
                    "target": "오류",
                    "pattern": "error",
                    "relevance": 0
                })
        
        print(f"    🔗 구조화된 관계: {len(structured_relationships)}개")
        for rel in structured_relationships:
            if rel["type"] == "direct_relationship":
                print(f"       #{rel['rank']} [{rel['pattern'].upper()}] {rel['source']} -[{rel['relationship']}]-> {rel['target']} (관련도: {rel['relevance']:.3f})")
            elif rel["type"] == "causal_chain":
                print(f"       #{rel['rank']} [CAUSAL] {rel['cause']} → {rel['intermediate']} → {rel['effect']} (관련도: {rel['relevance']:.3f})")
            elif rel["type"] == "network_relationship":
                print(f"       #{rel['rank']} [NETWORK] {rel['source']} -[{rel['relationship']}]-> {rel['target']} (관련도: {rel['relevance']:.3f})")
            else:
                print(f"       #{rel['rank']} [{rel['type'].upper()}] {rel['source']} -[{rel['relationship']}]-> {rel['target']} (관련도: {rel['relevance']:.3f})")
    
    synthesis_process["data_integration"] = {
        "news_count": len(structured_news),
        "relationship_count": len(structured_relationships),
        "integration_strategy": integration_strategy
    }
    
    # Step 3: LLM 기반 통합 답변 생성
    print(f"  🤖 Step 5.3: LLM 기반 통합 답변 생성...")
    
    # 답변 생성 프롬프트 (전략별)
    if integration_strategy == "hybrid_synthesis":
        synthesis_prompt = f"""
당신은 뉴스 분석 전문가입니다. 실제 뉴스 기사와 관계 그래프 정보를 통합하여 정확하고 포괄적인 답변을 제공하세요.

질문: {question}

=== 실제 뉴스 기사 정보 ===
{chr(10).join([f"[뉴스 {news['rank']}] {news['title']} ({news['date']})\\n내용: {news['content_snippet']}\\n출처: {news['url']}\\n관련도: {news['relevance']:.3f}\\n" for news in structured_news]) if structured_news else "관련 뉴스 없음"}

=== 관계 그래프 정보 ===
{chr(10).join([f"[관계 {rel['rank']}] {rel['source']} -[{rel['relationship']}]-> {rel['target']} (패턴: {rel['pattern']}, 관련도: {rel['relevance']:.3f})" if rel['type'] in ['direct_relationship', 'network_relationship', 'unknown', 'error'] else f"[인과 {rel['rank']}] {rel['cause']} → {rel['intermediate']} → {rel['effect']} (관련도: {rel['relevance']:.3f})" for rel in structured_relationships]) if structured_relationships else "관계 정보 없음"}

답변 생성 지침:
1. 뉴스 기사의 구체적 사실을 답변의 핵심으로 사용
2. 그래프 관계로 맥락과 배경 설명 보강
3. 날짜, 출처, 구체적 수치 등 사실 정보 명시
4. 추측이나 일반 지식 사용 금지
5. 자연스럽고 체계적인 문단 구성
6. 정보의 신뢰도와 한계 명시

답변:
"""
    elif integration_strategy == "news_focused":
        synthesis_prompt = f"""
실제 뉴스 기사를 바탕으로 정확한 답변을 제공하세요.

질문: {question}

=== 뉴스 기사 정보 ===
{chr(10).join([f"[뉴스 {news['rank']}] {news['title']} ({news['date']})\\n내용: {news['content_snippet']}\\n출처: {news['url']}\\n" for news in structured_news])}

답변은 오직 제공된 뉴스 기사 내용만을 기반으로 작성하세요.
"""
    elif integration_strategy == "graph_focused":
        synthesis_prompt = f"""
관계 그래프 정보를 바탕으로 답변을 제공하세요.

질문: {question}

=== 관계 정보 ===
{chr(10).join([f"[관계 {rel['rank']}] {rel['source']} -[{rel['relationship']}]-> {rel['target']}" if rel['type'] in ['direct_relationship', 'network_relationship', 'unknown', 'error'] else f"[인과 {rel['rank']}] {rel['cause']} → {rel['intermediate']} → {rel['effect']}" for rel in structured_relationships])}

답변은 오직 제공된 관계 정보만을 기반으로 작성하세요.
"""
    else:
        synthesis_prompt = f"""
검색된 구체적 정보가 부족합니다. 질문에 대해 답변 불가능함을 명시하세요.

질문: {question}

답변: 죄송합니다. '{question}'에 대한 구체적이고 신뢰할 수 있는 정보를 찾을 수 없습니다. 
더 정확한 답변을 위해서는 추가 정보가 필요합니다.
"""
    
    try:
        start_llm = time.time()
        final_answer = agent.llm.invoke(synthesis_prompt, config={"callbacks": [langfuse_handler]}).content
        llm_time = time.time() - start_llm
        
        print(f"    ✅ LLM 답변 생성 완료 (소요시간: {llm_time:.2f}초)")
        print(f"    📝 답변 길이: {len(final_answer)}자")
        
        # Step 4: 신뢰도 및 메타데이터 계산
        print(f"  📊 Step 5.4: 신뢰도 및 메타데이터 계산...")
        
        # 신뢰도 계산 (정교한 알고리즘)
        base_confidence = 0.3
        
        # 뉴스 기반 신뢰도 가산
        if chroma_results:
            news_confidence = min(chroma_quality.get("avg_relevance", 0) * 0.4, 0.4)
            quality_bonus = chroma_quality.get("high_quality_ratio", 0) * 0.2
            recency_bonus = 0.1 if chroma_quality.get("has_recent_content", False) else 0
            base_confidence += news_confidence + quality_bonus + recency_bonus
        
        # 그래프 기반 신뢰도 가산
        if neo4j_results:
            graph_confidence = min(len(neo4j_results) / 10 * 0.2, 0.2)
            diversity_bonus = min(neo4j_quality.get("relationship_diversity", 0) / 5 * 0.1, 0.1)
            base_confidence += graph_confidence + diversity_bonus
        
        # 통합 보너스
        if chroma_results and neo4j_results:
            base_confidence += 0.1
        
        final_confidence = min(base_confidence, 1.0)
        
        # 출처 정리
        sources = []
        for news in structured_news:
            if news["url"]:
                sources.append(news["url"])
        
        print(f"    🎯 최종 신뢰도: {final_confidence:.3f}")
        print(f"    📚 참고 출처: {len(sources)}개")
        
        synthesis_process["end_time"] = time.time()
        synthesis_process["total_time"] = synthesis_process["end_time"] - synthesis_process["start_time"]
        synthesis_process["success"] = True
        synthesis_process["final_confidence"] = final_confidence
        synthesis_process["answer_length"] = len(final_answer)
        
        agent.log_step("synthesis", synthesis_process)
        
        return {
            "final_answer": final_answer,
            "confidence_score": final_confidence,
            "sources": sources,
            "synthesis_process": synthesis_process
        }
        
    except Exception as e:
        print(f"    ❌ 답변 생성 오류: {e}")
        
        synthesis_process["success"] = False
        synthesis_process["error"] = str(e)
        
        return {
            "final_answer": f"죄송합니다. '{question}'에 대한 답변을 생성하는 중 오류가 발생했습니다: {e}",
            "confidence_score": 0.0,
            "sources": [],
            "synthesis_process": synthesis_process
        }

def create_enhanced_hybrid_rag_graph():
    """강화된 하이브리드 RAG 그래프 생성"""
    checkpointer = InMemorySaver()
    
    graph = StateGraph(DetailedHybridRAGState)
    
    # 노드 추가 (순차적 + 병렬 조합)
    graph.add_node("analyze_question", analyze_question_strategy)
    graph.add_node("chroma_search", enhanced_chroma_search_node)
    graph.add_node("neo4j_search", enhanced_neo4j_search_node)
    graph.add_node("quality_assessment", quality_assessment_node)
    graph.add_node("synthesis", enhanced_synthesis_node)
    
    # 엣지 정의 (개선된 플로우)
    graph.add_edge(START, "analyze_question")
    
    # 검색 전략에 따른 병렬 실행
    graph.add_edge("analyze_question", "chroma_search")
    graph.add_edge("analyze_question", "neo4j_search")
    
    # 품질 평가 후 통합
    graph.add_edge("chroma_search", "quality_assessment")
    graph.add_edge("neo4j_search", "quality_assessment")
    graph.add_edge("quality_assessment", "synthesis")
    graph.add_edge("synthesis", END)
    
    return graph.compile(checkpointer=checkpointer)

def main():
    """강화된 하이브리드 RAG 시스템 테스트"""
    print("🚀 Enhanced Hybrid RAG Agent (상세 과정 추적)")
    print("=" * 60)
    
    # 그래프 생성
    enhanced_graph = create_enhanced_hybrid_rag_graph()
    
    # 테스트 질문
    test_questions = [
        # "최근 정치적 사건들이 국정 운영에 미친 영향은?",
        # "삼성전자 이재용 회장의 무죄 판결에 대해 알려주세요",
        # "최근 한국 경제 상황은 어떤가요?",
        "현재 한국 경제는 어떤상황인가요? 코스피는 왜 상승하나요?",
        "코스피 상승에 미국과의 관계",
        ""
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*10} 질문 {i} {'='*10}")
        print(f"❓ {question}")
        print("=" * 60)
        
        # 초기 상태
        initial_state = {
            "question": question,
            "search_strategy": {},
            "chroma_process": {},
            "neo4j_process": {},
            "quality_assessment": {},
            "synthesis_process": {},
            "chroma_results": [],
            "neo4j_results": [],
            "final_answer": "",
            "sources": [],
            "confidence_score": 0.0,
            "iteration_count": 0,
            "execution_log": []
        }
        
        # 실행
        config = {"configurable": {"thread_id": f"enhanced_session_{i}"}, "callbacks": [langfuse_handler]}
        
        try:
            start_total = time.time()
            result = enhanced_graph.invoke(initial_state, config=config)
            total_time = time.time() - start_total
            
            print(f"\n🎉 실행 완료 (총 소요시간: {total_time:.2f}초)")
            print("=" * 60)
            
            print(f"\n📋 최종 답변:")
            print(result["final_answer"])
            
            print(f"\n📊 결과 요약:")
            print(f"  🎯 신뢰도: {result['confidence_score']:.3f}")
            print(f"  📰 뉴스 출처: {len(result.get('chroma_results', []))}개")
            print(f"  🔗 그래프 관계: {len(result.get('neo4j_results', []))}개")
            print(f"  ⏱️ 총 처리시간: {total_time:.2f}초")
            
            if result.get("sources"):
                print(f"\n🔍 참고 출처:")
                for j, source in enumerate(result["sources"][:3], 1):
                    print(f"  {j}. {source}")
            
            # 실행 로그 요약
            quality_assessment = result.get("quality_assessment", {})
            if quality_assessment:
                overall_quality = quality_assessment.get("overall_quality", {})
                print(f"\n📈 품질 평가:")
                print(f"  📄 뉴스 품질: {quality_assessment.get('chroma_quality', {}).get('avg_relevance', 0):.3f}")
                print(f"  🔗 그래프 품질: {quality_assessment.get('neo4j_quality', {}).get('result_count', 0)}개 관계")
                print(f"  🏆 종합 등급: {overall_quality.get('quality_tier', 'unknown').upper()}")
        
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            import traceback
            print(traceback.format_exc())
        
        print("\n" + "=" * 60)
        
        # 다음 질문 전 잠시 대기
        if i < len(test_questions):
            print("다음 질문으로 이동 중...")
            time.sleep(1)

if __name__ == "__main__":
    main() 