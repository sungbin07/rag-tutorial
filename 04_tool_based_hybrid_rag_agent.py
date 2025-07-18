#!/usr/bin/env python3
"""
Tool-based Hybrid RAG Agent: ChromaDB + Neo4j + LangGraph with Tools
LLM이 동적으로 도구를 선택하여 검색하고 추론하는 Agent 시스템
"""

from typing import TypedDict, List, Dict, Optional, Annotated
import json
import time
from datetime import datetime
import os
import asyncio

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

# LangGraph imports  
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages

# Database imports
import chromadb
from neo4j import GraphDatabase

# Environment and logging
from dotenv import load_dotenv
from langfuse.langchain import CallbackHandler

load_dotenv()

# Langfuse setup
langfuse_handler = CallbackHandler() if os.getenv('LANGFUSE_PUBLIC_KEY') else None

# State definition with message passing
class ToolAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    search_results: Dict
    final_answer: str
    confidence_score: float
    sources: List[str]
    execution_log: List[Dict]

# Tool functions
@tool
def analyze_question_tool(question: str) -> str:
    """
    질문을 분석하여 검색 전략과 핵심 키워드를 추출합니다.
    
    Args:
        question: 사용자 질문
        
    Returns:
        JSON 형식의 분석 결과 (질문 유형, 키워드, 검색 전략 등)
    """
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
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
        
        config = {"callbacks": [langfuse_handler]} if langfuse_handler else {}
        response = llm.invoke(analysis_prompt, config=config)
        
        # JSON 파싱 시도
        try:
            strategy = json.loads(response.content)
        except:
            # 파싱 실패시 기본 전략
            strategy = {
                "question_type": "일반 질문",
                "keywords": question.split()[:3],
                "time_scope": "최신",
                "search_priority": "parallel",
                "complexity": "complex",
                "reasoning": "자동 분석 실패로 기본 전략 적용"
            }
        
        print(f"🎯 질문 분석 완료: {strategy['question_type']}")
        print(f"🔑 키워드: {', '.join(strategy['keywords'])}")
        print(f"🎯 검색 전략: {strategy['search_priority']}")
        
        return json.dumps(strategy, ensure_ascii=False)
        
    except Exception as e:
        error_result = {
            "question_type": "오류",
            "keywords": question.split()[:3],
            "search_priority": "parallel",
            "complexity": "complex",
            "reasoning": f"분석 중 오류 발생: {e}"
        }
        return json.dumps(error_result, ensure_ascii=False)

@tool
def search_chroma_news_tool(query: str, keywords: Optional[str] = None) -> str:
    """
    ChromaDB에서 뉴스 기사를 의미적 유사도로 검색합니다.
    
    Args:
        query: 검색할 질문이나 쿼리
        keywords: 추가 키워드 (JSON 문자열 형태)
        
    Returns:
        JSON 형식의 검색 결과
    """
    try:
        print(f"🔍 ChromaDB 뉴스 검색 시작: '{query}'")
        
        # ChromaDB 클라이언트 초기화
        chroma_client = chromadb.PersistentClient(path="../chroma_db_news_3")
        news_collection = chroma_client.get_collection("naver_news")
        
        # 임베딩 생성
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        query_embedding = embeddings.embed_query(query)
        
        # 검색 실행
        results = news_collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            include=["documents", "metadatas", "distances"]
        )
        
        # 결과 처리
        search_results = []
        if results["documents"] and results["documents"][0]:
            for i, (doc, metadata, distance) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0], 
                results["distances"][0]
            )):
                relevance_score = 1 - distance
                
                # 키워드 보너스 계산
                keyword_bonus = 0
                if keywords:
                    try:
                        keyword_list = json.loads(keywords) if isinstance(keywords, str) else keywords
                        if isinstance(keyword_list, list):
                            for keyword in keyword_list:
                                if keyword.lower() in doc.lower():
                                    keyword_bonus += 0.1
                    except:
                        pass
                
                final_score = min(relevance_score + keyword_bonus, 1.0)
                
                search_results.append({
                    "content": doc[:300] + "..." if len(doc) > 300 else doc,
                    "title": metadata.get("title", "제목 없음"),
                    "url": metadata.get("url", ""),
                    "published_date": metadata.get("published_date", ""),
                    "relevance_score": final_score,
                    "rank": i + 1
                })
        
        result = {
            "success": True,
            "results_count": len(search_results),
            "results": search_results,
            "search_method": "semantic_embedding"
        }
        
        print(f"✅ ChromaDB 검색 완료: {len(search_results)}개 결과")
        return json.dumps(result, ensure_ascii=False)
        
    except Exception as e:
        print(f"❌ ChromaDB 검색 오류: {e}")
        error_result = {
            "success": False,
            "error": str(e),
            "results": []
        }
        return json.dumps(error_result, ensure_ascii=False)

@tool  
def search_neo4j_graph_tool(query: str, keywords: Optional[str] = None) -> str:
    """
    Neo4j 그래프 데이터베이스에서 관계 정보를 검색합니다.
    
    Args:
        query: 검색할 질문이나 쿼리
        keywords: 검색에 사용할 키워드 (JSON 문자열 형태)
        
    Returns:
        JSON 형식의 그래프 관계 결과
    """
    try:
        print(f"🔗 Neo4j 그래프 검색 시작: '{query}'")
        
        # Neo4j 연결
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        username = os.getenv("NEO4J_USERNAME", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")
        
        driver = GraphDatabase.driver(uri, auth=(username, password))
        
        # 키워드 파싱
        search_keywords = []
        if keywords:
            try:
                keyword_list = json.loads(keywords) if isinstance(keywords, str) else keywords
                if isinstance(keyword_list, list):
                    search_keywords = keyword_list
            except:
                search_keywords = query.split()[:3]
        else:
            search_keywords = query.split()[:3]
        
        print(f"🔑 검색 키워드: {', '.join(search_keywords)}")
        
        # 다양한 검색 패턴 시도
        search_patterns = [
            # 패턴 1: 직접 관계 검색
            """
            MATCH (a)-[r]->(b)
            WHERE any(keyword IN $keywords WHERE 
                a.name CONTAINS keyword OR b.name CONTAINS keyword)
            RETURN a.name as source, type(r) as relationship, b.name as target,
                   'direct' as pattern_type
            LIMIT 5
            """,
            # 패턴 2: 인과관계 체인
            """
            MATCH (a)-[:원인이다]->(b)-[:결과이다]->(c)
            WHERE any(keyword IN $keywords WHERE 
                a.name CONTAINS keyword OR b.name CONTAINS keyword OR c.name CONTAINS keyword)
            RETURN a.name as cause, b.name as intermediate, c.name as effect,
                   'causal' as pattern_type
            LIMIT 3
            """,
            # 패턴 3: 키워드 기반 네트워크
            """
            MATCH (center)-[r]-(connected)
            WHERE any(keyword IN $keywords WHERE center.name CONTAINS keyword)
            RETURN center.name as center_entity, type(r) as relationship, 
                   connected.name as related_entity, 'network' as pattern_type
            LIMIT 4
            """
        ]
        
        all_results = []
        
        with driver.session() as session:
            for i, pattern in enumerate(search_patterns, 1):
                try:
                    results = session.run(pattern, {"keywords": search_keywords}).data()
                    
                    if results:
                        print(f"  📋 패턴 {i}: {len(results)}개 관계 발견")
                        all_results.extend(results)
                    
                except Exception as pattern_error:
                    print(f"  ⚠️ 패턴 {i} 실행 오류: {pattern_error}")
                    continue
        
        driver.close()
        
        # 중복 제거 및 정리
        unique_results = []
        seen_relationships = set()
        
        for result in all_results:
            # 관계 키 생성
            if "relationship" in result:
                key = f"{result.get('source', '')}-{result['relationship']}-{result.get('target', '')}"
            elif "cause" in result:
                key = f"{result['cause']}-{result['intermediate']}-{result['effect']}"
            else:
                key = str(result)
            
            if key not in seen_relationships:
                seen_relationships.add(key)
                unique_results.append(result)
        
        final_result = {
            "success": True,
            "results_count": len(unique_results),
            "results": unique_results[:8],  # 상위 8개만 선택
            "search_keywords": search_keywords
        }
        
        print(f"✅ Neo4j 검색 완료: {len(unique_results)}개 관계")
        return json.dumps(final_result, ensure_ascii=False)
        
    except Exception as e:
        print(f"❌ Neo4j 검색 오류: {e}")
        error_result = {
            "success": False,
            "error": str(e),
            "results": []
        }
        return json.dumps(error_result, ensure_ascii=False)

@tool
def synthesize_answer_tool(question: str, chroma_results: str, neo4j_results: str) -> str:
    """
    ChromaDB와 Neo4j 검색 결과를 통합하여 최종 답변을 생성합니다.
    
    Args:
        question: 원본 질문
        chroma_results: ChromaDB 검색 결과 (JSON 문자열)
        neo4j_results: Neo4j 검색 결과 (JSON 문자열)
        
    Returns:
        JSON 형식의 최종 답변과 메타데이터
    """
    try:
        print(f"🧠 답변 통합 생성 시작")
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        
        # 결과 파싱
        try:
            if isinstance(chroma_results, str):
                chroma_data = json.loads(chroma_results)
            elif isinstance(chroma_results, dict):
                chroma_data = chroma_results
            elif isinstance(chroma_results, list):
                chroma_data = {"results": chroma_results}
            else:
                chroma_data = {"results": []}
                
            if isinstance(neo4j_results, str):
                neo4j_data = json.loads(neo4j_results)
            elif isinstance(neo4j_results, dict):
                neo4j_data = neo4j_results
            elif isinstance(neo4j_results, list):
                neo4j_data = {"results": neo4j_results}
            else:
                neo4j_data = {"results": []}
        except:
            chroma_data = {"results": []}
            neo4j_data = {"results": []}
        
        # 뉴스 정보 정리
        news_context = ""
        sources = []
        if chroma_data.get("results"):
            news_context = "\n=== 관련 뉴스 정보 ===\n"
            for i, news in enumerate(chroma_data["results"][:3], 1):
                news_context += f"[뉴스 {i}] {news.get('title', '제목없음')}\n"
                news_context += f"내용: {news.get('content', '내용없음')}\n"
                news_context += f"날짜: {news.get('published_date', '날짜없음')}\n"
                news_context += f"관련도: {news.get('relevance_score', 0):.3f}\n\n"
                
                if news.get('url'):
                    sources.append(news['url'])
        
        # 그래프 관계 정리
        graph_context = ""
        if neo4j_data.get("results"):
            graph_context = "\n=== 관련 그래프 정보 ===\n"
            for i, rel in enumerate(neo4j_data["results"][:5], 1):
                if "relationship" in rel:
                    graph_context += f"[관계 {i}] {rel.get('source', '')} -[{rel['relationship']}]-> {rel.get('target', '')}\n"
                elif "cause" in rel:
                    graph_context += f"[인과 {i}] {rel['cause']} → {rel['intermediate']} → {rel['effect']}\n"
                else:
                    graph_context += f"[기타 {i}] {rel}\n"
        
        # 통합 답변 생성 프롬프트
        synthesis_prompt = f"""
당신은 뉴스 분석 전문가입니다. 실제 뉴스 기사와 관계 그래프 정보를 통합하여 정확하고 포괄적인 답변을 제공하세요.

질문: {question}

{news_context}

{graph_context}

답변 생성 지침:
1. 뉴스 기사의 구체적 사실을 답변의 핵심으로 사용
2. 그래프 관계로 맥락과 배경 설명 보강  
3. 날짜, 출처, 구체적 수치 등 사실 정보 명시
4. 추측이나 일반 지식 사용 금지
5. 자연스럽고 체계적인 문단 구성
6. 정보의 신뢰도와 한계 명시

답변을 JSON 형식으로 제공하세요:
{{
    "answer": "상세한 답변 내용",
    "confidence": 0.0-1.0 사이의 신뢰도,
    "reasoning": "답변 생성 과정 설명",
    "limitations": "답변의 한계나 주의사항"
}}
"""
        
        config = {"callbacks": [langfuse_handler]} if langfuse_handler else {}
        response = llm.invoke(synthesis_prompt, config=config)
        
        # JSON 파싱 시도
        try:
            answer_data = json.loads(response.content)
        except:
            # 파싱 실패시 기본 구조
            answer_data = {
                "answer": response.content,
                "confidence": 0.7,
                "reasoning": "자동 생성된 답변",
                "limitations": "JSON 파싱 실패로 기본 형식 사용"
            }
        
        # 최종 결과 구성
        final_result = {
            "success": True,
            "answer": answer_data.get("answer", "답변 생성 실패"),
            "confidence_score": answer_data.get("confidence", 0.5),
            "reasoning": answer_data.get("reasoning", ""),
            "limitations": answer_data.get("limitations", ""),
            "sources": sources,
            "news_count": len(chroma_data.get("results", [])),
            "graph_relations_count": len(neo4j_data.get("results", []))
        }
        
        print(f"✅ 답변 생성 완료 (신뢰도: {final_result['confidence_score']:.3f})")
        return json.dumps(final_result, ensure_ascii=False)
        
    except Exception as e:
        print(f"❌ 답변 생성 오류: {e}")
        error_result = {
            "success": False,
            "error": str(e),
            "answer": f"답변 생성 중 오류가 발생했습니다: {e}",
            "confidence_score": 0.0
        }
        return json.dumps(error_result, ensure_ascii=False)

# Tool 리스트
tools = [
    analyze_question_tool,
    search_chroma_news_tool, 
    search_neo4j_graph_tool,
    synthesize_answer_tool
]

# Agent 노드 함수들
def agent_node(state: ToolAgentState):
    """LLM Agent가 도구를 선택하고 실행하는 노드"""
    messages = state["messages"]
    
    # Tool-calling이 가능한 LLM 생성
    llm_with_tools = ChatOpenAI(
        model="gpt-4o-mini", 
        temperature=0
    ).bind_tools(tools)
    
    # Agent 시스템 프롬프트
    agent_prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 뉴스 분석 전문가이자 하이브리드 RAG 시스템입니다.

다음 도구들을 사용하여 사용자의 질문에 정확하고 포괄적으로 답변하세요:

1. analyze_question_tool: 질문을 분석하고 검색 전략을 수립
2. search_chroma_news_tool: ChromaDB에서 관련 뉴스 검색
3. search_neo4j_graph_tool: Neo4j에서 관계 그래프 검색  
4. synthesize_answer_tool: 검색 결과를 통합하여 최종 답변 생성

작업 순서:
1. 먼저 질문을 분석하여 검색 전략을 수립하세요
2. 분석 결과에 따라 ChromaDB와 Neo4j에서 병렬 검색을 수행하세요
3. 검색 결과를 통합하여 최종 답변을 생성하세요
4. 답변에는 신뢰도, 출처, 한계점을 포함하세요

중요: 각 도구의 결과를 확인한 후 다음 단계를 진행하세요."""),
        ("placeholder", "{messages}")
    ])
    
    # 프롬프트 적용
    prompt_response = agent_prompt.invoke({"messages": messages})
    
    # LLM 응답 생성
    config = {"callbacks": [langfuse_handler]} if langfuse_handler else {}
    response = llm_with_tools.invoke(prompt_response.messages, config=config)
    
    return {"messages": [response]}

def tool_node(state: ToolAgentState):
    """도구 실행 노드"""
    messages = state["messages"]
    last_message = messages[-1]
    
    # ToolNode를 사용하여 도구 호출 실행
    tool_executor = ToolNode(tools)
    return tool_executor.invoke(state)

def should_continue(state: ToolAgentState):
    """다음 단계 결정 함수"""
    messages = state["messages"]
    last_message = messages[-1]
    
    # 마지막 메시지가 도구 호출을 포함하면 도구 실행
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    # 그렇지 않으면 종료
    else:
        return "end"

class ToolBasedHybridRAGAgent:
    """Tool-calling 기반 하이브리드 RAG Agent"""
    
    def __init__(self):
        self.checkpointer = InMemorySaver()
        self.graph = self._create_agent_graph()
        
    def _create_agent_graph(self):
        """Agent 그래프 생성"""
        # StateGraph 초기화
        workflow = StateGraph(ToolAgentState)
        
        # 노드 추가
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", tool_node)
        
        # 시작점 설정
        workflow.add_edge(START, "agent")
        
        # 조건부 엣지 추가
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "tools": "tools",
                "end": END
            }
        )
        
        # 도구 실행 후 다시 agent로
        workflow.add_edge("tools", "agent")
        
        # 그래프 컴파일
        return workflow.compile(checkpointer=self.checkpointer)
    
    def run(self, question: str, thread_id: str = None):
        """Agent 실행"""
        if thread_id is None:
            thread_id = f"session_{int(time.time())}"
        
        print(f"🚀 Tool-based Hybrid RAG Agent 시작")
        print(f"❓ 질문: {question}")
        print("=" * 60)
        
        # 초기 상태 설정
        initial_state = {
            "messages": [HumanMessage(content=question)],
            "question": question,
            "search_results": {},
            "final_answer": "",
            "confidence_score": 0.0,
            "sources": [],
            "execution_log": []
        }
        
        # 설정
        config = {
            "configurable": {"thread_id": thread_id},
            "callbacks": [langfuse_handler] if langfuse_handler else []
        }
        
        try:
            start_time = time.time()
            
            # 그래프 실행
            result = self.graph.invoke(initial_state, config=config)
            
            execution_time = time.time() - start_time
            
            # 결과 파싱
            final_result = self._parse_final_result(result, execution_time)
            
            print(f"\n🎉 실행 완료 (소요시간: {execution_time:.2f}초)")
            print("=" * 60)
            
            return final_result
            
        except Exception as e:
            print(f"❌ Agent 실행 오류: {e}")
            import traceback
            print(traceback.format_exc())
            
            return {
                "success": False,
                "error": str(e),
                "question": question,
                "final_answer": f"오류가 발생했습니다: {e}",
                "confidence_score": 0.0,
                "sources": [],
                "execution_time": 0
            }
    
    def _parse_final_result(self, result, execution_time):
        """결과 파싱 및 정리"""
        messages = result.get("messages", [])
        
        # 마지막 AI 메시지에서 최종 답변 찾기
        final_answer = ""
        confidence_score = 0.0
        sources = []
        
        # Tool 호출 결과들 수집
        tool_results = []
        for message in messages:
            if hasattr(message, 'content') and message.content:
                # synthesize_answer_tool의 결과인지 확인
                if isinstance(message.content, str):
                    try:
                        # JSON 파싱 시도
                        parsed_content = json.loads(message.content)
                        if isinstance(parsed_content, dict) and "answer" in parsed_content:
                            final_answer = parsed_content.get("answer", "")
                            confidence_score = parsed_content.get("confidence_score", 0.0)
                            sources = parsed_content.get("sources", [])
                            break
                    except:
                        # JSON이 아닌 경우 그대로 사용
                        if len(message.content) > 50:  # 충분히 긴 답변인 경우
                            final_answer = message.content
                            confidence_score = 0.7
        
        # 마지막 AI 메시지가 최종 답변인 경우
        if not final_answer:
            for message in reversed(messages):
                if hasattr(message, 'content') and hasattr(message, 'type'):
                    if message.type == 'ai' and message.content:
                        final_answer = message.content
                        confidence_score = 0.5
                        break
        
        if not final_answer:
            final_answer = "답변을 생성하지 못했습니다."
        
        # Tool 사용 통계
        tool_calls_count = 0
        used_tools = set()
        
        for message in messages:
            if hasattr(message, 'tool_calls') and message.tool_calls:
                tool_calls_count += len(message.tool_calls)
                for tool_call in message.tool_calls:
                    used_tools.add(tool_call['name'])
        
        return {
            "success": True,
            "question": result.get("question", ""),
            "final_answer": final_answer,
            "confidence_score": confidence_score,
            "sources": sources,
            "execution_time": execution_time,
            "tool_calls_count": tool_calls_count,
            "used_tools": list(used_tools),
            "message_count": len(messages)
        }

def main():
    """메인 실행 함수"""
    print("🚀 Tool-based Hybrid RAG Agent 시스템")
    print("=" * 60)
    
    # Agent 초기화
    agent = ToolBasedHybridRAGAgent()
    
    # 테스트 질문들
    test_questions = [
        "최근 정치적 사건들이 국정 운영에 미친 영향은?",
        "최근 한국 경제 상황은 어떤가요?",
        "코스피 지수가 최근 상승한 이유는 무엇인가요?",
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*10} 질문 {i} {'='*10}")
        
        try:
            result = agent.run(question, thread_id=f"test_session_{i}")
            
            print(f"\n📋 최종 답변:")
            print(result["final_answer"])
            
            print(f"\n📊 실행 통계:")
            print(f"  🎯 신뢰도: {result['confidence_score']:.3f}")
            print(f"  🔧 사용된 도구: {', '.join(result.get('used_tools', []))}")
            print(f"  📞 도구 호출 횟수: {result.get('tool_calls_count', 0)}회")
            print(f"  💬 메시지 수: {result.get('message_count', 0)}개")
            print(f"  ⏱️ 실행 시간: {result['execution_time']:.2f}초")
            
            if result.get("sources"):
                print(f"\n🔍 참고 출처:")
                for j, source in enumerate(result["sources"][:3], 1):
                    print(f"  {j}. {source}")
                    
        except Exception as e:
            print(f"❌ 질문 {i} 실행 오류: {e}")
        
        print("\n" + "=" * 60)
        
        # 다음 질문 전 잠시 대기
        if i < len(test_questions):
            print("다음 질문으로 이동 중...")
            time.sleep(1)

if __name__ == "__main__":
    main() 