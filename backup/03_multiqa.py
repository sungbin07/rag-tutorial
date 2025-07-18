#!/usr/bin/env python3
"""
Neo4j 기반 Multi-hop QA 데이터 생성 시스템

이 모듈은 구축된 Neo4j 그래프 데이터베이스를 활용하여 
multi-hop reasoning이 필요한 QA 데이터셋을 자동으로 생성합니다.

주요 기능:
1. Neo4j 그래프에서 2-3 hop 경로 탐색
2. 경로 기반 자연어 질문 자동 생성
3. Reasoning trace 포함 답변 생성
4. 다양한 질문 유형 템플릿 적용
5. 품질 평가 및 필터링
"""

import os
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from pathlib import Path

# LangChain imports
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

# Langfuse imports
from langfuse.langchain import CallbackHandler
from langfuse import Langfuse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class QuestionType(Enum):
    """질문 유형 분류"""
    ENTITY_RELATIONSHIP = "entity_relationship"  # 엔티티 간 관계
    CAUSAL_INFERENCE = "causal_inference"        # 인과 관계 추론
    TEMPORAL_SEQUENCE = "temporal_sequence"      # 시간적 순서
    MULTI_HOP_FACT = "multi_hop_fact"           # 다중 홉 사실 확인
    COMPARATIVE = "comparative"                  # 비교 분석
    AGGREGATIVE = "aggregative"                 # 집계 정보


class DifficultyLevel(Enum):
    """난이도 분류"""
    EASY = "easy"      # 2-hop, 직접적 관계
    MEDIUM = "medium"  # 2-3 hop, 중간 추론
    HARD = "hard"      # 3+ hop, 복잡한 추론


@dataclass
class GraphPath:
    """그래프 경로 정보"""
    start_node: Dict[str, Any]
    end_node: Dict[str, Any]
    path_length: int
    relationships: List[Dict[str, Any]]
    path_description: str


@dataclass
class MultiHopQA:
    """Multi-hop QA 데이터 구조"""
    question: str
    answer: str
    reasoning_trace: List[str]
    question_type: QuestionType
    difficulty: DifficultyLevel
    confidence_score: float
    source_path: GraphPath
    metadata: Dict[str, Any]


class Neo4jMultiHopQAGenerator:
    """Neo4j 기반 Multi-hop QA 생성기 (Langfuse 통합)"""
    
    def __init__(self, enable_langfuse: bool = True):
        """초기화 및 연결 설정"""
        self.neo4j_uri = os.getenv("NEO4J_URI")
        self.neo4j_username = os.getenv("NEO4J_USERNAME") 
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")
        
        if not all([self.neo4j_uri, self.neo4j_username, self.neo4j_password]):
            raise ValueError("Neo4j 연결 정보가 환경변수에 설정되지 않았습니다.")
        
        # Neo4j 연결
        self.graph = Neo4jGraph(
            url=self.neo4j_uri,
            username=self.neo4j_username,
            password=self.neo4j_password
        )
        
        # Langfuse 설정
        self.enable_langfuse = enable_langfuse
        if enable_langfuse:
            try:
                self.langfuse_handler = CallbackHandler()
                logger.info("Langfuse 연동 활성화")
            except Exception as e:
                logger.warning(f"Langfuse 연동 실패, 비활성화됨: {e}")
                self.enable_langfuse = False
                self.langfuse_handler = None
        else:
            self.langfuse_handler = None
        
        # LLM 설정
        self.llm = ChatOpenAI(
            model="gpt-4o-mini", 
            temperature=0.3
        )
        
        # Cypher QA Chain 설정
        self.cypher_chain = GraphCypherQAChain.from_llm(
            llm=self.llm,
            graph=self.graph,
            allow_dangerous_requests=True,
            verbose=False
        )
        
        logger.info("Neo4j Multi-hop QA Generator 초기화 완료")

    def get_graph_schema(self) -> str:
        """그래프 스키마 정보 조회"""
        try:
            self.graph.refresh_schema()
            return self.graph.schema
        except Exception as e:
            logger.error(f"스키마 조회 오류: {e}")
            return ""

    def discover_multihop_paths(self, max_paths: int = 50) -> List[GraphPath]:
        """Multi-hop 경로 탐색"""
        logger.info("Multi-hop 경로 탐색 시작...")
        
        # 실제 데이터베이스 구조에 맞는 2-3 hop 경로 탐색 쿼리
        path_queries = [
            # 2-hop 경로: Article -> Category -> Article
            """
            MATCH path = (start:Article)-[r1:BELONGS_TO]->(middle:Category)-[r2:BELONGS_TO]-(end:Article)
            WHERE start <> end
            AND start.title IS NOT NULL 
            AND end.title IS NOT NULL
            WITH path, start, middle, end, r1, r2
            RETURN 
                start, middle, end,
                type(r1) as rel1_type, type(r2) as rel2_type,
                length(path) as path_length,
                path
            ORDER BY rand()
            LIMIT $limit
            """,
            
            # 2-hop 경로: Article -> Source -> Article  
            """
            MATCH path = (start:Article)-[r1:PUBLISHED]-(middle:Source)-[r2:PUBLISHED]-(end:Article)
            WHERE start <> end
            AND start.title IS NOT NULL 
            AND end.title IS NOT NULL
            WITH path, start, middle, end, r1, r2
            RETURN 
                start, middle, end,
                type(r1) as rel1_type, type(r2) as rel2_type,
                length(path) as path_length,
                path
            ORDER BY rand()
            LIMIT $limit
            """,
            
            # 3-hop 경로: Article -> Category -> Article -> Category
            """
            MATCH path = (start:Article)-[r1:BELONGS_TO]->(n1:Category)-[r2:BELONGS_TO]-(n2:Article)-[r3:BELONGS_TO]->(end:Category)
            WHERE start <> n2 AND n1 <> end
            AND start.title IS NOT NULL 
            AND n2.title IS NOT NULL
            WITH path, start, n1, n2, end, r1, r2, r3
            RETURN 
                start, n1, n2, end,
                type(r1) as rel1_type, type(r2) as rel2_type, type(r3) as rel3_type,
                length(path) as path_length,
                path
            ORDER BY rand()
            LIMIT $limit
            """
        ]
        
        all_paths = []
        
        for query in path_queries:
            try:
                results = self.graph.query(query, {"limit": max_paths // 2})
                for result in results:
                    path = self._parse_path_result(result)
                    if path:
                        all_paths.append(path)
            except Exception as e:
                logger.error(f"경로 탐색 쿼리 실행 오류: {e}")
                continue
        
        logger.info(f"총 {len(all_paths)}개의 multi-hop 경로 발견")
        return all_paths[:max_paths]

    def _get_node_label(self, node: Dict[str, Any]) -> str:
        """노드의 라벨(타입) 추출"""
        # 쿼리에서 명시적으로 지정한 노드 타입 기반
        if isinstance(node, dict):
            # 노드 딕셔너리에서 라벨 정보를 찾기
            if 'title' in node:  # Article 노드
                return "Article"
            elif 'name' in node and len(node.get('name', '')) < 20:  # Category, Source 등
                return "Category" if node.get('name') in ['일반', '경제', '사회', '국제', '스포츠', 'IT', '정치', '문화', '세계'] else "Source"
            else:
                return "Node"
        return "Node"

    def _get_node_display_name(self, node: Dict[str, Any]) -> str:
        """노드의 표시용 이름 추출"""
        if isinstance(node, dict):
            # Article 노드의 경우 title 사용 (첫 50자만)
            if 'title' in node:
                title = node['title']
                return title[:50] + "..." if len(title) > 50 else title
            # Category, Source 등의 경우 name 사용
            elif 'name' in node:
                return node['name']
            # 기타 식별 가능한 속성들
            elif 'content' in node:
                content = node['content']
                return content[:30] + "..." if len(content) > 30 else content
            elif 'id' in node:
                return f"ID:{node['id']}"
            else:
                return "Unknown"
        return str(node)[:50]

    def _parse_path_result(self, result: Dict[str, Any]) -> Optional[GraphPath]:
        """경로 결과 파싱"""
        try:
            path_length = result.get('path_length', 0)
            
            if path_length == 2:
                start_node = result['start']
                end_node = result['end']
                relationships = [
                    {'type': result['rel1_type'], 'direction': '->'},
                    {'type': result['rel2_type'], 'direction': '->'}
                ]
            elif path_length == 3:
                start_node = result['start']
                end_node = result['end']
                relationships = [
                    {'type': result['rel1_type'], 'direction': '->'},
                    {'type': result['rel2_type'], 'direction': '->'},
                    {'type': result['rel3_type'], 'direction': '->'}
                ]
            else:
                return None
            
            # 경로 설명 생성 - 노드 타입 추출
            start_label = self._get_node_label(start_node)
            end_label = self._get_node_label(end_node)
            
            # 노드별로 적절한 속성 선택
            start_name = self._get_node_display_name(start_node)
            end_name = self._get_node_display_name(end_node)
            
            path_description = f"{start_label}({start_name})"
            for rel in relationships:
                path_description += f" -{rel['type']}-> "
            path_description += f"{end_label}({end_name})"
            
            return GraphPath(
                start_node=dict(start_node),
                end_node=dict(end_node),
                path_length=path_length,
                relationships=relationships,
                path_description=path_description
            )
            
        except Exception as e:
            logger.error(f"경로 파싱 오류: {e}")
            return None

    def generate_question_from_path(self, path: GraphPath) -> Optional[MultiHopQA]:
        """경로 기반 질문 생성 (Langfuse 추적 포함)"""
        
        # 질문 생성 프롬프트 템플릿
        question_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 지식 그래프 경로를 분석하여 자연스러운 한국어 질문을 생성하는 전문가입니다.

주어진 그래프 경로를 바탕으로 multi-hop reasoning이 필요한 질문을 생성해주세요.

질문 생성 가이드라인:
1. 자연스럽고 명확한 한국어 질문
2. 시작 노드에서 끝 노드까지의 연결 관계를 탐색하는 질문
3. 단순한 사실 확인이 아닌 추론이 필요한 질문
4. 실제 사용자가 물어볼 법한 실용적인 질문

응답 형식:
{{
    "question": "생성된 질문",
    "question_type": "entity_relationship|causal_inference|temporal_sequence|multi_hop_fact|comparative|aggregative",
    "difficulty": "easy|medium|hard",
    "reasoning_strategy": "이 질문을 답하기 위한 추론 전략 설명"
}}"""),
            ("human", """그래프 경로 정보:
- 시작 노드: {start_node}
- 끝 노드: {end_node}  
- 경로 길이: {path_length}
- 관계들: {relationships}
- 경로 설명: {path_description}

이 경로를 기반으로 질문을 생성해주세요.""")
        ])
        
        try:
            # Langfuse config 설정
            config = {"callbacks": [self.langfuse_handler]} if self.enable_langfuse else {}
            
            # 질문 생성
            response = self.llm.invoke(
                question_prompt.format_messages(
                    start_node=path.start_node,
                    end_node=path.end_node,
                    path_length=path.path_length,
                    relationships=path.relationships,
                    path_description=path.path_description
                ),
                config=config
            )
            
            # JSON 파싱
            try:
                question_data = json.loads(response.content)
            except json.JSONDecodeError:
                # JSON 형식이 아닌 경우 단순 텍스트로 처리
                question_data = {
                    "question": response.content,
                    "question_type": "multi_hop_fact",
                    "difficulty": "medium",
                    "reasoning_strategy": "그래프 경로 기반 추론"
                }
            
            # 답변 및 reasoning trace 생성
            answer_data = self._generate_answer_with_trace(
                question_data["question"], 
                path
            )
            
            # 최종 QA 객체 생성
            qa = MultiHopQA(
                question=question_data["question"],
                answer=answer_data["answer"],
                reasoning_trace=answer_data["reasoning_trace"],
                question_type=QuestionType(question_data["question_type"]),
                difficulty=DifficultyLevel(question_data["difficulty"]),
                confidence_score=answer_data["confidence"],
                source_path=path,
                metadata={
                    "reasoning_strategy": question_data.get("reasoning_strategy", ""),
                    "generation_timestamp": datetime.now().isoformat()
                }
            )
            
            return qa
            
        except Exception as e:
            logger.error(f"질문 생성 오류: {e}")
            return None

    def _generate_answer_with_trace(self, question: str, path: GraphPath) -> Dict[str, Any]:
        """질문에 대한 답변과 reasoning trace 생성 (Langfuse 추적 포함)"""
        
        # Reasoning trace 생성 프롬프트
        trace_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 그래프 기반 추론 전문가입니다. 주어진 질문과 그래프 경로를 바탕으로 
단계별 추론 과정과 최종 답변을 생성해주세요.

응답 형식:
{{
    "reasoning_trace": [
        "1단계: 첫 번째 추론 단계",
        "2단계: 두 번째 추론 단계", 
        "3단계: 최종 결론 도출"
    ],
    "answer": "최종 답변",
    "confidence": 0.85
}}"""),
            ("human", """질문: {question}

그래프 경로 정보:
- 경로 설명: {path_description}
- 시작 노드: {start_node}
- 끝 노드: {end_node}
- 관계 체인: {relationships}

이 정보를 바탕으로 단계별 추론 과정과 답변을 생성해주세요.""")
        ])
        
        try:
            # Langfuse config 설정
            config = {"callbacks": [self.langfuse_handler]} if self.enable_langfuse else {}
            
            response = self.llm.invoke(
                trace_prompt.format_messages(
                    question=question,
                    path_description=path.path_description,
                    start_node=path.start_node,
                    end_node=path.end_node,
                    relationships=" -> ".join([rel['type'] for rel in path.relationships])
                ),
                config=config
            )
            
            try:
                result = json.loads(response.content)
                return result
                
            except json.JSONDecodeError:
                # 파싱 실패 시 기본 구조 반환
                fallback_result = {
                    "reasoning_trace": [
                        f"1단계: {self._get_node_display_name(path.start_node)}에서 시작",
                        f"2단계: {' -> '.join([rel['type'] for rel in path.relationships])} 관계를 따라 이동",
                        f"3단계: {self._get_node_display_name(path.end_node)}에 도달하여 정보 확인"
                    ],
                    "answer": response.content,
                    "confidence": 0.7
                }
                return fallback_result
                
        except Exception as e:
            logger.error(f"추론 trace 생성 오류: {e}")
            
            error_result = {
                "reasoning_trace": ["추론 과정 생성 중 오류 발생"],
                "answer": "답변을 생성할 수 없습니다.",
                "confidence": 0.0
            }
            return error_result

    def validate_qa_quality(self, qa: MultiHopQA) -> bool:
        """QA 품질 검증"""
        
        # 기본 검증
        if not qa.question or not qa.answer:
            return False
            
        if len(qa.question) < 10 or len(qa.answer) < 10:
            return False
            
        if qa.confidence_score < 0.5:
            return False
            
        # 추가 품질 검증 로직
        if "오류" in qa.answer or "생성할 수 없습니다" in qa.answer:
            return False
            
        return True

    async def generate_multihop_dataset(
        self, 
        target_size: int = 100,
        min_confidence: float = 0.6
    ) -> List[MultiHopQA]:
        """Multi-hop QA 데이터셋 생성 (Langfuse 추적 포함)"""
        
        logger.info(f"Multi-hop QA 데이터셋 생성 시작 (목표: {target_size}개)")
        
        # 1. 그래프 경로 탐색
        paths = self.discover_multihop_paths(max_paths=target_size * 2)
        
        if not paths:
            logger.warning("탐색된 그래프 경로가 없습니다.")
            return []
        
        # 2. 병렬로 QA 생성
        qa_tasks = []
        for path in paths:
            qa_tasks.append(self._generate_qa_async(path))
        
        # 3. 비동기 실행
        qa_results = await asyncio.gather(*qa_tasks, return_exceptions=True)
        
        # 4. 결과 수집 및 필터링
        valid_qas = []
        for result in qa_results:
            if isinstance(result, MultiHopQA) and self.validate_qa_quality(result):
                if result.confidence_score >= min_confidence:
                    valid_qas.append(result)
        
        # 5. 다양성 확보를 위한 선별
        final_qas = self._diversify_qa_selection(valid_qas, target_size)
        
        logger.info(f"최종 생성된 QA 개수: {len(final_qas)}")
        return final_qas

    async def _generate_qa_async(self, path: GraphPath) -> Optional[MultiHopQA]:
        """비동기 QA 생성"""
        try:
            return self.generate_question_from_path(path)
        except Exception as e:
            logger.error(f"비동기 QA 생성 오류: {e}")
            return None

    def _diversify_qa_selection(self, qas: List[MultiHopQA], target_size: int) -> List[MultiHopQA]:
        """QA 다양성 확보"""
        if len(qas) <= target_size:
            return qas
        
        # 질문 유형별 분류
        type_groups = {}
        for qa in qas:
            qtype = qa.question_type.value
            if qtype not in type_groups:
                type_groups[qtype] = []
            type_groups[qtype].append(qa)
        
        # 각 유형에서 균등하게 선택
        selected = []
        per_type = target_size // len(type_groups)
        
        for qtype, group in type_groups.items():
            # 신뢰도 기준 정렬
            group.sort(key=lambda x: x.confidence_score, reverse=True)
            selected.extend(group[:per_type])
        
        # 부족한 만큼 추가 선택
        remaining = target_size - len(selected)
        if remaining > 0:
            all_remaining = [qa for qa in qas if qa not in selected]
            all_remaining.sort(key=lambda x: x.confidence_score, reverse=True)
            selected.extend(all_remaining[:remaining])
        
        return selected[:target_size]

    def save_dataset(self, qas: List[MultiHopQA], output_path: str) -> None:
        """데이터셋 저장"""
        
        # QA 객체를 JSON 직렬화 가능한 형태로 변환
        qa_dicts = []
        for qa in qas:
            qa_dict = asdict(qa)
            # Enum 객체를 문자열로 변환
            qa_dict['question_type'] = qa.question_type.value
            qa_dict['difficulty'] = qa.difficulty.value
            
            # source_path 안의 DateTime 등 복잡한 객체들을 문자열로 변환
            if 'source_path' in qa_dict:
                path_dict = qa_dict['source_path']
                # 딕셔너리 내의 모든 값을 문자열로 변환 (JSON 직렬화 가능하도록)
                for key, value in path_dict.items():
                    if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                        path_dict[key] = str(value)
                        
                # 노드 정보도 문자열로 변환
                if 'start_node' in path_dict and isinstance(path_dict['start_node'], dict):
                    for k, v in path_dict['start_node'].items():
                        if not isinstance(v, (str, int, float, bool, list, dict, type(None))):
                            path_dict['start_node'][k] = str(v)
                            
                if 'end_node' in path_dict and isinstance(path_dict['end_node'], dict):
                    for k, v in path_dict['end_node'].items():
                        if not isinstance(v, (str, int, float, bool, list, dict, type(None))):
                            path_dict['end_node'][k] = str(v)
            
            qa_dicts.append(qa_dict)
        
        output_data = {
            "metadata": {
                "total_count": len(qas),
                "generation_timestamp": datetime.now().isoformat(),
                "generator_version": "1.0.0",
                "source": "neo4j_multihop_generator"
            },
            "statistics": self._calculate_statistics(qas),
            "qa_pairs": qa_dicts
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"데이터셋 저장 완료: {output_path}")

    def _calculate_statistics(self, qas: List[MultiHopQA]) -> Dict[str, Any]:
        """데이터셋 통계 계산"""
        if not qas:
            return {}
        
        type_counts = {}
        difficulty_counts = {}
        
        for qa in qas:
            # 질문 유형 통계
            qtype = qa.question_type.value
            type_counts[qtype] = type_counts.get(qtype, 0) + 1
            
            # 난이도 통계
            difficulty = qa.difficulty.value
            difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
        
        return {
            "question_type_distribution": type_counts,
            "difficulty_distribution": difficulty_counts,
            "average_confidence": sum(qa.confidence_score for qa in qas) / len(qas),
            "average_path_length": sum(qa.source_path.path_length for qa in qas) / len(qas),
            "reasoning_trace_avg_length": sum(len(qa.reasoning_trace) for qa in qas) / len(qas)
        }

    def generate_sample_queries(self) -> List[str]:
        """샘플 질의 생성 (디버깅/테스트용)"""
        
        sample_queries = [
            # 기본 연결성 확인
            "MATCH (n) RETURN labels(n) as node_types, count(n) as count ORDER BY count DESC LIMIT 10",
            
            # 관계 유형 확인  
            "MATCH ()-[r]->() RETURN type(r) as rel_type, count(r) as count ORDER BY count DESC LIMIT 10",
            
            # 2-hop 경로 샘플
            "MATCH (a)-[r1]->(b)-[r2]->(c) RETURN a.name, type(r1), b.name, type(r2), c.name LIMIT 5",
            
            # 허브 노드 확인
            "MATCH (n) WITH n, size((n)--()) as degree WHERE degree > 3 RETURN n.name, labels(n), degree ORDER BY degree DESC LIMIT 10"
        ]
        
        return sample_queries

    def test_connection_and_schema(self) -> Dict[str, Any]:
        """연결 및 스키마 테스트"""
        try:
            # 연결 테스트
            result = self.graph.query("MATCH (n) RETURN count(n) as total_nodes LIMIT 1")
            total_nodes = result[0]['total_nodes'] if result else 0
            
            # 스키마 정보
            schema = self.get_graph_schema()
            
            # 샘플 쿼리 실행
            sample_results = {}
            for i, query in enumerate(self.generate_sample_queries()):
                try:
                    sample_results[f"query_{i+1}"] = self.graph.query(query)
                except Exception as e:
                    sample_results[f"query_{i+1}"] = f"Error: {str(e)}"
            
            return {
                "connection_status": "success",
                "total_nodes": total_nodes,
                "schema": schema,
                "sample_query_results": sample_results
            }
            
        except Exception as e:
            return {
                "connection_status": "failed",
                "error": str(e)
            }


async def main():
    """메인 실행 함수 (Langfuse 통합)"""
    
    print("🚀 Neo4j Multi-hop QA 데이터 생성 시스템 시작 (Langfuse 통합)")
    print("=" * 70)
    
    try:
        # 1. 생성기 초기화
        generator = Neo4jMultiHopQAGenerator(enable_langfuse=True)
        
        # Langfuse 상태 확인
        if generator.enable_langfuse:
            print("✅ Langfuse 연동 활성화됨")
            print("📊 Langfuse 대시보드에서 실시간 추적이 가능합니다")
        else:
            print("⚠️ Langfuse 연동 비활성화됨")
        
        # 2. 연결 및 스키마 테스트
        print("\n📊 Neo4j 연결 및 스키마 테스트...")
        test_result = generator.test_connection_and_schema()
        
        if test_result["connection_status"] == "failed":
            print(f"❌ Neo4j 연결 실패: {test_result['error']}")
            return
        
        print(f"✅ Neo4j 연결 성공 (총 노드 수: {test_result['total_nodes']})")
        print(f"📋 스키마 정보: {test_result['schema'][:200]}...")
        
        # 3. Multi-hop QA 생성
        print("\n🔍 Multi-hop QA 데이터 생성 중...")
        print("  📈 Langfuse에서 실시간 추적 가능합니다...")
        
        qa_dataset = await generator.generate_multihop_dataset(
            target_size=20,      # 생성할 QA 개수 (테스트용으로 줄임)
            min_confidence=0.6   # 최소 신뢰도
        )
        
        if not qa_dataset:
            print("❌ QA 데이터 생성 실패")
            return
        
        # 4. 결과 출력
        print(f"\n✅ {len(qa_dataset)}개의 Multi-hop QA 생성 완료!")
        
        # 샘플 QA 출력 (Langfuse trace ID 포함)
        print("\n📝 생성된 QA 샘플:")
        for i, qa in enumerate(qa_dataset[:3], 1):
            print(f"\n--- QA {i} ---")
            print(f"질문: {qa.question}")
            print(f"답변: {qa.answer}")
            print(f"유형: {qa.question_type.value}")
            print(f"난이도: {qa.difficulty.value}")
            print(f"신뢰도: {qa.confidence_score:.2f}")
            print(f"추론 단계: {len(qa.reasoning_trace)}단계")
            for j, step in enumerate(qa.reasoning_trace, 1):
                print(f"  {j}. {step}")
            print(f"그래프 경로: {qa.source_path.path_description}")
            
            # Langfuse 추적 상태 표시
            if generator.enable_langfuse:
                print("🔗 Langfuse에서 추적됨")
        
        # 5. 데이터셋 저장
        output_path = f"data/multihop_qa_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        generator.save_dataset(qa_dataset, output_path)
        
        # 6. 통계 출력
        stats = generator._calculate_statistics(qa_dataset)
        print(f"\n📈 데이터셋 통계:")
        print(f"  - 평균 신뢰도: {stats['average_confidence']:.2f}")
        print(f"  - 평균 경로 길이: {stats['average_path_length']:.1f}")
        print(f"  - 평균 추론 단계: {stats['reasoning_trace_avg_length']:.1f}")
        print(f"  - 질문 유형 분포: {stats['question_type_distribution']}")
        print(f"  - 난이도 분포: {stats['difficulty_distribution']}")
        
        print(f"\n💾 데이터셋 저장 완료: {output_path}")
        
        # 7. Langfuse 정보 출력
        if generator.enable_langfuse:
            print(f"\n🔍 Langfuse 추적 정보:")
            print(f"  - 모든 QA 생성 과정이 Langfuse에 기록되었습니다")
            print(f"  - 대시보드에서 성능 지표와 추론 과정을 확인할 수 있습니다")
        
        print("\n🎉 Multi-hop QA 데이터 생성 완료!")
        
    except Exception as e:
        logger.error(f"실행 중 오류 발생: {e}")
        print(f"❌ 오류 발생: {e}")


if __name__ == "__main__":
    asyncio.run(main())
