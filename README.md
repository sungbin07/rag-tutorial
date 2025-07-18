# 🤖 Hybrid RAG Agent Project

네이버 뉴스 기반 하이브리드 RAG(Retrieval-Augmented Generation) 시스템으로, ChromaDB와 Neo4j를 결합하여 뉴스 데이터에 대한 지능적인 질의응답을 제공합니다.

## 📋 프로젝트 개요

이 프로젝트는 다음과 같은 핵심 기능을 제공합니다:

- **뉴스 크롤링**: 네이버 뉴스 카테고리별 최신 뉴스 수집
- **벡터 검색**: ChromaDB를 통한 의미적 유사도 기반 검색
- **그래프 검색**: Neo4j를 통한 관계 기반 검색
- **하이브리드 RAG**: 두 검색 방식을 결합한 고도화된 답변 생성
- **Human-in-the-Loop**: 사람의 피드백을 반영한 답변 개선

## 🏗️ 아키텍처

```
📊 Data Pipeline:
뉴스 크롤링 → 데이터 전처리 → 벡터화(ChromaDB) + 관계 추출(Neo4j)

🤖 RAG Pipeline:
사용자 질문 → 검색 전략 결정 → 하이브리드 검색 → 답변 생성 → 품질 평가
```

### 핵심 컴포넌트

- **ChromaDB**: 뉴스 원문의 벡터 임베딩 저장 및 의미적 검색
- **Neo4j**: 뉴스에서 추출한 개체 간 관계 그래프 저장 및 관계 검색
- **LangGraph**: 워크플로우 오케스트레이션 및 상태 관리
- **LangChain**: LLM 인터페이스 및 프롬프트 관리
- **Langfuse**: 추적 및 모니터링

## 📁 프로젝트 구조

```
rag-project/
├── 01_create_newsdata.py          # 뉴스 데이터 수집
├── 01_make_chromadb.ipynb         # ChromaDB 벡터 DB 생성
├── 02_generate_triplets.py        # 뉴스에서 관계 트리플릿 추출
├── 03_upload_to_neo4j.py          # Neo4j 그래프 DB 업로드
├── 04_hybrid_rag_agent.py         # 메인 하이브리드 RAG 에이전트
├── 04_tool_based_hybrid_rag_agent.py  # 도구 기반 RAG 에이전트
├── 04_hitl_demo.py                # Human-in-the-Loop 데모
├── 99_agent.py                    # 고급 에이전트 (메모리, HITL)
├── agent.py                       # 기본 에이전트
├── news_crawler.py                # 뉴스 크롤러 모듈
├── main.py                        # 메인 엔트리 포인트
├── data/                          # 데이터 저장소
│   ├── naver_news.json           # 크롤링된 뉴스 데이터
│   ├── triplets_output.json      # 추출된 관계 트리플릿
│   └── final_query.cyp           # Neo4j 쿼리 파일
└── chroma_db_news_3/              # ChromaDB 데이터베이스
```

## 🚀 시작하기

### 1. 환경 설정

```bash
# Python 3.12+ 필요
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정

`.env` 파일을 생성하고 다음 변수들을 설정하세요:

```env
# OpenAI API
OPENAI_API_KEY=your_openai_api_key

# Neo4j 설정
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password

# Langfuse 추적 (선택사항)
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com
```

## 📋 사용법

### 단계별 데이터 파이프라인 실행

#### 1단계: 뉴스 데이터 수집
```bash
python 01_create_newsdata.py
```
- 네이버 뉴스 카테고리별 최신 뉴스 크롤링
- `data/naver_news.json`에 저장

#### 2단계: ChromaDB 벡터 데이터베이스 생성
```bash
# Jupyter 노트북 실행
jupyter notebook 01_make_chromadb.ipynb
```
- 뉴스 텍스트를 임베딩하여 ChromaDB에 저장
- `chroma_db_news_3/` 디렉터리에 벡터 DB 생성

#### 3단계: 관계 트리플릿 추출
```bash
python 02_generate_triplets.py
```
- LLM을 사용하여 뉴스에서 개체 간 관계 추출
- `data/triplets_output.json`에 저장

#### 4단계: Neo4j 그래프 데이터베이스 업로드
```bash
python 03_upload_to_neo4j.py
```
- 추출된 관계를 Neo4j 그래프 데이터베이스에 업로드

### RAG 에이전트 실행

#### 기본 하이브리드 RAG 에이전트
```bash
python 04_hybrid_rag_agent.py
```

#### 도구 기반 RAG 에이전트
```bash
python 04_tool_based_hybrid_rag_agent.py
```

#### Human-in-the-Loop 데모
```bash
python 04_hitl_demo.py
```

#### 고급 에이전트 (메모리 & HITL)
```bash
python 99_agent.py
```

## 🔧 주요 기능

### 1. 하이브리드 검색
- **의미적 검색**: ChromaDB를 통한 벡터 유사도 기반 검색
- **관계 검색**: Neo4j를 통한 그래프 관계 기반 검색
- **지능적 전략**: 질문 유형에 따른 최적 검색 전략 자동 선택

### 2. 품질 관리
- **신뢰도 점수**: 답변의 신뢰성 평가
- **소스 추적**: 답변 근거가 된 뉴스 기사 추적
- **반복 개선**: 품질이 낮은 경우 자동 재시도

### 3. Human-in-the-Loop
- **피드백 수집**: 사용자 피드백을 통한 답변 개선
- **대화 메모리**: 이전 대화 내용 학습 및 활용
- **개인화**: 사용자 선호도 학습

### 4. 모니터링 및 추적
- **Langfuse 통합**: 실행 과정 상세 추적
- **성능 메트릭**: 검색 정확도 및 응답 시간 측정
- **로그 분석**: 상세한 실행 로그 제공

## 🎯 사용 예시

### 질문 예시
```
질문: "삼성전자의 실적 부진 원인은 무엇인가요?"

답변 과정:
1. 검색 전략 결정: 하이브리드 (의미적 + 관계)
2. ChromaDB 검색: 삼성전자 관련 뉴스 검색
3. Neo4j 검색: 삼성전자와 연관된 원인 관계 탐색
4. 결과 통합: 두 검색 결과를 종합하여 답변 생성
5. 품질 평가: 신뢰도 점수 산출

답변: "삼성전자의 실적 부진은 주로 반도체 시장 침체와 
스마트폰 경쟁 심화로 인한 것으로 분석됩니다..."
```

## 🔍 고급 설정

### ChromaDB 설정
```python
# chroma_db_news_3/ 디렉터리 위치 변경
CHROMA_DB_PATH = "custom_path/chroma_db"
```

### Neo4j 스키마 커스터마이징
```cypher
// 노드 타입: Phenomenon, Cause, Policy, Company, Person, etc.
// 관계 타입: 원인이다, 영향을준다, 정책이다, etc.
```

