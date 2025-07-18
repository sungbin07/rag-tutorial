from langchain_community.graphs import Neo4jGraph
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from typing import TypedDict, List
import os

# env
from dotenv import load_dotenv
load_dotenv()



# ✅ Neo4j 연결
print("🔗 Connecting to Neo4j...")
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
)
print("✅ Connected to Neo4j successfully")

# ✅ LLM 초기화
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# ✅ 경로 프롬프트 템플릿
reasoning_prompt = PromptTemplate.from_template("""
다음은 사건들 간의 인과적 흐름입니다:

1. {from_node} →({rel1})→ {mid_node} →({rel2})→ {to_node}

이 흐름을 기반으로 자연스러운 질문과 그에 대한 답변을 생성해 주세요.

Q:
A:
""")

# ✅ LangGraph 상태 정의
class QAState(TypedDict):
    paths: List[dict]
    outputs: List[str]

# ✅ 경로를 추출하는 노드
def get_paths(_: dict) -> QAState:
    cypher = """
    MATCH (a)-[r1]->(b)-[r2]->(c)
    RETURN a.name AS from_node, type(r1) AS rel1,
           b.name AS mid_node, type(r2) AS rel2,
           c.name AS to_node
    LIMIT 5
    """
    records = graph.query(cypher)
    return {"paths": records, "outputs": []}

# ✅ LLM 호출 노드
def run_llm(state: QAState) -> QAState:
    new_outputs = []
    for path in state["paths"]:
        prompt = reasoning_prompt.format(**path)
        result = llm.invoke(prompt)
        new_outputs.append(result.content)
    return {"paths": state["paths"], "outputs": new_outputs}

# ✅ LangGraph Workflow 정의
builder = StateGraph(QAState)
builder.add_node("query_paths", get_paths)
builder.add_node("llm_reasoning", run_llm)
builder.set_entry_point("query_paths")
builder.add_edge("query_paths", "llm_reasoning")
builder.add_edge("llm_reasoning", END)

# ✅ 실행
app = builder.compile()
result = app.invoke({})

# ✅ 결과 출력
for i, output in enumerate(result["outputs"], 1):
    print(f"\n🔹 QA {i}")
    print(output)
