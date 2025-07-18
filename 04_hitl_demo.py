#!/usr/bin/env python3
"""
HITL (Human-in-the-Loop) 실제 시연 스크립트

LangGraph의 interrupt()와 실제 터미널 상호작용을 결합한 데모
"""

import sys
import os
from pathlib import Path
from typing import TypedDict, List, Optional
import time

# 99_agent.py의 모든 기능을 import
exec(open('99_agent.py').read())

def interactive_hitl_demo():
    """실제 터미널에서 HITL 기능을 시연하는 함수"""
    print("🤖 GraphRAG Agent - HITL 실제 시연")
    print("=" * 50)
    
    # 사용자 질문 입력
    print("\n📝 질문을 입력해주세요:")
    question = input("질문: ").strip()
    if not question:
        question = "삼성전자 위축의 정치적 원인은 무엇인가요?"
        print(f"기본 질문 사용: {question}")
    
    # 초기 상태 설정
    state = {
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
    
    print(f"\n🚀 질문 처리를 시작합니다: '{question}'")
    print("=" * 50)
    
    # 단계별 실행
    try:
        # 1. QA 단계
        print("\n[1단계] 🔍 QA - 데이터베이스 검색")
        print("-" * 30)
        qa_result = qa_node(state)
        state.update(qa_result)
        
        print(f"✅ Cypher 쿼리 생성: {state['cypher'][:60]}...")
        print(f"✅ 검색 결과: {len(state['result'])}개 항목")
        
        # 2. Memory 단계
        print("\n[2단계] 🧠 MEMORY - 대화 기록 저장")
        print("-" * 30)
        memory_result = memory_node(state)
        state.update(memory_result)
        
        history_count = len(state.get('conversation_history', []))
        preferences = state.get('user_preferences', {})
        print(f"✅ 대화 기록: {history_count}개")
        print(f"✅ 사용자 선호도: {preferences}")
        
        # 3. Self-Reflection 단계
        print("\n[3단계] 🤔 SELF-REFLECTION - 품질 자체 평가")
        print("-" * 30)
        reflection_result = self_reflection_node(state)
        state.update(reflection_result)
        
        print(f"✅ 품질 점수: {state.get('quality_score', 0)}/5")
        print(f"✅ 신뢰도: {state.get('confidence_score', 0):.2f}")
        print(f"✅ 관련성: {state.get('result_relevance', 'unknown')}")
        
        # 4. Quality Gate 단계
        print("\n[4단계] 🚪 QUALITY GATE - 검토 필요성 판단")
        print("-" * 30)
        quality_result = quality_gate_node(state)
        state.update(quality_result)
        
        needs_review = state.get('needs_human_review', False)
        
        if needs_review:
            print("⚠️  품질 기준 미달 - 인간 검토가 필요합니다!")
            
            # 5. Human Review 단계 (실제 HITL)
            print("\n[5단계] 👤 HUMAN REVIEW - 인간 검토")
            print("-" * 30)
            
            # 현재 상태 표시
            current_summary = state.get('summary', '')
            if current_summary:
                print(f"📄 현재 요약:\n{current_summary[:300]}...\n")
            
            print("현재 결과를 검토해주세요:")
            print(f"• 검색된 데이터: {len(state['result'])}개")
            print(f"• 품질 점수: {state.get('quality_score', 0)}/5")
            print(f"• 신뢰도: {state.get('confidence_score', 0):.2f}")
            print(f"• 관련성: {state.get('result_relevance', 'unknown')}")
            
            print("\n다음 중 선택하세요:")
            print("1. 👍 승인 (현재 결과로 진행)")
            print("2. 🔄 개선 요청 (다시 시도)")
            print("3. 🚫 거부 (종료)")
            
            # 사용자 피드백 받기
            while True:
                try:
                    choice = input("\n선택 (1/2/3): ").strip()
                    
                    if choice == "1":
                        feedback = "승인"
                        print(f"✅ {feedback} - 결과를 승인하고 계속 진행합니다.")
                        break
                    elif choice == "2":
                        improvement = input("구체적인 개선 요청사항을 입력하세요: ").strip()
                        feedback = f"개선 {improvement}" if improvement else "개선 필요"
                        print(f"🔄 {feedback}")
                        state["needs_refinement"] = True
                        break
                    elif choice == "3":
                        feedback = "거부"
                        print(f"🚫 {feedback} - 처리를 중단합니다.")
                        return
                    else:
                        print("❌ 1, 2, 3 중에서 선택해주세요.")
                        
                except (EOFError, KeyboardInterrupt):
                    feedback = "승인"
                    print(f"\n⚡ 입력 중단 - 자동으로 {feedback}")
                    break
            
            state["human_feedback"] = feedback
            
            # 6. Refinement 단계 (필요시)
            if state.get("needs_refinement", False):
                print("\n[6단계] 🔧 REFINEMENT - 개선 작업")
                print("-" * 30)
                
                refinement_result = refinement_node(state)
                state.update(refinement_result)
                
                print(f"✅ 개선 반복: {state.get('iteration_count', 0)}번째")
                print("개선 제안이 생성되었습니다.")
                
                # 재시도할지 물어보기
                retry = input("\n개선된 버전으로 다시 시도하시겠습니까? (y/n): ").strip().lower()
                if retry == 'y':
                    print("🔄 개선된 버전으로 재시도...")
                    # 실제로는 QA부터 다시 시작해야 하지만, 데모에서는 생략
                    print("(실제 구현에서는 QA 단계부터 다시 실행됩니다)")
        else:
            print("✅ 품질 기준 통과 - 자동으로 계속 진행합니다.")
        
        # 7. Narration 단계
        print("\n[7단계] 📖 NARRATION - 내러티브 생성")
        print("-" * 30)
        narration_result = narration_node(state)
        state.update(narration_result)
        
        print("✅ 자연어 내러티브가 생성되었습니다.")
        
        # 8. Reasoning 단계
        print("\n[8단계] 🧠 REASONING - 최종 추론")
        print("-" * 30)
        reasoning_result = reasoning_node(state)
        state.update(reasoning_result)
        
        print("✅ 추론 분석이 완료되었습니다.")
        
        # 최종 결과 표시
        print("\n" + "=" * 50)
        print("🎉 HITL 시연 완료!")
        print("=" * 50)
        
        print(f"\n📊 최종 상태:")
        print(f"• 질문: {state['question']}")
        print(f"• 검색 결과: {len(state['result'])}개")
        print(f"• 품질 점수: {state.get('quality_score', 0)}/5")
        print(f"• 신뢰도: {state.get('confidence_score', 0):.2f}")
        print(f"• 인간 피드백: {state.get('human_feedback', 'N/A')}")
        print(f"• 반복 횟수: {state.get('iteration_count', 0)}")
        
        # 최종 요약 출력
        final_summary = state.get('summary', '')
        if final_summary:
            print(f"\n📝 최종 분석 결과:")
            print(final_summary[:500])
            if len(final_summary) > 500:
                print("...")
        
    except Exception as e:
        print(f"\n❌ 실행 중 오류가 발생했습니다: {e}")
        import traceback
        traceback.print_exc()

def main():
    """메인 함수"""
    print("🤖 GraphRAG Agent - HITL 실제 시연 도구")
    print("이 도구는 실제 터미널에서 Human-in-the-Loop 기능을 시연합니다.")
    
    try:
        interactive_hitl_demo()
    except KeyboardInterrupt:
        print("\n⚡ 사용자가 중단했습니다. 프로그램을 종료합니다.")
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류: {e}")

if __name__ == "__main__":
    main() 