import os
from datetime import datetime
from dotenv import load_dotenv
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# .env 파일에서 API 키 로드
load_dotenv()

# 테스트를 위한 최소한의 문서
docs = [
    Document(
        page_content="12월: 정부의 저출산 정책 발표",
        metadata={"created_at": "2024-12-01"},
    ),
    Document(
        page_content="10월: 정부의 AI 산업 정책 발표",
        metadata={"created_at": "2024-10-01"},
    ),
    Document(
        page_content="2월: 정부의 의대 정원 결정",
        metadata={"created_at": "2024-02-01"},
    ),
]

print("1. FAISS 벡터 저장소를 생성합니다.")
try:
    vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())
    print("   - 성공: FAISS 벡터 저장소 생성 완료")
except Exception as e:
    print(f"   - 실패: {e}")
    exit()

# decay_rate=0.0 으로 시간 가중치 비활성화
retriever = TimeWeightedVectorStoreRetriever(
    vectorstore=vectorstore,
    decay_rate=0.0,
    k=3
)

print("\n2. TimeWeightedVectorStoreRetriever를 사용하여 문서를 검색합니다.")
query = "정부 정책"
try:
    # get_relevant_documents를 직접 호출하고, now 파라미터 전달
    retrieved_docs = retriever.get_relevant_documents(query, now=datetime.now())

    print(f"   - 쿼리: '{query}'")
    print(f"   - 검색된 문서 개수: {len(retrieved_docs)}")

    if retrieved_docs:
        print("\n--- 검색 결과 ---")
        for doc in retrieved_docs:
            print(f"- 생성일: {doc.metadata.get('created_at', 'N/A')}, 내용: {doc.page_content}")
    else:
        print("   - 결과가 없습니다.")

except Exception as e:
    print(f"   - 검색 중 오류 발생: {e}")