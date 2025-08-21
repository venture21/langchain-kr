"""
Google Shopping API를 활용한 갤럭시 플립7 검색 시스템
LangChain OutputParser와 Gemini 모델을 활용한 구조화된 출력
"""

import os
from typing import List, Optional
from dotenv import load_dotenv
from serpapi import GoogleSearch
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# 환경 변수 로드
load_dotenv()


class ProductInfo(BaseModel):
    """제품 정보 스키마"""
    rank: int = Field(description="검색 결과 순위")
    title: str = Field(description="제품명")
    price: str = Field(description="제품 가격")
    store: str = Field(description="판매처")
    link: str = Field(description="제품 링크")
    rating: Optional[float] = Field(description="평점", default=None)
    reviews: Optional[int] = Field(description="리뷰 수", default=None)
    shipping: Optional[str] = Field(description="배송 정보", default=None)


class ShoppingResults(BaseModel):
    """쇼핑 검색 결과 스키마"""
    query: str = Field(description="검색 쿼리")
    total_results: int = Field(description="총 검색 결과 수")
    products: List[ProductInfo] = Field(description="상위 10개 제품 정보")
    summary: str = Field(description="검색 결과 요약")


def search_google_shopping(query: str, limit: int = 10) -> dict:
    """
    Google Shopping API를 사용하여 제품 검색
    
    Args:
        query: 검색할 제품명
        limit: 반환할 결과 수 (기본값: 10)
    
    Returns:
        검색 결과 딕셔너리
    """
    params = {
        "engine": "google_shopping",
        "q": query,
        "api_key": os.getenv("SERPAPI_API_KEY"),
        "gl": "kr",  # 한국 지역 설정
        "hl": "ko",  # 한국어 설정
        "num": limit
    }
    
    search = GoogleSearch(params)
    results = search.get_dict()
    
    return results


def process_with_gemini(search_results: dict, query: str) -> ShoppingResults:
    """
    Gemini 모델을 사용하여 검색 결과를 구조화된 형태로 처리
    
    Args:
        search_results: Google Shopping API 검색 결과
        query: 원본 검색 쿼리
    
    Returns:
        구조화된 ShoppingResults 객체
    """
    # Pydantic 출력 파서 생성
    output_parser = PydanticOutputParser(pydantic_object=ShoppingResults)
    
    # Gemini 모델 초기화
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    # 프롬프트 템플릿 생성
    prompt_template = PromptTemplate(
        template="""
        다음 Google Shopping 검색 결과를 분석하여 구조화된 형태로 정리해주세요.
        
        검색 쿼리: {query}
        
        검색 결과:
        {search_results}
        
        위 검색 결과에서 상위 10개의 제품 정보를 추출하고, 
        각 제품의 순위, 제품명, 가격, 판매처, 링크, 평점, 리뷰 수, 배송 정보를 정리해주세요.
        또한 전체 검색 결과에 대한 간단한 요약도 제공해주세요.
        
        {format_instructions}
        """,
        input_variables=["query", "search_results"],
        partial_variables={"format_instructions": output_parser.get_format_instructions()}
    )
    
    # 프롬프트 생성 및 실행
    prompt = prompt_template.format(
        query=query,
        search_results=str(search_results.get("shopping_results", [])[:10])
    )
    
    # Gemini 모델 실행
    response = llm.invoke(prompt)
    
    # 결과 파싱
    parsed_result = output_parser.parse(response.content)
    
    return parsed_result


def format_results(results: ShoppingResults) -> str:
    """
    결과를 보기 좋게 포맷팅
    
    Args:
        results: ShoppingResults 객체
    
    Returns:
        포맷팅된 문자열
    """
    output = []
    output.append(f"\n{'='*80}")
    output.append(f"검색 쿼리: {results.query}")
    output.append(f"총 검색 결과: {results.total_results}개")
    output.append(f"{'='*80}\n")
    
    for product in results.products:
        output.append(f"순위 {product.rank}:")
        output.append(f"  제품명: {product.title}")
        output.append(f"  가격: {product.price}")
        output.append(f"  판매처: {product.store}")
        output.append(f"  링크: {product.link}")
        if product.rating:
            output.append(f"  평점: {product.rating} / 5.0")
        if product.reviews:
            output.append(f"  리뷰 수: {product.reviews}개")
        if product.shipping:
            output.append(f"  배송: {product.shipping}")
        output.append("-" * 40)
    
    output.append(f"\n{'='*80}")
    output.append("검색 결과 요약:")
    output.append(results.summary)
    output.append(f"{'='*80}\n")
    
    return "\n".join(output)


def main():
    """메인 실행 함수"""
    # 검색 쿼리
    query = "갤럭시 플립7"
    
    print(f"'{query}' 검색 중...")
    print("Google Shopping API로 데이터 수집 중...")
    
    try:
        # Google Shopping 검색 실행
        search_results = search_google_shopping(query, limit=10)
        
        # 검색 결과가 있는지 확인
        if not search_results.get("shopping_results"):
            print("검색 결과가 없습니다.")
            return
        
        print(f"검색 결과 {len(search_results.get('shopping_results', []))}개 발견")
        print("Gemini 모델로 결과 처리 중...")
        
        # Gemini 모델로 결과 처리
        structured_results = process_with_gemini(search_results, query)
        
        # 결과 출력
        formatted_output = format_results(structured_results)
        print(formatted_output)
        
        # JSON 파일로 저장 (선택사항)
        import json
        with open("galaxy_flip7_results.json", "w", encoding="utf-8") as f:
            json.dump(structured_results.dict(), f, ensure_ascii=False, indent=2)
        print("\n결과가 'galaxy_flip7_results.json' 파일에 저장되었습니다.")
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()