"""
SerpAPI Google Local API를 사용한 상일동역 주변 식당 검색
LangChain OutputParser를 활용한 결과 정리
"""

import os
from typing import List
from serpapi import GoogleSearch
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()


# Pydantic 모델 정의 - 식당 정보 구조화
class Restaurant(BaseModel):
    """식당 정보를 담는 데이터 모델"""
    name: str = Field(description="식당 이름")
    address: str = Field(description="식당 주소")
    rating: float = Field(description="평점", default=0.0)
    reviews: int = Field(description="리뷰 수", default=0)
    place_id: str = Field(description="Google Place ID", default="")


class RestaurantList(BaseModel):
    """식당 목록을 담는 데이터 모델"""
    restaurants: List[Restaurant] = Field(description="검색된 식당 목록")


def search_restaurants_near_sangildong():
    """
    SerpAPI를 사용하여 상일동역 주변 식당 검색
    """
    # SerpAPI 파라미터 설정
    params = {
        "engine": "google_local",
        "q": "식당 near 상일동역",  # 검색 쿼리
        "location": "Seoul, South Korea",  # 위치 설정
        "google_domain": "google.co.kr",  # 한국 구글 도메인
        "gl": "kr",  # 국가 코드
        "hl": "ko",  # 언어 코드
        "api_key": os.getenv("SERPAPI_API_KEY")  # API 키
    }
    
    # 검색 실행
    search = GoogleSearch(params)
    results = search.get_dict()
    
    return results


def parse_restaurant_results(results):
    """
    LangChain OutputParser를 사용하여 검색 결과 파싱
    """
    # Gemini 모델 초기화
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    # OutputParser 설정
    parser = PydanticOutputParser(pydantic_object=RestaurantList)
    
    # 프롬프트 템플릿 생성
    prompt = PromptTemplate(
        template="""다음은 상일동역 주변 식당 검색 결과입니다.
        
                검색 결과:
                {search_results}

                위 검색 결과에서 식당 정보를 추출하여 정리해주세요.
                각 식당의 이름, 주소, 평점, 리뷰 수, place_id를 포함해주세요.

                {format_instructions}""",
        input_variables=["search_results"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    # 체인 생성 및 실행
    chain = prompt | llm | parser
    
    # 로컬 결과 추출
    local_results = results.get("local_results", [])
    
    # 결과가 있는 경우에만 파싱
    if local_results:
        # 결과를 문자열로 변환
        results_str = str(local_results)
        
        # LLM으로 파싱
        parsed_results = chain.invoke({"search_results": results_str})
        return parsed_results
    else:
        return RestaurantList(restaurants=[])


def display_restaurants(restaurant_list: RestaurantList):
    """
    파싱된 식당 정보를 보기 좋게 출력
    """
    print("\n" + "="*60)
    print("🍽️  상일동역 주변 식당 검색 결과")
    print("="*60 + "\n")
    
    if not restaurant_list.restaurants:
        print("검색 결과가 없습니다.")
        return
    
    for idx, restaurant in enumerate(restaurant_list.restaurants, 1):
        print(f"[{idx}] {restaurant.name}")
        print(f"    📍 주소: {restaurant.address}")
        if restaurant.rating > 0:
            print(f"    ⭐ 평점: {restaurant.rating} ({restaurant.reviews}개 리뷰)")
        print()


def main():
    """
    메인 실행 함수
    """
    try:
        # API 키 확인
        if not os.getenv("SERPAPI_API_KEY"):
            print("❌ SERPAPI_API_KEY 환경변수를 설정해주세요.")
            return
        
        if not os.getenv("GOOGLE_API_KEY"):
            print("❌ GOOGLE_API_KEY 환경변수를 설정해주세요.")
            return
        
        print("🔍 상일동역 주변 식당을 검색중...")
        
        # 1. SerpAPI로 식당 검색
        search_results = search_restaurants_near_sangildong()
        
        # 2. LangChain OutputParser로 결과 파싱
        parsed_restaurants = parse_restaurant_results(search_results)
        
        # 3. 결과 출력
        display_restaurants(parsed_restaurants)
        
        # JSON 형태로도 저장 (선택사항)
        import json
        with open("restaurants_sangildong.json", "w", encoding="utf-8") as f:
            json.dump(
                parsed_restaurants.dict(), 
                f, 
                ensure_ascii=False, 
                indent=2
            )
        print("\n✅ 결과가 'restaurants_sangildong.json' 파일에 저장되었습니다.")
        
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")


if __name__ == "__main__":
    main()