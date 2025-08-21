"""
Google Shopping API 직접 파싱 버전
실제 API 응답을 직접 처리하여 Gemini 모델로 분석
"""

import os
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from serpapi import GoogleSearch
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import HumanMessage
from pydantic import BaseModel, Field

# 환경 변수 로드
load_dotenv()


class ProductDetail(BaseModel):
    """제품 상세 정보"""
    rank: int = Field(description="검색 결과 순위")
    title: str = Field(description="제품명")
    price: str = Field(description="제품 가격")
    store: str = Field(description="판매처 이름")
    link: str = Field(description="제품 구매 링크")
    rating: Optional[float] = Field(description="제품 평점", default=None)
    reviews: Optional[int] = Field(description="리뷰 개수", default=None)
    shipping: Optional[str] = Field(description="배송 정보", default=None)
    thumbnail: Optional[str] = Field(description="제품 이미지 URL", default=None)


class ShoppingAnalysis(BaseModel):
    """쇼핑 검색 분석 결과"""
    search_query: str = Field(description="검색한 제품명")
    total_found: int = Field(description="찾은 제품 수")
    top_10_products: List[ProductDetail] = Field(description="상위 10개 제품 리스트")
    price_range: str = Field(description="가격대 분석")
    best_deal: str = Field(description="가장 좋은 거래 추천")
    summary: str = Field(description="전체 검색 결과 요약")


class GoogleShoppingSearcher:
    """Google Shopping 검색 클래스"""
    
    def __init__(self):
        """초기화"""
        self.serpapi_key = os.getenv("SERPAPI_API_KEY")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        
        if not self.serpapi_key:
            raise ValueError("SERPAPI_API_KEY 환경 변수를 설정해주세요.")
        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY 환경 변수를 설정해주세요.")
        
        # Gemini 모델 초기화
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0,
            google_api_key=self.google_api_key
        )
        
        # Output Parser 초기화
        self.output_parser = PydanticOutputParser(pydantic_object=ShoppingAnalysis)
    
    def search_products(self, query: str, country: str = "kr") -> Dict[Any, Any]:
        """
        제품 검색 실행
        
        Args:
            query: 검색할 제품명
            country: 국가 코드 (기본값: kr)
        
        Returns:
            검색 결과 딕셔너리
        """
        params = {
            "engine": "google_shopping",
            "q": query,
            "api_key": self.serpapi_key,
            "gl": country,
            "hl": "ko" if country == "kr" else "en",
            "num": 20  # 더 많은 결과를 가져와서 상위 10개 선택
        }
        
        search = GoogleSearch(params)
        return search.get_dict()
    
    def extract_product_info(self, product_data: dict, rank: int) -> dict:
        """
        개별 제품 정보 추출
        
        Args:
            product_data: 제품 데이터 딕셔너리
            rank: 순위
        
        Returns:
            정리된 제품 정보
        """
        return {
            "rank": rank,
            "title": product_data.get("title", "제목 없음"),
            "price": product_data.get("price", "가격 정보 없음"),
            "store": product_data.get("source", "판매처 정보 없음"),
            "link": product_data.get("link", ""),
            "rating": product_data.get("rating"),
            "reviews": product_data.get("reviews"),
            "shipping": product_data.get("delivery", "배송 정보 없음"),
            "thumbnail": product_data.get("thumbnail", "")
        }
    
    def analyze_with_gemini(self, products: List[dict], query: str) -> ShoppingAnalysis:
        """
        Gemini 모델을 사용한 제품 분석
        
        Args:
            products: 제품 정보 리스트
            query: 검색 쿼리
        
        Returns:
            분석된 ShoppingAnalysis 객체
        """
        # 프롬프트 템플릿 생성
        prompt_template = PromptTemplate(
            template="""
            당신은 온라인 쇼핑 전문가입니다. 다음 제품 검색 결과를 분석해주세요.
            
            검색 제품: {query}
            
            검색된 제품 목록:
            {products}
            
            위 데이터를 분석하여 다음을 수행해주세요:
            1. 상위 10개 제품을 선정하고 각 제품의 상세 정보를 정리
            2. 전체 가격대 분석 (최저가, 최고가, 평균가 등)
            3. 가장 좋은 거래 추천 (가격, 평점, 배송 등 고려)
            4. 전체 검색 결과에 대한 종합적인 요약
            
            {format_instructions}
            
            중요: 실제 데이터를 기반으로 정확한 정보를 제공해주세요.
            """,
            input_variables=["query", "products"],
            partial_variables={"format_instructions": self.output_parser.get_format_instructions()}
        )
        
        # 프롬프트 생성
        prompt = prompt_template.format(
            query=query,
            products=str(products[:10])  # 상위 10개만 전달
        )
        
        # Gemini 실행
        response = self.llm.invoke(prompt)
        
        # 결과 파싱
        try:
            parsed_result = self.output_parser.parse(response.content)
        except Exception as e:
            print(f"파싱 오류: {e}")
            # 대체 파싱 방법
            parsed_result = self.fallback_parsing(products, query)
        
        return parsed_result
    
    def fallback_parsing(self, products: List[dict], query: str) -> ShoppingAnalysis:
        """
        파싱 실패 시 대체 방법
        
        Args:
            products: 제품 리스트
            query: 검색 쿼리
        
        Returns:
            ShoppingAnalysis 객체
        """
        top_10 = []
        for i, product in enumerate(products[:10], 1):
            top_10.append(ProductDetail(
                rank=i,
                title=product.get("title", ""),
                price=product.get("price", ""),
                store=product.get("store", ""),
                link=product.get("link", ""),
                rating=product.get("rating"),
                reviews=product.get("reviews"),
                shipping=product.get("shipping"),
                thumbnail=product.get("thumbnail")
            ))
        
        return ShoppingAnalysis(
            search_query=query,
            total_found=len(products),
            top_10_products=top_10,
            price_range="가격 정보 분석 중",
            best_deal="최적 거래 분석 중",
            summary=f"{query}에 대한 {len(products)}개의 결과를 찾았습니다."
        )
    
    def search_and_analyze(self, query: str) -> ShoppingAnalysis:
        """
        검색 및 분석 통합 실행
        
        Args:
            query: 검색할 제품명
        
        Returns:
            분석 결과
        """
        print(f"🔍 '{query}' 검색 시작...")
        
        # 1. Google Shopping 검색
        search_results = self.search_products(query)
        shopping_results = search_results.get("shopping_results", [])
        
        if not shopping_results:
            print("❌ 검색 결과가 없습니다.")
            return None
        
        print(f"✅ {len(shopping_results)}개의 제품 발견")
        
        # 2. 제품 정보 추출
        products = []
        for i, product in enumerate(shopping_results[:10], 1):
            products.append(self.extract_product_info(product, i))
        
        # 3. Gemini로 분석
        print("🤖 Gemini 모델로 분석 중...")
        analysis = self.analyze_with_gemini(products, query)
        
        return analysis
    
    def display_results(self, analysis: ShoppingAnalysis):
        """
        결과를 보기 좋게 출력
        
        Args:
            analysis: 분석 결과
        """
        print("\n" + "="*100)
        print(f"🛍️  {analysis.search_query} 검색 결과")
        print("="*100)
        print(f"📊 총 {analysis.total_found}개 제품 중 상위 10개")
        print("-"*100)
        
        for product in analysis.top_10_products:
            print(f"\n#{product.rank} {product.title}")
            print(f"   💰 가격: {product.price}")
            print(f"   🏪 판매처: {product.store}")
            if product.rating:
                print(f"   ⭐ 평점: {product.rating}/5.0", end="")
                if product.reviews:
                    print(f" ({product.reviews}개 리뷰)")
                else:
                    print()
            if product.shipping:
                print(f"   🚚 배송: {product.shipping}")
            print(f"   🔗 링크: {product.link[:50]}...")
        
        print("\n" + "="*100)
        print("📈 가격대 분석")
        print("-"*100)
        print(analysis.price_range)
        
        print("\n" + "="*100)
        print("💡 추천 거래")
        print("-"*100)
        print(analysis.best_deal)
        
        print("\n" + "="*100)
        print("📝 요약")
        print("-"*100)
        print(analysis.summary)
        print("="*100 + "\n")


def main():
    """메인 실행 함수"""
    # 검색 객체 생성
    searcher = GoogleShoppingSearcher()
    
    # 갤럭시 플립7 검색 및 분석
    query = "갤럭시 플립7"
    
    try:
        # 검색 및 분석 실행
        results = searcher.search_and_analyze(query)
        
        if results:
            # 결과 출력
            searcher.display_results(results)
            
            # JSON 파일로 저장
            import json
            with open("galaxy_flip7_analysis.json", "w", encoding="utf-8") as f:
                json.dump(results.dict(), f, ensure_ascii=False, indent=2)
            print("💾 결과가 'galaxy_flip7_analysis.json' 파일에 저장되었습니다.")
        
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()