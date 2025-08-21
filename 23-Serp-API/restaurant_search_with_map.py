"""
SerpAPI Google Local API를 사용한 상일동역 주변 식당 검색
LangChain OutputParser를 활용한 결과 정리
Google Maps 지도 표시 기능 추가
"""

import os
import folium
import webbrowser
from typing import List, Optional, Tuple
from serpapi import GoogleSearch
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import googlemaps
from datetime import datetime

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
    latitude: Optional[float] = Field(description="위도", default=None)
    longitude: Optional[float] = Field(description="경도", default=None)
    phone: Optional[str] = Field(description="전화번호", default=None)
    hours: Optional[str] = Field(description="영업시간", default=None)


class RestaurantList(BaseModel):
    """식당 목록을 담는 데이터 모델"""
    restaurants: List[Restaurant] = Field(description="검색된 식당 목록")


def get_coordinates_from_address(address: str, gmaps_client) -> Tuple[Optional[float], Optional[float]]:
    """
    Google Maps Geocoding API를 사용하여 주소를 좌표로 변환
    """
    try:
        geocode_result = gmaps_client.geocode(address)
        if geocode_result:
            location = geocode_result[0]['geometry']['location']
            return location['lat'], location['lng']
    except Exception as e:
        print(f"좌표 변환 실패: {e}")
    return None, None


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
    LangChain OutputParser를 사용하여 검색 결과 파싱 및 좌표 정보 추가
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
                가능하다면 전화번호와 영업시간도 추출해주세요.

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
        # 직접 SerpAPI 결과에서 좌표 정보 추출
        restaurants_with_coords = []
        
        for result in local_results:
            restaurant = Restaurant(
                name=result.get("title", ""),
                address=result.get("address", ""),
                rating=result.get("rating", 0.0),
                reviews=result.get("reviews", 0),
                place_id=result.get("place_id", ""),
                latitude=result.get("gps_coordinates", {}).get("latitude"),
                longitude=result.get("gps_coordinates", {}).get("longitude"),
                phone=result.get("phone", ""),
                hours=result.get("hours", "")
            )
            restaurants_with_coords.append(restaurant)
        
        return RestaurantList(restaurants=restaurants_with_coords)
    else:
        return RestaurantList(restaurants=[])


def create_map_with_restaurants(restaurant_list: RestaurantList, center_name: str = "상일동역"):
    """
    Folium을 사용하여 식당 위치를 표시한 지도 생성
    """
    # Google Maps Client 초기화 (좌표 변환용)
    gmaps = googlemaps.Client(key=os.getenv("GOOGLE_MAPS_API_KEY"))
    
    # 상일동역 중심 좌표 가져오기
    center_lat, center_lng = get_coordinates_from_address(f"{center_name}, 서울", gmaps)
    
    if not center_lat or not center_lng:
        # 기본 상일동역 좌표 (백업)
        center_lat, center_lng = 37.5567, 127.1660
    
    # Folium 지도 생성
    m = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=15,
        tiles='OpenStreetMap'
    )
    
    # 중심점 마커 추가 (상일동역)
    folium.Marker(
        location=[center_lat, center_lng],
        popup=f"<b>{center_name}</b>",
        tooltip=center_name,
        icon=folium.Icon(color='red', icon='star')
    ).add_to(m)
    
    # 식당 마커 추가
    for idx, restaurant in enumerate(restaurant_list.restaurants, 1):
        # 좌표가 없는 경우 주소로 좌표 검색
        if not restaurant.latitude or not restaurant.longitude:
            if restaurant.address:
                lat, lng = get_coordinates_from_address(restaurant.address, gmaps)
                if lat and lng:
                    restaurant.latitude = lat
                    restaurant.longitude = lng
        
        # 좌표가 있는 경우만 마커 추가
        if restaurant.latitude and restaurant.longitude:
            # 팝업 내용 생성
            popup_html = f"""
            <div style="width: 200px;">
                <h4>{restaurant.name}</h4>
                <p><b>주소:</b> {restaurant.address}</p>
                {'<p><b>평점:</b> ⭐ ' + str(restaurant.rating) + f' ({restaurant.reviews}개 리뷰)</p>' if restaurant.rating > 0 else ''}
                {'<p><b>전화:</b> ' + restaurant.phone + '</p>' if restaurant.phone else ''}
                {'<p><b>영업시간:</b> ' + restaurant.hours + '</p>' if restaurant.hours else ''}
            </div>
            """
            
            # 마커 색상 (평점에 따라)
            if restaurant.rating >= 4.5:
                color = 'green'
            elif restaurant.rating >= 4.0:
                color = 'blue'
            elif restaurant.rating >= 3.5:
                color = 'orange'
            else:
                color = 'gray'
            
            folium.Marker(
                location=[restaurant.latitude, restaurant.longitude],
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"{idx}. {restaurant.name} (⭐{restaurant.rating})",
                icon=folium.Icon(color=color, icon='cutlery', prefix='fa')
            ).add_to(m)
    
    # 범례 추가
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 200px; height: 120px; 
                background-color: white; z-index:9999; font-size:14px;
                border:2px solid grey; border-radius: 5px; padding: 10px">
        <p style="margin: 5px;"><b>식당 평점 범례</b></p>
        <p style="margin: 5px;"><i class="fa fa-map-marker" style="color:green"></i> 4.5 이상</p>
        <p style="margin: 5px;"><i class="fa fa-map-marker" style="color:blue"></i> 4.0 - 4.5</p>
        <p style="margin: 5px;"><i class="fa fa-map-marker" style="color:orange"></i> 3.5 - 4.0</p>
        <p style="margin: 5px;"><i class="fa fa-map-marker" style="color:gray"></i> 3.5 미만</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m


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
        if restaurant.phone:
            print(f"    📞 전화: {restaurant.phone}")
        if restaurant.latitude and restaurant.longitude:
            print(f"    🗺️  좌표: ({restaurant.latitude:.6f}, {restaurant.longitude:.6f})")
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
        
        if not os.getenv("GOOGLE_MAPS_API_KEY"):
            print("⚠️  GOOGLE_MAPS_API_KEY가 설정되지 않았습니다. 일부 기능이 제한될 수 있습니다.")
        
        print("🔍 상일동역 주변 식당을 검색중...")
        
        # 1. SerpAPI로 식당 검색
        search_results = search_restaurants_near_sangildong()
        
        # 2. LangChain OutputParser로 결과 파싱
        parsed_restaurants = parse_restaurant_results(search_results)
        
        # 3. 결과 출력
        display_restaurants(parsed_restaurants)
        
        # 4. 지도 생성 및 저장
        print("\n🗺️  지도를 생성중...")
        restaurant_map = create_map_with_restaurants(parsed_restaurants)
        
        # 지도를 HTML 파일로 저장
        map_file = "restaurants_sangildong_map.html"
        restaurant_map.save(map_file)
        print(f"✅ 지도가 '{map_file}' 파일에 저장되었습니다.")
        
        # 자동으로 브라우저에서 지도 열기
        webbrowser.open(f"file://{os.path.abspath(map_file)}")
        print("🌐 브라우저에서 지도를 여는 중...")
        
        # JSON 형태로도 저장 (선택사항)
        import json
        with open("restaurants_sangildong.json", "w", encoding="utf-8") as f:
            json.dump(
                parsed_restaurants.dict(), 
                f, 
                ensure_ascii=False, 
                indent=2
            )
        print("✅ 결과가 'restaurants_sangildong.json' 파일에 저장되었습니다.")
        
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")


if __name__ == "__main__":
    main()