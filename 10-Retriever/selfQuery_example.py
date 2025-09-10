

from dotenv import load_dotenv

# API 키 정보 로드
load_dotenv()

# 1. 문서 데이터 생성
docs = [
    Document(
        page_content="초미세먼지를 99.9% 걸러내는 강력한 공기청정기로, 스마트 센서가 실시간으로 공기 질을 감지합니다.",
        metadata={"year": 2024, "category": "일반가전", "user_rating": 4.9},
    ),
    Document(
        page_content="장애물 회피 기능이 뛰어난 로봇 청소기로, 한번 충전으로 넓은 공간을 깨끗하게 청소합니다.",
        metadata={"year": 2023, "category": "일반가전", "user_rating": 4.5},
    ),
    Document(
        page_content="8K 화질을 지원하는 초슬림 디자인의 스마트 TV입니다. AI 사운드 최적화 기술로 몰입감을 극대화합니다.",
        metadata={"year": 2024, "category": "일반가전", "user_rating": 4.8},
    ),
    Document(
        page_content="혁신적인 듀얼 카메라 시스템을 탑재한 최신 스마트폰입니다. 초고속 프로세서로 모든 앱을 부드럽게 실행합니다.",
        metadata={"year": 2024, "category": "스마트폰", "user_rating": 5.0},
    ),
    Document(
        page_content="가성비가 뛰어난 보급형 스마트폰입니다. 5G를 지원하며, 대용량 배터리로 하루 종일 사용이 가능합니다.",
        metadata={"year": 2024, "category": "스마트폰", "user_rating": 4.6},
    ),
    Document(
        page_content="접이식 디스플레이를 탑재해 휴대성과 활용성을 모두 잡은 폴더블 스마트폰입니다. 튼튼한 힌지로 내구성을 강화했습니다.",
        metadata={"year": 2023, "category": "스마트폰", "user_rating": 4.8},
    ),
    Document(
        page_content="피부 속부터 차오르는 수분감을 선사하는 히알루론산 세럼입니다. 모든 피부 타입에 적합하며, 끈적임 없이 빠르게 흡수됩니다.",
        metadata={"year": 2024, "category": "화장품", "user_rating": 4.7},
    ),
    Document(
        page_content="자외선 차단 효과와 미백 기능을 동시에 갖춘 선크림입니다. 민감한 피부에도 자극 없이 순하게 작용합니다.",
        metadata={"year": 2023, "category": "화장품", "user_rating": 4.9},
    ),
    Document(
        page_content="강렬한 발색력을 자랑하는 매트 립스틱입니다. 입술에 부드럽게 밀착되어 오랜 시간 유지됩니다.",
        metadata={"year": 2024, "category": "화장품", "user_rating": 4.5},
    ),
    Document(
        page_content="제로백 3초대의 강력한 성능을 자랑하는 순수 전기차입니다. 1회 충전으로 500km 이상 주행이 가능합니다.",
        metadata={"year": 2024, "category": "자동차", "user_rating": 4.9},
    ),
    Document(
        page_content="패밀리 SUV의 정석으로 불리는 모델입니다. 넓은 실내 공간과 최첨단 안전 기능을 탑재했습니다.",
        metadata={"year": 2023, "category": "자동차", "user_rating": 4.8},
    ),
    Document(
        page_content="세련된 디자인과 뛰어난 연비를 갖춘 하이브리드 세단입니다. 도심 주행에 최적화된 성능을 제공합니다.",
        metadata={"year": 2024, "category": "자동차", "user_rating": 4.7},
    ),
]

# 2. ChromaDB 벡터 스토어 생성
embedding = OpenAIEmbeddings()
vector_store = Chroma.from_documents(docs, embedding)

# 3. SelfQueryRetriever를 위한 메타데이터 스키마 정의
metadata_field_info = [
    AttributeInfo(
        name="category",
        description="제품의 종류. '일반가전', '스마트폰', '화장품', '자동차' 중 하나.",
        type="string",
    ),
    AttributeInfo(
        name="year",
        description="제품 출시 연도. 숫자 값.",
        type="integer",
    ),
    AttributeInfo(
        name="user_rating",
        description="사용자 평점. 1.0에서 5.0 사이의 숫자 값.",
        type="float",
    ),
]

# 4. SelfQueryRetriever 인스턴스 생성 (수정된 부분)
document_content_description = "다양한 제품에 대한 설명"
llm = ChatOpenAI(temperature=0)

examples = [
    (
        "2024년에 출시된 스마트폰에 대해 알려줘",
        {
            "query": "스마트폰",
            "filter": 'and(eq("category", "스마트폰"), eq("year", 2024))',
        },
    ),
    (
        "평점이 4.8점 이상인 자동차 찾아줘",
        {
            "query": "자동차",
            "filter": 'and(eq("category", "자동차"), gte("user_rating", 4.8))',
        },
    ),
    (
        "2023년에 나온 화장품이나 자동차는?",
        {
            "query": "화장품 자동차",
            "filter": 'or(and(eq("category", "화장품"), eq("year", 2023)), and(eq("category", "자동차"), eq("year", 2023)))',
        },
    ),
]

# load_query_constructor_runnable를 사용하여 쿼리 생성자 생성
query_constructor = load_query_constructor_runnable(
    llm,
    document_content_description,
    metadata_field_info,
    examples=examples,
    allowed_comparators=[
        Comparator.EQ,
        Comparator.GT,
        Comparator.GTE,
        Comparator.LT,
        Comparator.LTE,
    ],
    allowed_operators=[Operator.AND, Operator.OR, Operator.NOT],
)

# QueryConstructor 인스턴스를 직접 SelfQueryRetriever에 전달
retriever = SelfQueryRetriever(
    query_constructor=query_constructor, vectorstore=vector_store
)

# 5. 검색 및 결과 출력
queries = [
    "2024년에 출시된 스마트폰에 대해 알려줘",
    "평점이 4.8점 이상인 자동차 찾아줘",
    "공기청정기 같은 일반가전 제품에 대해 알려줘",
    "2023년에 나온 화장품이나 자동차는?",
]

for query in queries:
    print(f"--- 사용자 쿼리: {query} ---")
    retrieved_docs = retriever.invoke(query)
    for doc in retrieved_docs:
        print(f"  - 문서: {doc.page_content}")
        print(f"    (메타데이터: {doc.metadata})")
    print("\n")
