import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PandasDataFrameOutputParser
from langchain.prompts import PromptTemplate

# CSV 파일 불러오기
df = pd.read_csv("./data/storageInfo.csv")

# 데이터 확인
print("=== 원본 데이터 ===")
print(df.head())
print(f"\n데이터 크기: {df.shape}")
print(f"컬럼명: {df.columns.tolist()}")

# PandasDataFrameOutputParser 설정
parser = PandasDataFrameOutputParser(dataframe=df)

# Format instructions 확인
print("\n=== Parser Format Instructions ===")
print(parser.get_format_instructions()[:500])  # 일부만 출력

# LLM 초기화 (GPT-3.5-turbo 사용 권장)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# 프롬프트 템플릿 생성
prompt = PromptTemplate(
    template="""Answer the user query.
{format_instructions}

User Query: {query}

Remember to format your response exactly as specified in the instructions.
For example:
- To get a column: "column:column_name"
- To get a row: "row:row_number"
- To get mean: "mean:column_name"
- To count values: "value_counts:column_name"
""",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# 분석 쿼리 실행 함수
def analyze_data(query):
    """데이터프레임에 대한 쿼리를 실행하는 함수"""
    chain = prompt | llm | parser
    try:
        result = chain.invoke({"query": query})
        return result
    except Exception as e:
        print(f"Error: {e}")
        return None

# 다양한 분석 예제
print("\n=== 데이터 분석 예제 ===")

# 1. 특정 컬럼 조회
print("\n1. category 컬럼 조회:")
query1 = "Get the category column"
result1 = analyze_data(query1)
if result1:
    print(f"카테고리 종류: {set(result1['category'].values())}")
    print(f"총 데이터 수: {len(result1['category'])}")

# 2. storageDays 컬럼 조회
print("\n2. storageDays 컬럼 조회:")
query2 = "Get the storageDays column"
result2 = analyze_data(query2)
if result2:
    storage_days = pd.Series(result2['storageDays'])
    print(f"평균 보관일수: {storage_days.mean():.1f}일")
    print(f"최대 보관일수: {storage_days.max()}일")
    print(f"최소 보관일수: {storage_days.min()}일")

# 3. 첫 번째 행 조회
print("\n3. 첫 번째 행 데이터:")
query3 = "Retrieve the first row (row 0)"
result3 = analyze_data(query3)
if result3:
    for key, value in result3['0'].items():
        print(f"  {key}: {value}")

# 4. 특정 행 범위의 name 컬럼
print("\n4. 처음 5개 식품명:")
query4 = "Get the name column for rows 0 to 4"
result4 = analyze_data(query4)
if result4:
    for idx, name in result4['name'].items():
        if int(idx) < 5:
            print(f"  {idx}: {name}")

# 직접 pandas로 분석 (비교용)
print("\n=== Pandas 직접 분석 ===")

# 카테고리별 개수
print("\n카테고리별 식품 개수:")
category_counts = df['category'].value_counts()
print(category_counts)

# 카테고리별 평균 보관일수
print("\n카테고리별 평균 보관일수:")
category_avg = df.groupby('category')['storageDays'].mean().round(1)
print(category_avg)

# 보관일수 30일 이상 식품
print("\n보관일수 30일 이상인 식품:")
long_storage = df[df['storageDays'] >= 30][['name', 'storageDays', 'category']]
print(long_storage.head(10))

# 기술 통계량
print("\n=== 기술 통계량 ===")
print(df['storageDays'].describe())