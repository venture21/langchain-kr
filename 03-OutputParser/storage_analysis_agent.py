"""
Pandas DataFrame Agent를 사용한 storageInfo.csv 데이터 분석 및 시각화
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드
df = pd.read_csv("./data/storageInfo.csv")

print("=== 데이터 정보 ===")
print(f"데이터 크기: {df.shape}")
print(f"컬럼: {df.columns.tolist()}")
print("\n데이터 샘플:")
print(df.head())

# LLM 초기화
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Pandas DataFrame Agent 생성
agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=True,  # 실행 과정 출력
    agent_type=AgentType.OPENAI_FUNCTIONS,
    allow_dangerous_code=True  # 코드 실행 허용
)

# 분석 쿼리 함수
def run_analysis(query):
    """에이전트를 통해 분석 쿼리 실행"""
    print(f"\n질문: {query}")
    print("-" * 50)
    try:
        result = agent.run(query)
        print(f"답변: {result}")
        return result
    except Exception as e:
        print(f"Error: {e}")
        return None

# 1. 데이터 탐색 분석
print("\n" + "="*60)
print("1. 데이터 탐색 분석")
print("="*60)

queries = [
    "데이터프레임의 기본 정보를 요약해줘. 전체 행과 열의 개수, 각 컬럼의 데이터 타입을 알려줘.",
    "category 컬럼에는 어떤 종류의 값들이 있고, 각각 몇 개씩 있는지 알려줘.",
    "storageDays의 평균, 중앙값, 최대값, 최소값을 계산해줘.",
    "카테고리별로 평균 보관일수를 계산하고, 가장 긴 카테고리와 짧은 카테고리를 알려줘.",
    "보관일수가 30일 이상인 식품은 몇 개이고, 그 중 상위 5개의 이름과 보관일수를 알려줘."
]

for query in queries:
    run_analysis(query)

# 2. 심화 분석
print("\n" + "="*60)
print("2. 심화 분석")
print("="*60)

advanced_queries = [
    "과일과 야채 카테고리의 평균 보관일수를 비교해줘. 어느 쪽이 더 오래 보관 가능한가?",
    "보관일수가 가장 긴 상위 10개 식품과 가장 짧은 하위 10개 식품을 비교 분석해줘.",
    "storageMethod 컬럼에서 '냉장'이라는 단어가 포함된 식품은 몇 개인지, '냉동'이 포함된 식품은 몇 개인지 알려줘.",
    "name 컬럼에서 가장 자주 나타나는 단어 5개를 찾아줘. (예: 배추, 양배추에서 '배추')",
]

for query in advanced_queries:
    run_analysis(query)

# 3. 시각화를 위한 데이터 준비
print("\n" + "="*60)
print("3. 데이터 시각화")
print("="*60)

# 시각화 설정
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('식품 보관 데이터 분석', fontsize=16, fontweight='bold')

# 1. 카테고리별 식품 개수
ax1 = axes[0, 0]
category_counts = df['category'].value_counts()
category_counts.plot(kind='bar', ax=ax1, color='skyblue')
ax1.set_title('카테고리별 식품 개수')
ax1.set_xlabel('카테고리')
ax1.set_ylabel('개수')
ax1.tick_params(axis='x', rotation=45)

# 2. 카테고리별 평균 보관일수
ax2 = axes[0, 1]
category_avg = df.groupby('category')['storageDays'].mean().sort_values()
category_avg.plot(kind='barh', ax=ax2, color='lightcoral')
ax2.set_title('카테고리별 평균 보관일수')
ax2.set_xlabel('평균 보관일수 (일)')
ax2.set_ylabel('카테고리')

# 3. 보관일수 분포
ax3 = axes[0, 2]
df['storageDays'].hist(bins=20, ax=ax3, color='lightgreen', edgecolor='black')
ax3.set_title('보관일수 분포')
ax3.set_xlabel('보관일수 (일)')
ax3.set_ylabel('빈도')
ax3.axvline(df['storageDays'].mean(), color='red', linestyle='--', label=f'평균: {df["storageDays"].mean():.1f}일')
ax3.legend()

# 4. 보관일수 상위 10개 식품
ax4 = axes[1, 0]
top10 = df.nlargest(10, 'storageDays')[['name', 'storageDays']]
ax4.barh(range(len(top10)), top10['storageDays'].values, color='gold')
ax4.set_yticks(range(len(top10)))
ax4.set_yticklabels(top10['name'].values)
ax4.set_title('보관일수 상위 10개 식품')
ax4.set_xlabel('보관일수 (일)')

# 5. 카테고리별 보관일수 박스플롯
ax5 = axes[1, 1]
df.boxplot(column='storageDays', by='category', ax=ax5)
ax5.set_title('카테고리별 보관일수 분포')
ax5.set_xlabel('카테고리')
ax5.set_ylabel('보관일수 (일)')
plt.sca(ax5)
plt.xticks(rotation=45)

# 6. 보관일수 구간별 분포
ax6 = axes[1, 2]
bins = [0, 7, 14, 30, 60, 200]
labels = ['1주 이하', '1-2주', '2주-1달', '1-2달', '2달 이상']
df['storage_range'] = pd.cut(df['storageDays'], bins=bins, labels=labels)
storage_range_counts = df['storage_range'].value_counts()
colors = plt.cm.Set3(range(len(storage_range_counts)))
wedges, texts, autotexts = ax6.pie(storage_range_counts.values, 
                                     labels=storage_range_counts.index,
                                     autopct='%1.1f%%',
                                     colors=colors)
ax6.set_title('보관일수 구간별 비율')

plt.tight_layout()
plt.savefig('storage_analysis_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n시각화가 'storage_analysis_visualization.png' 파일로 저장되었습니다.")

# 4. 인사이트 도출
print("\n" + "="*60)
print("4. 주요 인사이트 도출")
print("="*60)

insight_queries = [
    "데이터를 전체적으로 분석했을 때, 가장 주목할 만한 3가지 인사이트를 도출해줘.",
    "식품 보관 관리를 위한 실용적인 조언 3가지를 데이터를 기반으로 제시해줘.",
]

for query in insight_queries:
    run_analysis(query)

# 5. 추가 통계 분석
print("\n" + "="*60)
print("5. 추가 통계 정보")
print("="*60)

# 카테고리별 상세 통계
print("\n카테고리별 상세 통계:")
category_stats = df.groupby('category')['storageDays'].agg([
    ('개수', 'count'),
    ('평균', 'mean'),
    ('중앙값', 'median'),
    ('최소', 'min'),
    ('최대', 'max'),
    ('표준편차', 'std')
]).round(1)
print(category_stats)

# 보관 방법 키워드 분석
print("\n보관 방법 주요 키워드:")
storage_keywords = ['냉장', '냉동', '실온', '서늘한', '밀폐', '신문지', '랩']
for keyword in storage_keywords:
    count = df['storageMethod'].str.contains(keyword, na=False).sum()
    percentage = (count / len(df)) * 100
    print(f"  '{keyword}': {count}개 ({percentage:.1f}%)")