from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.utils import ConfigurableFieldSpec
from dotenv import load_dotenv

# API KEY 정보로드
load_dotenv()

# SQLChatMessageHistory 객체를 생성하고 세션 ID와 데이터베이스 연결 파일을 설정
chat_message_history = SQLChatMessageHistory(
    session_id="sql_history", connection="sqlite:///sqlite.db"
)

prompt = ChatPromptTemplate.from_messages(
    [
        # 시스템 메시지
        ("system", "You are a helpful assistant."),
        # 대화 기록을 위한 Placeholder
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),  # 질문
    ]
)

# chain 을 생성합니다.
chain = prompt | ChatOpenAI(model_name="gpt-4.1-mini") | StrOutputParser()

def get_chat_history(user_id, conversation_id):
    return SQLChatMessageHistory(
        table_name=user_id,
        session_id=conversation_id,
        connection="sqlite:///sqlite.db",
    )



config_fields = [
    ConfigurableFieldSpec(
        id="user_id",
        annotation=str,
        name="User ID",
        description="Unique identifier for a user.",
        default="",
        is_shared=True,
    ),
    ConfigurableFieldSpec(
        id="conversation_id",
        annotation=str,
        name="Conversation ID",
        description="Unique identifier for a conversation.",
        default="",
        is_shared=True,
    ),
]

chain_with_history = RunnableWithMessageHistory(
    chain,             # chain을 설정합니다.
    get_chat_history,  # 대화 기록을 가져오는 함수를 설정합니다.
    input_messages_key="question",  # 입력 메시지의 키를 "question"으로 설정
    history_messages_key="chat_history",  # 대화 기록 메시지의 키를 "history"로 설정
    history_factory_config=config_fields,  # 대화 기록 조회시 참고할 파라미터를 설정합니다.
)

# config 설정
config = {"configurable": {"user_id": "user1", "conversation_id": "conversation1"}}

