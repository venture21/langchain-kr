# Title : LangChain과 Gradio를 활용한 대화형 채팅봇 애플리케이션

# 이 애플리케이션은 다음과 같은 주요 기능을 제공합니다:
# 1. 사용자별 대화 관리 (SQLite DB 활용)
# 2. 여러 채팅방 생성 및 관리
# 3. 대화 내역 영구 저장 및 불러오기
# 4. 실시간 스트리밍 응답
# 5. 대화 컨텍스트 유지 (메모리 기능)

import gradio as gr
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.utils import ConfigurableFieldSpec
from dotenv import load_dotenv
import os

# RAG 관련 임포트
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_core.runnables import RunnablePassthrough

# ============================================================================
# 1. 초기 설정 및 환경 변수 로드
# ============================================================================

# .env 파일에서 API KEY 정보 로드 (OPENAI_API_KEY 등)
load_dotenv()

# 전역 변수로 vectorstore와 retriever 관리
vectorstore = None
retriever = None
current_contexts = []  # 현재 검색된 컨텍스트 저장

# ============================================================================
# 2. LangChain 프롬프트 및 체인 설정
# ============================================================================

# ChatGPT에게 전달할 프롬프트 템플릿 설정
# system: AI의 역할 정의
# MessagesPlaceholder: 이전 대화 내역이 삽입될 위치
# human: 사용자의 현재 질문
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="chat_history"),  # 대화 이력이 여기에 삽입됨
        ("human", "{question}"),
    ]
)

# RAG를 위한 프롬프트 템플릿
rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a helpful assistant. Use the following pieces of context to answer the question.
        If you don't know the answer based on the context, just say that you don't know.
        
        Context:
        {context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

# 사용 가능한 모델 목록
AVAILABLE_MODELS = {
    "GPT-4o-mini": "OpenAI의 경량화된 GPT-4 모델",
    "GPT-4o": "OpenAI의 최신 GPT-4 모델",
    "Gemini-2.5-Flash": "Google의 최신 빠른 응답 Gemini 모델",
    "Gemini-2.5-Pro": "Google의 고성능 Gemini Pro 모델",
}


def get_llm_chain(model_name="GPT-4o-mini"):
    """
    선택한 모델에 따라 적절한 LLM 체인을 생성하는 함수

    Args:
        model_name (str): 사용할 모델 이름

    Returns:
        chain: LangChain 파이프라인 체인
    """
    # Gemini 모델인 경우
    if "Gemini" in model_name:
        if model_name == "Gemini-2.5-Flash":
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.7,
                google_api_key=os.getenv("GOOGLE_API_KEY"),
            )
        elif model_name == "Gemini-2.5-Pro":
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-pro",
                temperature=0.7,
                google_api_key=os.getenv("GOOGLE_API_KEY"),
            )
    # OpenAI 모델인 경우
    else:
        if model_name == "GPT-4o-mini":
            llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)
        elif model_name == "GPT-4o":
            llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)
        else:
            # 기본값
            llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)

    # LLM 체인 생성: 프롬프트 -> LLM -> 문자열 파서
    return prompt | llm | StrOutputParser()


# 기본 체인 생성 (GPT-4o-mini)
default_chain = get_llm_chain("GPT-4o-mini")

# ============================================================================
# 3. 데이터베이스 및 메모리 관리 함수
# ============================================================================


def get_chat_history(user_id, conversation_id):
    """
    사용자별, 대화별 채팅 기록을 가져오는 함수

    Args:
        user_id (str): 사용자 식별자 (테이블명으로 사용)
        conversation_id (str): 대화 식별자 (세션 ID로 사용)

    Returns:
        SQLChatMessageHistory: SQLite DB에 연결된 채팅 기록 객체

    동작 방식:
    - 각 사용자는 별도의 테이블로 관리됨
    - 각 대화는 session_id로 구분됨
    - chat_history.db 파일에 모든 데이터 저장
    """
    return SQLChatMessageHistory(
        table_name=user_id,
        session_id=conversation_id,
        connection="sqlite:///chat_history.db",
    )


# 설정 필드 정의: 런타임에 동적으로 설정 가능한 필드들
config_fields = [
    ConfigurableFieldSpec(
        id="user_id",
        annotation=str,
        name="User ID",
        description="Unique identifier for a user.",
        default="",
        is_shared=True,  # 세션 간 공유
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


def get_chain_with_history(model_name="GPT-4o-mini", use_rag=False):
    """
    선택한 모델로 메시지 히스토리를 포함한 체인을 생성

    Args:
        model_name (str): 사용할 모델 이름
        use_rag (bool): RAG 사용 여부

    Returns:
        RunnableWithMessageHistory: 히스토리가 포함된 체인
    """
    if use_rag and retriever is not None:
        chain = get_rag_chain(model_name)
    else:
        chain = get_llm_chain(model_name)
    
    return RunnableWithMessageHistory(
        chain,  # 선택된 모델의 LLM 체인
        get_chat_history,  # 히스토리 가져오는 함수
        input_messages_key="question",  # 입력 메시지 키
        history_messages_key="chat_history",  # 히스토리 메시지 키
        history_factory_config=config_fields,  # 설정 필드
    )


# ============================================================================
# 4. 채팅 응답 처리 함수
# ============================================================================


def chat_response(message, history, user_id, conversation_id, model_name):
    """
    채팅 응답을 생성하는 핵심 함수

    Args:
        message (str): 사용자가 입력한 메시지
        history (list): 현재까지의 대화 기록 [(user_msg, ai_msg), ...]
        user_id (str): 사용자 식별자
        conversation_id (str): 채팅방 식별자
        model_name (str): 사용할 AI 모델 이름

    Yields:
        list: 업데이트된 대화 기록 (스트리밍)

    주요 기능:
    1. 입력 검증 (사용자명, 채팅방 이름 확인)
    2. 특수문자 처리 (공백을 언더스코어로 변환)
    3. 선택된 모델로 LLM 스트리밍 응답 생성
    4. 실시간으로 UI 업데이트
    """

    # 사용자명과 채팅방 이름 검증
    if not user_id or not user_id.strip():
        history = history or []
        history.append((message, "⚠️ 사용자명을 입력해 주세요."))
        return history

    if not conversation_id or not conversation_id.strip():
        history = history or []
        history.append((message, "⚠️ 채팅방 이름을 입력해 주세요."))
        return history

    # 드롭다운에서 선택한 값에 "(N개 메시지)" 형식이 포함된 경우 제거
    if " (" in conversation_id and conversation_id.endswith(")"):
        conversation_id = conversation_id.split(" (")[0]

    # 공백을 언더스코어로 변환 (SQLite 테이블명 제약)
    user_id = user_id.strip().replace(" ", "_")
    conversation_id = conversation_id.strip().replace(" ", "_")

    # 디버깅: 현재 사용 중인 테이블, 세션, 모델 정보 콘솔 출력
    print(
        f"📊 DB 정보: 테이블명={user_id}, 세션ID={conversation_id}, 모델={model_name}"
    )

    # LangChain 설정 객체 생성
    config = {"configurable": {"user_id": user_id, "conversation_id": conversation_id}}

    # history가 None인 경우 빈 리스트로 초기화
    if history is None:
        history = []

    try:
        # 선택된 모델로 체인 생성
        chain_with_history = get_chain_with_history(model_name)

        # 스트리밍 응답 생성
        response = ""
        # chain_with_history.stream()은 토큰 단위로 응답을 스트리밍
        for chunk in chain_with_history.stream({"question": message}, config=config):
            response += chunk
            # 매 청크마다 전체 대화 기록을 업데이트하여 UI에 반영
            updated_history = history + [(message, response)]
            yield updated_history
    except Exception as e:
        # 오류 발생시 오류 메시지를 대화에 추가
        error_msg = f"오류가 발생했습니다: {str(e)}"
        if "GOOGLE_API_KEY" in str(e):
            error_msg += "\n💡 Gemini 모델을 사용하려면 .env 파일에 GOOGLE_API_KEY를 설정해주세요."
        history.append((message, error_msg))
        yield history


# ============================================================================
# 5. UI 제어 함수들
# ============================================================================


def clear_chat():
    """
    채팅 화면을 초기화하는 함수
    Returns: None (Gradio가 빈 채팅창으로 업데이트)
    """
    return None


def create_new_chat(user_id, conversation_id):
    """
    새로운 채팅방을 시작하는 함수

    Args:
        user_id (str): 사용자 식별자
        conversation_id (str): 새 채팅방 이름

    Returns:
        tuple: (채팅창 업데이트, 입력창 업데이트, 상태 메시지)
    """
    if not user_id or not conversation_id:
        return (
            gr.update(),
            gr.update(),
            "⚠️ 사용자명과 채팅방 이름을 모두 입력해 주세요.",
        )

    # 채팅창 초기화, 입력창 비우기, 성공 메시지 표시
    return (
        gr.update(value=[]),
        gr.update(value=""),
        f"✅ 새 채팅방 '{conversation_id}'이(가) 사용자 '{user_id}'로 생성되었습니다.",
    )


def update_conversation_list(user_id):
    """
    사용자 변경시 채팅방 목록을 업데이트하는 기본 함수

    Args:
        user_id (str): 사용자 식별자

    Returns:
        gr.update: 드롭다운 업데이트 객체
    """
    if not user_id:
        return gr.update(choices=[], value="")

    conversations = get_conversation_list(user_id)
    return gr.update(choices=conversations, value="")


def refresh_lists():
    """
    사용자 목록과 채팅방 목록을 새로고침하는 함수

    Returns:
        tuple: (사용자 드롭다운 업데이트, 채팅방 드롭다운 업데이트, 상태 메시지)
    """
    users = get_user_list()
    return gr.update(choices=users), gr.update(), "🔄 목록이 새로고침되었습니다."


def update_conversation_on_focus(user_id):
    """
    드롭다운 클릭시 채팅방 목록을 실시간으로 업데이트하는 함수
    각 채팅방의 메시지 개수 정보도 함께 표시

    Args:
        user_id (str): 사용자 식별자

    Returns:
        gr.update: 메시지 개수 정보가 포함된 채팅방 목록

    특징:
    - 드롭다운 클릭시마다 DB에서 최신 정보 조회
    - "채팅방명 (N개 메시지)" 형식으로 표시
    """
    if not user_id:
        return gr.update(choices=[], value="")

    conversations = get_conversation_list(user_id)
    conversation_with_info = []

    import sqlite3
    import json
    import os
    from datetime import datetime

    if os.path.exists("chat_history.db"):
        user_id_db = user_id.strip().replace(" ", "_")
        conn = sqlite3.connect("chat_history.db")
        cursor = conn.cursor()

        # 각 채팅방의 메시지 개수를 조회하여 정보 추가
        for conv in conversations:
            cursor.execute(
                f"SELECT COUNT(*) FROM {user_id_db} WHERE session_id=?;", (conv,)
            )
            count = cursor.fetchone()[0]
            conversation_with_info.append(f"{conv} ({count}개 메시지)")

        conn.close()

    return gr.update(
        choices=conversation_with_info if conversation_with_info else conversations
    )


def load_selected_chat(user_id, conversation_id):
    """
    선택한 채팅방의 대화 내역을 불러오는 함수

    Args:
        user_id (str): 사용자 식별자
        conversation_id (str): 채팅방 식별자

    Returns:
        tuple: (채팅창 업데이트, 상태 메시지)
    """
    if not user_id or not conversation_id:
        return gr.update(), "⚠️ 사용자와 채팅방을 모두 선택해 주세요."

    # 메시지 개수 정보 "(N개 메시지)" 부분 제거
    if " (" in conversation_id and conversation_id.endswith(")"):
        conversation_id = conversation_id.split(" (")[0]

    history = load_conversation_history(user_id, conversation_id)
    if history:
        return (
            gr.update(value=history),
            f"✅ 채팅방 '{conversation_id}'의 대화 내역을 불러왔습니다.",
        )
    else:
        return (
            gr.update(value=[]),
            f"ℹ️ 채팅방 '{conversation_id}'에 저장된 대화가 없습니다.",
        )


# ============================================================================
# 6. 데이터베이스 조회 함수들
# ============================================================================


def show_db_info():
    """
    현재 데이터베이스에 저장된 모든 채팅 정보를 표시하는 함수

    Returns:
        str: 포맷팅된 데이터베이스 정보 문자열

    표시 정보:
    - 모든 사용자 목록
    - 각 사용자의 채팅방 목록
    - 각 채팅방의 메시지 개수
    """
    import sqlite3

    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()

    # SQLite의 sqlite_master 테이블에서 모든 테이블 조회
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    info = "📊 **데이터베이스 정보:**\n\n"

    for table in tables:
        table_name = table[0]
        # 각 테이블(사용자)의 세션(채팅방) 목록 조회
        cursor.execute(f"SELECT DISTINCT session_id FROM {table_name};")
        sessions = cursor.fetchall()

        info += f"👤 **사용자: {table_name}**\n"
        for session in sessions:
            # 각 세션의 메시지 개수 조회
            cursor.execute(
                f"SELECT COUNT(*) FROM {table_name} WHERE session_id=?;", (session[0],)
            )
            count = cursor.fetchone()[0]
            info += f"  - 채팅방: {session[0]} (메시지 {count}개)\n"
        info += "\n"

    conn.close()
    return info


def get_user_list():
    """
    저장된 모든 사용자 목록을 가져오는 함수

    Returns:
        list: 사용자명 리스트 (각 사용자는 별도 테이블로 저장됨)
    """
    import sqlite3
    import os

    if not os.path.exists("chat_history.db"):
        return []

    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()

    # 모든 테이블명 = 사용자명
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    conn.close()

    return [table[0] for table in tables] if tables else []


def get_conversation_list(user_id):
    """
    특정 사용자의 모든 채팅방 목록을 가져오는 함수

    Args:
        user_id (str): 사용자 식별자

    Returns:
        list: 채팅방 ID 리스트 (최신순 정렬)
    """
    import sqlite3
    import os

    if not user_id or not os.path.exists("chat_history.db"):
        return []

    user_id = user_id.strip().replace(" ", "_")

    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()

    # 테이블 존재 확인
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (user_id,)
    )
    table_exists = cursor.fetchone()

    if not table_exists:
        conn.close()
        return []

    # 해당 사용자의 모든 세션 ID를 최신순으로 가져오기
    cursor.execute(f"SELECT DISTINCT session_id FROM {user_id} ORDER BY id DESC;")
    sessions = cursor.fetchall()
    conn.close()

    return [session[0] for session in sessions] if sessions else []


def load_conversation_history(user_id, conversation_id):
    """
    선택한 채팅방의 전체 대화 내역을 불러오는 함수

    Args:
        user_id (str): 사용자 식별자
        conversation_id (str): 채팅방 식별자

    Returns:
        list: [(사용자 메시지, AI 응답), ...] 형식의 대화 기록

    데이터 처리:
    - JSON 형식으로 저장된 메시지를 파싱
    - human/ai 타입별로 페어링하여 대화 형식으로 변환
    """
    import sqlite3
    import json
    import os

    if not user_id or not conversation_id or not os.path.exists("chat_history.db"):
        return []

    user_id = user_id.strip().replace(" ", "_")
    conversation_id = conversation_id.strip().replace(" ", "_")

    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()

    # 테이블 존재 확인
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (user_id,)
    )
    table_exists = cursor.fetchone()

    if not table_exists:
        conn.close()
        return []

    # 해당 채팅방의 메시지를 시간순으로 가져오기
    cursor.execute(
        f"SELECT message FROM {user_id} WHERE session_id=? ORDER BY id;",
        (conversation_id,),
    )
    messages = cursor.fetchall()
    conn.close()

    # JSON 메시지를 대화 형식으로 변환
    history = []
    for msg in messages:
        msg_data = json.loads(msg[0])
        if msg_data["type"] == "human":
            human_msg = msg_data["data"]["content"]
        elif msg_data["type"] == "ai":
            ai_msg = msg_data["data"]["content"]
            # human과 ai 메시지를 페어로 만들어 추가
            history.append((human_msg, ai_msg))

    return history


# ============================================================================
# 7. Gradio UI 구성
# ============================================================================

# Gradio Blocks를 사용한 커스텀 UI 생성
with gr.Blocks(title="LangChain 채팅봇", theme=gr.themes.Soft()) as demo:
    # 앱 헤더
    gr.Markdown(
        """
        # 🤖 LangChain SQLite 기반 채팅봇
        
        채팅 기록이 SQLite 데이터베이스에 자동으로 저장됩니다.
        """
    )

    with gr.Row():
        # 왼쪽 사이드바: 설정 및 제어 패널
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ 설정")

            # 사용자 선택/입력 드롭다운
            existing_users = get_user_list()
            user_id_dropdown = gr.Dropdown(
                label="기존 사용자 선택",
                choices=existing_users,
                value="",
                allow_custom_value=True,  # 새 사용자명 입력 가능
                info="기존 사용자를 선택하거나 새 이름을 입력하세요",
            )

            # 채팅방 선택/입력 드롭다운
            conversation_dropdown = gr.Dropdown(
                label="채팅방 선택",
                choices=[],
                value="",
                allow_custom_value=True,  # 새 채팅방명 입력 가능
                info="기존 채팅방을 선택하거나 새 이름을 입력하세요",
            )

            # 모델 선택 드롭다운
            model_dropdown = gr.Dropdown(
                label="AI 모델 선택",
                choices=list(AVAILABLE_MODELS.keys()),
                value="GPT-4o-mini",
                info="사용할 AI 모델을 선택하세요",
            )

            # 제어 버튼들
            with gr.Row():
                load_chat_btn = gr.Button("📂 채팅방 불러오기", variant="secondary")
                new_chat_btn = gr.Button("🆕 새 채팅 시작", variant="primary")

            with gr.Row():
                refresh_btn = gr.Button("🔄 목록 새로고침")
                clear_btn = gr.Button("🗑️ 화면 초기화")

            db_info_btn = gr.Button("📊 DB 정보 보기", variant="secondary")

            # 상태 표시 텍스트박스
            status_text = gr.Textbox(
                label="상태", interactive=False, show_label=True, lines=5
            )

            # 사용 방법 안내
            gr.Markdown(
                """
                ### 📝 사용 방법
                1. **사용자명**과 **채팅방 이름**을 입력하세요
                2. 메시지를 입력하고 전송하세요
                3. 같은 설정으로 재접속하면 이전 대화가 이어집니다
                
                ### 💡 팁
                - 사용자명과 채팅방 이름으로 대화를 관리합니다
                - 새로운 대화를 시작하려면 채팅방 이름을 변경하세요
                """
            )

        # 오른쪽 메인 영역: 채팅 인터페이스
        with gr.Column(scale=2):
            # 채팅 디스플레이
            chatbot = gr.Chatbot(
                height=600,
                label="채팅 내역",
                show_copy_button=True,  # 복사 버튼 표시
                type="tuples",  # (사용자, AI) 튜플 형식
            )

            # 메시지 입력창
            msg = gr.Textbox(
                label="메시지 입력",
                placeholder="메시지를 입력하고 Enter를 누르세요...",
                lines=2,
            )

            # 전송 및 재시도 버튼
            with gr.Row():
                submit_btn = gr.Button("📤 전송", variant="primary")
                retry_btn = gr.Button("🔄 재시도")

    # ============================================================================
    # 8. 이벤트 핸들러 연결
    # ============================================================================

    # Enter 키로 메시지 전송
    msg.submit(
        chat_response,
        inputs=[msg, chatbot, user_id_dropdown, conversation_dropdown, model_dropdown],
        outputs=[chatbot],
        show_progress=True,
    ).then(
        lambda: "", outputs=[msg]  # 입력창 비우기
    ).then(
        # 대화 후 채팅방 목록 자동 업데이트 (메시지 개수 반영)
        update_conversation_on_focus,
        inputs=[user_id_dropdown],
        outputs=[conversation_dropdown],
    )

    # 전송 버튼 클릭
    submit_btn.click(
        chat_response,
        inputs=[msg, chatbot, user_id_dropdown, conversation_dropdown, model_dropdown],
        outputs=[chatbot],
        show_progress=True,
    ).then(lambda: "", outputs=[msg]).then(
        # 대화 후 채팅방 목록 자동 업데이트
        update_conversation_on_focus,
        inputs=[user_id_dropdown],
        outputs=[conversation_dropdown],
    )

    # 재시도 버튼 (입력창을 비우지 않음)
    retry_btn.click(
        chat_response,
        inputs=[msg, chatbot, user_id_dropdown, conversation_dropdown, model_dropdown],
        outputs=[chatbot],
        show_progress=True,
    )

    # 화면 초기화
    clear_btn.click(clear_chat, outputs=[chatbot])

    # 새 채팅 시작
    new_chat_btn.click(
        create_new_chat,
        inputs=[user_id_dropdown, conversation_dropdown],
        outputs=[chatbot, msg, status_text],
    )

    # 사용자 변경시 채팅방 목록 업데이트
    user_id_dropdown.change(
        update_conversation_on_focus,
        inputs=[user_id_dropdown],
        outputs=[conversation_dropdown],
    )

    # 채팅방 드롭다운 클릭시 최신 목록 업데이트
    conversation_dropdown.focus(
        update_conversation_on_focus,
        inputs=[user_id_dropdown],
        outputs=[conversation_dropdown],
    )

    # 목록 새로고침
    refresh_btn.click(
        refresh_lists, outputs=[user_id_dropdown, conversation_dropdown, status_text]
    )

    # 채팅방 불러오기
    load_chat_btn.click(
        load_selected_chat,
        inputs=[user_id_dropdown, conversation_dropdown],
        outputs=[chatbot, status_text],
    )

    # DB 정보 표시
    db_info_btn.click(show_db_info, outputs=[status_text])

# ============================================================================
# 9. 애플리케이션 실행
# ============================================================================

if __name__ == "__main__":
    # Gradio 서버 설정 및 실행
    demo.queue()  # 스트리밍을 위한 큐 활성화 (필수)
    demo.launch(
        server_name="0.0.0.0",  # 모든 네트워크 인터페이스에서 접근 가능
        server_port=7860,  # 기본 포트
        share=False,  # True로 설정시 임시 공개 URL 생성
        show_error=True,  # 오류 메시지 표시
    )
