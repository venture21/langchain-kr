import os
import time
import nest_asyncio
from dotenv import load_dotenv
from llama_parse import LlamaParse
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import gradio as gr
import asyncio
from typing import Dict, Tuple, Optional
import tempfile
import shutil

# 환경 설정
load_dotenv()
nest_asyncio.apply()


class LlamaParseRAGSystem:
    """LlamaParse를 활용한 RAG 시스템"""

    def __init__(self):
        self.api_key = os.environ.get("LLAMA_CLOUD_API_KEY")
        self.openai_key = os.environ.get("OPENAI_API_KEY")
        self.embeddings = None
        self.llm = None
        self.vectorstore = None
        self.qa_chain = None
        self.current_documents = None
        self.current_file_path = None  # 원본 파일 경로 저장
        self.parsed_text = None  # 파싱된 텍스트 저장
        self.text_chunks = None  # 분할된 텍스트 청크 저장

        # API 키 확인
        if not self.api_key:
            print("⚠️ 경고: LLAMA_CLOUD_API_KEY가 설정되지 않았습니다.")
        if not self.openai_key:
            print("⚠️ 경고: OPENAI_API_KEY가 설정되지 않았습니다.")

    def initialize_models(self):
        """모델 초기화"""
        if self.openai_key:
            self.embeddings = OpenAIEmbeddings()
            self.llm = ChatOpenAI(model="gpt-4", temperature=0)

    async def parse_document(
        self, file_obj, mode: str, custom_instruction: str = None
    ) -> Tuple[str, Dict, str, str]:
        """
        선택된 모드로 문서 파싱

        Returns:
            Tuple[str, Dict, str, str]: (상태 메시지, 메타데이터, 원본 파일 경로, 파싱된 텍스트)
        """
        if not file_obj:
            return "파일을 선택해주세요.", {}, None, ""

        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(file_obj.name)[1]
        ) as tmp_file:
            shutil.copy(file_obj.name, tmp_file.name)
            file_path = tmp_file.name

        start_time = time.time()
        metadata = {
            "mode": mode,
            "file_name": os.path.basename(file_obj.name),
            "file_size": os.path.getsize(file_obj.name) / 1024,  # KB
        }

        try:
            # 모드별 파서 설정
            parser_config = {
                "api_key": self.api_key,
                "result_type": "markdown",
                "language": "ko",
                "verbose": True,
            }

            # 커스텀 지시사항 추가
            if custom_instruction:
                parser_config["system_prompt"] = custom_instruction

            # 모드별 설정
            if mode == "default":
                # 기본 설정 사용
                if not custom_instruction:
                    parser_config[
                        "system_prompt"
                    ] = """
                    문서를 체계적으로 파싱하세요:
                    1. 제목과 섹션 구조 유지
                    2. 중요한 내용 강조
                    3. 표와 리스트 형식 보존
                    """

            elif mode == "fast":
                parser_config["fast_mode"] = True
                parser_config["result_type"] = "text"  # 빠른 처리를 위해 text 형식
                if not custom_instruction:
                    parser_config[
                        "system_prompt"
                    ] = """
                    빠르게 핵심 내용만 추출하세요:
                    1. 주요 텍스트 내용 중심
                    2. 간단한 구조 유지
                    """

            elif mode == "auto":
                parser_config["auto_mode"] = True
                if not custom_instruction:
                    parser_config[
                        "system_prompt"
                    ] = """
                    문서 유형을 자동으로 감지하여 최적의 방법으로 파싱하세요.
                    """

            elif mode == "vendor_multimodal":
                if not self.openai_key:
                    return "OpenAI API 키가 필요합니다.", metadata

                parser_config.update(
                    {
                        "use_vendor_multimodal_model": True,
                        "vendor_multimodal_model_name": "openai-gpt4o",
                        "vendor_multimodal_api_key": self.openai_key,
                    }
                )

                if not custom_instruction:
                    parser_config[
                        "system_prompt"
                    ] = """
                    문서를 정밀하게 분석하세요:
                    1. 모든 표를 마크다운 형식으로 변환
                    2. 이미지와 차트에 대한 상세한 설명
                    3. 수식과 다이어그램 정확히 추출
                    4. 복잡한 레이아웃 구조 보존
                    """

            elif mode == "premium":
                parser_config["premium_mode"] = True
                parser_config["page_separator"] = True

                if not custom_instruction:
                    parser_config[
                        "system_prompt"
                    ] = """
                    최고 품질로 문서를 파싱하세요:
                    1. 모든 구조 요소 보존
                    2. 레이아웃과 서식 유지
                    3. 메타데이터와 주석 포함
                    4. 페이지별 구분 유지
                    """

            # 파서 생성 및 실행
            parser = LlamaParse(**parser_config)
            documents = await parser.aload_data(file_path)

            # LangChain 형식으로 변환
            self.current_documents = []
            for doc in documents:
                if hasattr(doc, "to_langchain_format"):
                    self.current_documents.append(doc.to_langchain_format())

            # 원본 파일 경로와 파싱된 텍스트 저장
            self.current_file_path = file_obj.name
            self.parsed_text = "\n\n---\n\n".join(
                [doc.page_content for doc in self.current_documents]
            )

            # 메타데이터 업데이트
            elapsed_time = time.time() - start_time
            metadata.update(
                {
                    "processing_time": f"{elapsed_time:.2f}초",
                    "pages": len(documents),
                    "total_chars": sum(
                        len(doc.page_content) for doc in self.current_documents
                    ),
                    "success": True,
                }
            )

            # 임시 파일 삭제
            os.unlink(file_path)

            status_msg = f"""
### ✅ 문서 파싱 완료

**파일명:** {metadata['file_name']}
**파싱 모드:** {mode}
**처리 시간:** {metadata['processing_time']}
**페이지 수:** {metadata['pages']}
**총 문자 수:** {metadata['total_chars']:,}

이제 'Split & Embed' 버튼을 클릭하여 벡터 DB를 생성하세요.
"""

            return status_msg, metadata, self.current_file_path, self.parsed_text

        except Exception as e:
            # 임시 파일 삭제
            if os.path.exists(file_path):
                os.unlink(file_path)

            metadata["error"] = str(e)
            metadata["success"] = False
            return f"❌ 오류 발생: {str(e)}", metadata, None, ""

    def split_and_embed(
        self,
        splitter_type: str = "RecursiveCharacterTextSplitter",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separator: str = "\n",
        encoding_name: str = "cl100k_base",
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold_amount: float = 95.0,
        retriever_k: int = 3,
    ) -> Tuple[str, str]:
        """문서 분할 및 임베딩 생성"""
        if not self.current_documents:
            return "먼저 문서를 파싱해주세요.", ""

        if not self.openai_key:
            return "⚠️ 임베딩을 생성하려면 OpenAI API 키가 필요합니다.", ""

        try:
            self.initialize_models()

            # 선택된 splitter 유형에 따라 text_splitter 생성
            if splitter_type == "CharacterTextSplitter":
                text_splitter = CharacterTextSplitter(
                    separator=separator,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    length_function=len,
                )
                splitter_info = f"CharacterTextSplitter (separator: '{separator}')"

            elif splitter_type == "RecursiveCharacterTextSplitter":
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    separators=["\n\n", "\n", ". ", " ", ""],
                    length_function=len,
                )
                splitter_info = "RecursiveCharacterTextSplitter (계층적 분할)"

            elif splitter_type == "TokenTextSplitter":
                # TokenTextSplitter는 토큰 기반 분할
                text_splitter = TokenTextSplitter(
                    encoding_name=encoding_name,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                splitter_info = f"TokenTextSplitter (encoding: {encoding_name})"

            elif splitter_type == "SemanticChunker":
                # SemanticChunker는 의미적 유사성 기반 분할
                if not self.embeddings:
                    self.embeddings = OpenAIEmbeddings()

                text_splitter = SemanticChunker(
                    embeddings=self.embeddings,
                    breakpoint_threshold_type=breakpoint_threshold_type,
                    breakpoint_threshold_amount=breakpoint_threshold_amount,
                )
                splitter_info = f"SemanticChunker (threshold: {breakpoint_threshold_type}={breakpoint_threshold_amount})"

            else:
                return f"❌ 지원하지 않는 Splitter 유형: {splitter_type}"

            # 문서 분할
            splits = text_splitter.split_documents(self.current_documents)

            # 분할 결과 확인
            if not splits:
                return "❌ 문서 분할 실패: 분할된 청크가 없습니다."

            # 벡터 스토어 생성
            self.vectorstore = FAISS.from_documents(splits, self.embeddings)

            # 검색 설정 (k값 저장)
            self.retriever_k = retriever_k
            retriever = self.vectorstore.as_retriever(
                search_type="similarity", search_kwargs={"k": retriever_k}
            )

            # RAG 체인 생성
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
            )

            # 청크 통계 계산
            chunk_lengths = [len(doc.page_content) for doc in splits]
            avg_chunk_length = (
                sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0
            )
            min_chunk_length = min(chunk_lengths) if chunk_lengths else 0
            max_chunk_length = max(chunk_lengths) if chunk_lengths else 0

            # 분할된 텍스트 청크 저장 및 표시용 텍스트 생성
            self.text_chunks = splits
            chunks_display = ""
            for i, chunk in enumerate(splits, 1):
                chunks_display += f"### 청크 {i}\n"
                chunks_display += f"**길이:** {len(chunk.page_content)}자\n"
                chunks_display += f"**내용:**\n{chunk.page_content}\n\n----------------------------------\n\n"

            status_msg = f"""
### ✅ 벡터 DB 생성 완료

**Splitter 유형:** {splitter_info}
**청크 크기 설정:** {chunk_size}
**청크 오버랩:** {chunk_overlap}
**총 청크 수:** {len(splits)}
**평균 청크 길이:** {avg_chunk_length:.0f}자
**최소/최대 청크:** {min_chunk_length}자 / {max_chunk_length}자
**벡터 차원:** 1536 (OpenAI)
**검색 개수 (k):** {retriever_k}개

이제 질문을 입력하여 문서에 대해 질의할 수 있습니다.
"""

            return status_msg, chunks_display

        except Exception as e:
            return f"❌ 벡터 DB 생성 실패: {str(e)}", ""

    async def answer_question(self, question: str) -> tuple:
        """RAG 시스템을 통한 질문 응답"""
        if not self.qa_chain:
            return "먼저 벡터 DB를 생성해주세요.", []

        if not question.strip():
            return "질문을 입력해주세요.", []

        try:
            # 질문 응답
            result = self.qa_chain({"query": question})
            answer = result["result"]

            # 검색된 문서 정보 수집
            retrieved_docs = []
            if "source_documents" in result and result["source_documents"]:
                for i, doc in enumerate(result["source_documents"], 1):
                    doc_info = {
                        "순번": i,
                        "내용": doc.page_content[:500]
                        + ("..." if len(doc.page_content) > 500 else ""),
                        "메타데이터": str(doc.metadata) if doc.metadata else "없음",
                    }
                    retrieved_docs.append(doc_info)

            # 출처 문서 요약
            source_info = ""
            if retrieved_docs:
                source_info = f"\n\n### 📚 검색된 문서 ({len(retrieved_docs)}개)\n"
                for doc in retrieved_docs[:3]:  # 답변에는 처음 3개만 표시
                    source_info += f"\n**[문서 {doc['순번']}]**\n"
                    source_info += f"{doc['내용'][:200]}...\n"

            answer_text = f"""
### 💬 답변

{answer}

{source_info}
"""
            return answer_text, retrieved_docs

        except Exception as e:
            return f"❌ 답변 생성 실패: {str(e)}", []

    def clear_system(self):
        """시스템 초기화"""
        self.vectorstore = None
        self.qa_chain = None
        self.current_documents = None
        self.current_file_path = None
        self.parsed_text = None
        self.text_chunks = None
        return "시스템이 초기화되었습니다."


def create_gradio_interface():
    """Gradio 인터페이스 생성"""
    app = LlamaParseRAGSystem()

    with gr.Blocks(title="LlamaParse RAG System", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
        # 🦙 LlamaParse RAG System
        
        ### 문서 파싱 → 분할 → 임베딩 → 검색 기반 질의응답
        
        고급 문서 파싱과 RAG(Retrieval-Augmented Generation) 시스템을 통해 
        문서에 대한 정확한 질의응답을 수행합니다.
        """
        )

        # 상단 컨트롤 패널
        with gr.Row():
            # Step 1: 문서 파싱 (박스로 감싸기)
            with gr.Group():
                with gr.Column(min_width=600):
                    gr.Markdown("### 📄 Step 1: 문서 업로드 & 파싱")

                    file_input = gr.File(
                        label="문서 업로드",
                        file_types=[".pdf", ".docx", ".pptx", ".txt"],
                        type="filepath",
                    )

                    with gr.Row():
                        parsing_mode = gr.Dropdown(
                            choices=[
                                "default",
                                "fast",
                                "auto",
                                "vendor_multimodal",
                                "premium",
                            ],
                            value="default",
                            label="파싱 모드",
                            scale=2,
                        )
                        parse_btn = gr.Button(
                            "📝 Parse Document", variant="primary", scale=1.0
                        )

                    parse_status = gr.Markdown()

            # Step 간 구분선
            gr.Markdown(
                """
                <div style="width: 3px; height: 100%; background: linear-gradient(to bottom, #1f77b4, #ff7f0e); margin: 0; border-radius: 2px;"></div>
                """,
                elem_classes=["step-divider"],
            )

            with gr.Group():
                with gr.Column(min_width=600, scale=1.0):
                    gr.Markdown("### 🔪 Step 2: 텍스트 분할 & 임베딩")

                    with gr.Row():
                        splitter_type = gr.Dropdown(
                            choices=[
                                "RecursiveCharacterTextSplitter",
                                "CharacterTextSplitter",
                                "TokenTextSplitter",
                                "SemanticChunker",
                            ],
                            value="RecursiveCharacterTextSplitter",
                            label="Splitter 유형",
                            scale=2,
                        )
                        split_btn = gr.Button(
                            "🔗 Split & Embed", variant="primary", scale=1
                        )

                    with gr.Row():
                        chunk_size = gr.Slider(
                            100, 2000, 500, step=50, label="청크 크기", scale=1
                        )
                        chunk_overlap = gr.Slider(
                            0, 200, 0, step=10, label="오버랩", scale=1
                        )
                        retriever_k = gr.Slider(
                            1, 10, 3, step=1, label="검색 개수(k)", scale=1
                        )

                    split_status = gr.Markdown()

            # Step 간 구분선
            gr.Markdown(
                """
                <div style="width: 3px; height: 100%; background: linear-gradient(to bottom, #ff7f0e, #2ca02c); margin: 0; border-radius: 2px;"></div>
                """,
                elem_classes=["step-divider"],
            )

            with gr.Group():
                with gr.Column(min_width=600, scale=1.0):
                    gr.Markdown("### 💬 Step 3: 질의응답")

                    question_input = gr.Textbox(
                        label="질문 입력",
                        placeholder="문서에 대해 질문하세요...",
                        lines=2,
                    )

                    with gr.Row():
                        ask_btn = gr.Button("🔍 질문하기", variant="primary", scale=2)
                        clear_btn = gr.Button("🔄 초기화", scale=2)

                    answer_output = gr.Markdown(label="답변")

        # 문서 표시 영역
        gr.Markdown("### 📋 문서 분석 결과")

        with gr.Row():
            # 왼쪽: 파싱된 내용 (원본 문서 자리로 이동)
            with gr.Column(min_width=600, scale=1.0):
                gr.Markdown("#### 🔍 파싱된 내용")
                parsed_text_output = gr.Textbox(
                    label="파싱된 텍스트",
                    lines=20,
                    max_lines=30,
                    show_copy_button=True,
                    interactive=False,
                )

            # 오른쪽: 분할된 텍스트 청크 (파싱된 문서 자리로 이동)
            with gr.Column(min_width=600, scale=1.0):
                gr.Markdown("#### ✂️ 분할된 텍스트 청크")
                chunks_output = gr.Textbox(
                    label="텍스트 청크",
                    lines=20,
                    max_lines=30,
                    show_copy_button=True,
                    interactive=False,
                )

        # 검색된 문서 표시 (접을 수 있는 섹션)
        with gr.Accordion("🔎 검색된 문서 상세", open=False):
            retrieved_docs_output = gr.DataFrame(
                headers=["순번", "내용", "메타데이터"], label="검색된 문서"
            )

        # 숨겨진 설정들 (고급 사용자용)
        with gr.Accordion("⚙️ 고급 설정", open=False):
            custom_instruction = gr.Textbox(
                label="커스텀 파싱 지시사항",
                placeholder="예: 모든 표를 정확히 추출하고, 그래프의 데이터를 텍스트로 변환하세요.",
                lines=3,
            )

            # Splitter별 추가 설정
            separator = gr.Textbox(
                value="\n", label="구분자 (CharacterTextSplitter용)", visible=False
            )

            encoding_name = gr.Dropdown(
                choices=["cl100k_base", "p50k_base", "r50k_base"],
                value="cl100k_base",
                label="인코딩 (TokenTextSplitter용)",
                visible=False,
            )

            breakpoint_threshold_type = gr.Dropdown(
                choices=[
                    "percentile",
                    "standard_deviation",
                    "interquartile",
                    "gradient",
                ],
                value="percentile",
                label="임계값 유형 (SemanticChunker용)",
                visible=False,
            )

            breakpoint_threshold_amount = gr.Slider(
                minimum=50,
                maximum=99,
                value=95,
                step=2,
                label="임계값 (SemanticChunker용)",
                visible=False,
            )

            # Splitter 설명
            gr.Markdown(
                """
                **📌 파싱 모드별 특징:**
                
                | 모드 | 속도 | 정확도 | 적합한 문서 | 특징 |
                |------|------|--------|------------|------|
                | `default` | 중간 | 중간 | 일반 문서 | 균형잡힌 속도와 품질 |
                | `fast` | 빠름 | 낮음 | 간단한 텍스트 | 속도 우선, 단순 구조 |
                | `auto` | 가변 | 가변 | 모든 문서 | 문서 유형 자동 감지 |
                | `vendor_multimodal` | 느림 | 매우 높음 | 복잡한 표/차트 | AI 기반 고급 파싱 |
                | `premium` | 느림 | 최고 | 중요 문서 | 정확도 우선, 완벽한 구조 보존 |
                
                **Splitter 설명:**
                - `RecursiveCharacterTextSplitter`: 계층적 구분자로 자연스러운 분할 (추천)
                - `CharacterTextSplitter`: 단일 구분자 기반 단순 분할
                - `TokenTextSplitter`: 토큰 개수 기반 정확한 분할
                - `SemanticChunker`: AI 기반 의미적 유사성 분할 (실험적)
                """
            )

        # Splitter 유형 변경 시 관련 설정 표시/숨김
        def update_splitter_settings(splitter):
            return (
                gr.update(visible=splitter == "CharacterTextSplitter"),
                gr.update(visible=splitter == "TokenTextSplitter"),
                gr.update(visible=splitter == "SemanticChunker"),
                gr.update(visible=splitter == "SemanticChunker"),
            )

        splitter_type.change(
            update_splitter_settings,
            inputs=[splitter_type],
            outputs=[
                separator,
                encoding_name,
                breakpoint_threshold_type,
                breakpoint_threshold_amount,
            ],
        )

        # 메타데이터 표시 (접을 수 있는 섹션)
        with gr.Accordion("📊 파싱 메타데이터", open=False):
            metadata_output = gr.JSON(label="메타데이터")

        # 이벤트 핸들러
        async def handle_parse(f, m, i):
            status, metadata, file_path, parsed_text = await app.parse_document(f, m, i)
            return status, metadata, file_path, parsed_text

        def start_parsing(f, m, i):
            # 파싱 시작 전 버튼 비활성화 및 상태 표시
            return (
                gr.update(interactive=False, value="🔄 파싱 중..."),  # 버튼 비활성화
                "⏳ 문서를 파싱 중입니다. 잠시만 기다려주세요...",  # 상태 메시지
            )

        def complete_parsing(f, m, i):
            # 실제 파싱 수행 후 버튼 재활성화
            status, metadata, file_path, parsed_text = asyncio.run(
                handle_parse(f, m, i)
            )
            return (
                gr.update(interactive=True, value="📝 Parse Document"),  # 버튼 재활성화
                status,
                metadata,
                parsed_text,
            )

        # 파싱 버튼 클릭 시 2단계 처리
        parse_btn.click(
            start_parsing,
            inputs=[file_input, parsing_mode, custom_instruction],
            outputs=[parse_btn, parse_status],
        ).then(
            complete_parsing,
            inputs=[file_input, parsing_mode, custom_instruction],
            outputs=[parse_btn, parse_status, metadata_output, parsed_text_output],
        )

        split_btn.click(
            app.split_and_embed,
            inputs=[
                splitter_type,
                chunk_size,
                chunk_overlap,
                separator,
                encoding_name,
                breakpoint_threshold_type,
                breakpoint_threshold_amount,
                retriever_k,
            ],
            outputs=[split_status, chunks_output],
        )

        async def handle_question(q):
            answer, docs = await app.answer_question(q)
            return answer, docs

        ask_btn.click(
            lambda q: asyncio.run(handle_question(q)),
            inputs=[question_input],
            outputs=[answer_output, retrieved_docs_output],
        )

        def clear_all():
            status = app.clear_system()
            return status, "", "", []

        clear_btn.click(
            clear_all,
            outputs=[
                answer_output,
                parsed_text_output,
                chunks_output,
                retrieved_docs_output,
            ],
        )

        # 하단 정보
        gr.Markdown(
            """
        ---
        ### ℹ️ 사용 방법
        
        1. **문서 업로드**: PDF, DOCX, PPTX, TXT 파일을 업로드
        2. **파싱 모드 선택**: 문서 유형에 맞는 모드 선택
        3. **Parse Document**: 문서 파싱 실행
        4. **Split & Embed**: 텍스트 분할 방법과 검색 개수(k) 설정 후 벡터 DB 생성
        5. **질문하기**: 문서에 대한 질문 입력 (검색된 문서 확인 가능)
        
        ### ⚙️ 필요한 환경 변수
        - `LLAMA_CLOUD_API_KEY`: LlamaParse API 키
        - `OPENAI_API_KEY`: OpenAI API 키 (임베딩 및 GPT-4)
        
        ### 📌 파싱 모드별 특징
        
        | 모드 | 속도 | 정확도 | 적합한 문서 |
        |------|------|--------|------------|
        | Default | 중간 | 중간 | 일반 문서 |
        | Fast | 빠름 | 낮음 | 간단한 텍스트 |
        | Auto | 가변 | 가변 | 모든 문서 |
        | Vendor Multimodal | 느림 | 매우 높음 | 복잡한 표/차트 |
        | Premium | 느림 | 최고 | 중요 문서 |
        
        ### 🔪 Text Splitter별 특징
        
        | Splitter | 분할 방식 | 장점 | 적합한 용도 |
        |----------|----------|------|------------|
        | RecursiveCharacterTextSplitter | 계층적 구분자 | 문맥 유지, 자연스러운 분할 | 일반적인 문서 (추천) |
        | CharacterTextSplitter | 단일 구분자 | 단순하고 빠름 | 구조가 단순한 텍스트 |
        | TokenTextSplitter | 토큰 개수 기반 | 정확한 토큰 제어 | API 토큰 제한 고려 시 |
        | SemanticChunker | 의미적 유사성 | 의미 단위로 분할 | 논리적 단락 유지 필요 시 |
        
        ### 🎯 검색 개수 (k) 설정 가이드
        
        - **k=1~2**: 정확한 답변만 필요할 때
        - **k=3~5**: 균형잡힌 답변 (기본값: 3)
        - **k=6~10**: 포괄적인 정보 수집이 필요할 때
        
        ⚠️ k값이 클수록 더 많은 문서를 참조하지만, 응답 시간이 늘어날 수 있습니다.
        """
        )

    return demo


# 메인 실행
if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, debug=True)
