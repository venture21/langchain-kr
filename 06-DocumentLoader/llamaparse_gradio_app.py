import os
import time
import nest_asyncio
from dotenv import load_dotenv
from llama_parse import LlamaParse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import gradio as gr
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import json
import io
import tempfile
import shutil

# 환경 설정
load_dotenv()
nest_asyncio.apply()

class LlamaParseGradioApp:
    """Gradio를 통한 LlamaParse 테스트 애플리케이션"""
    
    def __init__(self):
        self.api_key = os.environ.get("LLAMA_CLOUD_API_KEY")
        self.openai_key = os.environ.get("OPENAI_API_KEY")
        self.embeddings = None
        self.llm = None
        self.vectorstore = None
        self.qa_chain = None
        
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
    
    async def parse_with_mode(self, 
                              file_obj, 
                              mode: str,
                              custom_instruction: str = None,
                              vendor_model: str = "openai-gpt4o") -> Tuple[str, Dict]:
        """
        선택된 모드로 파일 파싱
        
        Returns:
            Tuple[str, Dict]: (파싱된 텍스트, 메타데이터)
        """
        if not file_obj:
            return "파일을 선택해주세요.", {}
        
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_obj.name)[1]) as tmp_file:
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
                "verbose": True
            }
            
            # 커스텀 지시사항 추가
            if custom_instruction:
                parser_config["parsing_instruction"] = custom_instruction
            
            # 모드별 설정
            if mode == "default":
                # 기본 설정 사용
                pass
                
            elif mode == "fast":
                parser_config["fast_mode"] = True
                parser_config["result_type"] = "text"  # 빠른 처리를 위해 text 형식
                
            elif mode == "vendor_multimodal":
                if not self.openai_key and "openai" in vendor_model:
                    return "OpenAI API 키가 필요합니다.", metadata
                
                parser_config.update({
                    "use_vendor_multimodal_model": True,
                    "vendor_multimodal_model_name": vendor_model,
                    "vendor_multimodal_api_key": self.openai_key,
                })
                
                if not custom_instruction:
                    parser_config["parsing_instruction"] = """
                    문서를 정밀하게 분석하세요:
                    1. 모든 표를 마크다운 형식으로 변환
                    2. 이미지와 차트에 대한 상세한 설명
                    3. 수식과 다이어그램 정확히 추출
                    """
                    
            elif mode == "premium":
                parser_config["premium_mode"] = True
                parser_config["page_separator"] = True
                
                if not custom_instruction:
                    parser_config["parsing_instruction"] = """
                    최고 품질로 문서를 파싱하세요:
                    1. 모든 구조 요소 보존
                    2. 레이아웃과 서식 유지
                    3. 메타데이터와 주석 포함
                    """
                    
            elif mode == "auto":
                parser_config["auto_mode"] = True
                
            elif mode == "continuous":
                parser_config["continuous_mode"] = True
                parser_config["batch_size"] = 5
                
            elif mode == "spreadsheet":
                # 엑셀/CSV 파일 체크
                if not file_obj.name.endswith(('.xlsx', '.xls', '.csv')):
                    return "Spreadsheet 모드는 엑셀 또는 CSV 파일만 지원합니다.", metadata
                
                parser_config["spreadsheet_mode"] = True
                parser_config["result_type"] = "csv"
                parser_config["preserve_formatting"] = True
                
            elif mode == "audio":
                # 오디오 파일 체크
                if not file_obj.name.endswith(('.mp3', '.wav', '.m4a', '.aac')):
                    return "Audio 모드는 오디오 파일만 지원합니다.", metadata
                
                parser_config["audio_mode"] = True
                parser_config["include_timestamps"] = True
                parser_config["result_type"] = "text"
            
            # 파서 생성 및 실행
            parser = LlamaParse(**parser_config)
            documents = await parser.aload_data(file_path)
            
            # 결과 처리
            if mode == "spreadsheet" and documents:
                # CSV 데이터를 DataFrame으로 변환하여 표시
                csv_content = documents[0].text
                df = pd.read_csv(io.StringIO(csv_content))
                result_text = f"### 📊 스프레드시트 데이터\n\n"
                result_text += f"**행 수:** {len(df)}\n"
                result_text += f"**열 수:** {len(df.columns)}\n\n"
                result_text += "**데이터 미리보기:**\n\n"
                result_text += df.head(10).to_markdown(index=False)
                
            elif mode == "audio" and documents:
                # 오디오 전사 결과 포맷팅
                result_text = f"### 🎵 오디오 전사 결과\n\n"
                result_text += documents[0].text
                
                # 타임스탬프 파싱 (있는 경우)
                if "[" in documents[0].text and "]" in documents[0].text:
                    result_text += "\n\n### 타임스탬프별 내용\n"
                    import re
                    pattern = r'\[(\d{2}:\d{2}:\d{2})\]\s*(.*?)(?=\[\d{2}:\d{2}:\d{2}\]|$)'
                    matches = re.findall(pattern, documents[0].text, re.DOTALL)
                    for timestamp, content in matches[:10]:  # 처음 10개만 표시
                        result_text += f"\n**[{timestamp}]** {content.strip()[:100]}..."
                        
            else:
                # 일반 문서 결과
                result_text = f"### 📄 파싱 결과 ({mode} 모드)\n\n"
                for i, doc in enumerate(documents[:3], 1):  # 처음 3페이지만 표시
                    if hasattr(doc, 'to_langchain_format'):
                        doc_content = doc.to_langchain_format().page_content
                    else:
                        doc_content = doc.text
                    
                    result_text += f"#### 페이지 {i}\n\n"
                    result_text += doc_content[:2000]  # 각 페이지당 2000자 제한
                    if len(doc_content) > 2000:
                        result_text += "\n\n... (내용이 너무 길어 생략되었습니다)"
                    result_text += "\n\n---\n\n"
            
            # 메타데이터 업데이트
            elapsed_time = time.time() - start_time
            metadata.update({
                "processing_time": f"{elapsed_time:.2f}초",
                "pages": len(documents),
                "total_chars": sum(len(doc.text if hasattr(doc, 'text') else str(doc)) for doc in documents),
                "success": True
            })
            
            # 임시 파일 삭제
            os.unlink(file_path)
            
            return result_text, metadata
            
        except Exception as e:
            # 임시 파일 삭제
            if os.path.exists(file_path):
                os.unlink(file_path)
            
            metadata["error"] = str(e)
            metadata["success"] = False
            return f"❌ 오류 발생: {str(e)}", metadata
    
    async def create_rag_system(self, file_obj, mode: str) -> str:
        """파싱된 문서로 RAG 시스템 생성"""
        if not self.openai_key:
            return "⚠️ RAG 시스템을 사용하려면 OpenAI API 키가 필요합니다."
        
        self.initialize_models()
        
        # 문서 파싱
        result_text, metadata = await self.parse_with_mode(file_obj, mode)
        
        if not metadata.get("success"):
            return result_text
        
        try:
            # 임시 파일로 저장
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_obj.name)[1]) as tmp_file:
                shutil.copy(file_obj.name, tmp_file.name)
                file_path = tmp_file.name
            
            # 파서 설정
            parser_config = {
                "api_key": self.api_key,
                "result_type": "markdown",
                "language": "ko"
            }
            
            if mode == "vendor_multimodal":
                parser_config.update({
                    "use_vendor_multimodal_model": True,
                    "vendor_multimodal_model_name": "openai-gpt4o",
                    "vendor_multimodal_api_key": self.openai_key,
                })
            elif mode == "premium":
                parser_config["premium_mode"] = True
            
            parser = LlamaParse(**parser_config)
            documents = await parser.aload_data(file_path)
            
            # LangChain 형식으로 변환
            docs = []
            for doc in documents:
                if hasattr(doc, 'to_langchain_format'):
                    docs.append(doc.to_langchain_format())
            
            # 텍스트 분할
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            splits = text_splitter.split_documents(docs)
            
            # 벡터 스토어 생성
            self.vectorstore = FAISS.from_documents(splits, self.embeddings)
            
            # RAG 체인 생성
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3})
            )
            
            # 임시 파일 삭제
            os.unlink(file_path)
            
            return f"✅ RAG 시스템 생성 완료!\n- 문서: {len(docs)}개\n- 청크: {len(splits)}개\n\n이제 질문을 입력해주세요."
            
        except Exception as e:
            return f"❌ RAG 시스템 생성 실패: {str(e)}"
    
    async def answer_question(self, question: str) -> str:
        """RAG 시스템을 통한 질문 응답"""
        if not self.qa_chain:
            return "먼저 RAG 시스템을 생성해주세요."
        
        try:
            answer = self.qa_chain.run(question)
            return f"### 💬 답변\n\n{answer}"
        except Exception as e:
            return f"❌ 답변 생성 실패: {str(e)}"
    
    async def compare_modes(self, file_obj, modes: List[str]) -> Tuple[str, pd.DataFrame]:
        """여러 모드 성능 비교"""
        if not file_obj:
            return "파일을 선택해주세요.", pd.DataFrame()
        
        if not modes:
            return "비교할 모드를 선택해주세요.", pd.DataFrame()
        
        results = []
        comparison_text = "### 📊 모드별 비교 결과\n\n"
        
        for mode in modes:
            comparison_text += f"#### {mode} 모드 처리 중...\n"
            
            result_text, metadata = await self.parse_with_mode(file_obj, mode)
            
            if metadata.get("success"):
                results.append({
                    "모드": mode,
                    "처리 시간": metadata.get("processing_time", "N/A"),
                    "페이지 수": metadata.get("pages", 0),
                    "총 문자 수": f"{metadata.get('total_chars', 0):,}",
                    "상태": "✅ 성공"
                })
                comparison_text += f"✅ 완료: {metadata.get('processing_time', 'N/A')}\n\n"
            else:
                results.append({
                    "모드": mode,
                    "처리 시간": "N/A",
                    "페이지 수": 0,
                    "총 문자 수": 0,
                    "상태": f"❌ 실패: {metadata.get('error', 'Unknown')[:30]}"
                })
                comparison_text += f"❌ 실패: {metadata.get('error', 'Unknown')[:50]}\n\n"
        
        # DataFrame 생성
        df = pd.DataFrame(results)
        
        # 분석 결과 추가
        comparison_text += "\n### 📈 분석 결과\n\n"
        
        successful = [r for r in results if "✅" in r["상태"]]
        if successful:
            # 가장 빠른 모드
            fastest = min(successful, key=lambda x: float(x["처리 시간"].replace("초", "")))
            comparison_text += f"**⚡ 가장 빠른 모드:** {fastest['모드']} ({fastest['처리 시간']})\n\n"
            
            # 가장 많은 텍스트 추출
            most_text = max(successful, key=lambda x: int(x["총 문자 수"].replace(",", "")))
            comparison_text += f"**📝 가장 많은 텍스트 추출:** {most_text['모드']} ({most_text['총 문자 수']}자)\n\n"
        
        return comparison_text, df


def create_gradio_interface():
    """Gradio 인터페이스 생성"""
    app = LlamaParseGradioApp()
    
    with gr.Blocks(title="LlamaParse Multi-Mode Tester", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🦙 LlamaParse Multi-Mode Tester
        
        다양한 파싱 모드를 테스트하고 비교할 수 있는 도구입니다.
        
        ### 📋 지원 모드
        - **Default**: 표준 파싱 모드
        - **Fast**: 빠른 처리 (정확도 낮음)
        - **Vendor Multimodal**: AI 모델 활용 (높은 정확도)
        - **Premium**: 최고 품질 파싱
        - **Auto**: 자동 최적화
        - **Continuous**: 연속 문서 처리
        - **Spreadsheet**: 엑셀/CSV 특화
        - **Audio**: 오디오 전사
        """)
        
        with gr.Tabs():
            # 탭 1: 단일 모드 테스트
            with gr.Tab("🔧 단일 모드 테스트"):
                with gr.Row():
                    with gr.Column(scale=1):
                        single_file = gr.File(
                            label="파일 선택",
                            file_types=[".pdf", ".docx", ".pptx", ".xlsx", ".xls", ".csv", ".txt", ".mp3", ".wav", ".m4a"]
                        )
                        single_mode = gr.Dropdown(
                            choices=["default", "fast", "vendor_multimodal", "premium", "auto", "continuous", "spreadsheet", "audio"],
                            value="default",
                            label="파싱 모드"
                        )
                        vendor_model = gr.Dropdown(
                            choices=["openai-gpt4o", "openai-gpt-4-vision-preview", "anthropic-claude-3-opus"],
                            value="openai-gpt4o",
                            label="Vendor Model (Multimodal 모드용)",
                            visible=True
                        )
                        custom_instruction = gr.Textbox(
                            label="커스텀 파싱 지시사항 (선택사항)",
                            placeholder="예: 모든 표를 마크다운 형식으로 변환하고, 이미지 설명을 포함하세요.",
                            lines=3
                        )
                        parse_btn = gr.Button("🚀 파싱 시작", variant="primary")
                    
                    with gr.Column(scale=2):
                        single_output = gr.Markdown(label="파싱 결과")
                        single_metadata = gr.JSON(label="메타데이터")
                
                # 이벤트 핸들러
                def update_vendor_visibility(mode):
                    return gr.update(visible=(mode == "vendor_multimodal"))
                
                single_mode.change(
                    update_vendor_visibility,
                    inputs=[single_mode],
                    outputs=[vendor_model]
                )
                
                parse_btn.click(
                    lambda f, m, i, v: asyncio.run(app.parse_with_mode(f, m, i, v)),
                    inputs=[single_file, single_mode, custom_instruction, vendor_model],
                    outputs=[single_output, single_metadata]
                )
            
            # 탭 2: 모드 비교
            with gr.Tab("📊 모드 비교"):
                with gr.Row():
                    with gr.Column():
                        compare_file = gr.File(
                            label="파일 선택",
                            file_types=[".pdf", ".docx", ".pptx", ".txt"]
                        )
                        compare_modes = gr.CheckboxGroup(
                            choices=["default", "fast", "vendor_multimodal", "premium", "auto"],
                            value=["default", "fast"],
                            label="비교할 모드 선택"
                        )
                        compare_btn = gr.Button("📈 비교 시작", variant="primary")
                    
                    with gr.Column():
                        compare_output = gr.Markdown(label="비교 결과")
                        compare_table = gr.DataFrame(label="성능 비교표")
                
                compare_btn.click(
                    lambda f, m: asyncio.run(app.compare_modes(f, m)),
                    inputs=[compare_file, compare_modes],
                    outputs=[compare_output, compare_table]
                )
            
            # 탭 3: RAG 시스템
            with gr.Tab("🤖 RAG 시스템"):
                gr.Markdown("""
                ### RAG (Retrieval-Augmented Generation) 시스템
                문서를 파싱하고 벡터 데이터베이스를 생성한 후 질문에 답변합니다.
                """)
                
                with gr.Row():
                    with gr.Column():
                        rag_file = gr.File(
                            label="문서 선택",
                            file_types=[".pdf", ".docx", ".txt"]
                        )
                        rag_mode = gr.Dropdown(
                            choices=["default", "vendor_multimodal", "premium"],
                            value="default",
                            label="파싱 모드"
                        )
                        create_rag_btn = gr.Button("🔨 RAG 시스템 생성", variant="primary")
                        rag_status = gr.Markdown()
                
                with gr.Row():
                    with gr.Column():
                        question_input = gr.Textbox(
                            label="질문 입력",
                            placeholder="문서에 대해 질문하세요...",
                            lines=2
                        )
                        ask_btn = gr.Button("💬 질문하기")
                    
                    with gr.Column():
                        answer_output = gr.Markdown(label="답변")
                
                create_rag_btn.click(
                    lambda f, m: asyncio.run(app.create_rag_system(f, m)),
                    inputs=[rag_file, rag_mode],
                    outputs=[rag_status]
                )
                
                ask_btn.click(
                    lambda q: asyncio.run(app.answer_question(q)),
                    inputs=[question_input],
                    outputs=[answer_output]
                )
            
            # 탭 4: 배치 처리
            with gr.Tab("📁 배치 처리"):
                gr.Markdown("""
                ### 여러 파일 일괄 처리
                여러 파일을 동시에 업로드하고 처리합니다.
                """)
                
                with gr.Row():
                    with gr.Column():
                        batch_files = gr.File(
                            label="파일들 선택 (여러 개 가능)",
                            file_count="multiple",
                            file_types=[".pdf", ".docx", ".txt"]
                        )
                        batch_mode = gr.Dropdown(
                            choices=["default", "fast", "auto"],
                            value="auto",
                            label="파싱 모드"
                        )
                        batch_btn = gr.Button("🚀 배치 처리 시작", variant="primary")
                    
                    with gr.Column():
                        batch_output = gr.Markdown(label="처리 결과")
                        batch_summary = gr.DataFrame(label="처리 요약")
                
                async def process_batch(files, mode):
                    if not files:
                        return "파일을 선택해주세요.", pd.DataFrame()
                    
                    results = []
                    output_text = f"### 📁 배치 처리 결과\n\n"
                    output_text += f"**총 {len(files)}개 파일 처리**\n\n"
                    
                    for file in files:
                        output_text += f"처리 중: {os.path.basename(file.name)}...\n"
                        result_text, metadata = await app.parse_with_mode(file, mode)
                        
                        results.append({
                            "파일명": os.path.basename(file.name),
                            "크기(KB)": f"{metadata.get('file_size', 0):.1f}",
                            "처리시간": metadata.get('processing_time', 'N/A'),
                            "페이지": metadata.get('pages', 0),
                            "상태": "✅" if metadata.get('success') else "❌"
                        })
                    
                    df = pd.DataFrame(results)
                    
                    # 요약 통계
                    success_count = sum(1 for r in results if r["상태"] == "✅")
                    output_text += f"\n### 📊 요약\n"
                    output_text += f"- 성공: {success_count}/{len(files)}\n"
                    output_text += f"- 실패: {len(files) - success_count}/{len(files)}\n"
                    
                    return output_text, df
                
                batch_btn.click(
                    lambda f, m: asyncio.run(process_batch(f, m)),
                    inputs=[batch_files, batch_mode],
                    outputs=[batch_output, batch_summary]
                )
        
        # 하단 정보
        gr.Markdown("""
        ---
        ### ℹ️ 참고사항
        - **API 키 필요**: 환경 변수에 `LLAMA_CLOUD_API_KEY` 설정 필요
        - **Vendor Multimodal**: OpenAI API 키 추가 필요 (`OPENAI_API_KEY`)
        - **파일 크기 제한**: 대용량 파일은 처리 시간이 오래 걸릴 수 있음
        - **오디오 파일**: mp3, wav, m4a 형식 지원
        """)
    
    return demo


# 메인 실행
if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )