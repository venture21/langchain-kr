import gradio as gr
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    SpacyTextSplitter,
    SentenceTransformersTokenTextSplitter,
    NLTKTextSplitter,
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
import os
import tiktoken
 

def get_text_splitter(splitter_type, chunk_size, chunk_overlap):
    """
    선택된 타입에 따라 적절한 텍스트 스플리터를 반환하는 함수
    """
    if splitter_type == "CharacterTextSplitter":
        return CharacterTextSplitter(
            separator="\n\n",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    elif splitter_type == "RecursiveCharacterTextSplitter":
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

    elif splitter_type == "CharacterTextSplitter (tiktoken)":
        return CharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    elif splitter_type == "TokenTextSplitter":
        return TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    elif splitter_type == "SpacyTextSplitter":
        return SpacyTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            pipeline="ko_core_news_sm",  # 한국어 모델 사용
        )

    elif splitter_type == "SentenceTransformersTokenTextSplitter":
        return SentenceTransformersTokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )

    elif splitter_type == "NLTKTextSplitter":
        import nltk

        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")

        return NLTKTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    elif splitter_type == "SemanticChunker":
        # SemanticChunker는 chunk_size와 chunk_overlap을 사용하지 않음
        # OpenAI API 키가 필요함
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "SemanticChunker를 사용하려면 OPENAI_API_KEY 환경변수가 필요합니다."
            )

        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        return SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95,
        )

    else:
        raise ValueError(f"알 수 없는 스플리터 타입: {splitter_type}")


def split_text(file_path, splitter_type, chunk_size, chunk_overlap):
    """
    텍스트 파일을 읽어서 선택된 TextSplitter로 분할하는 함수
    """
    if not file_path:
        return "파일을 선택해주세요.", "파일을 선택해주세요.", "통계 정보가 없습니다."

    try:
        # 파일 읽기
        with open(file_path, encoding="utf-8") as f:
            original_text = f.read()

        # 선택된 텍스트 스플리터 가져오기
        try:
            text_splitter = get_text_splitter(splitter_type, chunk_size, chunk_overlap)
        except Exception as e:
            return original_text, f"스플리터 생성 오류: {str(e)}", f"오류: {str(e)}"

        # 텍스트를 청크로 분할
        chunks = text_splitter.create_documents([original_text])

        # 모든 청크를 문자열로 포맷팅
        formatted_chunks = ""
        for i, chunk in enumerate(chunks):
            formatted_chunks += (
                f"=== Chunk {i+1} (길이: {len(chunk.page_content)}자) ===\n"
            )
            formatted_chunks += chunk.page_content
            formatted_chunks += "\n\n" + "=" * 50 + "\n\n"

        # 청크 통계 정보
        stats = f"사용된 스플리터: {splitter_type}\n"
        stats += f"총 청크 개수: {len(chunks)}\n"
        stats += f"표시된 청크: {len(chunks)}개 (전체)\n"

        # SemanticChunker는 chunk_size와 overlap을 사용하지 않음
        if splitter_type != "SemanticChunker":
            stats += f"설정된 chunk_size: {chunk_size}\n"
            stats += f"설정된 chunk_overlap: {chunk_overlap}"
        else:
            stats += "SemanticChunker는 의미 기반으로 자동 분할"

        return original_text, formatted_chunks, stats

    except Exception as e:
        return f"오류 발생: {str(e)}", f"오류 발생: {str(e)}", f"오류 발생: {str(e)}"


def load_sample_file():
    """샘플 파일 경로를 반환하는 함수"""
    sample_path = "./data/appendix-keywords.txt"
    if os.path.exists(sample_path):
        return sample_path
    return None


# Gradio 인터페이스 생성
with gr.Blocks(title="Text Splitter Demo") as demo:
    gr.Markdown("# 다양한 TextSplitter 데모")
    gr.Markdown(
        "텍스트 파일을 업로드하거나 샘플 파일을 사용하여 다양한 TextSplitter의 작동 방식을 확인해보세요."
    )

    with gr.Row():
        with gr.Column(scale=1):
            # 파일 업로드 컴포넌트
            file_input = gr.File(
                label="텍스트 파일 선택", file_types=[".txt"], type="filepath"
            )

            # 샘플 파일 로드 버튼
            sample_btn = gr.Button(
                "샘플 파일 사용 (appendix-keywords.txt)", variant="secondary"
            )

            gr.Markdown("### 분할 설정")

            # 스플리터 타입 선택 드롭다운
            splitter_dropdown = gr.Dropdown(
                choices=[
                    "CharacterTextSplitter",
                    "RecursiveCharacterTextSplitter",
                    "CharacterTextSplitter (tiktoken)",
                    "TokenTextSplitter",
                    "SpacyTextSplitter",
                    "SentenceTransformersTokenTextSplitter",
                    "NLTKTextSplitter",
                    "SemanticChunker",
                ],
                value="CharacterTextSplitter",
                label="Text Splitter 선택",
                info="사용할 텍스트 분할 방법을 선택하세요",
            )

            # chunk_size 슬라이더
            chunk_size_slider = gr.Slider(
                minimum=50,
                maximum=1000,
                value=210,
                step=10,
                label="Chunk Size",
                info="각 청크의 최대 크기 (SemanticChunker 제외)",
            )

            # chunk_overlap 슬라이더
            chunk_overlap_slider = gr.Slider(
                minimum=0,
                maximum=200,
                value=0,
                step=10,
                label="Chunk Overlap",
                info="청크 간 중복되는 크기 (SemanticChunker 제외)",
            )

            # 분할 실행 버튼
            split_btn = gr.Button("텍스트 분할 실행", variant="primary")

            # 주의사항
            gr.Markdown(
                """
            ### 참고사항
            - **SpacyTextSplitter**: 한국어 모델(ko_core_news_sm) 설치 필요
            - **SemanticChunker**: OPENAI_API_KEY 환경변수 필요
            - **NLTKTextSplitter**: punkt 토크나이저 자동 다운로드
            """
            )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 원본 텍스트")
            original_output = gr.Textbox(
                label="원본 내용", lines=20, max_lines=30, interactive=False
            )

        with gr.Column():
            gr.Markdown("### 분할된 청크 (전체)")
            chunks_output = gr.Textbox(
                label="청크 결과", lines=15, max_lines=25, interactive=False
            )

            gr.Markdown("### 청크 결과 통계")
            stats_output = gr.Textbox(
                label="통계 정보", lines=5, max_lines=5, interactive=False
            )

    # 이벤트 핸들러 연결
    def handle_file_and_split(file_path, splitter_type, chunk_size, chunk_overlap):
        if file_path:
            return split_text(file_path, splitter_type, chunk_size, chunk_overlap)
        return "파일을 선택해주세요.", "파일을 선택해주세요.", "통계 정보가 없습니다."

    # SemanticChunker 선택 시 슬라이더 비활성화
    def update_sliders(splitter_type):
        is_semantic = splitter_type == "SemanticChunker"
        return (
            gr.update(interactive=not is_semantic),
            gr.update(interactive=not is_semantic),
        )

    splitter_dropdown.change(
        fn=update_sliders,
        inputs=splitter_dropdown,
        outputs=[chunk_size_slider, chunk_overlap_slider],
    )

    # 샘플 파일 로드
    def load_sample():
        sample_path = load_sample_file()
        if sample_path:
            return sample_path
        return None

    sample_btn.click(fn=load_sample, outputs=file_input)

    # 파일이 업로드되면 자동으로 분할 실행
    file_input.change(
        fn=handle_file_and_split,
        inputs=[file_input, splitter_dropdown, chunk_size_slider, chunk_overlap_slider],
        outputs=[original_output, chunks_output, stats_output],
    )

    # 분할 버튼 클릭 시 실행
    split_btn.click(
        fn=handle_file_and_split,
        inputs=[file_input, splitter_dropdown, chunk_size_slider, chunk_overlap_slider],
        outputs=[original_output, chunks_output, stats_output],
    )

    # 스플리터 변경 시 자동으로 재분할
    splitter_dropdown.change(
        fn=handle_file_and_split,
        inputs=[file_input, splitter_dropdown, chunk_size_slider, chunk_overlap_slider],
        outputs=[original_output, chunks_output, stats_output],
    )

    # 슬라이더 값 변경 시 자동으로 재분할
    chunk_size_slider.change(
        fn=handle_file_and_split,
        inputs=[file_input, splitter_dropdown, chunk_size_slider, chunk_overlap_slider],
        outputs=[original_output, chunks_output, stats_output],
    )

    chunk_overlap_slider.change(
        fn=handle_file_and_split,
        inputs=[file_input, splitter_dropdown, chunk_size_slider, chunk_overlap_slider],
        outputs=[original_output, chunks_output, stats_output],
    )

if __name__ == "__main__":
    demo.launch(share=False)
