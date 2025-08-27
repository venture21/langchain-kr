import gradio as gr
import os
from dotenv import load_dotenv
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

# --- Configuration ---
# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()

# Semantic Chunker를 사용하려면 OpenAI API 키가 필요합니다.
# .env 파일에 OPENAI_API_KEY="YOUR_API_KEY" 형식으로 키를 추가해주세요.
if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"]:
    print("경고: OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다. SemanticChunker는 사용할 수 없습니다.")

# --- Data Loading ---
def load_file_content(filepath):
    """업로드된 파일의 내용을 읽어 반환합니다."""
    if filepath is None:
        return "파일을 업로드해주세요."
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"파일을 읽는 중 오류가 발생했습니다: {e}"

# --- Core Splitting Logic ---
def split_text(
    text_content,
    splitter_choice,
    char_chunk_size,
    char_chunk_overlap,
    rec_chunk_size,
    rec_chunk_overlap,
    tok_chunk_size,
    tok_chunk_overlap,
    sem_threshold_type,
    sem_threshold_amount,
):
    """선택된 분할기에 따라 텍스트를 분할합니다."""
    splitter = None
    if not text_content or text_content.strip() == "파일을 업로드해주세요.":
        return "오류: 분할할 텍스트가 없습니다. 파일을 먼저 업로드하세요.", []
    try:
        if splitter_choice == "CharacterTextSplitter":
            splitter = CharacterTextSplitter(
                separator="\n\n",
                chunk_size=char_chunk_size,
                chunk_overlap=char_chunk_overlap,
                is_separator_regex=False,
            )
            chunks = splitter.split_text(text_content)
        elif splitter_choice == "RecursiveCharacterTextSplitter":
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=rec_chunk_size,
                chunk_overlap=rec_chunk_overlap,
            )
            chunks = splitter.split_text(text_content)
        elif splitter_choice == "TokenTextSplitter":
            splitter = TokenTextSplitter(
                chunk_size=tok_chunk_size,
                chunk_overlap=tok_chunk_overlap,
            )
            chunks = splitter.split_text(text_content)
        elif splitter_choice == "SemanticChunker":
            if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"]:
                raise ValueError("OpenAI API Key가 설정되지 않았습니다. SemanticChunker를 사용할 수 없습니다.")
            embeddings = OpenAIEmbeddings()
            splitter = SemanticChunker(
                embeddings,
                breakpoint_threshold_type=sem_threshold_type,
                breakpoint_threshold_amount=sem_threshold_amount,
            )
            chunks = splitter.split_text(text_content)
        else:
            return "분할기를 선택해주세요.", []

        chunk_df = [[i + 1, chunk] for i, chunk in enumerate(chunks)]
        return f"{len(chunks)}개의 청크로 분할되었습니다.", chunk_df

    except Exception as e:
        return f"오류: {e}", []


# --- UI Helper ---
def update_visibility(choice):
    """선택된 분할기에 따라 파라미터 UI를 업데이트합니다."""
    is_char = choice == "CharacterTextSplitter"
    is_rec = choice == "RecursiveCharacterTextSplitter"
    is_tok = choice == "TokenTextSplitter"
    is_sem = choice == "SemanticChunker"
    return (
        gr.update(visible=is_char),
        gr.update(visible=is_rec),
        gr.update(visible=is_tok),
        gr.update(visible=is_sem),
    )

# --- Gradio App ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📚 LangChain Text Splitter 비교 도구")
    gr.Markdown(
        "다양한 Text Splitter가 텍스트를 어떻게 분할하는지 비교하고 테스트합니다. "
        "왼쪽에서 파일을 업로드하고, 오른쪽에서 분할기 종류와 설정을 조절한 후 '텍스트 분할' 버튼을 누르세요."
        "\n**참고:** `SemanticChunker`를 사용하려면 `.env` 파일에 `OPENAI_API_KEY` 설정이 필요합니다."
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. 원본 텍스트")
            file_uploader = gr.File(
                label="텍스트 파일 업로드 (.txt)", 
                file_types=[".txt"], 
                type="filepath"
            )
            file_content_display = gr.Textbox(
                label="파일 내용", lines=25, interactive=False, value="파일을 업로드해주세요."
            )

        with gr.Column(scale=1):
            gr.Markdown("### 2. 분할 설정 및 결과")
            splitter_choice = gr.Radio(
                [
                    "CharacterTextSplitter",
                    "RecursiveCharacterTextSplitter",
                    "TokenTextSplitter",
                    "SemanticChunker",
                ],
                label="Text Splitter 선택",
                value="RecursiveCharacterTextSplitter",
            )

            with gr.Group(visible=False) as char_params:
                char_chunk_size = gr.Slider(10, 2000, value=200, label="Chunk Size")
                char_chunk_overlap = gr.Slider(0, 500, value=50, label="Chunk Overlap")

            with gr.Group(visible=True) as recursive_params:
                rec_chunk_size = gr.Slider(10, 2000, value=200, label="Chunk Size")
                rec_chunk_overlap = gr.Slider(0, 500, value=50, label="Chunk Overlap")

            with gr.Group(visible=False) as token_params:
                tok_chunk_size = gr.Slider(10, 1000, value=100, label="Chunk Size (tokens)")
                tok_chunk_overlap = gr.Slider(0, 200, value=20, label="Chunk Overlap (tokens)")

            with gr.Group(visible=False) as semantic_params:
                sem_threshold_type = gr.Dropdown(
                    ["percentile", "standard_deviation", "interquartile"],
                    label="Breakpoint Threshold Type",
                    value="percentile",
                )
                sem_threshold_amount = gr.Slider(
                    0.0, 1.0, value=0.95, step=0.01, label="Breakpoint Threshold Amount"
                )

            split_button = gr.Button("텍스트 분할", variant="primary")
            
            chunk_count_output = gr.Textbox(label="결과", interactive=False)
            chunk_display = gr.DataFrame(
                headers=["Chunk #", "Content"],
                wrap=True,
                row_count=(10, "dynamic"),
                col_count=(2, "fixed"),
                label="분할된 청크"
            )

    # --- Event Handlers ---
    file_uploader.upload(
        fn=load_file_content,
        inputs=[file_uploader],
        outputs=[file_content_display],
    )

    splitter_choice.change(
        fn=update_visibility,
        inputs=[splitter_choice],
        outputs=[char_params, recursive_params, token_params, semantic_params],
    )

    split_button.click(
        fn=split_text,
        inputs=[
            file_content_display,
            splitter_choice,
            char_chunk_size,
            char_chunk_overlap,
            rec_chunk_size,
            rec_chunk_overlap,
            tok_chunk_size,
            tok_chunk_overlap,
            sem_threshold_type,
            sem_threshold_amount,
        ],
        outputs=[chunk_count_output, chunk_display],
    )

if __name__ == "__main__":
    demo.launch()
