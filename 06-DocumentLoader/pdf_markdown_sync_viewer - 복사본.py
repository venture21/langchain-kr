import os
import gradio as gr
import nest_asyncio
from dotenv import load_dotenv
from llama_parse import LlamaParse
import fitz  # PyMuPDF for PDF display
import base64
import json
from pathlib import Path
import pickle
import re

# 환경 변수 로드
load_dotenv()

# jupyter 환경에서 asyncio를 사용할 수 있도록 설정
nest_asyncio.apply()


class PDFMarkdownViewer:
    def __init__(self):
        self.current_pdf_path = None
        self.pdf_pages = []
        self.markdown_pages = []
        self.current_page = 0
        self.total_pages = 0
        
    def pdf_to_page_images(self, pdf_path):
        """PDF 파일을 페이지별 이미지로 변환"""
        try:
            pdf_document = fitz.open(pdf_path)
            pages = []
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                # 페이지를 이미지로 변환
                mat = fitz.Matrix(2, 2)  # 2x zoom for better quality
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                
                # base64로 인코딩
                img_base64 = base64.b64encode(img_data).decode()
                pages.append(img_base64)
            
            pdf_document.close()
            return pages
        
        except Exception as e:
            print(f"PDF 이미지 변환 오류: {str(e)}")
            return []
    
    def parse_pdf_to_markdown_pages(self, pdf_file_path):
        """PDF를 페이지별 Markdown으로 변환"""
        try:
            # parsing instruction 설정
            parsing_instruction = (
                "You are parsing a document. Please extract all content including tables in markdown format. "
                "Preserve the structure and formatting as much as possible. "
                "Mark page breaks clearly with '---PAGE_BREAK---' marker."
            )
            
            # LlamaParse 설정
            parser = LlamaParse(
                use_vendor_multimodal_model=True,
                vendor_multimodal_model_name="openai-gpt4o",
                vendor_multimodal_api_key=os.environ.get("OPENAI_API_KEY"),
                result_type="markdown",
                language="ko",
                parsing_instruction=parsing_instruction,
            )
            
            # PDF 파일 파싱
            parsed_docs = parser.load_data(file_path=pdf_file_path)
            
            # 페이지별로 분리
            markdown_pages = []
            for doc in parsed_docs:
                # LangChain 형식으로 변환
                content = doc.to_langchain_format().page_content
                
                # 페이지 구분자로 분리 (LlamaParse가 페이지를 구분하는 경우)
                # 또는 문서 구조에 따라 적절히 분리
                pages = content.split('---PAGE_BREAK---')
                if len(pages) == 1:
                    # 페이지 구분자가 없으면 전체를 하나의 페이지로
                    # 또는 문단/섹션 기준으로 분리 가능
                    markdown_pages.append(content)
                else:
                    markdown_pages.extend(pages)
            
            return markdown_pages
        
        except Exception as e:
            print(f"Markdown 변환 오류: {str(e)}")
            return []
    
    def save_conversion(self, pdf_path, pdf_pages, markdown_pages):
        """변환 결과를 MD 파일로 저장"""
        try:
            # 저장 디렉토리 생성
            save_dir = Path("conversions")
            save_dir.mkdir(exist_ok=True)
            
            # 파일명 기반으로 저장 (중복 방지)
            base_name = Path(pdf_path).stem
            md_path = save_dir / f"{base_name}.md"
            
            # 중복 파일명 처리
            counter = 1
            while md_path.exists():
                md_path = save_dir / f"{base_name}_{counter}.md"
                counter += 1
            
            # Markdown 파일로 저장 (PDF 경로 정보 포함)
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(f"# {base_name}\n\n")
                f.write(f"원본 PDF 경로: {pdf_path}\n")
                f.write(f"변환 날짜: {Path(pdf_path).stat().st_mtime}\n")
                f.write(f"총 페이지: {len(markdown_pages)}\n\n")
                f.write("<!-- PDF_PATH_MARKER -->\n\n")
                
                for i, page in enumerate(markdown_pages):
                    f.write(f"## 페이지 {i+1}\n\n")
                    f.write(page)
                    f.write(f"\n\n{'='*50}\n\n")
            
            return str(md_path)
        
        except Exception as e:
            print(f"저장 오류: {str(e)}")
            return None
    
    def get_available_conversions(self):
        """저장된 MD 파일 목록 가져오기"""
        save_dir = Path("conversions")
        if not save_dir.exists():
            return []
        
        files = list(save_dir.glob("*.md"))
        return [str(f) for f in files]
    
    def get_all_pdf_html(self):
        """모든 PDF 페이지를 연속된 HTML로 반환"""
        if not self.pdf_pages:
            return "<p style='text-align: center; color: #666;'>PDF가 로드되지 않았습니다.</p>"
        
        html_parts = []
        for i, page_base64 in enumerate(self.pdf_pages):
            html_parts.append(f'''
                <div class="pdf-page" data-page="{i}">
                    <div class="page-number">페이지 {i+1}</div>
                    <img src="data:image/png;base64,{page_base64}" style="width: 100%; margin-bottom: 20px;">
                </div>
            ''')
        return ''.join(html_parts)
    
    def get_all_markdown_content(self):
        """모든 Markdown 페이지를 연속된 내용으로 반환"""
        if not self.markdown_pages:
            return "Markdown 내용이 없습니다."
        
        content_parts = []
        for i, page_content in enumerate(self.markdown_pages):
            content_parts.append(f"## 페이지 {i+1}\n\n{page_content}\n\n---\n\n")
        
        return ''.join(content_parts)


# 전역 뷰어 인스턴스
viewer = PDFMarkdownViewer()


def process_pdf(pdf_file):
    """PDF 파일 처리 및 변환"""
    if pdf_file is None:
        return (
            "PDF 파일을 업로드해주세요.",
            "",
            "파일을 업로드해주세요."
        )
    
    try:
        # PDF를 페이지별 이미지로 변환
        viewer.pdf_pages = viewer.pdf_to_page_images(pdf_file.name)
        
        # Markdown으로 변환
        viewer.markdown_pages = viewer.parse_pdf_to_markdown_pages(pdf_file.name)
        
        # 페이지 수가 다른 경우 조정
        if len(viewer.markdown_pages) < len(viewer.pdf_pages):
            # Markdown 페이지가 적으면 빈 페이지 추가
            while len(viewer.markdown_pages) < len(viewer.pdf_pages):
                viewer.markdown_pages.append("*(이 페이지의 Markdown 변환 내용이 없습니다)*")
        elif len(viewer.markdown_pages) > len(viewer.pdf_pages):
            # Markdown이 더 많으면 합치기
            combined = "\n\n".join(viewer.markdown_pages[len(viewer.pdf_pages):])
            viewer.markdown_pages = viewer.markdown_pages[:len(viewer.pdf_pages)-1]
            viewer.markdown_pages[-1] += "\n\n" + combined
        
        viewer.total_pages = len(viewer.pdf_pages)
        viewer.current_page = 0
        viewer.current_pdf_path = pdf_file.name
        
        # 변환 결과 저장
        save_path = viewer.save_conversion(
            pdf_file.name, 
            viewer.pdf_pages, 
            viewer.markdown_pages
        )
        
        # 모든 페이지를 스크롤 방식으로 표시
        if viewer.total_pages > 0:
            pdf_html = viewer.get_all_pdf_html()
            markdown_content = viewer.get_all_markdown_content()
            
            return (
                pdf_html,
                markdown_content,
                f"✅ 변환 완료! 총 {viewer.total_pages}페이지 저장됨 (위치: {save_path})"
            )
        else:
            return (
                "PDF 페이지를 불러올 수 없습니다.",
                "",
                "❌ 변환 실패"
            )
    
    except Exception as e:
        return (
            f"<p style='color: red;'>오류: {str(e)}</p>",
            f"오류: {str(e)}",
            f"❌ 오류 발생: {str(e)}"
        )


def refresh_display():
    """전체 내용을 새로고침"""
    if not viewer.pdf_pages or not viewer.markdown_pages:
        return "PDF를 먼저 업로드하세요.", "내용이 없습니다."
    
    pdf_html = viewer.get_all_pdf_html()
    markdown_content = viewer.get_all_markdown_content()
    
    return pdf_html, markdown_content


def load_saved_conversion(file_path):
    """저장된 MD 파일과 원본 PDF 불러오기"""
    if not file_path:
        return (
            "파일을 선택하세요.",
            "",
            "파일을 선택하세요."
        )
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # MD 파일에서 원본 PDF 경로 추출
        pdf_path = None
        pdf_path_pattern = r'원본 PDF 경로: (.+)'
        pdf_match = re.search(pdf_path_pattern, content)
        
        if pdf_match:
            pdf_path = pdf_match.group(1).strip()
            print(f"[DEBUG] PDF 경로 찾음: {pdf_path}")
        else:
            print(f"[DEBUG] PDF 경로를 찾을 수 없음 (레거시 파일). 자동 검색을 시도합니다.")
            # 레거시 파일의 경우 같은 이름의 PDF를 자동으로 찾기
            md_filename = Path(file_path).stem
            
            # 검색할 경로들
            search_paths = [
                Path(file_path).parent,  # 같은 폴더 (conversions)
                Path(file_path).parent.parent / "data",  # data 폴더
                Path(file_path).parent.parent,  # 상위 폴더
            ]
            
            for search_dir in search_paths:
                if search_dir.exists():
                    potential_pdf = search_dir / f"{md_filename}.pdf"
                    if potential_pdf.exists():
                        pdf_path = str(potential_pdf)
                        print(f"[DEBUG] 자동으로 찾은 PDF: {pdf_path}")
                        break
        
        # PDF 파일이 존재하는지 확인하고 로드
        viewer.pdf_pages = []
        if pdf_path:
            if Path(pdf_path).exists():
                print(f"[DEBUG] PDF 파일 존재 확인됨: {pdf_path}")
                try:
                    # PDF를 이미지로 변환
                    viewer.pdf_pages = viewer.pdf_to_page_images(pdf_path)
                    viewer.current_pdf_path = pdf_path
                    print(f"[DEBUG] PDF 페이지 로드 완료: {len(viewer.pdf_pages)}페이지")
                except Exception as pdf_error:
                    print(f"[DEBUG] PDF 로드 오류: {pdf_error}")
                    viewer.pdf_pages = []
            else:
                print(f"[DEBUG] PDF 파일이 존재하지 않음: {pdf_path}")
                pdf_path = None  # 파일이 없으면 None으로 설정
        else:
            print(f"[DEBUG] PDF를 찾을 수 없음")
        
        # MD 파일에서 페이지별 내용 추출
        viewer.markdown_pages = []
        
        # "## 페이지 X" 패턴으로 분리
        page_pattern = r'## 페이지 (\d+)'
        pages = re.split(page_pattern, content)
        
        if len(pages) > 1:
            # 첫 번째 부분(제목 등)은 제외하고 페이지별로 처리
            for i in range(2, len(pages), 2):  # 페이지 번호와 내용이 번갈아 나타남
                if i < len(pages):
                    page_content = pages[i].strip()
                    # 구분선 제거
                    page_content = page_content.replace('='*50, '').strip()
                    viewer.markdown_pages.append(page_content)
        else:
            # 페이지 구분이 없으면 전체 내용을 하나로
            viewer.markdown_pages = [content]
        
        # 페이지 수 맞추기
        if viewer.pdf_pages and viewer.markdown_pages:
            max_pages = max(len(viewer.pdf_pages), len(viewer.markdown_pages))
            
            # PDF 페이지가 부족하면 빈 페이지 추가
            while len(viewer.pdf_pages) < max_pages:
                viewer.pdf_pages.append("")
                
            # Markdown 페이지가 부족하면 빈 페이지 추가
            while len(viewer.markdown_pages) < max_pages:
                viewer.markdown_pages.append("*(이 페이지의 내용이 없습니다)*")
                
            viewer.total_pages = max_pages
        else:
            viewer.total_pages = max(len(viewer.pdf_pages), len(viewer.markdown_pages))
        
        viewer.current_page = 0
        
        # 결과 생성
        if viewer.pdf_pages and any(viewer.pdf_pages):  # PDF가 있으면
            pdf_html = viewer.get_all_pdf_html()
            if pdf_match:  # 원래 경로에서 찾은 경우
                status_msg = f"✅ 불러오기 완료: {Path(file_path).name} (PDF + Markdown, 총 {viewer.total_pages}페이지)"
            else:  # 자동 검색으로 찾은 경우
                status_msg = f"✅ 불러오기 완료: {Path(file_path).name} (PDF 자동 검색 성공 + Markdown, 총 {viewer.total_pages}페이지)"
        else:  # PDF가 없거나 로드 실패시
            if pdf_path:
                pdf_html = f"<p style='text-align: center; color: #666; font-size: 18px;'>⚠️ 원본 PDF를 찾을 수 없습니다<br><small>{pdf_path}</small></p>"
                status_msg = f"⚠️ MD 파일만 불러옴: {Path(file_path).name} (원본 PDF 없음, 총 {viewer.total_pages}페이지)"
            else:
                # PDF 경로 정보가 아예 없는 경우 (레거시 파일이고 PDF도 못 찾은 경우)
                pdf_html = "<p style='text-align: center; color: #666; font-size: 18px;'>📄 레거시 MD 파일입니다<br><small>PDF 파일을 찾을 수 없어 Markdown만 표시됩니다</small></p>"
                status_msg = f"ℹ️ 레거시 MD 파일 불러옴: {Path(file_path).name} (PDF 없음, 총 {viewer.total_pages}페이지)"
        
        markdown_content = viewer.get_all_markdown_content()
        
        return (
            pdf_html,
            markdown_content,
            status_msg
        )
        
    except Exception as e:
        return (
            f"<p style='color: red;'>오류: {str(e)}</p>",
            f"오류: {str(e)}",
            f"❌ 파일 불러오기 실패: {str(e)}"
        )


def load_markdown_file(md_file):
    """Markdown 파일만 불러와서 메인 뷰어에 전체 내용으로 표시"""
    if not md_file:
        return (
            "Markdown 파일을 선택하세요.",
            "",
            "Markdown 파일을 선택하세요."
        )
    
    try:
        with open(md_file.name, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 페이지별로 분리 시도
        # 방법 1: "--- 페이지 X ---" 패턴
        page_pattern = r'---\s*페이지\s*(\d+)\s*---'
        pages = re.split(page_pattern, content)
        
        if len(pages) > 1:
            viewer.markdown_pages = []
            for i in range(1, len(pages), 2):
                if i+1 < len(pages):
                    viewer.markdown_pages.append(pages[i+1].strip())
        else:
            # 방법 2: 섹션 구분자로 분리
            sections = content.split('\n---\n')
            if len(sections) > 1:
                viewer.markdown_pages = sections
            else:
                # 방법 3: 큰 단락으로 분리
                paragraphs = content.split('\n\n\n')
                viewer.markdown_pages = paragraphs if len(paragraphs) > 1 else [content]
        
        # PDF가 로드되어 있으면 페이지 수 맞추기
        if viewer.pdf_pages:
            if len(viewer.markdown_pages) < len(viewer.pdf_pages):
                while len(viewer.markdown_pages) < len(viewer.pdf_pages):
                    viewer.markdown_pages.append("*(이 페이지의 Markdown이 없습니다)*")
            elif len(viewer.markdown_pages) > len(viewer.pdf_pages):
                # 초과분은 마지막 페이지에 합치기
                excess = viewer.markdown_pages[len(viewer.pdf_pages):]
                viewer.markdown_pages = viewer.markdown_pages[:len(viewer.pdf_pages)]
                viewer.markdown_pages[-1] += "\n\n" + "\n\n".join(excess)
            
            viewer.total_pages = len(viewer.pdf_pages)
        else:
            # PDF가 없으면 Markdown 페이지 수로 설정
            viewer.total_pages = len(viewer.markdown_pages)
            viewer.pdf_pages = [""] * viewer.total_pages  # 빈 PDF 페이지
        
        viewer.current_page = 0
        
        # 모든 페이지를 스크롤 방식으로 표시
        if viewer.pdf_pages and viewer.pdf_pages[0]:
            pdf_html = viewer.get_all_pdf_html()
        else:
            pdf_html = "<p style='text-align: center; color: #666;'>PDF가 로드되지 않았습니다.</p>"
        
        markdown_content = viewer.get_all_markdown_content()
        
        return (
            pdf_html,
            markdown_content,
            f"✅ Markdown 로드 완료: {Path(md_file.name).name} (총 {viewer.total_pages}페이지)"
        )
    
    except Exception as e:
        return (
            f"<p style='color: red;'>오류: {str(e)}</p>",
            f"오류: {str(e)}",
            f"❌ 파일 로드 실패: {str(e)}"
        )


def get_saved_files():
    """저장된 파일 목록 가져오기"""
    files = viewer.get_available_conversions()
    if files:
        # 파일 전체 경로를 반환 (드롭다운에는 파일명만 표시)
        choices = files
        labels = [str(Path(f).name) for f in files]
        return gr.update(choices=list(zip(labels, files)), value=files[0] if files else None)
    return gr.update(choices=[], value=None)


# Custom CSS
custom_css = """
.pdf-container {
    height: 800px;
    overflow-y: auto;
    border: 2px solid #ddd;
    padding: 10px;
    background-color: #f9f9f9;
    border-radius: 5px;
    scroll-behavior: smooth;
}

.markdown-container {
    height: 800px;
    overflow-y: auto;
    border: 2px solid #ddd;
    padding: 20px;
    background-color: white;
    border-radius: 5px;
    scroll-behavior: smooth;
}

.pdf-page {
    margin-bottom: 30px;
    padding-bottom: 20px;
    border-bottom: 2px solid #eee;
}

.page-number {
    background-color: #007acc;
    color: white;
    padding: 5px 15px;
    border-radius: 15px;
    font-size: 14px;
    font-weight: bold;
    margin-bottom: 10px;
    text-align: center;
    display: inline-block;
}

.gradio-container {
    max-width: 1600px !important;
}

/* 스크롤바 커스터마이징 */
.pdf-container::-webkit-scrollbar,
.markdown-container::-webkit-scrollbar {
    width: 12px;
}

.pdf-container::-webkit-scrollbar-track,
.markdown-container::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 6px;
}

.pdf-container::-webkit-scrollbar-thumb,
.markdown-container::-webkit-scrollbar-thumb {
    background: #888;
    border-radius: 6px;
}

.pdf-container::-webkit-scrollbar-thumb:hover,
.markdown-container::-webkit-scrollbar-thumb:hover {
    background: #555;
}
"""

# Gradio 인터페이스
with gr.Blocks(title="PDF-Markdown 동기화 뷰어", theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown(
        """
        # 📄 PDF-Markdown 동기화 뷰어
        
        **PDF와 변환된 Markdown을 페이지별로 동기화하여 표시합니다.**
        """
    )
    
    with gr.Tabs():
        with gr.Tab("📤 새로운 변환"):
            with gr.Row():
                pdf_input = gr.File(
                    label="PDF 파일 업로드",
                    file_types=[".pdf"],
                    type="filepath"
                )
                convert_btn = gr.Button("🔄 변환하기", variant="primary")
            
            status_text = gr.Textbox(
                label="상태",
                value="PDF 파일을 선택한 후 '변환하기' 버튼을 클릭하세요.",
                interactive=False
            )
        
        with gr.Tab("📥 이전 변환 불러오기"):
            with gr.Row():
                with gr.Column():
                    saved_files = gr.Dropdown(
                        label="저장된 변환 파일 (PDF + Markdown)",
                        choices=[],
                        interactive=True
                    )
                    refresh_btn = gr.Button("🔄 목록 새로고침")
                    load_btn = gr.Button("📂 불러오기 (PDF + Markdown)", variant="primary")
            
            load_status = gr.Textbox(
                label="상태",
                value="불러올 파일을 선택하세요. 원본 PDF와 Markdown을 함께 불러옵니다.",
                interactive=False
            )
        
        with gr.Tab("📝 Markdown만 불러오기"):
            with gr.Row():
                with gr.Column():
                    md_file_input = gr.File(
                        label="Markdown 파일 선택",
                        file_types=[".md"],
                        type="filepath"
                    )
                    load_md_btn = gr.Button("📄 Markdown 불러오기", variant="primary")
            
            md_load_status = gr.Textbox(
                label="상태",
                value="Markdown 파일을 선택한 후 '불러오기' 버튼을 클릭하세요. 메인 뷰어에 페이지별로 표시됩니다.",
                interactive=False
            )
    
    gr.Markdown("---")
    gr.Markdown("### 📚 PDF와 Markdown 스크롤 뷰어")
    gr.Markdown("*좌우 패널을 각각 스크롤하여 내용을 확인하세요.*")
    
    # PDF와 Markdown 표시 영역
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📖 원본 PDF")
            pdf_display = gr.HTML(
                label="PDF 전체 페이지",
                elem_classes=["pdf-container"]
            )
        
        with gr.Column(scale=1):
            gr.Markdown("### 📝 변환된 Markdown")
            markdown_display = gr.Markdown(
                label="Markdown 전체 내용",
                elem_classes=["markdown-container"]
            )
    
    # 이벤트 연결
    
    # 새로운 변환 - 버튼 클릭시에만 변환 실행
    convert_btn.click(
        fn=process_pdf,
        inputs=[pdf_input],
        outputs=[pdf_display, markdown_display, status_text]
    )
    
    # 저장된 변환 불러오기
    refresh_btn.click(
        fn=get_saved_files,
        outputs=[saved_files]
    )
    
    load_btn.click(
        fn=lambda x: load_saved_conversion(x) if x else ("파일을 선택하세요.", "", "파일을 선택하세요."),
        inputs=[saved_files],
        outputs=[pdf_display, markdown_display, load_status]
    )
    
    # Markdown 파일만 불러오기
    load_md_btn.click(
        fn=load_markdown_file,
        inputs=[md_file_input],
        outputs=[pdf_display, markdown_display, md_load_status]
    )
    
    # 초기 파일 목록 로드
    demo.load(
        fn=get_saved_files,
        outputs=[saved_files]
    )
    
    gr.Markdown(
        """
        ---
        ### 💡 사용 방법:
        1. **새로운 변환**: PDF 파일을 선택한 후 '변환하기' 버튼을 클릭하세요.
        2. **이전 변환 불러오기**: 저장된 MD 파일을 선택하면 원본 PDF와 Markdown을 함께 불러옵니다.
        3. **Markdown만 불러오기**: 기존 Markdown 파일을 불러와서 확인할 수 있습니다.
        4. **스크롤 뷰어**: 좌우 패널을 각각 스크롤하여 전체 내용을 확인하세요.
        5. **연속 표시**: 모든 PDF 페이지와 Markdown 내용이 연속으로 표시됩니다.
        
        ### 📁 저장 위치:
        - 변환 결과는 `conversions/` 폴더에 `.md` 파일로 저장됩니다.
        - 동일한 파일명이 있으면 자동으로 숫자가 추가됩니다 (예: `document_1.md`, `document_2.md`)
        - MD 파일에는 원본 PDF 경로 정보와 페이지별 내용이 포함됩니다.
        
        ### 🔧 특징:
        - PDF 페이지별로 구분선과 페이지 번호가 표시됩니다.
        - Markdown도 페이지별로 구분되어 표시됩니다.
        - 각 패널을 독립적으로 스크롤할 수 있습니다.
        """
    )

if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )