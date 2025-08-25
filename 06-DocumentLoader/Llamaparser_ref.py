import os
import nest_asyncio
from dotenv import load_dotenv

load_dotenv()
# jupyter 환경에서 asyncio를 사용할 수 있도록 설정
# 이는 Jupyter Notebook에서 비동기 작업을 지원하기 위해 필요합니다.
nest_asyncio.apply()


# LlamaParse와 같은 파서 객체가 이전에 초기화되었다고 가정합니다.
# from llama_parse import LlamaParse
# documents = LlamaParse(result_type="markdown")


def pdf_parser(pdf_file_path: str):
    """
    PDF 파일을 파싱하여 그 내용을 Markdown 파일로 저장합니다.

    Args:
        pdf_file_path (str): 처리할 PDF 파일의 경로.
    """
    print(f"🔄 '{pdf_file_path}' 파일 파싱을 시작합니다...")

    try:
        # parsing instruction 을 지정합니다.
        parsing_instruction = (
            "You are parsing a AI Report. Please extract tables in markdown format."
        )

        # LlamaParse 설정
        parser = LlamaParse(
            use_vendor_multimodal_model=True,
            vendor_multimodal_model_name="openai-gpt4o",
            vendor_multimodal_api_key=os.environ["OPENAI_API_KEY"],
            result_type="markdown",
            language="ko",
            parsing_instruction=parsing_instruction,
        )

        # 1. LlamaParse를 사용하여 PDF 파일을 로드합니다.
        # 'documents' 객체는 이 함수 외부에서 미리 정의되어 있어야 합니다.
        parsed_docs = documents.load_data(file_path=pdf_file_path)

        # 2. LangChain 형식의 도큐먼트로 변환합니다.
        docs = [doc.to_langchain_format() for doc in parsed_docs]

        # 3. 저장할 Markdown 파일의 경로를 생성합니다. (확장자 변경)
        file_root, _ = os.path.splitext(pdf_file_path)
        output_file_path = file_root + ".md"

        # 4. 모든 페이지의 내용을 하나의 텍스트로 합칩니다.
        #    페이지 사이는 두 줄로 띄어 가독성을 높입니다.
        full_text = "\n\n".join([doc.page_content for doc in docs])

        # 5. 추출된 전체 텍스트를 .md 파일로 저장합니다.
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(full_text)

        print(f"✅ 파일 저장 완료: {output_file_path}")

    except FileNotFoundError:
        print(f"❌ 오류: 파일을 찾을 수 없습니다 - {pdf_file_path}")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")


# --- 함수 사용 예시 ---
# 이 코드를 실행하기 전에 'documents' 파서 객체를 초기화해야 합니다.
file_to_parse = "data/디지털정부혁신추진계획.pdf"
pdf_parser(file_to_parse)
