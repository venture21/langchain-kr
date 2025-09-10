from dotenv import load_dotenv
import os
from langchain_upstage import UpstageDocumentParseLoader

# 환경변수 로드
load_dotenv()

file_path = "./data/[삼성전자]반기보고서_12.pdf"  # ex: ./document.pdf

loader = UpstageDocumentParseLoader(
    file_path,
    model="document-parse-250618",
    ocr="auto",
    coordinates=True,
    base64_encoding=["figure"],
)
pages = loader.load()  # or loader.lazy_load()

# pages[0].page_content를 html로 출력하기
print(pages[0].page_content)
