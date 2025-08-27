import os
import nest_asyncio
from dotenv import load_dotenv
from llama_parse import LlamaParse
from typing import List, Dict, Any
import pandas as pd

# 환경 변수 로드
load_dotenv()
nest_asyncio.apply()

class LlamaParseMultiMode:
    """다양한 파싱 모드를 지원하는 LlamaParse 클래스"""
    
    def __init__(self, api_key: str = None):
        """
        초기화 메서드
        
        Args:
            api_key: LlamaParse API 키 (없으면 환경변수에서 로드)
        """
        self.api_key = api_key or os.environ.get("LLAMA_CLOUD_API_KEY")
        if not self.api_key:
            raise ValueError("LlamaParse API 키가 필요합니다.")
    
    def parse_default(self, file_path: str) -> List[Dict]:
        """
        기본 모드로 문서 파싱
        일반적인 PDF, DOCX 문서에 적합
        """
        print(f"📄 Default 모드로 '{file_path}' 파싱 중...")
        
        parser = LlamaParse(
            api_key=self.api_key,
            result_type="markdown",
            language="ko",
            verbose=True
        )
        
        documents = parser.load_data(file_path)
        return [doc.to_langchain_format() for doc in documents]
    
    def parse_fast(self, file_path: str) -> List[Dict]:
        """
        Fast 모드로 빠른 파싱
        대량 문서 처리나 빠른 미리보기에 적합
        """
        print(f"⚡ Fast 모드로 '{file_path}' 파싱 중...")
        
        parser = LlamaParse(
            api_key=self.api_key,
            result_type="text",  # 빠른 처리를 위해 text 형식
            fast_mode=True,
            language="ko"
        )
        
        documents = parser.load_data(file_path)
        return [doc.to_langchain_format() for doc in documents]
    
    def parse_vendor_multimodal(self, file_path: str, 
                                vendor_model: str = "openai-gpt4o",
                                parsing_instruction: str = None) -> List[Dict]:
        """
        Vendor Multimodal 모드로 고급 파싱
        복잡한 표, 차트, 이미지가 포함된 문서에 적합
        
        Args:
            file_path: 파일 경로
            vendor_model: 사용할 벤더 모델 (openai-gpt4o, anthropic-claude-3, etc.)
            parsing_instruction: 파싱 지시사항
        """
        print(f"🤖 Vendor Multimodal 모드 ({vendor_model})로 '{file_path}' 파싱 중...")
        
        if not parsing_instruction:
            parsing_instruction = """
            다음 규칙에 따라 문서를 파싱하세요:
            1. 모든 표는 마크다운 형식으로 변환
            2. 이미지에 대한 설명 포함
            3. 차트와 그래프의 데이터 추출
            4. 중요한 숫자와 통계 강조
            """
        
        # OpenAI 모델 사용 시
        if "openai" in vendor_model:
            vendor_api_key = os.environ.get("OPENAI_API_KEY")
        # Anthropic Claude 사용 시
        elif "anthropic" in vendor_model:
            vendor_api_key = os.environ.get("ANTHROPIC_API_KEY")
        else:
            vendor_api_key = None
        
        parser = LlamaParse(
            api_key=self.api_key,
            use_vendor_multimodal_model=True,
            vendor_multimodal_model_name=vendor_model,
            vendor_multimodal_api_key=vendor_api_key,
            result_type="markdown",
            language="ko",
            parsing_instruction=parsing_instruction,
            verbose=True
        )
        
        documents = parser.load_data(file_path)
        return [doc.to_langchain_format() for doc in documents]
    
    def parse_premium(self, file_path: str, 
                     parsing_instruction: str = None,
                     page_separator: bool = True) -> List[Dict]:
        """
        Premium 모드로 최고 품질 파싱
        복잡한 레이아웃의 문서나 정확도가 중요한 경우 사용
        """
        print(f"💎 Premium 모드로 '{file_path}' 파싱 중...")
        
        if not parsing_instruction:
            parsing_instruction = """
            최고 품질로 문서를 파싱하세요:
            1. 모든 구조 요소 보존
            2. 헤더, 푸터, 페이지 번호 유지
            3. 각주와 참조 정확히 매핑
            4. 다단 레이아웃 구조 유지
            """
        
        parser = LlamaParse(
            api_key=self.api_key,
            premium_mode=True,
            result_type="markdown",
            language="ko",
            parsing_instruction=parsing_instruction,
            page_separator=page_separator,
            verbose=True
        )
        
        documents = parser.load_data(file_path)
        return [doc.to_langchain_format() for doc in documents]
    
    def parse_auto(self, file_path: str) -> List[Dict]:
        """
        Auto 모드로 자동 최적화 파싱
        문서 유형을 자동 감지하여 최적의 방법 선택
        """
        print(f"🔄 Auto 모드로 '{file_path}' 파싱 중...")
        
        parser = LlamaParse(
            api_key=self.api_key,
            auto_mode=True,
            result_type="markdown",
            language="ko",
            verbose=True
        )
        
        documents = parser.load_data(file_path)
        return [doc.to_langchain_format() for doc in documents]
    
    def parse_continuous(self, file_paths: List[str]) -> Dict[str, List[Dict]]:
        """
        Continuous 모드로 연속 문서 파싱
        여러 문서를 효율적으로 배치 처리
        """
        print(f"📚 Continuous 모드로 {len(file_paths)}개 문서 파싱 중...")
        
        parser = LlamaParse(
            api_key=self.api_key,
            continuous_mode=True,
            result_type="markdown",
            language="ko",
            batch_size=5,  # 배치 크기 설정
            verbose=True
        )
        
        results = {}
        for file_path in file_paths:
            print(f"  - {file_path} 처리 중...")
            documents = parser.load_data(file_path)
            results[file_path] = [doc.to_langchain_format() for doc in documents]
        
        return results
    
    def parse_spreadsheet(self, file_path: str, 
                         sheet_name: str = None,
                         preserve_formatting: bool = True) -> pd.DataFrame:
        """
        Spreadsheet 모드로 엑셀/CSV 파싱
        표 구조와 데이터 관계를 유지하며 파싱
        """
        print(f"📊 Spreadsheet 모드로 '{file_path}' 파싱 중...")
        
        parser = LlamaParse(
            api_key=self.api_key,
            result_type="csv",  # CSV 형식으로 결과 반환
            spreadsheet_mode=True,
            preserve_formatting=preserve_formatting,
            target_sheet=sheet_name,
            verbose=True
        )
        
        documents = parser.load_data(file_path)
        
        # DataFrame으로 변환
        if documents:
            # CSV 문자열을 DataFrame으로 변환
            import io
            csv_content = documents[0].text
            df = pd.read_csv(io.StringIO(csv_content))
            return df
        return pd.DataFrame()
    
    def parse_audio(self, audio_path: str, 
                   language: str = "ko",
                   include_timestamps: bool = True) -> Dict[str, Any]:
        """
        Audio 모드로 오디오 파일 전사
        음성을 텍스트로 변환
        """
        print(f"🎵 Audio 모드로 '{audio_path}' 전사 중...")
        
        parser = LlamaParse(
            api_key=self.api_key,
            audio_mode=True,
            language=language,
            include_timestamps=include_timestamps,
            result_type="text",
            verbose=True
        )
        
        documents = parser.load_data(audio_path)
        
        # 전사 결과 구조화
        result = {
            "file_path": audio_path,
            "transcription": documents[0].text if documents else "",
            "metadata": documents[0].metadata if documents else {}
        }
        
        if include_timestamps:
            # 타임스탬프가 포함된 경우 파싱
            result["timestamps"] = self._parse_timestamps(result["transcription"])
        
        return result
    
    def _parse_timestamps(self, text: str) -> List[Dict]:
        """타임스탬프 파싱 헬퍼 함수"""
        # 예: "[00:00:15] 텍스트" 형식 파싱
        import re
        pattern = r'\[(\d{2}:\d{2}:\d{2})\]\s*(.*?)(?=\[\d{2}:\d{2}:\d{2}\]|$)'
        matches = re.findall(pattern, text, re.DOTALL)
        
        return [{"timestamp": time, "text": content.strip()} 
                for time, content in matches]
    
    def save_to_markdown(self, documents: List[Dict], output_path: str):
        """파싱 결과를 마크다운 파일로 저장"""
        full_text = "\n\n---\n\n".join([doc.page_content for doc in documents])
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_text)
        
        print(f"✅ 저장 완료: {output_path}")


# 사용 예제
if __name__ == "__main__":
    # LlamaParseMultiMode 인스턴스 생성
    parser = LlamaParseMultiMode()
    
    # 1. 일반 PDF 파싱 (Default 모드)
    print("\n=== Default 모드 예제 ===")
    docs_default = parser.parse_default("data/sample.pdf")
    parser.save_to_markdown(docs_default, "output/sample_default.md")
    
    # 2. 빠른 파싱 (Fast 모드)
    print("\n=== Fast 모드 예제 ===")
    docs_fast = parser.parse_fast("data/large_document.pdf")
    parser.save_to_markdown(docs_fast, "output/large_document_fast.md")
    
    # 3. 복잡한 표가 있는 문서 (Vendor Multimodal 모드)
    print("\n=== Vendor Multimodal 모드 예제 ===")
    instruction = "모든 표를 정확히 추출하고, 차트의 데이터를 텍스트로 변환하세요."
    docs_vendor = parser.parse_vendor_multimodal(
        "data/report_with_charts.pdf",
        vendor_model="openai-gpt4o",
        parsing_instruction=instruction
    )
    parser.save_to_markdown(docs_vendor, "output/report_multimodal.md")
    
    # 4. 고품질 파싱 (Premium 모드)
    print("\n=== Premium 모드 예제 ===")
    docs_premium = parser.parse_premium(
        "data/complex_layout.pdf",
        parsing_instruction="모든 레이아웃 요소를 정확히 보존하세요."
    )
    parser.save_to_markdown(docs_premium, "output/complex_premium.md")
    
    # 5. 자동 모드 (Auto 모드)
    print("\n=== Auto 모드 예제 ===")
    docs_auto = parser.parse_auto("data/mixed_content.pdf")
    parser.save_to_markdown(docs_auto, "output/mixed_auto.md")
    
    # 6. 대량 문서 처리 (Continuous 모드)
    print("\n=== Continuous 모드 예제 ===")
    file_list = [
        "data/doc1.pdf",
        "data/doc2.pdf",
        "data/doc3.pdf"
    ]
    results_continuous = parser.parse_continuous(file_list)
    for file_path, docs in results_continuous.items():
        output_name = os.path.basename(file_path).replace(".pdf", "_continuous.md")
        parser.save_to_markdown(docs, f"output/{output_name}")
    
    # 7. 엑셀 파일 파싱 (Spreadsheet 모드)
    print("\n=== Spreadsheet 모드 예제 ===")
    df = parser.parse_spreadsheet(
        "data/sales_data.xlsx",
        sheet_name="2024년도",
        preserve_formatting=True
    )
    print(f"데이터프레임 shape: {df.shape}")
    print(df.head())
    df.to_csv("output/sales_data.csv", index=False)
    
    # 8. 오디오 파일 전사 (Audio 모드)
    print("\n=== Audio 모드 예제 ===")
    audio_result = parser.parse_audio(
        "data/meeting_recording.mp3",
        language="ko",
        include_timestamps=True
    )
    
    # 전사 결과 저장
    with open("output/meeting_transcript.txt", "w", encoding="utf-8") as f:
        f.write(f"파일: {audio_result['file_path']}\n\n")
        f.write("=== 전사 내용 ===\n")
        f.write(audio_result['transcription'])
        
        if audio_result.get('timestamps'):
            f.write("\n\n=== 타임스탬프별 내용 ===\n")
            for item in audio_result['timestamps']:
                f.write(f"[{item['timestamp']}] {item['text']}\n")
    
    print("\n✨ 모든 파싱 작업 완료!")