import os
import nest_asyncio
from dotenv import load_dotenv
from llama_parse import LlamaParse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import asyncio
from typing import List, Dict

load_dotenv()
nest_asyncio.apply()

class AdvancedLlamaParseRAG:
    """모드별 특화 RAG 시스템 구현"""
    
    def __init__(self):
        self.api_key = os.environ.get("LLAMA_CLOUD_API_KEY")
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    async def parse_research_paper_pipeline(self, pdf_path: str):
        """
        연구 논문 처리 파이프라인
        Premium 모드 + Vendor Multimodal 조합
        """
        print("📚 연구 논문 분석 파이프라인 시작...")
        
        # Step 1: Premium 모드로 전체 구조 파싱
        parser_premium = LlamaParse(
            api_key=self.api_key,
            premium_mode=True,
            result_type="markdown",
            parsing_instruction="""
            학술 논문 파싱:
            1. 초록, 서론, 방법론, 결과, 결론 섹션 구분
            2. 모든 참고문헌 보존
            3. 수식과 알고리즘 정확히 추출
            4. 그림과 표 캡션 유지
            """,
            language="en",
            page_separator=True
        )
        
        docs_structure = await parser_premium.aload_data(pdf_path)
        
        # Step 2: Vendor Multimodal로 그래프/차트 분석
        parser_visual = LlamaParse(
            api_key=self.api_key,
            use_vendor_multimodal_model=True,
            vendor_multimodal_model_name="openai-gpt4o",
            vendor_multimodal_api_key=os.environ.get("OPENAI_API_KEY"),
            result_type="markdown",
            parsing_instruction="""
            Focus on:
            1. Extract all data from graphs and charts
            2. Describe experimental results from figures
            3. Convert mathematical equations to LaTeX
            4. Explain diagrams and flowcharts
            """
        )
        
        docs_visual = await parser_visual.aload_data(pdf_path)
        
        # Step 3: 결과 통합 및 벡터 스토어 생성
        all_docs = []
        for doc in docs_structure:
            all_docs.append(doc.to_langchain_format())
        for doc in docs_visual:
            all_docs.append(doc.to_langchain_format())
        
        # 텍스트 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(all_docs)
        
        # 벡터 스토어 생성
        vectorstore = FAISS.from_documents(splits, self.embeddings)
        
        # RAG 체인 생성
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5})
        )
        
        return qa_chain, vectorstore
    
    async def parse_financial_report(self, file_path: str):
        """
        재무 보고서 특화 파싱
        Spreadsheet + Vendor Multimodal 모드 조합
        """
        print("💰 재무 보고서 분석 시작...")
        
        # 파일 확장자 확인
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext in ['.xlsx', '.xls', '.csv']:
            # Spreadsheet 모드로 표 데이터 추출
            parser_sheet = LlamaParse(
                api_key=self.api_key,
                spreadsheet_mode=True,
                result_type="csv",
                preserve_formatting=True
            )
            docs_sheet = await parser_sheet.aload_data(file_path)
            
        else:  # PDF 재무 보고서
            # Vendor Multimodal로 표와 차트 분석
            parser_financial = LlamaParse(
                api_key=self.api_key,
                use_vendor_multimodal_model=True,
                vendor_multimodal_model_name="openai-gpt4o",
                vendor_multimodal_api_key=os.environ.get("OPENAI_API_KEY"),
                result_type="markdown",
                parsing_instruction="""
                재무 보고서 분석:
                1. 모든 재무제표를 표 형식으로 추출
                2. 손익계산서, 재무상태표, 현금흐름표 구분
                3. 전년 대비 증감율 계산
                4. 주요 재무 지표 하이라이트
                5. 차트에서 트렌드 데이터 추출
                """,
                language="ko"
            )
            docs_sheet = await parser_financial.aload_data(file_path)
        
        return docs_sheet
    
    async def batch_process_documents(self, folder_path: str):
        """
        폴더 내 모든 문서 일괄 처리
        Continuous + Auto 모드 활용
        """
        print(f"📁 '{folder_path}' 폴더 일괄 처리 시작...")
        
        # 지원 파일 확장자
        supported_exts = ['.pdf', '.docx', '.pptx', '.xlsx', '.txt']
        
        # 파일 목록 수집
        files = []
        for root, dirs, filenames in os.walk(folder_path):
            for filename in filenames:
                if any(filename.lower().endswith(ext) for ext in supported_exts):
                    files.append(os.path.join(root, filename))
        
        print(f"발견된 파일: {len(files)}개")
        
        # Continuous 모드로 배치 처리
        parser_batch = LlamaParse(
            api_key=self.api_key,
            continuous_mode=True,
            auto_mode=True,  # 파일 유형별 자동 최적화
            result_type="markdown",
            language="ko",
            batch_size=10,
            max_workers=3  # 병렬 처리 워커 수
        )
        
        results = {}
        for file_path in files:
            try:
                print(f"  처리 중: {os.path.basename(file_path)}")
                docs = await parser_batch.aload_data(file_path)
                results[file_path] = [doc.to_langchain_format() for doc in docs]
            except Exception as e:
                print(f"  ❌ 오류: {file_path} - {str(e)}")
                results[file_path] = None
        
        # 결과 요약
        success_count = sum(1 for v in results.values() if v is not None)
        print(f"\n✅ 처리 완료: {success_count}/{len(files)} 파일")
        
        return results
    
    async def create_knowledge_base(self, documents: Dict[str, List]):
        """
        문서들로부터 통합 지식 베이스 생성
        """
        print("🧠 통합 지식 베이스 생성 중...")
        
        all_docs = []
        for file_path, docs in documents.items():
            if docs:
                for doc in docs:
                    # 메타데이터에 파일 정보 추가
                    doc.metadata['source_file'] = os.path.basename(file_path)
                    all_docs.append(doc)
        
        # 텍스트 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        splits = text_splitter.split_documents(all_docs)
        
        # 벡터 스토어 생성
        vectorstore = FAISS.from_documents(splits, self.embeddings)
        
        # 저장
        vectorstore.save_local("knowledge_base")
        print(f"✅ 지식 베이스 저장 완료: {len(splits)}개 청크")
        
        return vectorstore
    
    async def audio_meeting_analysis(self, audio_path: str):
        """
        회의 녹음 분석 파이프라인
        Audio 모드 + 요약 및 액션 아이템 추출
        """
        print("🎙️ 회의 녹음 분석 시작...")
        
        # Audio 모드로 전사
        parser_audio = LlamaParse(
            api_key=self.api_key,
            audio_mode=True,
            language="ko",
            include_timestamps=True,
            result_type="text"
        )
        
        docs = await parser_audio.aload_data(audio_path)
        transcript = docs[0].text if docs else ""
        
        # 전사 내용 분석
        analysis_prompt = f"""
        다음 회의 전사 내용을 분석하세요:
        
        {transcript}
        
        분석 항목:
        1. 회의 요약 (3-5문장)
        2. 주요 결정 사항
        3. 액션 아이템 (담당자, 기한 포함)
        4. 다음 단계
        5. 미해결 이슈
        """
        
        response = await self.llm.ainvoke(analysis_prompt)
        
        return {
            "transcript": transcript,
            "analysis": response.content,
            "metadata": docs[0].metadata if docs else {}
        }


class ComparativeParser:
    """모드별 성능 비교 도구"""
    
    def __init__(self):
        self.api_key = os.environ.get("LLAMA_CLOUD_API_KEY")
        self.results = {}
    
    async def compare_modes(self, file_path: str):
        """
        동일 파일을 여러 모드로 파싱하여 비교
        """
        import time
        
        modes = [
            ("default", {}),
            ("fast", {"fast_mode": True}),
            ("auto", {"auto_mode": True}),
            ("premium", {"premium_mode": True})
        ]
        
        print(f"\n📊 '{file_path}' 파일 모드별 비교 시작\n")
        
        for mode_name, mode_config in modes:
            print(f"🔄 {mode_name.upper()} 모드 테스트...")
            
            start_time = time.time()
            
            try:
                parser = LlamaParse(
                    api_key=self.api_key,
                    result_type="markdown",
                    language="ko",
                    **mode_config
                )
                
                docs = await parser.aload_data(file_path)
                
                elapsed_time = time.time() - start_time
                
                # 결과 저장
                self.results[mode_name] = {
                    "time": elapsed_time,
                    "pages": len(docs),
                    "total_chars": sum(len(doc.text) for doc in docs),
                    "success": True
                }
                
                print(f"  ✅ 완료: {elapsed_time:.2f}초, "
                      f"{len(docs)}페이지, "
                      f"{self.results[mode_name]['total_chars']:,}자")
                
            except Exception as e:
                self.results[mode_name] = {
                    "time": time.time() - start_time,
                    "error": str(e),
                    "success": False
                }
                print(f"  ❌ 오류: {str(e)}")
        
        # 비교 결과 출력
        self._print_comparison()
        
        return self.results
    
    def _print_comparison(self):
        """비교 결과 출력"""
        print("\n" + "="*60)
        print("📈 모드별 성능 비교 결과")
        print("="*60)
        
        # 성공한 모드만 필터링
        successful = {k: v for k, v in self.results.items() if v['success']}
        
        if successful:
            # 속도 순위
            speed_ranking = sorted(successful.items(), key=lambda x: x[1]['time'])
            print("\n⚡ 속도 순위:")
            for i, (mode, data) in enumerate(speed_ranking, 1):
                print(f"  {i}. {mode}: {data['time']:.2f}초")
            
            # 품질 지표 (추출된 텍스트 양)
            quality_ranking = sorted(successful.items(), 
                                    key=lambda x: x[1]['total_chars'], 
                                    reverse=True)
            print("\n📊 추출 품질 (텍스트 양):")
            for i, (mode, data) in enumerate(quality_ranking, 1):
                print(f"  {i}. {mode}: {data['total_chars']:,}자")
            
            # 추천 사항
            print("\n💡 추천 사항:")
            fastest = speed_ranking[0][0]
            best_quality = quality_ranking[0][0]
            
            print(f"  - 빠른 처리가 필요한 경우: {fastest} 모드")
            print(f"  - 최고 품질이 필요한 경우: {best_quality} 모드")
            
            if fastest == best_quality:
                print(f"  - 🏆 최적 선택: {fastest} 모드 (속도와 품질 모두 우수)")


# 실행 예제
async def main():
    # 1. 연구 논문 RAG 시스템
    rag_system = AdvancedLlamaParseRAG()
    
    # 논문 분석
    qa_chain, vectorstore = await rag_system.parse_research_paper_pipeline(
        "data/research_paper.pdf"
    )
    
    # 질문 응답
    question = "이 논문의 주요 기여점은 무엇인가요?"
    answer = qa_chain.run(question)
    print(f"\n질문: {question}")
    print(f"답변: {answer}")
    
    # 2. 모드 비교
    comparator = ComparativeParser()
    await comparator.compare_modes("data/sample.pdf")
    
    # 3. 일괄 처리
    batch_results = await rag_system.batch_process_documents("data/documents")
    knowledge_base = await rag_system.create_knowledge_base(batch_results)
    
    # 4. 회의 녹음 분석
    meeting_analysis = await rag_system.audio_meeting_analysis(
        "data/meeting.mp3"
    )
    print("\n회의 분석 결과:")
    print(meeting_analysis['analysis'])


if __name__ == "__main__":
    asyncio.run(main())