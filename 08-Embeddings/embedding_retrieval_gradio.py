import os
import gradio as gr
from typing import List, Tuple
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain_upstage import UpstageEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document

# 상위 폴더의 .env 파일 로드
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
load_dotenv(env_path)

# 환경 변수 설정 확인
def check_env_vars():
    required_vars = []
    if not os.getenv("OPENAI_API_KEY"):
        required_vars.append("OPENAI_API_KEY")
    if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
        required_vars.append("HUGGINGFACEHUB_API_TOKEN")
    if not os.getenv("UPSTAGE_API_KEY"):
        required_vars.append("UPSTAGE_API_KEY")
    
    if required_vars:
        print(f"경고: 다음 환경 변수가 설정되지 않았습니다: {', '.join(required_vars)}")
        print("일부 임베딩 모델이 작동하지 않을 수 있습니다.")

# 임베딩 모델 초기화 함수
def get_embedding_model(model_choice: str):
    """선택한 임베딩 모델을 반환합니다."""
    try:
        if model_choice == "OpenAI (text-embedding-3-small)":
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
            return OpenAIEmbeddings(model="text-embedding-3-small")
        
        elif model_choice == "HuggingFace (multilingual-e5-large-instruct)":
            if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
                raise ValueError("HUGGINGFACEHUB_API_TOKEN이 설정되지 않았습니다.")
            model_name = "intfloat/multilingual-e5-large-instruct"
            return HuggingFaceEndpointEmbeddings(
                model=model_name,
                task="feature-extraction",
                huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
            )
        
        elif model_choice == "Upstage (solar-embedding-1-large)":
            if not os.getenv("UPSTAGE_API_KEY"):
                raise ValueError("UPSTAGE_API_KEY가 설정되지 않았습니다.")
            # 쿼리와 문서 모두에 사용할 수 있는 기본 모델 사용
            return UpstageEmbeddings(model="solar-embedding-1-large-passage")
        
        else:
            raise ValueError(f"지원하지 않는 모델: {model_choice}")
    except Exception as e:
        raise Exception(f"임베딩 모델 초기화 실패: {str(e)}")

# 문서 로드 및 전처리
def load_and_split_documents(file_path: str) -> List[Document]:
    """파일을 로드하고 문장 단위로 분할합니다."""
    try:
        # 파일 내용 직접 읽기
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 빈 줄과 불필요한 공백 제거
        lines = content.strip().split('\n')
        
        # 실제 내용이 있는 줄만 추출 (빈 줄 제외)
        sentences = []
        for line in lines:
            line = line.strip()
            # 따옴표로 둘러싸인 실제 문장만 추출
            if line and '"' in line:
                # 따옴표 사이의 내용 추출
                start = line.find('"')
                end = line.rfind('"')
                if start != -1 and end != -1 and start < end:
                    sentence = line[start+1:end]
                    if sentence:  # 빈 문장이 아닌 경우만 추가
                        sentences.append(sentence)
        
        # Document 객체로 변환
        documents = [Document(page_content=sentence, metadata={"index": i}) 
                    for i, sentence in enumerate(sentences)]
        
        return documents
    except Exception as e:
        raise Exception(f"문서 로드 실패: {str(e)}")

# 벡터 스토어 생성 또는 로드
def create_or_load_vectorstore(documents: List[Document], embedding_model, persist_directory: str = "./chroma_db"):
    """ChromaDB 벡터 스토어를 생성하거나 로드합니다."""
    try:
        # 기존 컬렉션 삭제하고 새로 생성 (매번 새로운 임베딩으로 시작)
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            persist_directory=persist_directory,
            collection_name="myembedding_collection"
        )
        return vectorstore
    except Exception as e:
        raise Exception(f"벡터 스토어 생성 실패: {str(e)}")

# 유사 문서 검색
def search_similar_documents(vectorstore, query: str, k: int = 3, with_score: bool = True) -> List[Tuple[str, float]]:
    """쿼리와 유사한 문서를 검색합니다."""
    try:
        if with_score:
            # 유사도 점수와 함께 검색
            results = vectorstore.similarity_search_with_score(query, k=k)
            # 결과 포맷팅
            formatted_results = []
            for doc, score in results:
                formatted_results.append((doc.page_content, score))
        else:
            # 점수 없이 검색 (기본 순서로 정렬됨)
            results = vectorstore.similarity_search(query, k=k)
            # 결과 포맷팅 (점수는 None으로 설정)
            formatted_results = []
            for doc in results:
                formatted_results.append((doc.page_content, None))
        
        return formatted_results
    except Exception as e:
        raise Exception(f"문서 검색 실패: {str(e)}")

# Gradio 인터페이스를 위한 메인 함수
class EmbeddingRetriever:
    def __init__(self):
        self.vectorstore = None
        self.current_model = None
        self.documents = None
        self.current_file_path = None
        
    def initialize(self, embedding_choice: str, file_obj):
        """선택한 임베딩 모델로 시스템을 초기화합니다."""
        try:
            # 파일 경로 결정
            if file_obj is not None:
                # 업로드된 파일 사용
                file_path = file_obj.name
                self.current_file_path = file_path
            else:
                # 기본 파일 사용
                file_path = "data/myEmbedding.txt"
                self.current_file_path = file_path
            
            # 문서 로드
            self.documents = load_and_split_documents(file_path)
            if not self.documents:
                return f"오류: 문서를 로드할 수 없습니다. ({file_path})", "", self.get_all_chunks()
            
            # 임베딩 모델 초기화
            embedding_model = get_embedding_model(embedding_choice)
            self.current_model = embedding_choice
            
            # 벡터 스토어 생성
            self.vectorstore = create_or_load_vectorstore(self.documents, embedding_model)
            
            # 로드된 문서 정보
            doc_info = f"📂 파일: {os.path.basename(file_path)}\n"
            doc_info += f"📊 총 {len(self.documents)}개의 문장이 로드되었습니다.\n\n"
            doc_info += "로드된 문장 예시 (처음 5개):\n"
            for i, doc in enumerate(self.documents[:5], 1):
                doc_info += f"{i}. {doc.page_content}\n"
            
            # 전체 청크 정보
            chunks_info = self.get_all_chunks()
            
            return f"✅ 성공적으로 초기화되었습니다!\n모델: {embedding_choice}\n파일: {os.path.basename(file_path)}", doc_info, chunks_info
        except Exception as e:
            return f"❌ 초기화 실패: {str(e)}", "", self.get_all_chunks()
    
    def get_all_chunks(self):
        """분할된 모든 청크를 반환합니다."""
        if self.documents is None or len(self.documents) == 0:
            return "아직 로드된 문서가 없습니다."
        
        chunks_text = f"📄 총 {len(self.documents)}개의 청크로 분할됨\n"
        chunks_text += "="*50 + "\n\n"
        
        for i, doc in enumerate(self.documents, 1):
            chunks_text += f"【청크 #{i}】\n"
            chunks_text += f"내용: {doc.page_content}\n"
            if doc.metadata:
                chunks_text += f"메타데이터: {doc.metadata}\n"
            chunks_text += "-"*40 + "\n\n"
        
        return chunks_text
    
    def search(self, query: str, top_k: int, search_method: str):
        """쿼리를 검색합니다."""
        try:
            if self.vectorstore is None:
                return "먼저 임베딩 모델을 선택하고 초기화 버튼을 클릭하세요."
            
            if not query.strip():
                return "검색할 쿼리를 입력하세요."
            
            # 검색 방법에 따라 점수 포함 여부 결정
            with_score = (search_method == "similarity_search_with_score")
            
            # 유사 문서 검색
            results = search_similar_documents(self.vectorstore, query, k=top_k, with_score=with_score)
            
            # 결과 포맷팅
            output = f"🔍 검색 쿼리: '{query}'\n"
            output += f"📊 사용 모델: {self.current_model}\n"
            output += f"🔧 검색 방법: {search_method}\n"
            output += f"📝 상위 {top_k}개 결과:\n"
            output += "="*50 + "\n\n"
            
            for i, (content, score) in enumerate(results, 1):
                output += f"🏆 순위 {i}\n"
                output += f"📄 문장: {content}\n"
                
                if score is not None:
                    # ChromaDB의 거리 점수를 유사도로 변환 (낮을수록 유사)
                    similarity = 1 / (1 + score) if score > 0 else 1.0
                    output += f"💯 유사도 점수: {similarity:.4f} (거리: {score:.4f})\n"
                else:
                    output += f"💯 유사도 점수: (점수 없음 - 순서로만 정렬됨)\n"
                
                output += "-"*40 + "\n\n"
            
            return output
        except Exception as e:
            return f"❌ 검색 실패: {str(e)}"

# Gradio 앱 생성
def create_gradio_app():
    retriever = EmbeddingRetriever()
    
    with gr.Blocks(title="🔍 임베딩 기반 문서 검색 시스템") as app:
        gr.Markdown("""
        # 🔍 임베딩 기반 문서 검색 시스템
        
        이 시스템은 LangChain과 ChromaDB를 사용하여 문서를 임베딩하고 유사한 문장을 검색합니다.
        
        ## 사용 방법:
        1. (선택사항) 분석할 텍스트 파일을 업로드하거나 기본 파일을 사용하세요
        2. 원하는 임베딩 모델을 선택하세요
        3. '초기화' 버튼을 클릭하여 문서를 로드하고 임베딩을 생성하세요
        4. 검색할 쿼리를 입력하고 '검색' 버튼을 클릭하세요
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ 설정")
                
                # 파일 업로드 위젯 추가
                file_upload = gr.File(
                    label="📁 문서 파일 선택 (선택사항)",
                    file_types=[".txt"],
                    type="filepath",
                    value=None
                )
                gr.Markdown("*기본값: data/myEmbedding.txt*")
                
                embedding_choice = gr.Dropdown(
                    choices=[
                        "OpenAI (text-embedding-3-small)",
                        "HuggingFace (multilingual-e5-large-instruct)",
                        "Upstage (solar-embedding-1-large)"
                    ],
                    label="임베딩 모델 선택",
                    value="OpenAI (text-embedding-3-small)"
                )
                
                init_btn = gr.Button("🚀 초기화", variant="primary")
                init_status = gr.Textbox(label="초기화 상태", lines=3)
                doc_info = gr.Textbox(label="로드된 문서 정보", lines=7)
            
            with gr.Column(scale=1):
                gr.Markdown("### 🔍 검색")
                query_input = gr.Textbox(
                    label="검색 쿼리",
                    placeholder="예: 트랜스포머, K-POP, 김치 등",
                    lines=2
                )
                
                search_method = gr.Radio(
                    choices=[
                        "similarity_search_with_score",
                        "similarity_search"
                    ],
                    value="similarity_search_with_score",
                    label="검색 방법 선택",
                    info="with_score: 유사도 점수 포함 | similarity_search: 순서만 반환"
                )
                
                top_k = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=3,
                    step=1,
                    label="검색 결과 개수"
                )
                
                search_btn = gr.Button("🔎 검색", variant="primary")
        
        # 탭을 사용하여 검색 결과와 청크 보기를 분리
        with gr.Tabs():
            with gr.TabItem("🔎 검색 결과"):
                search_output = gr.Textbox(
                    label="검색 결과",
                    lines=15,
                    max_lines=30
                )
            
            with gr.TabItem("📑 분할된 청크 보기"):
                chunks_display = gr.Textbox(
                    label="전체 청크 목록",
                    lines=20,
                    max_lines=40,
                    value="문서를 초기화하면 여기에 모든 청크가 표시됩니다."
                )
        
        # 예시 쿼리들
        gr.Examples(
            examples=[
                ["트랜스포머"],
                ["K-POP 아이돌"],
                ["머신러닝"],
                ["한국 음식"],
                ["미국 정치"],
                ["관광지"]
            ],
            inputs=query_input,
            label="예시 쿼리"
        )
        
        # 이벤트 핸들러
        init_btn.click(
            fn=retriever.initialize,
            inputs=[embedding_choice, file_upload],
            outputs=[init_status, doc_info, chunks_display]
        )
        
        search_btn.click(
            fn=retriever.search,
            inputs=[query_input, top_k, search_method],
            outputs=search_output
        )
        
        # 엔터키로도 검색 가능
        query_input.submit(
            fn=retriever.search,
            inputs=[query_input, top_k, search_method],
            outputs=search_output
        )
    
    return app

# 메인 실행
if __name__ == "__main__":
    # 환경 변수 확인
    check_env_vars()
    
    # Gradio 앱 실행
    app = create_gradio_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )