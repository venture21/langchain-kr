import os
import gradio as gr
from typing import List, Dict, Any, Optional, Tuple
import tempfile
import shutil
from pathlib import Path
import re
from bs4 import BeautifulSoup

# LangChain imports
from llama_parse import LlamaParse  # LlamaParse is from llama-parse package
from langchain_upstage import UpstageDocumentParseLoader  # For table extraction
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

# Environment variables
from dotenv import load_dotenv

# Load .env from parent directory
load_dotenv(dotenv_path="../.env")


class PDFRAGService:
    def __init__(self):
        """Initialize the RAG service with required components."""
        self.llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None

        # Storage for parsed content
        self.text_docs = []
        self.image_docs = []
        self.table_docs = []

        # Storage for Upstage HTML output
        self.upstage_html_content = ""
        self.extracted_tables_html = ""

        # Temporary directory for vector store
        self.vector_store_path = "./chroma_db"

    def extract_tables_from_html(self, html_content: str) -> List[str]:
        """
        Extract table content from HTML using BeautifulSoup.

        Args:
            html_content: HTML content from Upstage DocumentParser

        Returns:
            List of table contents as strings
        """
        try:
            print(f"🔍 Debug: HTML content length: {len(html_content)}")
            print(f"🔍 Debug: HTML preview: {html_content[:500]}...")
            
            soup = BeautifulSoup(html_content, "html.parser")
            tables = soup.find_all("table")
            
            print(f"🔍 Debug: Found {len(tables)} tables in HTML")

            table_contents = []
            for table_idx, table in enumerate(tables, 1):
                print(f"🔍 Debug: Processing table {table_idx}")
                
                # Convert table HTML to markdown-like format for better readability
                table_text = []

                # Process headers
                headers = table.find_all("th")
                print(f"🔍 Debug: Table {table_idx} - Found {len(headers)} headers")
                if headers:
                    header_row = " | ".join([th.get_text(strip=True) for th in headers])
                    table_text.append(header_row)
                    table_text.append("-" * len(header_row))
                    print(f"🔍 Debug: Table {table_idx} - Header: {header_row}")

                # Process rows
                rows = table.find_all("tr")
                print(f"🔍 Debug: Table {table_idx} - Found {len(rows)} rows")
                
                for row_idx, row in enumerate(rows):
                    cells = row.find_all(["td", "th"])
                    if cells:
                        cell_texts = [cell.get_text(strip=True) for cell in cells]
                        row_text = " | ".join(cell_texts)
                        
                        # Only add non-empty rows and avoid duplicate headers
                        if row_text and row_text.strip() and (not headers or row_text != (header_row if 'header_row' in locals() else "")):
                            table_text.append(row_text)
                            if row_idx < 3:  # Show first few rows for debugging
                                print(f"🔍 Debug: Table {table_idx} Row {row_idx}: {row_text}")

                if table_text:
                    final_table = "\n".join(table_text)
                    table_contents.append(final_table)
                    print(f"🔍 Debug: Table {table_idx} - Final length: {len(final_table)} characters")
                else:
                    print(f"🔍 Debug: Table {table_idx} - No content extracted")

            print(f"🔍 Debug: Total extracted tables: {len(table_contents)}")
            return table_contents
            
        except Exception as e:
            print(f"Error extracting tables from HTML: {e}")
            import traceback
            traceback.print_exc()
            return []

    def save_to_markdown(
        self,
        text_docs: List[Document],
        image_docs: List[Document],
        table_docs: List[Document],
        pdf_path: str,
    ) -> None:
        """
        Save parsed content to load_data.md file.

        Args:
            text_docs: List of text documents
            image_docs: List of image documents
            table_docs: List of table documents
            pdf_path: Original PDF file path
        """
        try:
            from datetime import datetime

            # Create markdown content
            markdown_content = []

            # Add header with timestamp
            markdown_content.append(f"# PDF Parsing Results")
            markdown_content.append(f"**Source File:** {Path(pdf_path).name}")
            markdown_content.append(
                f"**Parsed Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            markdown_content.append(
                f"**Total Documents:** Text({len(text_docs)}) | Images({len(image_docs)}) | Tables({len(table_docs)})"
            )
            markdown_content.append("\n---\n")

            # Add text content
            if text_docs:
                markdown_content.append("## 📄 Text Content")
                markdown_content.append(f"*Found {len(text_docs)} text documents*\n")
                for i, doc in enumerate(text_docs, 1):
                    markdown_content.append(f"### Text Document {i}")
                    if doc.metadata:
                        markdown_content.append(f"**Metadata:** {doc.metadata}")
                    markdown_content.append("```")
                    # Limit content length for readability
                    content = doc.page_content[:2000]
                    if len(doc.page_content) > 2000:
                        content += "\n... (truncated)"
                    markdown_content.append(content)
                    markdown_content.append("```\n")

            # Add image content
            if image_docs:
                markdown_content.append("## 🖼️ Image Content")
                markdown_content.append(f"*Found {len(image_docs)} image documents*\n")
                for i, doc in enumerate(image_docs, 1):
                    markdown_content.append(f"### Image Document {i}")
                    if doc.metadata:
                        markdown_content.append(f"**Metadata:** {doc.metadata}")
                    markdown_content.append("```")
                    # Limit content length for readability
                    content = doc.page_content[:1000]
                    if len(doc.page_content) > 1000:
                        content += "\n... (truncated)"
                    markdown_content.append(content)
                    markdown_content.append("```\n")

            # Add table content
            if table_docs:
                markdown_content.append("## 📊 Table Content")
                markdown_content.append(f"*Found {len(table_docs)} table documents*\n")
                for i, doc in enumerate(table_docs, 1):
                    markdown_content.append(f"### Table Document {i}")
                    if doc.metadata:
                        markdown_content.append(f"**Metadata:** {doc.metadata}")
                    markdown_content.append("```")
                    # For tables, show more content as they're structured
                    content = doc.page_content[:3000]
                    if len(doc.page_content) > 3000:
                        content += "\n... (truncated)"
                    markdown_content.append(content)
                    markdown_content.append("```\n")

            # Add summary statistics
            markdown_content.append("---\n")
            markdown_content.append("## 📊 Summary Statistics")
            markdown_content.append(
                f"- **Total Text Characters:** {sum(len(doc.page_content) for doc in text_docs)}"
            )
            markdown_content.append(
                f"- **Total Image Characters:** {sum(len(doc.page_content) for doc in image_docs)}"
            )
            markdown_content.append(
                f"- **Total Table Characters:** {sum(len(doc.page_content) for doc in table_docs)}"
            )
            markdown_content.append(
                f"- **Average Text Document Length:** {sum(len(doc.page_content) for doc in text_docs) // len(text_docs) if text_docs else 0}"
            )
            markdown_content.append(
                f"- **Average Image Document Length:** {sum(len(doc.page_content) for doc in image_docs) // len(image_docs) if image_docs else 0}"
            )
            markdown_content.append(
                f"- **Average Table Document Length:** {sum(len(doc.page_content) for doc in table_docs) // len(table_docs) if table_docs else 0}"
            )

            # Write to file
            output_path = Path("load_data.md")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("\n".join(markdown_content))

            print(f"✅ Parsed content saved to {output_path.absolute()}")

        except Exception as e:
            print(f"⚠️ Warning: Could not save to markdown file: {e}")

    def save_tables_to_markdown(
        self, table_docs: List[Document], pdf_path: str
    ) -> None:
        """
        Save table documents to table.md file in markdown format.

        Args:
            table_docs: List of table documents
            pdf_path: Original PDF file path
        """
        try:
            from datetime import datetime

            print(
                f"🔍 Debug: Starting save_tables_to_markdown with {len(table_docs)} tables"
            )

            # Create markdown content
            markdown_content = []

            # Add header with timestamp
            markdown_content.append("# 📊 Extracted Tables")
            markdown_content.append(f"**Source File:** {Path(pdf_path).name}")
            markdown_content.append(
                f"**Extraction Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            markdown_content.append(f"**Total Tables:** {len(table_docs)}")
            markdown_content.append("\n---\n")

            # Add statistics
            if table_docs:
                total_chars = sum(len(doc.page_content) for doc in table_docs)
                avg_chars = total_chars // len(table_docs) if table_docs else 0

                markdown_content.append("## 📈 Statistics")
                markdown_content.append(f"- **Total Tables:** {len(table_docs)}")
                markdown_content.append(f"- **Total Characters:** {total_chars:,}")
                markdown_content.append(
                    f"- **Average Table Size:** {avg_chars:,} characters"
                )
                markdown_content.append("\n---\n")

                # Add each table
                for i, doc in enumerate(table_docs, 1):
                    # Get metadata
                    page = (
                        doc.metadata.get("page", "Unknown")
                        if doc.metadata
                        else "Unknown"
                    )
                    table_idx = (
                        doc.metadata.get("table_index", "Unknown")
                        if doc.metadata
                        else "Unknown"
                    )
                    source = (
                        doc.metadata.get("source", "Unknown")
                        if doc.metadata
                        else "Unknown"
                    )

                    # Table header
                    markdown_content.append(
                        f"## Table {i} (Page {page}, Index {table_idx})"
                    )
                    markdown_content.append(f"*Extracted by: {source}*")
                    markdown_content.append("")

                    # Table content - format as code block to preserve structure
                    markdown_content.append("```")
                    markdown_content.append(doc.page_content)
                    markdown_content.append("```")
                    markdown_content.append("")

                    # Add metadata if available
                    if doc.metadata:
                        markdown_content.append("**Metadata:**")
                        for key, value in doc.metadata.items():
                            markdown_content.append(f"- **{key}:** {value}")
                        markdown_content.append("")

                    markdown_content.append("---\n")
            else:
                markdown_content.append("## No Tables Found")
                markdown_content.append("No tables were extracted from the document.")
                markdown_content.append("\nThis could be because:")
                markdown_content.append("- The document doesn't contain tables")
                markdown_content.append(
                    "- The Upstage DocumentParser couldn't detect table structures"
                )
                markdown_content.append("- There was an error during table extraction")

            # Write to file
            output_path = Path("table.md")
            print(f"🔍 Debug: Attempting to write to {output_path.absolute()}")

            with open(output_path, "w", encoding="utf-8") as f:
                content_to_write = "\n".join(markdown_content)
                f.write(content_to_write)
                print(f"🔍 Debug: Written {len(content_to_write)} characters to file")

            print(f"✅ Table content saved to {output_path.absolute()}")

        except Exception as e:
            print(f"⚠️ Error: Could not save tables to markdown file: {e}")
            import traceback

            print(f"🔍 Debug: Full error traceback:")
            traceback.print_exc()

    def parse_pdf_with_upstage_for_tables(self, pdf_path: str) -> List[Document]:
        """
        Parse PDF using Upstage DocumentParser specifically for table extraction.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of table documents
        """
        try:
            print("📊 Using Upstage DocumentParser for table extraction...")

            # Initialize Upstage DocumentParser
            upstage_loader = UpstageDocumentParseLoader(
                pdf_path,
                model="document-parse-250618",
                ocr="auto",
                coordinates=True,
                base64_encoding=["figure"],
            )

            # Load pages
            pages = upstage_loader.load()

            # Store full HTML content
            self.upstage_html_content = ""
            table_html_parts = []

            table_docs = []
            for page_num, page in enumerate(pages, 1):
                # Add page HTML to full content
                page_html = f"<h2>Page {page_num}</h2>\n{page.page_content}\n<hr>\n"
                self.upstage_html_content += page_html

                # Extract tables from HTML content
                soup = BeautifulSoup(page.page_content, "html.parser")
                page_tables = soup.find_all("table")

                if page_tables:
                    table_html_parts.append(f"<h3>📄 Page {page_num} Tables</h3>")
                    for table in page_tables:
                        # Store original HTML table
                        table_html_parts.append(str(table))
                        table_html_parts.append("<br><hr><br>")

                # Extract tables as text for processing
                tables = self.extract_tables_from_html(page.page_content)
                
                print(f"🔍 Debug: Page {page_num} - Found {len(page_tables)} HTML tables, extracted {len(tables)} text tables")
                
                for table_idx, table_content in enumerate(tables, 1):
                    if table_content:
                        print(f"🔍 Debug: Table {table_idx} content preview: {table_content[:200]}...")
                        table_doc = Document(
                            page_content=table_content,
                            metadata={
                                'type': 'table',
                                'page': page_num,
                                'table_index': table_idx,
                                'source': 'upstage',
                                'parser': 'document-parse-250618'
                            }
                        )
                        table_docs.append(table_doc)
                    else:
                        print(f"🔍 Debug: Table {table_idx} is empty or invalid")

            # Store extracted tables HTML using table_html_parts
            if table_html_parts:
                self.extracted_tables_html = (
                    "<h1>📊 Extracted Tables</h1>\n" + "\n".join(table_html_parts)
                )
            else:
                self.extracted_tables_html = "<p>No tables found in the document.</p>"

            print(f"✅ Extracted {len(table_docs)} tables using Upstage DocumentParser")
            return table_docs

        except Exception as e:
            print(f"⚠️ Warning: Could not extract tables with Upstage: {e}")
            return []

    def parse_pdf(
        self, pdf_path: str
    ) -> Tuple[List[Document], List[Document], List[Document]]:
        """
        Parse PDF using LlamaParse for text/images and Upstage for tables.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Tuple of (text_docs, image_docs, table_docs)
        """
        text_docs = []
        image_docs = []
        table_docs = []

        # First, extract tables using Upstage DocumentParser
        table_docs = self.parse_pdf_with_upstage_for_tables(pdf_path)

        # Debug: Print table_docs information
        print(f"🔍 Debug: table_docs length = {len(table_docs)}")
        for i, doc in enumerate(table_docs):
            print(f"🔍 Debug: Table {i+1} content length = {len(doc.page_content)}")
            print(f"🔍 Debug: Table {i+1} metadata = {doc.metadata}")

        # Save table content to table.md (always try to save, even if empty)
        self.save_tables_to_markdown(table_docs, pdf_path)

        # Then, use LlamaParse for text and image extraction
        try:
            print("📝 Using LlamaParse for text and image extraction...")

            # Initialize LlamaParse
            parser = LlamaParse(
                api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
                result_type="markdown",  # Can be "markdown" or "text"
                parsing_instruction="Extract text and images. Skip tables as they will be processed separately.",
                language="ko",
                verbose=True,
            )

            # Parse the document
            parsed_docs = parser.load_data(pdf_path)

            # Convert to LangChain Document format
            for doc in parsed_docs:
                # LlamaParse returns documents with text attribute
                content = doc.text if hasattr(doc, "text") else str(doc)

                # Create LangChain Document
                langchain_doc = Document(
                    page_content=content,
                    metadata=doc.metadata if hasattr(doc, "metadata") else {},
                )

                # Classify content (skip table detection as we use Upstage for tables)
                content_lower = content.lower()
                if "![" in content or "image" in content_lower:
                    langchain_doc.metadata["type"] = "image"
                    langchain_doc.metadata["source"] = "llamaparse"
                    image_docs.append(langchain_doc)
                # Skip table detection for LlamaParse content
                elif "|" not in content or content.count("|") <= 3:
                    langchain_doc.metadata["type"] = "text"
                    langchain_doc.metadata["source"] = "llamaparse"
                    text_docs.append(langchain_doc)

            # If no specific categorization, treat all as text
            if not text_docs and not image_docs and not table_docs:
                text_docs = [
                    Document(
                        page_content=doc.text if hasattr(doc, "text") else str(doc),
                        metadata={
                            **(doc.metadata if hasattr(doc, "metadata") else {}),
                            "type": "text",
                            "source": "llamaparse",
                        },
                    )
                    for doc in parsed_docs
                ]

            print(
                f"✅ LlamaParse extracted: Text({len(text_docs)}) | Images({len(image_docs)})"
            )

        except Exception as e:
            print(f"Error parsing PDF with LlamaParse: {e}")
            # Fallback to basic text extraction if LlamaParse fails
            from langchain_community.document_loaders import PyPDFLoader

            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            for doc in documents:
                doc.metadata["type"] = "text"
                doc.metadata["source"] = "pypdf"
            text_docs = documents

        # Save parsed content to load_data.md
        self.save_to_markdown(text_docs, image_docs, table_docs, pdf_path)

        return text_docs, image_docs, table_docs

    def process_pdf(self, file_path: str) -> Dict[str, Any]:
        """
        Process uploaded PDF file and create vector store.

        Args:
            file_path: Path to the uploaded PDF file

        Returns:
            Dictionary with processing results
        """
        try:
            # Parse PDF
            self.text_docs, self.image_docs, self.table_docs = self.parse_pdf(file_path)

            # Combine all documents
            all_docs = self.text_docs + self.image_docs + self.table_docs

            if not all_docs:
                return {
                    "success": False,
                    "error": "No content extracted from PDF",
                    "text_count": 0,
                    "image_count": 0,
                    "table_count": 0,
                }

            # Split documents into chunks
            # Tables and images should remain as single chunks when possible
            split_docs = []

            # Define max tokens for single chunks (roughly 4 chars per token)
            MAX_CHUNK_SIZE = 50000  # ~12,500 tokens max for images/tables

            # Process text documents - split into chunks
            for doc in self.text_docs:
                chunks = self.text_splitter.split_documents([doc])
                split_docs.extend(chunks)

            # Process image documents - keep as single chunks if not too large
            for doc in self.image_docs:
                doc.metadata["type"] = "image"
                if len(doc.page_content) > MAX_CHUNK_SIZE:
                    # For large images, use a larger chunk size splitter
                    image_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=MAX_CHUNK_SIZE,
                        chunk_overlap=1000,
                        length_function=len,
                        separators=["\n\n", "\n", ".", " ", ""],
                    )
                    image_chunks = image_splitter.split_documents([doc])
                    for idx, chunk in enumerate(image_chunks):
                        chunk.metadata["type"] = "image"
                        chunk.metadata["part"] = f"{idx+1}/{len(image_chunks)}"
                    split_docs.extend(image_chunks)
                else:
                    split_docs.append(doc)

            # Process table documents - keep as single chunks if not too large
            for doc in self.table_docs:
                doc.metadata["type"] = "table"
                if len(doc.page_content) > MAX_CHUNK_SIZE:
                    # For large tables, use a larger chunk size splitter
                    table_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=MAX_CHUNK_SIZE,
                        chunk_overlap=1000,
                        length_function=len,
                        separators=["\n\n", "\n", "|", " ", ""],
                    )
                    table_chunks = table_splitter.split_documents([doc])
                    for idx, chunk in enumerate(table_chunks):
                        chunk.metadata["type"] = "table"
                        chunk.metadata["part"] = f"{idx+1}/{len(table_chunks)}"
                    split_docs.extend(table_chunks)
                else:
                    split_docs.append(doc)

            # Create vector store
            if os.path.exists(self.vector_store_path):
                shutil.rmtree(self.vector_store_path)

            self.vectorstore = Chroma.from_documents(
                documents=split_docs,
                embedding=self.embeddings,
                persist_directory=self.vector_store_path,
            )

            # Create retriever - increased k from 4 to 6 for more context
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity", search_kwargs={"k": 6}
            )

            # Create QA chain
            prompt_template = """주어진 컨텍스트를 바탕으로 질문에 답변해주세요.
            
컨텍스트:
{context}

질문: {question}

답변:"""

            PROMPT = PromptTemplate(
                template=prompt_template, input_variables=["context", "question"]
            )

            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT},
            )

            return {
                "success": True,
                "text_count": len(self.text_docs),
                "image_count": len(self.image_docs),
                "table_count": len(self.table_docs),
                "total_chunks": len(split_docs),
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "text_count": 0,
                "image_count": 0,
                "table_count": 0,
            }

    def similarity_search(self, query: str, k: int = 4) -> List[str]:
        """
        Perform similarity search on the vector store.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of similar documents
        """
        if not self.vectorstore:
            return ["Vector store not initialized. Please upload a PDF first."]

        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            results = []
            for i, doc in enumerate(docs, 1):
                # Display full content of each document
                doc_type = doc.metadata.get("type", "text") if doc.metadata else "text"
                source = (
                    doc.metadata.get("source", "unknown") if doc.metadata else "unknown"
                )
                result_text = f"**Result {i} - {doc_type.upper()} Document (Source: {source}):**\n"
                result_text += (
                    f"{doc.page_content}\n"  # Full content without truncation
                )
                if doc.metadata:
                    result_text += f"\n*Metadata: {doc.metadata}*"
                results.append(result_text)
            return results
        except Exception as e:
            return [f"Error during search: {str(e)}"]

    def rag_query(self, query: str) -> str:
        """
        Perform RAG query using the QA chain.

        Args:
            query: User question

        Returns:
            Generated answer
        """
        if not self.qa_chain:
            return "QA chain not initialized. Please upload a PDF first."

        try:
            result = self.qa_chain.invoke({"query": query})

            answer = result.get("result", "No answer generated")
            sources = result.get("source_documents", [])

            # Format response with sources
            response = f"**답변:**\n{answer}\n\n"

            if sources:
                response += "=" * 80 + "\n"
                response += "**📚 참조 문서 (전체 내용):**\n"
                response += "=" * 80 + "\n\n"

                # Display ALL source documents with FULL content
                for i, doc in enumerate(sources, 1):
                    doc_type = (
                        doc.metadata.get("type", "text") if doc.metadata else "text"
                    )
                    source = (
                        doc.metadata.get("source", "unknown")
                        if doc.metadata
                        else "unknown"
                    )
                    response += f"\n{'─' * 70}\n"
                    response += f"**[문서 {i}/{len(sources)}] {doc_type.upper()} TYPE (Parser: {source})**\n"
                    response += f"{'─' * 70}\n\n"

                    # Full document content without any truncation
                    response += f"{doc.page_content}\n\n"

                    if doc.metadata:
                        response += f"📌 *Metadata: {doc.metadata}*\n"
                    response += f"{'─' * 70}\n\n"

                response += "=" * 80 + "\n"
                response += f"**총 {len(sources)}개의 참조 문서가 사용되었습니다.**\n"

            return response

        except Exception as e:
            return f"Error during RAG query: {str(e)}"


# Gradio Interface
def create_gradio_interface():
    """Create and configure Gradio interface."""

    # Initialize RAG service
    rag_service = PDFRAGService()

    def process_pdf_file(file):
        """Process uploaded PDF file."""
        if file is None:
            return "No file uploaded", "", "", "", ""

        # Process the PDF
        result = rag_service.process_pdf(file.name)

        if result["success"]:
            status = f"✅ PDF processed successfully!\n"
            status += f"Total chunks created: {result.get('total_chunks', 0)}\n"
            status += f"📊 Tables extracted with Upstage DocumentParser\n"
            status += f"📝 Text & Images extracted with LlamaParse"

            text_info = f"Text Documents: {result['text_count']}"
            image_info = f"Image Documents: {result['image_count']}"
            table_info = f"Table Documents: {result['table_count']}"

            return status, text_info, image_info, table_info, ""
        else:
            return f"❌ Error: {result['error']}", "", "", "", ""

    def search_similar(query):
        """Perform similarity search."""
        if not query:
            return "Please enter a search query."

        results = rag_service.similarity_search(query)
        return "\n\n---\n\n".join(results)

    def rag_answer(query):
        """Generate RAG answer."""
        if not query:
            return "Please enter a question."

        return rag_service.rag_query(query)

    def get_upstage_html():
        """Get the Upstage HTML output."""
        if not rag_service.upstage_html_content:
            return "<p>No Upstage HTML content available. Please upload and process a PDF first.</p>"
        return rag_service.upstage_html_content

    def get_extracted_tables():
        """Get only the extracted tables."""
        if not rag_service.extracted_tables_html:
            return "<p>No tables extracted. Please upload and process a PDF first.</p>"
        return rag_service.extracted_tables_html

    def clear_vectorstore():
        """Clear the vector store and reset."""
        if os.path.exists(rag_service.vector_store_path):
            shutil.rmtree(rag_service.vector_store_path)
        rag_service.vectorstore = None
        rag_service.retriever = None
        rag_service.qa_chain = None
        rag_service.text_docs = []
        rag_service.image_docs = []
        rag_service.table_docs = []
        rag_service.upstage_html_content = ""
        rag_service.extracted_tables_html = ""
        return (
            "Vector store cleared!",
            "",
            "",
            "",
            "",
            "",
            "",
            "<p>Cleared</p>",
            "<p>Cleared</p>",
        )

    # Create Gradio interface
    with gr.Blocks(title="PDF RAG Service with Hybrid Parsing") as app:
        gr.Markdown("# 📚 PDF RAG Service with Hybrid Parsing")
        gr.Markdown(
            "Upload a PDF file to create a RAG system using:\n"
            "- **Upstage DocumentParser** for accurate table extraction\n"
            "- **LlamaParse** for text and image processing"
        )

        with gr.Row():
            with gr.Column(scale=1):
                # File upload section
                pdf_file = gr.File(
                    label="Upload PDF", file_types=[".pdf"], type="filepath"
                )
                process_btn = gr.Button("Process PDF", variant="primary")
                clear_btn = gr.Button("Clear Vector Store", variant="secondary")

                # Status display
                status_text = gr.Textbox(
                    label="Processing Status", lines=4, interactive=False
                )

                # Document count display
                with gr.Row():
                    text_count = gr.Textbox(label="📄 Text", interactive=False, scale=1)
                    image_count = gr.Textbox(
                        label="🖼️ Images", interactive=False, scale=1
                    )
                    table_count = gr.Textbox(
                        label="📊 Tables", interactive=False, scale=1
                    )

            with gr.Column(scale=2):
                # Search section
                with gr.Tab("Similarity Search"):
                    search_input = gr.Textbox(
                        label="Search Query",
                        placeholder="Enter search terms...",
                        lines=1,
                    )
                    search_btn = gr.Button("Search", variant="primary")
                    # Changed from Markdown to Textbox for better handling of long content
                    search_output = gr.Textbox(
                        label="Search Results",
                        value="",
                        lines=30,  # Increased lines for full content display
                        max_lines=50,
                        interactive=False,
                    )

                # RAG Q&A section
                with gr.Tab("RAG Q&A"):
                    rag_input = gr.Textbox(
                        label="Question",
                        placeholder="Ask a question about the document...",
                        lines=2,
                    )
                    rag_btn = gr.Button("Get Answer", variant="primary")
                    # Changed from Markdown to Textbox for better handling of long content
                    rag_output = gr.Textbox(
                        label="Answer & Full Reference Documents",
                        value="",
                        lines=40,  # Increased lines for full content display
                        max_lines=100,
                        interactive=False,
                    )

                # Upstage HTML Output section
                with gr.Tab("Upstage HTML Output"):
                    upstage_refresh_btn = gr.Button(
                        "🔄 Refresh HTML Content", variant="secondary"
                    )
                    upstage_html_output = gr.HTML(
                        label="Upstage Document Parser HTML Output",
                        value="<p>Upload and process a PDF to see Upstage HTML output.</p>",
                    )

                # Tables Only section
                with gr.Tab("Tables Only"):
                    tables_refresh_btn = gr.Button(
                        "🔄 Refresh Tables", variant="secondary"
                    )
                    tables_html_output = gr.HTML(
                        label="Extracted Tables from Document",
                        value="<p>Upload and process a PDF to see extracted tables.</p>",
                    )

        # Event handlers
        process_btn.click(
            fn=process_pdf_file,
            inputs=[pdf_file],
            outputs=[status_text, text_count, image_count, table_count, search_output],
        ).then(fn=get_upstage_html, inputs=[], outputs=[upstage_html_output]).then(
            fn=get_extracted_tables, inputs=[], outputs=[tables_html_output]
        )

        search_btn.click(
            fn=search_similar, inputs=[search_input], outputs=[search_output]
        )

        rag_btn.click(fn=rag_answer, inputs=[rag_input], outputs=[rag_output])

        clear_btn.click(
            fn=clear_vectorstore,
            inputs=[],
            outputs=[
                status_text,
                text_count,
                image_count,
                table_count,
                search_output,
                rag_output,
                search_input,
                upstage_html_output,
                tables_html_output,
            ],
        )

        # Upstage HTML refresh button
        upstage_refresh_btn.click(
            fn=get_upstage_html, inputs=[], outputs=[upstage_html_output]
        )

        # Tables refresh button
        tables_refresh_btn.click(
            fn=get_extracted_tables, inputs=[], outputs=[tables_html_output]
        )

        # Add examples
        gr.Examples(
            examples=[
                ["What is the main topic of this document?"],
                ["Summarize the key findings"],
                ["What are the conclusions?"],
                ["Show me all tables in this document"],
                ["What data is presented in the tables?"],
            ],
            inputs=rag_input,
        )

    return app


if __name__ == "__main__":
    # Create and launch the app
    app = create_gradio_interface()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, debug=True)
