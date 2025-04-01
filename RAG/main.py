# test용 web interface
import gradio as gr
import os

# ollama, ollama embedding 툴
from langchain_ollama import OllamaEmbeddings, OllamaLLM

# Vector DB - ChromaDB
from langchain_chroma import Chroma

# pdf load
from langchain_community.document_loaders import PyPDFLoader

# text splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

# DB <--> LLM 간 pipeline
from langchain.chains import RetrievalQA


class TestRAG:
    # PDF_PATH = "dk_UI_test.pdf"
    PDF_PATH = "for_RAG_test.pdf"
    DB_PATH = "./chroma_db"

    def __init__(self):
        self.qa_chain = None

    def load_pdf(self):
        loader = PyPDFLoader(self.PDF_PATH)
        return loader.load()

    def split_text(self, documents, chunk_size=1000, chunk_overlap=200):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return text_splitter.split_documents(documents)

    def ignore_utf_error(self, text):
        return text.encode("utf-8", errors="ignore").decode("utf-8")

    def initialize_vectorstore(self, documents):
        embedding_function = OllamaEmbeddings(model="gemma2")
        vectorstore = Chroma(embedding_function=embedding_function, persist_directory=self.DB_PATH)

        existing_docs = vectorstore.get(include=["metadatas"])

        existing_filenames = {meta.get("source", "") for meta in existing_docs.get("metadatas", []) if "source" in meta}

        if os.path.basename(self.PDF_PATH) in existing_filenames:
            print(f"VectorDB에 이미 존재하는 문서입니다: {self.PDF_PATH}")
            print(f"VetorDB 문서 목록: {existing_filenames}")
        else:
            print(f"* * * * NOW EMBEDDING - {self.PDF_PATH} * * * *")
            for document in documents:
                document.page_content = self.ignore_utf_error(document.page_content)
                document.metadata["source"] = os.path.basename(self.PDF_PATH)

            vectorstore.add_documents(documents)
            print(f"* * * * EMBEDDING SUCCESS - {self.PDF_PATH} * * * *")

        return vectorstore

    def setup_qa_chain(self, vectorstore):
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        llm = OllamaLLM(model="gemma2")

        if vectorstore._collection.count() > 0:
            return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
        return None

    def chat(self, query):
        if self.qa_chain is None:
            return "Vector store is empty."

        response = self.qa_chain.invoke({"query": query})
        return response["result"]

    def run(self):
        print("* * * * LOADING pdf * * * *")
        documents = self.load_pdf()
        docs = self.split_text(documents)

        print("* * * * INITIALIZING Vectorstore * * * *")
        vectorstore = self.initialize_vectorstore(docs)

        self.qa_chain = self.setup_qa_chain(vectorstore)

        iface = gr.Interface(fn=self.chat, inputs="text", outputs="text", title="Local RAG(test)")
        iface.launch()


if __name__ == "__main__":
    TestRAG().run()
