import ollama
import chromadb

# ChromaDB 설정
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="test_embeddings")


class TestRAG:
    def __init__(self, model='gemma2'):
        self.model = model
        self.context = ""  # 문서 내용을 저장할 변수

    def embed_and_store(self, text):
        """텍스트를 임베딩 후 ChromaDB에 저장"""
        response = ollama.embeddings(model=self.model, prompt=text)
        embedding = response['embedding']

        # ChromaDB에 저장
        collection.add(
            ids=[str(len(collection.get()['ids']) + 1)],  # 고유 ID
            embeddings=[embedding],
            metadatas=[{"text": text}]
        )
        self.context = text  # 문서 내용을 저장하여 챗봇에 활용

    def embed_file(self, path):
        """파일을 읽어서 임베딩 후 저장"""
        with open(path, 'r', encoding='utf-8') as file:
            text = file.read()
        self.embed_and_store(text)  # 문서를 임베딩하고 저장

    def chat(self):
        """문서를 기반으로 대화"""
        print("📖 문서 기반 챗봇 시작! 'exit'을 입력하면 종료됩니다.")

        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                print("----CLOSED----.")
                break

            # 문서 내용을 컨텍스트로 추가하여 대화 요청
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant. Use the given document as context."},
                {"role": "system", "content": f"Document Context: {self.context}"},
                {"role": "user", "content": user_input}
            ]

            response = ollama.chat(model=self.model, messages=messages)
            print(f"Gemma: {response['message']['content']}")


if __name__ == '__main__':
    test_rag = TestRAG()
    test_rag.embed_file('embed_test.txt')  # 문서를 임베딩 후 저장
    test_rag.chat()  # 챗봇 시작
