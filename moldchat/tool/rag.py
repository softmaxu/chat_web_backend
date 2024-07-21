from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

class RagAdapter:
    def __init__(self, model_name, docs_path, db_path):
        self.model_name = model_name
        self.db_path = db_path
        self.loader = DirectoryLoader(docs_path, glob="*")
        self.embedding_function = SentenceTransformerEmbeddings(model_name=model_name)
        self.db = Chroma(persist_directory=db_path, embedding_function=self.embedding_function)

    def load_documents(self):
        return self.loader.load()

    def split_documents(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
        return text_splitter.split_documents(documents)

    def create_chroma_db(self, documents):
        self.db.from_documents(documents)

    def query_chroma_db(self, query):
        return self.db.similarity_search(query)

    def get_page_content(self, doc):
        return doc.page_content

    def get_file_path(self, doc):
        return doc.file_path