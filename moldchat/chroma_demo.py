# import
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# load the document and split it into chunks
loader = TextLoader("all.txt")
documents = loader.load()

# split it into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
docs = text_splitter.split_documents(documents)

# create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="/data/usr/jy/asset/tokenizer/m3e-base")

# load it into Chroma
db = Chroma.from_documents(docs, embedding_function, persist_directory="./chroma_db")
db = Chroma(persist_directory="./chroma_demo_db", embedding_function=embedding_function)

# query it
query = "抛光树脂"
docs = db.similarity_search(query)

# print results
print(docs[0].page_content)