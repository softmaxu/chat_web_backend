import torch, jieba
import logging
from modelscope import GenerationConfig, snapshot_download
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import json,string,time,os,re
import shutil
import numpy as np
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextStreamer
from langchain.prompts import PromptTemplate

time_str=str(int(time.time()))
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class RAG_Chroma:
    def __init__(self, db_path="./chroma_db/yeya", embedding_path="/data/usr/jy/asset/tokenizer/m3e-base") -> None:
        self.db_path=db_path
        self.embedding_function = SentenceTransformerEmbeddings (model_name = embedding_path)
        if not os.path.exists(self.db_path):
            self.create()
        else:
            self.db=Chroma(persist_directory=self.db_path, embedding_function = self.embedding_function)

    def create(self, file_path:str="yeya-text12456.txt", chunk_size=500, chunk_overlap=50):
        loader = TextLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n",":","：",".","。",";","；"])
        docs = text_splitter.split_documents(documents)
        self.db = Chroma.from_documents(docs, self.embedding_function, persist_directory=self.db_path)
        self.db.persist()

    def query(self, query:str):
        docs = self.db.similarity_search(query)
        return docs
    
    def delete(self):
        if os.path.exists(self):
            # 使用shutil.rmtree删除目录及其所有内容
            shutil.rmtree(self.db_path)
            print(f"数据库目录 {self.db_path} 已被删除。")
            return True
        else:
            print(f"数据库目录 {self.db_path} 不存在。")
        return False

class LLM_Adapter:
    def __init__(self, model_dir, model_name, system_msg, rag_prompt) -> None:
        self.model_dir=model_dir
        self.model_name=model_name
        self.model_path=os.path.join(model_dir,model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, device_map="cuda:0", 
                              trust_remote_code=True, torch_dtype=torch.float16, )
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, device_map="cuda:0", 
                              trust_remote_code=True, torch_dtype=torch.float16,)
        self.model.generation_config = GenerationConfig.from_pretrained(self.model_path)
        self.system_msg=system_msg
        self.rag_prompt=rag_prompt
        self.rag=RAG_Chroma()
        
    
    def predict(self, message):
        messages=self._convert_format(message)
        use_RAG=messages[-1].get("use_RAG",False)
        if "use_RAG" not in messages[-1]:
            logging.warning("key use_RAG doesn't exist in messages[-1]")
        logging.debug(f"use_RAG {use_RAG}")
        logging.debug(f"messages {messages}")
        rag_to_user=None
        if use_RAG:
            query=messages[-1]["content"]
            rag_docs=self.rag.query(query)[0].page_content
            rag_to_user="在知识库中检索到以下内容：\r\n"+rag_docs
            rag_prompt=self.rag_prompt.format(rag_docs=rag_docs)
            print(rag_prompt)
            messages.insert(-1,self._gen_chat_msg("assistant",rag_prompt))
            logging.info(messages[-2:])
        return rag_to_user, self.model.chat(self.tokenizer, messages, stream=True)
    
    def _gen_chat_msg(self, role, content):
        return {"role": role, "content": content}
    
    def _convert_format(self, messages, max_len=10):
        res=[]
        if (type(messages) is list):
            res.append(self._gen_chat_msg("system",self.system_msg))
            for m in messages[-max_len:]:
                if m["type"]=="sent":
                    res.append(self._gen_chat_msg("user",m["text"]))
                elif m["type"]=="received":
                    res.append(self._gen_chat_msg("assistant",m["text"]))
            page_name=messages[-1]["page"]
            if page_name=="聊天对话":
                res[-1]["use_RAG"]=False
            elif page_name=="知识库问答":
                res[-1]["use_RAG"]=True
        return res
        



