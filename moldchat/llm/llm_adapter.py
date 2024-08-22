import torch, jieba
import logging
from modelscope import GenerationConfig, snapshot_download
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import json,string,time,os,re
import shutil
import numpy as np
from langchain_community.document_loaders import TextLoader, DirectoryLoader
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
    def __init__(self, db_path="/data/usr/jy/chat_web/back/moldchat/chroma_db/yeya", embedding_path="/data/usr/jy/asset/tokenizer/m3e-base") -> None:
        self.db_path=db_path
        self.embedding_function = SentenceTransformerEmbeddings (model_name = embedding_path)
        if os.path.exists(self.db_path):
            self.db=Chroma(persist_directory=self.db_path, embedding_function = self.embedding_function)
            
    def _get_loader(self, file_path:str="yeya-text12456.txt", is_file_path_dir=False):
        if is_file_path_dir:
            loader = DirectoryLoader(file_path, glob="*")
        else:
            loader = TextLoader(file_path)
        return loader
    
    def _get_documents(self, loader, chunk_size, chunk_overlap, separators=["\n\n",":","：",".","。",";","；"]):
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=separators)
        docs = text_splitter.split_documents(docs)
        return docs

    def create(self, file_path, is_file_path_dir=False, chunk_size=300, chunk_overlap=30):
        print("create file_path", file_path)
        loader=self._get_loader(file_path, is_file_path_dir)
        docs=self._get_documents(loader, chunk_size, chunk_overlap)
        self.db = Chroma.from_documents(docs, self.embedding_function, persist_directory=self.db_path)
        self.db.persist()

    def query(self, query:str, top_k=4):
        docs = self.db.similarity_search(query, k=top_k)
        return docs
    
    def insert(self, file_path, is_file_path_dir=False, chunk_size=300, chunk_overlap=30):
        loader=self._get_loader(file_path, is_file_path_dir)
        docs=self._get_documents(loader, chunk_size, chunk_overlap)
        self.db.add_documents(docs)
        
    def select(self, keyword:str, limit : int=None):
        if keyword:
            res = self.db.get(where_document={"$contains": keyword}, limit=limit)
        else:
            res = self.db.get(limit=limit)
        return res
            
        
    def drop(self):
        if os.path.exists(self):
            # 使用shutil.rmtree删除目录及其所有内容
            shutil.rmtree(self.db_path)
            print(f"数据库目录 {self.db_path} 已被删除。")
            return True
        else:
            print(f"数据库目录 {self.db_path} 不存在。")
        return False

class LLM_Adapter:
    def __init__(self, model_dir, model_name, system_msg, rag_prompt, rag_db_dir) -> None:
        self.model_dir=model_dir
        self.model_name=model_name
        self.model_path=os.path.join(model_dir,model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, device_map="cuda:0", 
                              trust_remote_code=True, torch_dtype=torch.float16, )
        # 如果不量化，在这里填上device_map
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, 
                              trust_remote_code=True, torch_dtype=torch.float16,)
        self.model.generation_config = GenerationConfig.from_pretrained(self.model_path)
        self.model = self.model.quantize(8).cuda()  # 使用量化删掉device_map
        self.system_msg=system_msg
        self.rag_prompt=rag_prompt
        self.rag_db_directory=rag_db_dir
        
    
    def predict(self, message):
        messages=self._convert_format(message)
        use_RAG=messages[-1].get("use_RAG",False)
        rag_db_name=messages[-1].get("db_name",None)
        if not rag_db_name:
            rag_db_name="yeya"
        if use_RAG:
            rag_db_path=os.path.join(self.rag_db_directory, rag_db_name)
        if "use_RAG" not in messages[-1]:
            logging.warning("key use_RAG doesn't exist in messages[-1]")
        logging.debug(f"use_RAG {use_RAG}")
        logging.debug(f"messages {messages}")
        rag_to_user=None
        if use_RAG:
            query=messages[-1]["content"]
            print("rag_db_path", rag_db_path)
            self.rag=RAG_Chroma(db_path=rag_db_path)
            rag_docs=self.rag.query(query)[0].page_content
            rag_to_user=f"在知识库 {rag_db_name} 中检索到以下内容：\r\n"+rag_docs
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
                res[-1]["db_name"]=messages[-1]["rag_db"]
        return res
        



