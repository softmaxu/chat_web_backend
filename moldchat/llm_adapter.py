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
    def __init__(self, path="./chroma_db", embedding_path="/data/usr/jy/asset/m3e-base") -> None:
        self.path=path
        self.embedding_path=embedding_path

    def create(self, path:str="all.txt", chunk_size=500, chunk_overlap=50):
        loader = TextLoader(path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        docs = text_splitter.split_documents(documents)
        embedding_function = SentenceTransformerEmbeddings(model_name=self.embedding_path)
        self.db = Chroma.from_documents(docs, embedding_function)

    def query(self, query:str):
        docs = self.db.similarity_search(query)
        return docs
    
    def delete(self):
        if os.path.exists(self):
            # 使用shutil.rmtree删除目录及其所有内容
            shutil.rmtree(self.path)
            print(f"数据库目录 {self.path} 已被删除。")
            return True
        else:
            print(f"数据库目录 {self.path} 不存在。")
        return False

# class LangchainAdapter:
#     def __init__(self, model_id, max_new_tokens) -> None:
#         self.model_id=model_id
#         self.tokenizer=AutoTokenizer.from_pretrained(model_id,trust_remote_code=True)
#         self.model=AutoModelForCausalLM.from_pretrained(model_id)
#         self.pipe=pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, max_new_tokens=max_new_tokens)
#         self.hf=HuggingFacePipeline(pipeline=self.pipe)

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
        self.rag.create()
        
    
    def predict(self, message):
        messages=self._convert_format(message)
        use_RAG=messages[-1].get("use_RAG",False)
        if "use_RAG" not in messages[-1]:
            logging.warning("key use_RAG doesn't exist in messages[-1]")
        logging.debug(f"use_RAG {use_RAG}")
        logging.debug(f"messages {messages}")
        if use_RAG:
            query=messages[-1]["content"]
            rag_docs=self.rag.query(query)[0].page_content
            rag_to_user="在知识库中检索到以下内容：\r\n"+rag_docs
            a=a/0 # 未完成需要修改
            return rag_to_user
            rag_prompt=self.rag_prompt.format(rag_docs=rag_docs)
            messages.insert(-1,self._gen_chat_msg("assistant",rag_prompt))
            logging.info(messages[-2:])
        return self.model.chat(self.tokenizer, messages, stream=True)
    
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
        



