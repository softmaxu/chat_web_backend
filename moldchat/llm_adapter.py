import torch, jieba
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
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

time_str=str(int(time.time()))

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



class LLM_Adapter:
    def __init__(self, model_dir, model_name) -> None:
        self.model_dir=model_dir
        self.model_name=model_name
        self.model_path=os.path.join(model_dir,model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, device_map="cuda:0", 
                              trust_remote_code=True, torch_dtype=torch.float16, )
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, device_map="cuda:0", 
                              trust_remote_code=True, torch_dtype=torch.float16,)
        self.model.generation_config = GenerationConfig.from_pretrained(self.model_path)
        self.streamer = TextStreamer(tokenizer=self.tokenizer)
    
    def predict(self, message):
        messages=self._convert_format(message)
        return self.model.chat(self.tokenizer, messages)
    
    def _gen_chat_msg(self, role, content):
        return {"role": role, "content": content}
    
    def _convert_format(self, messages):
        res=[]
        if (type(messages) is list):
            res.append(self._gen_chat_msg("system",""))
            for m in messages:
                if m["type"]=="sent":
                    res.append(self._gen_chat_msg("user",m["text"]))
                elif m["type"]=="received":
                    res.append(self._gen_chat_msg("assistant",m["text"]))
        return res
        

# 微调后的模型路径
# model_dir = '/data/usr/jy/Baichuan2/fine-tune/mymodel/'
# # 微调后的模型名
# model_name = 'bc2_7b_chat_qav2_e10'

# model.generation_config = GenerationConfig.from_pretrained("baichuan-inc/Baichuan2-7B-Chat")

