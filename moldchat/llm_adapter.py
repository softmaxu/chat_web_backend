import torch, jieba
from transformers import AutoModelForCausalLM, AutoTokenizer
from modelscope import GenerationConfig, snapshot_download
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import json,string,time,os,re
import numpy as np
time_str=str(int(time.time()))

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
    
    def predict(self, message):
        messages=self._convert_format(message)
        return self.model.chat(self.tokenizer, messages)
    
    def _gen_chat_msg(self, role, content):
        return {"role": role, "content": content}
    
    def _convert_format(self, message):
        messages=[]
        if (type(message) is str):
            messages.append(self._gen_chat_msg("system",""))
            messages.append(self._gen_chat_msg("user","今天是几号"))
            messages.append(self._gen_chat_msg("assistant","今天是3月10日"))
            messages.append(self._gen_chat_msg("user",message))
        return messages
        

# 微调后的模型路径
# model_dir = '/data/usr/jy/Baichuan2/fine-tune/mymodel/'
# # 微调后的模型名
# model_name = 'bc2_7b_chat_qav2_e10'

# model.generation_config = GenerationConfig.from_pretrained("baichuan-inc/Baichuan2-7B-Chat")

