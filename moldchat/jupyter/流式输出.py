from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import os

model_dir = "/data/usr/jy/Langchain-Chatchat/webui_pages/fine_tune/final_model/"
model_name = "user_模型2401" 
model_path = os.path.join(model_dir, model_name)
tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path,trust_remote_code=True)

# Tokenized
text = "详细介绍一下模具是什么"
inputs = tokenizer(text, return_tensors="pt")
inputs.pop("token_type_ids", None)
streamer = TextStreamer(tokenizer=tokenizer)

# Generation
# model.generate(**inputs, streamer=streamer, max_new_tokens=5000)

res=model.chat(tokenizer, [{"role": "user", "content": text}])
print(res)