from transformers import AutoTokenizer, TFAutoModel
import os

# 设置环境变量
os.environ['HF_DATASETS_OFFLINE'] = "1"
os.environ['TRANSFORMERS_OFFLINE'] = "1"

tokenizer = AutoTokenizer.from_pretrained("E:\\hugging_face\\bert-base-chinese")

model = TFAutoModel.from_pretrained("E:\\hugging_face\\bert-base-chinese")

input_ids = tokenizer.encode("春眠不觉晓", return_tensors="tf")

embeddings = model(input_ids)
print(embeddings[0][0][0])
