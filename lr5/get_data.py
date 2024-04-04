import json
import copy as copy
import numpy as np
from tqdm import tqdm

bert_model = "E:\\hugging_face\\bert-base-chinese"
import char_until
from transformers import AutoTokenizer

# 输入BERT选择的名称
tokenizer = AutoTokenizer.from_pretrained(bert_model)


# 这一段代码用来查找某个序列arr中特定的一个小片段span
def search(arr, span):
    start_index = -1
    span_length = len(span)
    for i in range((len(arr) - span_length)):
        if arr[i:i + span_length] == span:
            start_index = i
            break
    end_index = start_index + len(span)
    return start_index, end_index


p_entitys = ["", '丈夫', '上映时间', '主持人', '主演', '主角', '作曲', '作者', '作词', '出品公司', '出生地', '出生日期', '创始人', '制片人', '号', '嘉宾',
             '国籍', '妻子', '字', '导演', '所属专辑', '改编自', '朝代', '歌手', '母亲', '毕业院校', '民族', '父亲', '祖籍', '编剧', '董事长', '身高',
             '连载网站']
max_length = 300
token_list = []
label_list = []
with open('../data/train_data.json', 'r', encoding="UTF-8") as f:
    data = json.load(f)

    for line in tqdm(data):
        text = line["text"]
        new_spo_list = line["new_spo_list"]

        for spo in new_spo_list:

            s_entity = spo["s"]["entity"]
            p_entity = spo["p"]["entity"]
            o_entity = spo["o"]["entity"]

            # 对每个实体依赖关系进行显性标注，需要注意的是不能使用原始的text
            text_p = p_entity + "[SEP]" + text
            token = tokenizer.encode(text_p)
            label = [0] * len(token)

            # 获取
            p_entity_token = p_entitys.index(p_entity)
            s_label = (2 * p_entity_token) - 1
            o_label = (2 * p_entity_token)

            # 使用BERT编码器进行编码
            s_entity_token = tokenizer.encode(s_entity)[1:-1]
            o_entity_token = tokenizer.encode(o_entity)[1:-1]

            s_entity_start_index, s_entity_end_index = search(token, s_entity_token)
            if s_entity_start_index != -1:
                for j in range(s_entity_start_index, s_entity_end_index):
                    label[j] = s_label

            o_entity_start_index, o_entity_end_index = search(token, o_entity_token)
            if o_entity_start_index != -1:
                for j in range(o_entity_start_index, o_entity_end_index):
                    label[j] = o_label

            token_list.append(token)
            label_list.append(label)  # 其中p_entitys是获取到的所有实体类别


token_list = char_until.pad_sequences(token_list,maxlen=max_length)
label_list = char_until.pad_sequences(label_list, maxlen=max_length).astype(np.float)

train_length = len(label_list)
def generator(batch_size = 12):
    batch_num = train_length//batch_size

    seed = int(np.random.random()*5217)
    np.random.seed(seed);np.random.shuffle(token_list)
    np.random.seed(seed);np.random.shuffle(label_list)

    while 1:
        for i in range(batch_num):
            start = batch_size * i
            end = batch_size * (i + 1)

            yield token_list[start:end],label_list[start:end]
