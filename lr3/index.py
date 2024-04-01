import numpy as np
from tqdm import tqdm
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel

strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0","/gpu:1"],cross_device_ops=tf.distribute.ReductionToOneDevice())
with strategy.scope():
    tokenizer = AutoTokenizer.from_pretrained("E:\\hugging_face\\bert-base-chinese")
    labels = []
    token_list = []
    with open('../data/ChnSentiCorp.txt', mode="r", encoding="UTF-8") as file:
        for line in tqdm(file.readlines()):
            line = line.strip().split(",")
            labels.append(int(line[0]))
            text = line[1]
            token = tokenizer.encode(text, truncation=True)
            token = token[:128] + [0] * (128 - len(token))
            token_list.append(token)
    labels = np.array(labels)
    token_list = np.array(token_list)
    model = TFAutoModel.from_pretrained("E:\\hugging_face\\bert-base-chinese")
    input_token = tf.keras.Input(shape=(128,), dtype=tf.int32, name="input1")
    embedding = model(input_token)[0]
    embedding = tf.keras.layers.Flatten()(embedding)
    output = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid, name="output")(embedding)
    model = tf.keras.Model(inputs=input_token, outputs=output)

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='binary_crossentropy',#loss=tf.keras.losses.binary_crossentropy(),
                  metrics=['accuracy'])

    model.fit(token_list, labels, batch_size=16, epochs=10)

#%%
