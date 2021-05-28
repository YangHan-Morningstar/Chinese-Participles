import tensorflow as tf
from utils import *
import numpy as np


data = load_data("./corpus/msr_training_data_processed.txt")
char2idx = load_dict("./dicts/char2idx.json")
tag2idx = load_dict("./dicts/tag2idx_for_transformer.json")
count = 0

x, y = [], []

for dic in data:
    text_content, label = dic["text"], dic["label"]
    x_list, y_list = [], []
    for char in text_content:
        if char in char2idx:
            x_list.append(char2idx[char])
        else:
            x_list.append(char2idx["UNK"])
    for tag in label:
        y_list.append(tag2idx[tag])

    x.append(x_list)
    y.append(y_list)

    print(str(count) + " has been appended!")

    count += 1


x = tf.keras.preprocessing.sequence.pad_sequences(x, dtype="int64", padding="post")
y = tf.keras.preprocessing.sequence.pad_sequences(y, dtype="int64", padding="post")
assert x.shape == y.shape

for i in range(len(x)):
    np.savez("./corpus/prepared_data/" + str(i) + ".npz", x=x[i], y=y[i])
    print(str(i) + " has been saved!")
