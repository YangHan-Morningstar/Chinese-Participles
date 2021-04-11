from utils import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import numpy as np


class BiGRU(object):
    def __init__(self, char_dict_filepath, tag_dict_filepath, max_length=128, hidden_size=256, embedding_dim=256, dropout_rate=0.5):
        super(BiGRU, self).__init__()
        self.char2idx = load_dict(char_dict_filepath)
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}
        self.tag2idx = load_dict(tag_dict_filepath)
        self.idx2tag = {idx: tag for tag, idx in self.tag2idx.items()}
        self.vocab_size = len(self.char2idx)
        self.tag_vocab_size = len(self.tag2idx)
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.first_bilstm_layers_unit = hidden_size
        self.second_bilstm_layers_unit = int(hidden_size / 2)
        self.dropout_rate = dropout_rate

    def prepare_data(self, source_data_filepath):
        train_data = load_data(source_data_filepath)
        x, y = [], []
        for dict_content in train_data:
            text, label = dict_content["text"], dict_content["label"]
            x_cache = [self.char2idx[char] for char in text]

            x.append(x_cache)
            y.append(label)
        x = pad_sequences(x, maxlen=self.max_length, padding="post")
        y = to_categorical(y, num_classes=self.tag_vocab_size)

        return x, y

    def train(self, source_data_file_path, batch_size, epochs):
        x, y = self.prepare_data(source_data_file_path)
        ck_callback = ModelCheckpoint('./rnn-dense_weights.{epoch:02d}-{categorical_accuracy:.4f}.hdf5',
                                      monitor="categorical_accuracy",
                                      mode="max", verbose=2,
                                      save_best_only=True,
                                      save_weights_only=True)
        rnn_model = self.get_model()
        rnn_model.compile(optimizer="rmsprop",
                          loss="categorical_crossentropy",
                          metrics=["categorical_accuracy"])
        rnn_model.summary()
        rnn_model.fit(x=x, y=y, batch_size=batch_size, epochs=epochs, callbacks=ck_callback)

    def get_model(self):
        text_inputs = tf.keras.layers.Input(shape=(None, ))
        text_embeddings_outputs = tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim)(text_inputs)
        bilstm_ouptuts_1 = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(
                self.first_bilstm_layers_unit, return_sequences=True))(text_embeddings_outputs)
        dropout1_outputs = tf.keras.layers.Dropout(self.dropout_rate)(bilstm_ouptuts_1)
        bilstm_ouptuts_2 = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(
                self.second_bilstm_layers_unit, return_sequences=True))(dropout1_outputs)
        dropout2_outputs = tf.keras.layers.Dropout(self.dropout_rate)(bilstm_ouptuts_2)
        dense1_outputs = tf.keras.layers.Dense(self.tag_vocab_size, activation="softmax")(dropout2_outputs)
        model = tf.keras.Model(inputs=text_inputs, outputs=dense1_outputs)

        model.summary()

        return model

    def test(self, weights_filepath):
        rnn_model = self.get_model()
        rnn_model.summary()
        rnn_model.load_weights(weights_filepath)

        while True:
            print("请输入文本：")
            text_content = input()
            text_idx = []
            for char in text_content:
                if char in self.char2idx:
                    text_idx.append(self.char2idx[char])
                else:
                    text_idx.append(self.char2idx["UNK"])
            text_inputs = np.array([text_idx])
            predictions = rnn_model.predict(x=text_inputs)
            predictions_num_list = np.argmax(predictions, axis=-1)[0]

            print(self.decode(text_content, predictions_num_list))

    def decode(self, text, num_list):
        text_participle = ""
        for i in range(len(text)):
            if self.idx2tag[num_list[i]] == "S" or self.idx2tag[num_list[i]] == "E":
                text_participle += text[i] + "  "
            else:
                text_participle += text[i]
        return text_participle


if __name__ == "__main__":
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    rnn_model = BiGRU("./dicts/char2idx.json", "dicts/tag2idx_for_dl.json")
    # 训练
    rnn_model.train("./corpus/msr_training_data_processed_for_dl.txt", batch_size=32, epochs=10)

    # 测试
    # rnn_model.test("./rnn-dense_weights.10-0.9926.hdf5")