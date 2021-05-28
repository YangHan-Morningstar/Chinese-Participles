# coding=utf-8
import tensorflow as tf
from transformer import *
from utils import *
import os
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import metrics


def decode(text, num_list, idx2tag):
    text_participle = ""
    for i in range(len(text)):
        if idx2tag[num_list[i]] == "S" or idx2tag[num_list[i]] == "E":
            text_participle += text[i] + "  "
        else:
            text_participle += text[i]
    return text_participle


def evaluate(inp_sentence, transformer, char2idx, idx2tag):

    input_idx = []
    for char in inp_sentence:
        if char in char2idx:
            input_idx.append(char2idx[char])
        else:
            input_idx.append(char2idx["UNK"])

    encoder_input = tf.expand_dims(input_idx, 0)

    enc_padding_mask = create_masks(encoder_input)

    predictions = transformer(encoder_input,
                              False,
                              enc_padding_mask)

    predicted_ids = tf.cast(tf.argmax(predictions, axis=-1), tf.int32).numpy().tolist()[0]

    return decode(inp_sentence, predicted_ids, idx2tag)


if __name__ == "__main__":

    char2idx = load_dict("./dicts/char2idx.json")
    tag2idx = load_dict("./dicts/tag2idx_for_transformer.json")

    num_layers = 4
    d_model = 128
    dff = 512
    num_heads = 8

    input_vocab_size = len(char2idx)
    target_vocab_size = len(tag2idx)
    dropout_rate = 0.1
    transformer = Transformer(num_layers, d_model, num_heads, dff,
                              input_vocab_size, target_vocab_size,
                              pe_input=input_vocab_size,
                              rate=dropout_rate)

    ckpt = tf.train.Checkpoint(transformer=transformer)
    ckpt.restore("./transformer_checkpoints/train/ckpt-4")

    idx2tag = {num: tag for tag, num in tag2idx.items()}

    # while True:
    #     print("请输入：")
    #     sentence = input()
    #
    #     result = evaluate(sentence,
    #                       transformer,
    #                       char2idx,
    #                       idx2tag)
    #
    #     print(result)

    # 计算评价指标
    test_dic = load_data("./corpus/msr_test_gold.txt")
    real = [content["label"] for content in test_dic]
    prediction = []
    for dic in test_dic:
        text_content = dic["text"]
        text_idx = []
        for char in text_content:
            if char in char2idx:
                text_idx.append(char2idx[char])
            else:
                text_idx.append(char2idx["UNK"])

        encoder_input = tf.expand_dims(text_idx, 0)
        enc_padding_mask = create_masks(encoder_input)

        predictions = transformer(encoder_input,
                                  False,
                                  enc_padding_mask)

        predicted_ids = tf.cast(tf.argmax(predictions, axis=-1), tf.int32).numpy().tolist()[0]
        predicted_ids = [i - 1 for i in predicted_ids]
        print(predicted_ids)
        prediction.append(predicted_ids)

    real_for_report = MultiLabelBinarizer().fit_transform(real)
    predictions_for_report = MultiLabelBinarizer().fit_transform(prediction)
    class_list = [tag for tag, idx in tag2idx.items() if idx != 0]
    report = metrics.classification_report(real_for_report,
                                           predictions_for_report,
                                           target_names=class_list,
                                           digits=4)

    print(report)

