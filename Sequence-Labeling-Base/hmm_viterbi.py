import numpy as np
from utils import *
from tqdm import tqdm
from sklearn import metrics
from sklearn.preprocessing import MultiLabelBinarizer


class HMM_Viterbi:
    def __init__(self, char2idx_path, tag2idx_path):
        self.char2idx = load_dict(char2idx_path)
        self.tag2idx = load_dict(tag2idx_path)
        self.idx2tag = {v: k for k, v in self.tag2idx.items()}
        self.tag_size = len(self.tag2idx)
        self.vocab_size = max([v for _, v in self.char2idx.items()]) + 1
        self.transition = np.zeros([self.tag_size,
                                    self.tag_size])
        self.emission = np.zeros([self.tag_size,
                                  self.vocab_size])
        self.pi = np.zeros(self.tag_size)
        self.epsilon = 1e-8

    def fit(self, train_dic_path):
        print("初始化...")
        train_dic = load_data(train_dic_path)

        self.estimate_transition_and_initial_probs(train_dic)
        self.estimate_emission_probs(train_dic)

        self.pi = np.log(self.pi)
        self.transition = np.log(self.transition)
        self.emission = np.log(self.emission)
        print("初始化完成!")

    def estimate_emission_probs(self, train_dic):
        print("计算发射概率矩阵...")
        for dic in tqdm(train_dic):
            for char, tag in zip(dic["text"], dic["label"]):
                self.emission[self.tag2idx[tag],
                              self.char2idx[char]] += 1
        self.emission[self.emission == 0] = self.epsilon
        self.emission /= np.sum(self.emission, axis=1, keepdims=True)

    def estimate_transition_and_initial_probs(self, train_dic):
        print("计算转移概率矩阵与初始化概率向量...")
        for dic in tqdm(train_dic):
            for i, tag in enumerate(dic["label"][:-1]):
                if i == 0:
                    self.pi[self.tag2idx[tag]] += 1
                curr_tag = self.tag2idx[tag]
                next_tag = self.tag2idx[dic["label"][i+1]]
                self.transition[curr_tag, next_tag] += 1
        self.transition[self.transition == 0] = self.epsilon
        self.transition /= np.sum(self.transition, axis=1, keepdims=True)
        self.pi[self.pi == 0] = self.epsilon
        self.pi /= np.sum(self.pi)

    def get_current_char_emission_prob(self, char):
        # 计算char对应的发射概率向量
        # 如果当前字属于未知, 则设为均匀分布
        char_token = self.char2idx.get(char, 0)
        if char_token == 0:
            return np.log(np.ones(self.tag_size) / self.tag_size)
        return np.ravel(self.emission[:, char_token])

    def predict(self, text):
        # 预测并打印出预测结果
        # 维特比算法解码
        if len(text) == 0:
            raise NotImplementedError("输入文本为空!")
        best_tags_id = self.viterbi_decode(text)

        # 输出分词结果
        text_participle = ""
        for i in range(len(text)):
            if self.idx2tag[best_tags_id[i]] == "S" or self.idx2tag[best_tags_id[i]] == "E":
                text_participle += text[i] + "  "
            else:
                text_participle += text[i]
        return text_participle, best_tags_id

    def viterbi_decode(self, text):
        seq_len = len(text)

        T1_table = np.zeros([seq_len, self.tag_size])
        T2_table = np.zeros([seq_len, self.tag_size])

        # 得到第1时刻的发射概率
        start_char_emission_prob = self.get_current_char_emission_prob(text[0])
        # 计算第一步初始概率, 填入表中
        T1_table[0, :] = self.pi + start_char_emission_prob
        T2_table[0, :] = np.nan

        for i in range(1, seq_len):
            current_char_emission_prob = self.get_current_char_emission_prob(text[i])
            current_char_emission_prob = np.expand_dims(current_char_emission_prob, axis=0)
            prev_score = np.expand_dims(T1_table[i - 1, :], axis=-1)
            # 上一时刻转移概率向量 + 转移概率矩阵 + 当前时刻的发射概率向量（通过广播计算）
            curr_score = prev_score + self.transition + current_char_emission_prob

            T1_table[i, :] = np.max(curr_score, axis=0)
            T2_table[i, :] = np.argmax(curr_score, axis=0)

        # 回溯，计算最优路径
        best_tag_id = int(np.argmax(T1_table[-1, :]))
        best_tags = [best_tag_id, ]
        for i in range(seq_len-1, 0, -1):
            best_tag_id = int(T2_table[i, best_tag_id])
            best_tags.append(best_tag_id)
        return list(reversed(best_tags))

    def test(self, test_raw_data_filepath, test_processed_data_filepath):
        print("开始测试...")
        predictions = []
        with open(test_raw_data_filepath, "r", encoding="UTF-8") as f:
            for line in tqdm(f):
                line = line.strip().replace('\u3000', ' ')
                if not line:
                    continue
                _, prediction = model.predict(line)
                predictions.append(prediction)

        test_data = load_data(test_processed_data_filepath)
        real = [dict_content["label"] for dict_content in test_data]

        real_for_report = MultiLabelBinarizer().fit_transform(real)
        predictions_for_report = MultiLabelBinarizer().fit_transform(predictions)

        class_list = [tag for tag, idx in model.tag2idx.items()]

        report = metrics.classification_report(real_for_report, predictions_for_report, target_names=class_list,
                                               digits=4)
        print("测试完成！")
        return report


if __name__ == '__main__':
    model = HMM_Viterbi(char2idx_path="./dicts/char2idx.json",
                        tag2idx_path="./dicts/tag2idx.json")
    model.fit("./corpus/msr_training_data_processed.txt")
    report = model.test("./corpus/msr_test.utf8", "./corpus/msr_test_gold.txt")
    print(report)

    while True:
        print("请输入文本：")
        text = input()
        text_participle, _ = model.predict(text)
        print(text_participle)
