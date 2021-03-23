import sklearn_crfsuite
from sklearn_crfsuite import metrics
import joblib
from utils import *


class CRF(object):
    def __init__(self, model_path="./crf_model.pkl"):
        self.algorithm = "lbfgs"
        self.c1 = "0.1"
        self.c2 = "0.1"
        self.max_iterations = 100
        self.model_path = model_path
        self.model = None

    def initialize_model(self):
        print("初始化模型...")
        algorithm = self.algorithm
        c1 = float(self.c1)
        c2 = float(self.c2)
        max_iterations = int(self.max_iterations)
        self.model = sklearn_crfsuite.CRF(algorithm=algorithm,
                                          c1=c1,
                                          c2=c2,
                                          max_iterations=max_iterations,
                                          all_possible_transitions=True)

    def train(self, train_dic_path):
        self.initialize_model()
        x, y = self.load_train_data(train_dic_path)

        x_train, y_train = x[0: 80000], y[0: 80000]
        x_test, y_test = x[80000:], y[80000:]

        print("开始训练...")
        self.model.fit(x_train, y_train)

        print("开始测试...")
        labels = list(self.model.classes_)
        y_predict = self.model.predict(x_test)
        metrics.flat_f1_score(y_test, y_predict, average='weighted', labels=labels)
        sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
        print(metrics.flat_classification_report(y_test, y_predict, labels=sorted_labels, digits=3))

        self.save_model()

    def predict(self, text):
        word_lists = []
        for char in text:
            word_lists.append(char)
        word_lists = [["<BOS>"] + word_lists + ["<EOS>"]]
        word_grams = [self.segment_by_window(word_list) for word_list in word_lists]
        features = self.extract_feature(word_grams)
        y_predict = self.model.predict(features)[0]

        # 输出分词结果
        text_participle = ""
        for i in range(len(text)):
            if y_predict[i] == "S" or y_predict[i] == "E":
                text_participle += text[i] + "  "
            else:
                text_participle += text[i]
        print(text_participle)

    def load_model(self):
        print("加载模型...")
        self.model = joblib.load(self.model_path)

    def save_model(self):
        joblib.dump(self.model, self.model_path)
        print("模型已保存")

    def extract_feature(self, word_grams):
        print("提取特征...")
        features, feature_list = [], []
        for index in range(len(word_grams)):
            for i in range(len(word_grams[index])):
                word_gram = word_grams[index][i]
                feature = {'w-1': word_gram[0],
                           'w': word_gram[1],
                           'w+1': word_gram[2],
                           'w-1:w': word_gram[0] + word_gram[1],
                           'w:w+1': word_gram[1] + word_gram[2],
                           'bias': 1.0}
                feature_list.append(feature)
            features.append(feature_list)
            feature_list = []
        return features

    def segment_by_window(self, words_list=None, window=3):
        """窗口切分"""
        words = []
        begin, end = 0, window
        for _ in range(1, len(words_list)):
            if end > len(words_list):
                break
            words.append(words_list[begin:end])
            begin = begin + 1
            end = end + 1
        return words

    def load_train_data(self, train_dic_path):
        train_dict = load_data(train_dic_path)
        word_seq, tag_seq = [], []
        for single_data_dict in train_dict:
            text, label = single_data_dict["text"], single_data_dict["label"]
            text = ["<BOS>"] + text + ["<EOS>"]
            word_seq.append(text)
            tag_seq.append(label)

        word_grams = [self.segment_by_window(word_list) for word_list in word_seq]
        features = self.extract_feature(word_grams)
        return features, tag_seq


if __name__ == "__main__":
    crf = CRF()

    # 训练
    crf.train("./corpus/msr_training_data_processed.txt")

    # 测试
    # crf.load_model()
    # while True:
    #     print("请输入中文句子：")
    #     text = input()
    #     crf.predict(text)
