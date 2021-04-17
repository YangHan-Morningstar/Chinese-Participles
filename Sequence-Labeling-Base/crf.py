import sklearn_crfsuite
import joblib
from utils import *
from sklearn import metrics
from sklearn.preprocessing import MultiLabelBinarizer


class CRF(object):
    def __init__(self, tag2idx_path, model_path="./crf_model.pkl"):
        self.algorithm = "lbfgs"
        self.c1 = "0.1"
        self.c2 = "0.1"
        self.max_iterations = 100
        self.model_path = model_path
        self.model = None
        self.tag2idx = load_dict(tag2idx_path)
        self.idx2tag = {v: k for k, v in self.tag2idx.items()}

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
        x_train, y_train = self.load_processed_data(train_dic_path)

        print("开始训练...")
        self.model.fit(x_train, y_train)

        self.save_model()
        print("训练完成，模型已保存")

    def test(self, test_dic_path):
        x_test, y_test_raw = self.load_processed_data(test_dic_path, need_split=True)

        labels = list(self.model.classes_)
        y_predict = self.model.predict(x_test)

        prediction = self.convert_tag_to_idx(y_predict)

        real_for_report = MultiLabelBinarizer().fit_transform(y_test_raw)
        prediction_for_report = MultiLabelBinarizer().fit_transform(prediction)

        class_list = [tag for tag, idx in self.tag2idx.items()]

        report = metrics.classification_report(real_for_report,
                                               prediction_for_report,
                                               target_names=class_list,
                                               digits=4)
        print(report)

    def convert_tag_to_idx(self, tag_array2d):
        idx_array2d = []
        for tag_array in tag_array2d:
            cache = [self.tag2idx[tag] for tag in tag_array]
            idx_array2d.append(cache)
        return idx_array2d

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

    def load_processed_data(self, data_dic_path, need_split=False):
        data_dict = load_data(data_dic_path)
        word_seq, tag_seq = [], []
        for single_data_dict in data_dict:
            text, label = single_data_dict["text"], single_data_dict["label"]
            if need_split:
                char_list = [char for char in text]
                text = ["<BOS>"] + char_list + ["<EOS>"]
            else:
                text = ["<BOS>"] + text + ["<EOS>"]
            word_seq.append(text)
            tag_seq.append(label)

        word_grams = [self.segment_by_window(word_list) for word_list in word_seq]
        features = self.extract_feature(word_grams)
        return features, tag_seq


if __name__ == "__main__":
    crf = CRF(tag2idx_path="./dicts/tag2idx.json")

    # 训练
    # crf.train("./corpus/msr_training_data_processed.txt")

    # 测试
    crf.load_model()
    crf.test("./corpus/msr_test_gold.txt")
    # while True:
    #     print("请输入中文句子：")
    #     text = input()
    #     crf.predict(text)
