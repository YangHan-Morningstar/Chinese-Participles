class FMM(object):
    def __init__(self, dict_path):
        super(FMM, self).__init__()
        self.dict_path = dict_path
        self.words_dict_list = self.get_word_dict()

    def get_word_dict(self):
        words_dict_list = []
        with open(self.dict_path, "r", encoding="utf-8") as word_dict_file:
            for word in word_dict_file:
                words_dict_list.append(word.strip())
        return words_dict_list

    def cut_words(self, raw_sentence):
        # 词典中最长的词
        word_max_length = max(len(word) for word in self.words_dict_list)

        sentence = raw_sentence.strip()
        sentence_length = len(sentence)

        # 从sentence中分出的词
        cut_word_list = []

        while sentence_length > 0:
            max_cut_length = min(word_max_length, sentence_length)
            sub_sentence = sentence[0: max_cut_length]
            while max_cut_length > 0:
                if sub_sentence in self.words_dict_list or max_cut_length == 1:
                    cut_word_list.append(sub_sentence)
                    break
                else:
                    max_cut_length -= 1
                    sub_sentence = sub_sentence[0: max_cut_length]
            sentence = sentence[max_cut_length:]
            sentence_length -= max_cut_length

        return cut_word_list


if __name__ == "__main__":
    fmm_algorithm = FMM("./dict/dict.txt")
    while True:
        print("请输入文本：")
        raw_sentence = input()
        print("/".join(fmm_algorithm.cut_words(raw_sentence)))
