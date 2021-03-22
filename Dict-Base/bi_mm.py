from fmm import FMM
from rmm import RMM


class BiMM(object):
    def __init__(self, dict_path):
        super(BiMM, self).__init__()
        self.dict_path = dict_path

        self.fmm_algorithm = FMM(self.dict_path)
        self.rmm_algorithm = RMM(self.dict_path)

    def cut_words(self, raw_sentence):
        fmm_cut_words_list = self.fmm_algorithm.cut_words(raw_sentence)
        rmm_cut_words_list = self.rmm_algorithm.cut_words(raw_sentence)

        fmm_cut_words_list_size = len(fmm_cut_words_list)
        rmm_cut_words_list_size = len(rmm_cut_words_list)

        if fmm_cut_words_list_size > rmm_cut_words_list_size:
            return rmm_cut_words_list
        elif fmm_cut_words_list_size < rmm_cut_words_list_size:
            return fmm_cut_words_list
        else:
            if fmm_cut_words_list == rmm_cut_words_list:
                return fmm_cut_words_list
            else:
                fmm_single_word_num, rmm_single_word_num = 0, 0
                for i in range(fmm_cut_words_list_size):
                    if len(fmm_cut_words_list[i]) == 1:
                        fmm_single_word_num += 1
                    if len(rmm_cut_words_list[i]) == 1:
                        rmm_single_word_num += 1
                if fmm_single_word_num > rmm_single_word_num:
                    return rmm_cut_words_list
                else:
                    return fmm_cut_words_list


if __name__ == "__main__":
    bimm_algorithm = BiMM("./dict/dict.txt")
    while True:
        print("请输入文本：")
        raw_sentence = input()
        print("/".join(bimm_algorithm.cut_words(raw_sentence)))
