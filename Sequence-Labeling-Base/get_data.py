import json
from tqdm import tqdm
from utils import extend_list


filepath = "./corpus/msr_training.utf8"
all_texts, all_labels = [], []


with open(filepath, "r", encoding="UTF-8") as f:
    for line in tqdm(f):
        content = ''
        content_seg, content_index = [], []
        text_cache, label_cache = [], []
        line = line.strip().replace('\u3000', ' ')
        if not line:
            continue
        line_seg = line.split(' ')
        for word in line_seg:
            if word != '':
                content_seg.append(word)
        for word in content_seg:
            content += word
        if len(content) <= 1:
            continue

        for word in content_seg:
            if len(word) == 1:
                label_cache.append("S")
            else:
                for i in range(len(word)):
                    if i == 0:
                        label_cache.append("B")
                    elif i == len(word) - 1:
                        label_cache.append("E")
                    else:
                        label_cache.append("M")
        for char in content:
            text_cache.append(char)

        all_texts.append(text_cache)
        all_labels.append(label_cache)

with open("./corpus/msr_training_data_processed.txt", "w+", encoding="utf-8") as f:
    for text, label in zip(all_texts, all_labels):
        f.write(str({"text": text, "label": label}))
        f.write("\n")


all_texts_word = extend_list(all_texts)
all_texts_labels = extend_list(all_labels)

unique_chars = list(set(all_texts_word))
unique_tags = list(set(all_texts_labels))

char2idx = {"UNK": 0}
for i, element in enumerate(unique_chars):
    char2idx[element] = i + 1
tag2idx = {element: i for i, element in enumerate(unique_tags)}

with open("./dicts/char2idx.json", "w+", encoding="utf-8") as f:
    f.write(json.dumps(char2idx, ensure_ascii=False))
with open("./dicts/tag2idx.json", "w+", encoding="utf-8") as f:
    f.write(json.dumps(tag2idx, ensure_ascii=False))
