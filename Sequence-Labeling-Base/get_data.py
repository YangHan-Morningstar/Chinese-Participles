import json
from tqdm import tqdm
from utils import extend_list


filepath = "./corpus/msr_training.utf8"
all_texts, all_labels = [], []


with open(filepath, "r", encoding="UTF-8") as f:
    for line in tqdm(f):
        content = ''
        content_seg, content_index = [], []
        text_cache, label_cache = '', []
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
            text_cache += char

        if len(text_cache) > 128:
            continue

        if len(label_cache) < 128:
            label_cache += ["S"] * (128 - len(label_cache))

        all_texts.append(text_cache)
        all_labels.append(label_cache)


all_texts_word = extend_list(all_texts)
all_texts_labels = extend_list(all_labels)

unique_chars = list(set(all_texts_word))
unique_tags = list(set(all_texts_labels))

tag2idx = {element: i for i, element in enumerate(unique_tags)}

all_labels_idx = []
for labels in all_labels:
    idx = [tag2idx[label] for label in labels]
    all_labels_idx.append(idx)

with open("corpus/msr_training_data_processed_for_dl.txt", "w+", encoding="utf-8") as f:
    for text, label in zip(all_texts, all_labels_idx):
        f.write(str({"text": text, "label": label}))
        f.write("\n")

with open("dicts/tag2idx_for_dl.json", "w+", encoding="utf-8") as f:
    f.write(json.dumps(tag2idx, ensure_ascii=False))
