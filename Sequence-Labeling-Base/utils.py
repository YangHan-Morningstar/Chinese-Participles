import json


def load_dict(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return [eval(i) for i in f.readlines()]


def extend_list(raw_list):
    extended_list = []
    for i in raw_list:
        extended_list.extend(i)
    return extended_list
