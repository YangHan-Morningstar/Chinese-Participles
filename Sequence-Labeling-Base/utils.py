import json
import tensorflow as tf
import numpy as np


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


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def load_npz_data(file_path):
    data = np.load(file_path)
    return data["x"], data["y"]
