import tensorflow as tf
import os
from utils import *
from transformer import *
import time


char2idx = load_dict("./dicts/char2idx.json")
tag2idx = load_dict("./dicts/tag2idx_for_transformer.json")
filepath_list = ["./corpus/prepared_data/" + str(i) + ".npz" for i in range(86908 + 1)]


# BUFFER_SIZE = 86908 + 1
BUFFER_SIZE = 20000
BATCH_SIZE = 128

dataset = tf.data.Dataset.from_tensor_slices((filepath_list))
dataset = dataset.map(lambda item1: tf.numpy_function(
          load_npz_data, [item1], [tf.int64, tf.int64]),
          num_parallel_calls=tf.data.experimental.AUTOTUNE)
# dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

num_layers = 4
d_model = 128
dff = 512
num_heads = 8

input_vocab_size = len(char2idx)
target_vocab_size = len(tag2idx)
dropout_rate = 0.1
max_position = 583

EPOCHS = 200

learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size,
                          pe_input=max_position,
                          rate=dropout_rate)


checkpoint_path = "./transformer_checkpoints/train"

ckpt = tf.train.Checkpoint(transformer=transformer,  optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')


train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]


@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):

    enc_padding_mask = create_masks(inp)

    with tf.GradientTape() as tape:
        predictions = transformer(inp,
                                  True,
                                  enc_padding_mask)
        loss = loss_function(tar, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(tar, predictions)


for epoch in range(EPOCHS):
    start = time.time()

    train_loss.reset_states()
    train_accuracy.reset_states()

    for (batch, (inp, tar)) in enumerate(dataset):
        # tf.config.experimental_run_functions_eagerly(True)
        train_step(inp, tar)

        if batch % 50 == 0:
            print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                epoch + 1, batch, train_loss.result(), train_accuracy.result()))

    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))

    print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, train_loss.result(), train_accuracy.result()))

    print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
