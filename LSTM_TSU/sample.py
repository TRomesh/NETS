import tensorflow as tf
import numpy as np

batch_size = 10
max_step = 5
class_dim = 3
last_step_only = False
print_acc = False
print_loss = True


def get_sample_data():
    logits = np.random.uniform(0, 1, size=(batch_size * max_step, class_dim))
    labels = np.random.randint(class_dim, size=(batch_size, max_step))
    indexes = np.random.randint(1, max_step + 1, size=(batch_size))
    return logits, labels, indexes


gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))

_logits = tf.placeholder(tf.float32, shape=[None, class_dim])
_labels = tf.placeholder(tf.int64, shape=[None, max_step])
_indexes = tf.placeholder(tf.int32, shape=[None])

if last_step_only:
    spread_indexes = tf.range(0, tf.shape(_labels)[0]) * max_step + (_indexes - 1)
    gathered_labels = tf.gather(tf.reshape(_labels, [-1]), spread_indexes)
    gathered_logits = tf.gather(_logits, spread_indexes)
    correct_prediction = tf.equal(gathered_labels, tf.argmax(gathered_logits, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))
else:
    tiled_indexes = tf.tile(tf.expand_dims(_indexes - 1, -1), [1, max_step])
    sample_range = tf.transpose(tf.tile(tf.expand_dims(tf.range(0, max_step), -1), [1,
        tf.shape(_labels)[0]]))
    index_mask = tf.less_equal(sample_range, tiled_indexes) 
    correct_prediction = tf.equal(_labels, tf.argmax(tf.reshape(_logits, [tf.shape(_labels)[0],
        tf.shape(_labels)[1], tf.shape(_logits)[1]]), 2))
    correct_prediction = tf.multiply(tf.cast(index_mask, tf.float32), tf.cast(correct_prediction,
        tf.float32))
    accuracy = tf.div(tf.reduce_sum(correct_prediction),  tf.cast(tf.reduce_sum(_indexes),
        tf.float32))


lengths_tiled = tf.tile(tf.expand_dims(_indexes-1, 1), [1, max_step])
range_tiled = tf.tile(tf.expand_dims(tf.range(0, max_step), 0), [tf.shape(_labels)[0], 1])
step_weights = tf.transpose(tf.cast(tf.less_equal(range_tiled, lengths_tiled), dtype=tf.float32), [1, 0])

step_logits = tf.reshape(_logits, [-1, max_step, class_dim])
step_logits = tf.unpack(tf.transpose(step_logits, [1, 0, 2]), max_step)
print('logit reshaped', step_logits)

step_labels = tf.unpack(tf.transpose(_labels, [1, 0]), max_step)
print('label reshaped', step_labels)
step_weights = tf.unpack(step_weights, max_step)
print('weight reshaped', step_weights)
penaltied_weights = []
for idx, step_weight in enumerate(step_weights):
    penaltied_weights.append(step_weight * ((idx+1) / len(step_weights)))
print('weight penaltied', penaltied_weights)


loss = tf.nn.seq2seq.sequence_loss_by_example(
    step_logits,
    step_labels,
    step_weights,
    average_across_timesteps=True)
#    average_across_batch=True)

loss_group = []
for idx, (_logit, _label, _weight) in enumerate(zip(step_logits, step_labels, step_weights)):
    step_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(_logit, _label) * _weight
    
    prev_labels = step_labels[:idx]
    negative_loss = 0
    for prev_label in prev_labels:
        negative_loss += (tf.nn.sparse_softmax_cross_entropy_with_logits(-_logit, prev_label) * _weight)
    
    # step_loss += negative_loss * 0.1
    loss_group.append(step_loss)

loss_sample = tf.add_n(loss_group)
loss_sample /= (tf.add_n(step_weights) + 1e-12)
loss_sample = tf.reduce_sum(loss_sample) / batch_size


sess.run(tf.global_variables_initializer())

logits, labels, indexes = get_sample_data()
feed_dict={_logits: logits, _labels: labels, _indexes: indexes}
print(logits)
print(labels)
print(indexes)

if last_step_only and print_acc:
    s, lo, la, cp, ac = sess.run([spread_indexes, gathered_logits, gathered_labels, correct_prediction, accuracy],
                                 feed_dict=feed_dict)
    print('spread indexes', s)
    print('gathered labels', la)
    print('gathered logits', lo)
    print('correct prediction', cp)
    print('accuracy', ac)
elif print_acc:
    ti, sr, im, cp, ac = sess.run([tiled_indexes, sample_range, index_mask, correct_prediction, accuracy],
                                  feed_dict=feed_dict)
    print('tiled indexes', ti)
    print('sample range', sr)
    print('index mask', im)
    print('correct prediction', cp)
    print('accuracy', ac)


if print_loss:
    lti, rti, w, l, nl, ls, pw = sess.run([lengths_tiled, range_tiled, step_weights, loss, negative_loss, loss_sample,
                                           penaltied_weights], feed_dict=feed_dict)
    print('2 tiles')
    print(lti)
    print(rti)
    print('resulting weight and loss')
    print(np.reshape(w, [max_step, batch_size]))
    print(np.reshape(pw, [max_step, batch_size]))
    print(l, np.mean(l))
    print(nl)
    print(ls)

