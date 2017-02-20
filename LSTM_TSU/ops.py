import tensorflow as tf


def dropout(x, keep_ratio):
    return tf.nn.dropout(x, keep_ratio)


def prelu(features, initializer=None, scope=None):
    """
    Implementation of [Parametric ReLU](https://arxiv.org/abs/1502.01852) borrowed from Keras.
    """
    with tf.variable_scope(scope, 'PReLU', initializer=initializer):
        alpha = tf.get_variable('alpha', features.get_shape().as_list()[1:])
        pos = tf.nn.relu(features)
        neg = alpha * (features - tf.abs(features)) * 0.5
        return pos + neg


def accuracy_score(labels, logits, indexes, params): 
    """
    labels:   shape=(batch_size, max_step), dtype=int64
    logits:   shape=(batch_size*max_step, class_dim), dtype=float32
    indexes:  shape=(batch_size,), dtype=int32, elements.val < max_step
    """
    last_step_only = params['eval_last_step_only']
    max_time_step = params['max_time_step']
   
    if last_step_only:
        spread_indexes = tf.range(0, tf.shape(labels)[0]) * max_time_step + (indexes - 1)
        gathered_labels = tf.gather(tf.reshape(labels, [-1]), spread_indexes)
        gathered_logits = tf.gather(logits, spread_indexes)
        correct_prediction = tf.equal(gathered_labels, tf.argmax(gathered_logits, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))
    else:
        tiled_indexes = tf.tile(tf.expand_dims(indexes-1, -1), [1, max_time_step])
        sample_range = tf.transpose(tf.tile(tf.expand_dims(tf.range(0, max_time_step), -1), [1, tf.shape(labels)[0]]))
        index_mask = tf.less_equal(sample_range, tiled_indexes) 
        correct_prediction = tf.equal(labels, tf.argmax(tf.reshape(logits, [tf.shape(labels)[0], tf.shape(labels)[1],
                                                                            tf.shape(logits)[1]]), 2))
        correct_prediction = tf.multiply(tf.cast(index_mask, tf.float32), tf.cast(correct_prediction, tf.float32))
        accuracy = tf.div(tf.reduce_sum(correct_prediction),  tf.cast(tf.reduce_sum(indexes), tf.float32))

    return accuracy


def top_n_accuracy_score(labels, logits, indexes, params):
    last_step_only = params['eval_last_step_only']
    max_time_step = params['max_time_step']
    top_n = params['acc_top_n']

    if last_step_only:
        spread_indexes = tf.range(0, tf.shape(labels)[0]) * max_time_step + (indexes - 1)
        gathered_labels = tf.gather(tf.reshape(labels, [-1]), spread_indexes)
        gathered_logits = tf.gather(logits, spread_indexes)
        correct_prediction = tf.nn.in_top_k(gathered_logits, gathered_labels, top_n)
        topn_accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))
    else: 
        tiled_indexes = tf.tile(tf.expand_dims(indexes-1, -1), [1, max_time_step])
        sample_range = tf.transpose(tf.tile(tf.expand_dims(tf.range(0, max_time_step), -1), [1, tf.shape(labels)[0]]))
        index_mask = tf.less_equal(sample_range, tiled_indexes) 
        correct_prediction = tf.reshape(tf.nn.in_top_k(logits, tf.reshape(labels, [-1]), top_n),
                                        [tf.shape(labels)[0], tf.shape(labels)[1]])
        correct_prediction = tf.multiply(tf.cast(index_mask, tf.float32), tf.cast(correct_prediction, tf.float32))
        topn_accuracy = tf.div(tf.reduce_sum(correct_prediction),  tf.cast(tf.reduce_sum(indexes), tf.float32))
    
    return topn_accuracy


def linear(inputs, output_dim, dropout_rate=1.0, activation=None, scope='Linear', reuse=None):
    with tf.variable_scope(scope, reuse=reuse) as scope:
        input_dim = inputs.get_shape()[-1]
        weights = tf.get_variable('Weights', [input_dim, output_dim],
                                  initializer=tf.random_normal_initializer())
        variable_summaries(weights, scope.name + '/Weights')
        biases = tf.get_variable('Biases', [output_dim],
                                 initializer=tf.constant_initializer(0.0))
        variable_summaries(biases, scope.name + '/Biases')
        if activation is None:
            return dropout((tf.matmul(inputs, weights) + biases), dropout_rate)
        else:
            return dropout(activation(tf.matmul(inputs, weights) + biases), dropout_rate)


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev/' + name, stddev)
        tf.summary.scalar('max/' + name, tf.reduce_max(var))
        tf.summary.scalar('min/' + name, tf.reduce_min(var))
        tf.summary.histogram(name, var)
