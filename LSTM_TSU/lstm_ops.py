import tensorflow as tf
from ops import *


def lstm_cell(n_hidden, number_of_layers, keep_prob):
    cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, activation=tf.tanh, state_is_tuple=True)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
    return tf.nn.rnn_cell.MultiRNNCell([cell] * number_of_layers, state_is_tuple=True)


def x_to_s(x_input, params):
    dim_slot_embedding = params['dim_slot_embedding']
    dim_input = params['dim_input']
    max_time_step = params['max_time_step']
    
    with tf.variable_scope("embeds", reuse=True):
        slot_embeddings = tf.get_variable("slot_embedding", [dim_input, dim_slot_embedding], dtype=tf.float32)
    x_input_embed = tf.nn.embedding_lookup(slot_embeddings, x_input)
    
    x_input_mask = tf.cast(tf.less_equal(tf.zeros(tf.shape(x_input), dtype=tf.int32), x_input), dtype=tf.float32)
    x_input_mask = tf.tile(tf.expand_dims(x_input_mask, -1), [1, 1, 1, dim_slot_embedding])
    x_input_embed = tf.multiply(x_input_embed, x_input_mask)
    x_input_embed = tf.reduce_sum(x_input_embed, 2)
    
    # (batch_size, n_steps) => (batch_size, n_steps, n_input) => (n_steps*batch_size, n_input) and split
    x_tr = tf.transpose(x_input_embed, [1, 0, 2])
    x_tr_reshape = tf.reshape(x_tr, [-1, dim_slot_embedding])
    x_tr_reshape_split = tf.split(0, max_time_step, x_tr_reshape)

    return x_tr_reshape_split


def x_to_wsu(x_input, y_title, user_idx, params):
    dim_we = params['dim_word_embedding']
    dim_ue = params['dim_user_embedding']
    dim_se = params['dim_slot_embedding']
    dim_hidden = params['dim_hidden']
    dim_input = params['dim_input']
    voca_size = params['voca_size']
    user_size = params['user_size']
    max_time_step = params['max_time_step']
    embed_word = params['embed_word']
    embed_slot = params['embed_slot']
    embed_user = params['embed_user']

    with tf.variable_scope("embeds", reuse=True):
        slot_embeddings = tf.get_variable("slot_embedding", [dim_input, dim_se], dtype=tf.float32)
        word_embeddings = tf.get_variable("word_embedding", [voca_size, dim_we], dtype=tf.float32)
        user_embeddings = tf.get_variable("user_embedding", [user_size, dim_ue], dtype=tf.float32)
   
    # get slot embedding
    x_input_embed = tf.nn.embedding_lookup(slot_embeddings, x_input)
    x_input_mask = tf.cast(tf.less_equal(tf.zeros(tf.shape(x_input), dtype=tf.int32), x_input), dtype=tf.float32)
    x_input_mask = tf.tile(tf.expand_dims(x_input_mask, -1), [1, 1, 1, dim_se])
    x_input_embed = tf.multiply(x_input_embed, x_input_mask)
    x_input_embed = tf.reduce_sum(x_input_embed, 2)
    
    # get title embedding
    y_title_embed = tf.nn.embedding_lookup(word_embeddings, y_title)
    y_title_mask = tf.cast(tf.less_equal(tf.zeros(tf.shape(y_title), dtype=tf.int32), y_title), dtype=tf.float32)
    y_title_mask = tf.tile(tf.expand_dims(y_title_mask, -1), [1, 1, 1, dim_we])
    y_title_embed = tf.multiply(y_title_embed, y_title_mask)
    # y_title_length = tf.reduce_sum(tf.cast(tf.less_equal(tf.zeros(tf.shape(y_title), dtype=tf.int32), y_title),
    #                                        dtype=tf.float32), 2)
    # y_title_length = tf.tile(tf.expand_dims(y_title_length, -1), [1, 1, dim_we])
    y_title_embed = tf.reduce_sum(y_title_embed, 2)
    # y_title_embed /= (y_title_length + 1e-5)

    # get user embedding
    user_embed = tf.nn.embedding_lookup(user_embeddings, user_idx)

    # concat
    cct1 = x_input_embed if embed_slot else tf.reduce_sum(tf.one_hot(x_input, dim_input), 2)
    cct2 = y_title_embed if embed_word else None
    cct3 = user_embed if embed_user else None

    if cct2 is not None and cct3 is not None:
        x_total = tf.concat(2, [cct1, cct2, cct3])
        print('W%sU model' % ('S' if embed_slot else 'X'))
    elif cct2 is not None:
        x_total = tf.concat(2, [cct1, cct2])
        print('W%s model' % ('S' if embed_slot else 'X'))
    elif cct3 is not None:
        x_total = tf.concat(2, [cct1, cct3])
        print('%sU model' % ('S' if embed_slot else 'X'))
    else:
        x_total = cct1
        print('%s model' % ('S' if embed_slot else 'X'))

    dim_we = dim_we if embed_word else 0
    dim_se = dim_se if embed_slot else dim_input
    dim_ue = dim_ue if embed_user else 0

    x_tr = tf.transpose(x_total, [1, 0, 2])
    x_reshape = tf.reshape(x_tr, [-1, dim_we + dim_se + dim_ue])
    print(x_reshape, str(dim_we + dim_se + dim_ue))

    # linear transformation
    x_linear = linear(x_reshape, 
            output_dim=dim_hidden,
            activation=tf.nn.sigmoid,
            scope='input-linear')
    x_split = tf.split(0, max_time_step, x_linear)

    return x_split


def post_concat_lstm(x_input, y_title, x_length, user_idx, lstm_dropout, params, scope):
    dim_rnn_cell = params['dim_rnn_cell']
    lstm_layer = params['lstm_layer']
    # voca_size = params['voca_size']
    user_size = params['user_size']
    dim_word_embedding = params['dim_word_embedding']
    dim_user_embedding = params['dim_user_embedding']

    with tf.variable_scope("embeds", reuse=True):
        # word_embeddings = tf.get_variable("word_embedding", [voca_size, dim_word_embedding],
        #         dtype=tf.float32)
        user_embeddings = tf.get_variable("user_embedding", [user_size, dim_user_embedding], dtype=tf.float32)

    with tf.variable_scope(scope or 'lstm'):
        cell = lstm_cell(dim_rnn_cell, lstm_layer, lstm_dropout)
        outputs, state = tf.nn.rnn(cell, x_input, sequence_length=x_length, dtype=tf.float32)
        outputs = tf.transpose(tf.pack(outputs), [1, 0, 2])

        # y_title_embed = tf.nn.embedding_lookup(word_embeddings, y_title)
        # y_title_mask = tf.cast(tf.less_equal(tf.zeros(tf.shape(y_title), dtype=tf.int32), y_title), dtype=tf.float32)
        # y_title_mask = tf.tile(tf.expand_dims(y_title_mask, -1), [1, 1, 1, dim_word_embedding])
        # y_title_embed = tf.multiply(y_title_embed, y_title_mask)
        # y_title_embed = tf.reduce_sum(y_title_embed, 2)

        user_embed = tf.nn.embedding_lookup(user_embeddings, user_idx)
        # concat = tf.concat(2, [outputs, y_title_embed, user_embed])
        concat = tf.concat(2, [outputs, y_title, user_embed])
        output = tf.reshape(concat, [-1, dim_rnn_cell + dim_word_embedding + dim_user_embedding])

        return output


def pre_concat_lstm(x_input, x_length, lstm_dropout, params, scope):
    dim_rnn_cell = params['dim_rnn_cell']
    lstm_layer = params['lstm_layer']

    with tf.variable_scope(scope or 'lstm'):
        cell = lstm_cell(dim_rnn_cell, lstm_layer, lstm_dropout)
        outputs, state = tf.nn.rnn(cell, x_input, sequence_length=x_length, dtype=tf.float32)
        outputs = tf.transpose(tf.pack(outputs), [1, 0, 2])
        output = tf.reshape(outputs, [-1, dim_rnn_cell])

        return output


def pre_concat_mlp(x_input, lstm_dropout, params, scope):
    # dim_rnn_cell = params['dim_rnn_cell']

    with tf.variable_scope(scope or 'mlp'):
        x_input = tf.transpose(tf.pack(x_input), [1, 0, 2])
        dim_input = x_input.get_shape().as_list()[-1]
        x_input = tf.reshape(x_input, [-1, dim_input])

        return x_input


def post_concat_mlp(x_input, y_title, x_length, user_idx, lstm_dropout, params, scope):
    # dim_rnn_cell = params['dim_rnn_cell']
    # lstm_layer = params['lstm_layer']
    # voca_size = params['voca_size']
    user_size = params['user_size']
    dim_word_embedding = params['dim_word_embedding']
    dim_slot_embedding = params['dim_slot_embedding']
    dim_user_embedding = params['dim_user_embedding']

    with tf.variable_scope("embeds", reuse=True):
        user_embeddings = tf.get_variable("user_embedding", [user_size, dim_user_embedding], dtype=tf.float32)

    with tf.variable_scope(scope or 'lstm'):
        outputs = tf.transpose(tf.pack(x_input), [1, 0, 2])

        user_embed = tf.nn.embedding_lookup(user_embeddings, user_idx)
        concat = tf.concat(2, [outputs, y_title, user_embed])
        output = tf.reshape(concat, [-1, dim_slot_embedding + dim_word_embedding + dim_user_embedding])

        return output
