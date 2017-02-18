import tensorflow as tf
import numpy as np

import dataset as d
from ops import *
from lstm_ops import *


class WSU_RnnModel(object):
    def __init__(self, params, initializer):

        # Saved params and session
        config = tf.ConfigProto(device_count={'GPU': 1})
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 1.0
        self.session = tf.Session(config=config)
        self.params = params

        # Hyper parameters
        self.dim_input = params['dim_input']
        self.dim_output = params['dim_output']
        self.dim_hidden = params['dim_hidden']
        self.batch_size = params['batch_size']
        self.dim_word_embedding = params['dim_word_embedding']
        self.dim_slot_embedding = params['dim_slot_embedding']
        self.dim_user_embedding = params['dim_user_embedding']
        self.user_size = params['user_size']
        self.voca_size = params['voca_size']
        self.max_slot_num = params['max_slot_num']
        self.max_title_len = params['max_title_len']
        self.max_grad_norm = params['max_grad_norm']
        self.max_time_step = params['max_time_step']
        self.learning_rate = params['learning_rate']
        self.decay_rate = params['decay_rate']
        self.decay_step = params['decay_step']
        self.cold_start = params['cold_start']
        self.pre_concat = params['pre_concat']
        self.we_trainable = params['we_trainable']
        self.ue_trainable = params['ue_trainable']
        self.embed_word = params['embed_word']
        self.embed_user = params['embed_user']

        # Embedding table initializer
        cal2vec_set, user2vec_set = initializer
        self.cal2vec = {
            'idx2vec': cal2vec_set[0],
            'idx2word': cal2vec_set[1],
            'word2idx': cal2vec_set[2],
            'count': cal2vec_set[3]
        }
        self.user2vec = {
            'idx2vec': user2vec_set[0],
            'user2idx': user2vec_set[1]
        }
        self.initialize_embedding(
                np.asarray(self.cal2vec['idx2vec'], dtype=np.float32),
                np.asarray(self.user2vec['idx2vec'], dtype=np.float32))

        # Input placeholders
        self.x = tf.placeholder(tf.int32, shape=[None, self.max_time_step, self.max_slot_num])
        self.y = tf.placeholder(tf.int64, shape=[None, self.max_time_step])
        self.x_title = tf.placeholder(tf.int32, shape=[None, self.max_time_step, self.max_title_len])
        self.y_title = tf.placeholder(tf.int32, shape=[None, self.max_time_step, self.max_title_len])
        self.user_idx = tf.placeholder(tf.int32, shape=[None, self.max_time_step])
        self.seq_len = tf.placeholder(tf.int32, shape=[None])
        self.lstm_dropout = tf.placeholder(tf.float32)

        # Hyper parameters (ops)
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(
                self.learning_rate, self.global_step,
                self.decay_step, self.decay_rate, staircase=True)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

        # Build model
        self.logits = None
        self.optimize_op = None
        self.optimize_user_op = None
        self.accuracy = None
        self.top5acc = None
        self.cost = None
        self.build_model()

        self.saver = tf.train.Saver(tf.global_variables())
        self.session.run(tf.global_variables_initializer())

        # Check for initializer
        print("Word embedding initialized", self.get_word_embedding()[0][:5])
        print("Slot embedding initialized", self.get_slot_embedding()[0][:5])
        user_prefix_prob_sum = sum(self.get_user_embedding()[0][:self.dim_user_embedding])
        # assert True if self.cold_start else 1. - user_prefix_prob_sum < 0.0001, 'Wrong UE'
        print("User embedding initialized", self.get_user_embedding()[0][:5], 'prefix_prob_sum', user_prefix_prob_sum)

    def build_model(self):
        self.logits = self.inference(self.x, 
                self.y_title, 
                self.seq_len, 
                self.user_idx,
                self.lstm_dropout)
        self.cost = self.loss(self.y, self.logits, self.seq_len)
        self.optimize_op = self.optimize(self.cost)
        if self.ue_trainable and self.embed_user:
            self.optimize_user_op = self.optimize_user(self.cost)

        self.accuracy = accuracy_score(self.y, self.logits, self.seq_len, self.params)
        self.top5acc = top_n_accuracy_score(self.y, self.logits, self.seq_len, self.params)
        
    def loss(self, labels, logits, indexes):
        lengths_tiled = tf.tile(tf.expand_dims(indexes-1, 1), [1, self.max_time_step])
        range_tiled = tf.tile(tf.expand_dims(tf.range(0, self.max_time_step), 0), [tf.shape(labels)[0], 1])
        step_weights = tf.transpose(tf.cast(tf.less_equal(range_tiled, lengths_tiled), dtype=tf.float32), [1, 0])
        step_weights = tf.unpack(step_weights, self.max_time_step)
        penaltied_weights = []
        for idx, step_weight in enumerate(step_weights):
            penaltied_weights.append(step_weight * ((idx+1) / len(step_weights)))
        step_logits = tf.reshape(logits, [-1, self.max_time_step, self.dim_output])
        step_logits = tf.unpack(tf.transpose(step_logits, [1, 0, 2]), self.max_time_step)
        step_labels = tf.unpack(tf.transpose(labels, [1, 0]), self.max_time_step)

        loss = tf.nn.seq2seq.sequence_loss(
            step_logits,
            step_labels,
            step_weights,
            average_across_timesteps=True,
            average_across_batch=True)

        return loss

    def inference(self, x, y_title, x_length, user_idxes, keep_prob):
        pass

    def optimize(self, cost):
        pass

    def optimize_user(self, cost):
        pass
    
    def run(self, data, train=False, validate=False, user_update=False, debug=False):
        eou = False
        d.week_itr = 0
        word_dict = self.cal2vec['word2idx']
        user_dict = self.user2vec['user2idx']
        cnt, total_acc1, total_acc5, total_loss = 0, 0.0, 0.0, 0.0
        results = []
        step = 0

        while not eou:
            batch_x, _, batch_y, batch_y_title, batch_seq_len, batch_user_idx, eou = \
                d.get_next_batch(data, user_dict, self.batch_size, self.max_time_step, word_dict,
                                 self.dim_input, self.params)
            lstm_dropout = self.params['lstm_dropout'] if train else 1.0

            feed_dict = {self.x: batch_x, self.y: batch_y, self.y_title: batch_y_title,
                         self.seq_len: batch_seq_len, self.user_idx: batch_user_idx,
                         self.lstm_dropout: lstm_dropout}

            acc1, acc5, loss, step = self.session.run(
                    fetches=[self.accuracy, self.top5acc, self.cost, self.global_step], 
                    feed_dict=feed_dict)

            if train:
                self.session.run(self.optimize_op, feed_dict=feed_dict)
            if user_update and self.embed_user:
                self.session.run(self.optimize_user_op, feed_dict=feed_dict) 
            if debug:
                results.append(self.debug(feed_dict))

            total_acc1 += acc1
            total_acc5 += acc5
            total_loss += loss
            cnt += 1

        if train:
            msg = 'Training'
        elif validate:
            msg = 'Validation'
        else:
            msg = 'Testing'
        print(msg, 'loss: %.3f' % (total_loss / cnt), 'Top1 Accuracy: %.3f%%' % (total_acc1 / cnt * 100),
              'Top5 Accuracy: %.3f%%' % (total_acc5 / cnt * 100), 'step: %d' % step)
        return (total_loss / cnt), (total_acc1 / cnt * 100), (total_acc5 / cnt * 100), results

    def debug(self, feed_dict):
        logits, labels, indexes = self.session.run(fetches=[self.logits, self.y, self.seq_len], feed_dict=feed_dict)
        return [logits, labels, indexes]

    def initialize_embedding(self, word_embed, user_embed):
        with tf.variable_scope("embeds"):
            word_embeddings = tf.get_variable("word_embedding",
                                              initializer=tf.constant(word_embed),
                                              trainable=self.we_trainable,
                                              dtype=tf.float32)
            slot_embeddings = tf.get_variable("slot_embedding",
                                              initializer=tf.random_uniform([self.dim_input, self.dim_slot_embedding], -0.1, 0.1),
                                              dtype=tf.float32)
            user_embeddings = tf.get_variable("user_embedding",
                                              initializer=tf.constant(user_embed),
                                              trainable=self.ue_trainable,
                                              dtype=tf.float32)

    def get_word_embedding(self):
        with tf.variable_scope("embeds", reuse=True):
            word_embeddings = tf.get_variable("word_embedding", [self.voca_size, self.dim_word_embedding],
                                              dtype=tf.float32)
            return word_embeddings.eval(session=self.session)

    def get_slot_embedding(self):
        with tf.variable_scope("embeds", reuse=True):
            slot_embeddings = tf.get_variable("slot_embedding", [self.dim_input, self.dim_slot_embedding],
                                              dtype=tf.float32)
            return slot_embeddings.eval(session=self.session)

    def get_user_embedding(self):
        with tf.variable_scope("embeds", reuse=True):
            user_embeddings = tf.get_variable("user_embedding", [self.user_size, self.dim_user_embedding],
                                              dtype=tf.float32)
            return user_embeddings.eval(session=self.session)
    
    @staticmethod
    def reset_graph():
        tf.reset_default_graph()

    def save(self, filepath):
        self.saver.save(self.session, filepath, global_step=self.global_step)
        print("Model saved as %s" % filepath)

    def load(self, filepath):
        self.saver.restore(self.session, filepath)
        print("Model restored from %s" % filepath)


class WSU_LstmModel(WSU_RnnModel):
    def __init__(self, params, initializer):
        super(WSU_LstmModel, self).__init__(params, initializer)

    def inference(self, x, y_title, x_length, user_idx, lstm_dropout):
        pre_concat = pre_concat_lstm(x_to_wsu(x, y_title, user_idx, self.params),
                x_length, 
                lstm_dropout, 
                self.params, 
                scope='pre-concat')
        print('pre-concat' if self.pre_concat else 'post-concat', 'version')
        output = linear(pre_concat if self.pre_concat else post_concat,
                output_dim=self.dim_output,
                scope='Output')
        return output

    def optimize(self, cost):
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), self.max_grad_norm)
        return self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)

    def optimize_user(self, cost):
        tvars = [v for v in tf.trainable_variables() if v.name == "embeds/user_embedding:0"]
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), self.max_grad_norm)
        return self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)


class WSU_MlpModel(WSU_RnnModel):
    def __init__(self, params, initializer):
        super(WSU_MlpModel, self).__init__(params, initializer)

    def inference(self, x, y_title, x_length, user_idx, lstm_dropout):
        pre_concat = pre_concat_mlp(x_to_wsu(x, y_title, user_idx, self.params),
                lstm_dropout,
                self.params,
                scope='pre-concat')
        '''
        post_concat = post_concat_mlp(x_to_s(x, self.params),
                y_title,
                x_length,
                user_idx,
                lstm_dropout,
                self.params,
                scope='post-concat')
        '''
        hidden = linear(pre_concat,
                output_dim=self.dim_hidden,
                dropout_rate=lstm_dropout,
                activation=tf.nn.relu,
                scope='Hiddden')
        
        output = linear(dropout(pre_concat, lstm_dropout),
                output_dim=self.dim_output,
                scope='Output')
        return output

    def optimize(self, cost):
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), self.max_grad_norm)
        return self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)

    def optimize_user(self, cost):
        tvars = [v for v in tf.trainable_variables() if v.name == "embeds/user_embedding:0"]
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), self.max_grad_norm)
        return self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
