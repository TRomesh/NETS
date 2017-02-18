import os
import time
import pprint
import numpy as np
import tensorflow as tf
from datetime import datetime

from dataset import *
from model import *
from util import *


FLAGS = tf.app.flags.FLAGS
# Basic parameters
tf.app.flags.DEFINE_integer('dim_input', 4 * 42, 'Dimension of input.')
tf.app.flags.DEFINE_integer('dim_output', 4 * 42, 'Dimension of output.')
tf.app.flags.DEFINE_integer('dim_word_embedding', 300, 'Dimension of word embedding.')
tf.app.flags.DEFINE_integer('dim_slot_embedding', 300, 'Dimension of slot embedding.')
tf.app.flags.DEFINE_integer('dim_user_embedding', 336, 'Dimension of user embedding.')
tf.app.flags.DEFINE_integer('user_size', 705, 'Size of total user (train/valid/test).')
tf.app.flags.DEFINE_integer('max_time_step', 116, 'Maximum time step of RNN.')
tf.app.flags.DEFINE_integer('max_slot_num', 12, 'Maximum number of slots per event.')
tf.app.flags.DEFINE_integer('max_title_len', 10, 'Maximum length of title tokens per event.')
tf.app.flags.DEFINE_integer('max_grad_norm', 5, 'Maximum gradient for gradient clipping.')
tf.app.flags.DEFINE_integer('num_week_slot', 336, 'Original number of slots (30 min) in a week.')
tf.app.flags.DEFINE_integer('num_day_slot', 48, 'Original number of slots (30 min) in a day.')
tf.app.flags.DEFINE_integer('slot_col_idx', 5, 'Index of start slot.')
tf.app.flags.DEFINE_integer('batch_size', 300, 'Size of mini-batch.')
tf.app.flags.DEFINE_integer('voca_size', 50000, 'Size of vocabulary in word embedding.')
tf.app.flags.DEFINE_integer('train_epoch', 100, 'Training epoch.')
tf.app.flags.DEFINE_float('train_ratio', 0.6, 'Dataset train ratio.')
tf.app.flags.DEFINE_float('test_ratio', 0.4, 'Dataset test ratio.')
tf.app.flags.DEFINE_float('seen_user_ratio', 0.605, 'Dataset seen user ratio.')
tf.app.flags.DEFINE_float('past_event_ratio', 0.650, 'Dataset past event ratio.')
tf.app.flags.DEFINE_float('acc_top_n', 5, 'N of top N accuracy.')
tf.app.flags.DEFINE_float('decay_rate', 0.95, 'Decay rate.')
tf.app.flags.DEFINE_float('decay_step', 100, 'Decay steps.')

# Validation hyper parameters
tf.app.flags.DEFINE_integer('valid_iteration', 1, 'Number of validations.')
tf.app.flags.DEFINE_integer('dim_hidden', 200, 'Dimension of input hidden layer.')
tf.app.flags.DEFINE_integer('dim_hidden_min', 300, 'Minimum dimension of input hidden layer.')
tf.app.flags.DEFINE_integer('dim_hidden_max', 449, 'Maximum dimension of input hidden layer.')
tf.app.flags.DEFINE_integer('dim_rnn_cell', 200, 'Dimension of RNN cell.')
tf.app.flags.DEFINE_integer('dim_rnn_cell_min', 100, 'Minimum dimension of RNN cell.')
tf.app.flags.DEFINE_integer('dim_rnn_cell_max', 399, 'Maximum dimension of RNN cell.')
tf.app.flags.DEFINE_float('learning_rate', 5e-3, 'Learning rate of the model.')
tf.app.flags.DEFINE_float('learning_rate_min', 5e-4, 'Minimum learning rate of the model.')
tf.app.flags.DEFINE_float('learning_rate_max', 5e-3, 'Maximum learning rate of the model.')
tf.app.flags.DEFINE_float('lstm_dropout', 0.5, 'Dropout of LSTM.')
tf.app.flags.DEFINE_float('lstm_dropout_min', 0.3, 'Minimum dropout of LSTM.')
tf.app.flags.DEFINE_float('lstm_dropout_max', 0.8, 'Maximum dropout of LSTM.')
tf.app.flags.DEFINE_integer('lstm_layer', 1, 'Number of layers in LSTM.')
tf.app.flags.DEFINE_integer('lstm_layer_min', 1, 'Minimum number of layers in LSTM.')
tf.app.flags.DEFINE_integer('lstm_layer_max', 1, 'Maximum number of layers in LSTM.')

# Model, task settings
tf.app.flags.DEFINE_boolean('default_params', True, 'True to use default hyper parameters.')
tf.app.flags.DEFINE_boolean('shuffle_data', True, 'True to shuffle dataset')
tf.app.flags.DEFINE_boolean('cold_start', False, 'True(False) to make the task cold(warm) start.')
tf.app.flags.DEFINE_boolean('pre_concat', True, 'True(False) to make the model pre(post) concat.')
tf.app.flags.DEFINE_boolean('embed_word', True, 'True(False) to embed word.')
tf.app.flags.DEFINE_boolean('embed_slot', True, 'True(False) to embed slot.')
tf.app.flags.DEFINE_boolean('embed_user', True, 'True(False) to embed user.')
tf.app.flags.DEFINE_boolean('mlp', False, 'True(False) to make lstm to mlp.')
tf.app.flags.DEFINE_boolean('we_trainable', False, 'True(False) to make cal2vec trainable.')
tf.app.flags.DEFINE_boolean('ue_trainable', False, 'True(False) to make user2vec trainable.')
tf.app.flags.DEFINE_boolean('load_pretrained', False, 'True to load pretrained model.')
tf.app.flags.DEFINE_boolean('load_test', False, 'True to load saved model when testing.')
tf.app.flags.DEFINE_boolean('save', False, 'True to save trained model')
tf.app.flags.DEFINE_boolean('train', True, 'True to train model')
tf.app.flags.DEFINE_boolean('test', False, 'True to test model')
tf.app.flags.DEFINE_boolean('eval_last_step_only', False, 'True to evaluate only last steps of RNN')

# File IO path
tf.app.flags.DEFINE_string('calendar_data_dir', './data/inputs', 'Calendar data directory.')
tf.app.flags.DEFINE_string('cal2vec_path', './data/embedding/glove_init (2-300) (special).pickle', 'Cal2vec path.')
tf.app.flags.DEFINE_string('user2vec_path', './data/embedding/user_vectors (concat_sf_tf).336d.pkl', 'User2vec path.')
tf.app.flags.DEFINE_string('avg2vec_path', './data/embedding/user_vectors (avg).336d.pkl', 'Averaged user2vec path.')
tf.app.flags.DEFINE_string('user_list_path', './data/user_list (705).txt', 'User list path.')
tf.app.flags.DEFINE_string('valid_result_path', './result/validation_' + datetime.now().strftime("%Y%m%d") + '.txt',
                           'Validation results.')
tf.app.flags.DEFINE_string('checkpoint_dir', os.getcwd() + '/result/model_' + datetime.now().strftime("%Y%m%d_%H%M%S")
                           + '_validate', 'Model save directory.')
tf.app.flags.DEFINE_string('load_model_name', 'model.ckpt', 'When load is True, specify.')
tf.app.flags.DEFINE_string('save_model_name', 'save.ckpt', 'When save is True, specify.')


def sample_parameters(params):
    combination = [
            params['dim_hidden'],
            params['dim_rnn_cell'],
            params['learning_rate'],
            params['lstm_dropout'],
            params['lstm_layer']
    ]

    if not params['default_params']:
        combination[0] = params['dim_hidden'] = int(np.random.uniform(
                params['dim_hidden_min'],
                params['dim_hidden_max']) // 50) * 50 
        combination[1] = params['dim_rnn_cell'] = int(np.random.uniform(
                params['dim_rnn_cell_min'],
                params['dim_rnn_cell_max']) // 50) * 50
        combination[2] = params['learning_rate'] = float('{0:.5f}'.format(np.random.uniform(
                params['learning_rate_min'],
                params['learning_rate_max'])))
        combination[3] = params['lstm_dropout'] = float('{0:.5f}'.format(np.random.uniform(
                params['lstm_dropout_min'],
                params['lstm_dropout_max'])))
        combination[4] = params['lstm_layer'] = int(np.random.uniform(
                params['lstm_layer_min'],
                params['lstm_layer_max']))

    return params, combination


def experiment(clf, dataset, params):
    min_cost = 99999
    max_top1 = 0
    max_top5 = 0
    max_top1_epoch = 0
    # test_epoch = 5      # Test for every 5 epochs
    total_duration = 0.
    nochange_cnt = 0
    early_stop = 3

    load_pretrained = params['load_pretrained']
    # load_test = params['load_test']
    cold_start = params['cold_start']
    save = params['save']
    train = params['train']
    test = params['test']
    checkpoint_dir = params['checkpoint_dir']
    load_model_name = params['load_model_name']
    save_model_name = params['save_model_name']
    train_epoch = params['train_epoch']
    # embed_word = params['embed_word']

    train_data, valid_data, test_data = dataset 
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    if load_pretrained:
        try:
            print("Resuming! => ", end='')
            clf.load(os.path.join(checkpoint_dir, load_model_name))
        except ValueError:
            print("Starting a new model" + os.path.join(checkpoint_dir, save_model_name))
            clf.save(os.path.join(checkpoint_dir, save_model_name))

    print('Input', params['dim_input'], 'output', params['dim_output'])
    print('[Model ready!]\n')

    if train:
        print('### TRAINING ###')
        start_time = time.time()
        for epoch in range(train_epoch):
            # epoch_time = time.time()

            print("Epoch #%d" % (epoch + 1))

            clf.run(train_data, train=True)
            total_duration = time.time() - start_time
            valid_cost, valid_acc1, valid_acc5, results = clf.run(valid_data, validate=True)
            if valid_acc1 > max_top1:
                max_top1 = valid_acc1
                max_top5 = valid_acc5
                max_top1_epoch = epoch

            if valid_cost < min_cost - 5e-3:
                nochange_cnt = 0
                min_cost = valid_cost
            else:
                nochange_cnt += 1
                print("no change count %d" % nochange_cnt)

            print("Elapsed Time: {:.2f}".format(total_duration / 60), "Minutes")
            print("Best Model Top1: %.3f%%" % max_top1, "Top5: %.3f%%" % max_top5, '@ epoch: %d' % (max_top1_epoch+1))

            # Save best performance model
            if valid_acc1 == max_top1 and save:
                clf.save(os.path.join(checkpoint_dir, save_model_name))

            # Early stop
            if nochange_cnt == early_stop:
                print("Early stopping applied.")
                clf.run(test_data, user_update=cold_start)
                break

            print()
        print('### END OF TRAINING ###\n')

    if test:
        print('### TESTING ###') 
        if train and save:
            clf.load(os.path.join(checkpoint_dir, save_model_name))
        elif train is False:
            clf.load(os.path.join(checkpoint_dir, load_model_name))
        clf.run(test_data, user_update=cold_start)
        print("Elapsed Time: {:.2f}".format(total_duration / 60), "Minutes", datetime.now())
        print('### END OF TESTING ###\n')
    
    clf.reset_graph()
    return max_top1, max_top5, max_top1_epoch


def main(_):
    saved_params = FLAGS.__flags
    pprint.PrettyPrinter().pprint(saved_params)
    if not os.path.exists(os.getcwd() + '/result'):
        os.mkdir(os.getcwd() + '/result')
    validation_writer = open(saved_params['valid_result_path'], 'a')
    model_name = 'MLP_' if saved_params['mlp'] else 'LSTM_'
    model_name += 'W' if saved_params['embed_word'] else ''
    model_name += 'S' if saved_params['embed_slot'] else 'X'
    model_name += 'U' if saved_params['embed_user'] else ''

    validation_writer.write(model_name + '\n')
    validation_writer.write("[dim_hidden, dim_rnn_cell, learning_rate, lstm_dropout, lstm_layer]\n")
    validation_writer.write("combination\ttop1\ttop5\tepoch\n")

    """
    Dataset consists of 5 elements:
        train, valid, test calendar data (3)
        cal2vec, user2vec (2)
    """
    dataset = get_data(saved_params)
    print('[Data loading done!]\n')
    
    for _ in range(saved_params['valid_iteration']):
        params, combination = sample_parameters(saved_params.copy())
        print(model_name, saved_params['valid_result_path'], 'parameter sets: ', end='')
        pprint.PrettyPrinter().pprint(combination)
        assert params['cold_start'] == params['ue_trainable'], 'Turn on(off) ue_trainable for cold(warm) start'

        if not params['mlp']:
            wsu_model = WSU_LstmModel(params, dataset[3:])
        else:
            wsu_model = WSU_MlpModel(params, dataset[3:])
        top1, top5, ep = experiment(wsu_model, dataset[:3], params)
        
        validation_writer.write(str(combination) + '\t')
        validation_writer.write(str(top1) + '\t' + str(top5) + '\tEp:' + str(ep) + '\n')

    validation_writer.close()

if __name__ == "__main__":
    tf.app.run()
