import os
import pickle
import nltk
import math
import numpy as np
from operator import itemgetter
from collections import OrderedDict
from random import shuffle

from util import *

nltk.download('punkt')
week_itr = 0
print('[Utility download done!]\n')


def load_cal2vec(_cal2vec_path):
    with open(_cal2vec_path, 'rb') as f:
        _final_embeddings, _reverse_dictionary, _dictionary, _count = pickle.load(f)
        return _final_embeddings, _reverse_dictionary, _dictionary, _count


def get_calendar_data(data_dir, user_list_path):
    user_infos = list()
    user2idx = OrderedDict()
    user_list = list()
    total_week = 0
    total_event = 0
    max_year_week = 0

    found_user_list = os.path.exists(user_list_path)

    if found_user_list:
        print("Loading user list...", user_list_path, 'Done!')
        with open(user_list_path, 'r') as f:
            for line in f:
                user_list.append(line.strip())

    print("Loading input data...", data_dir, end=' ')

    for subdir, _, files in os.walk(data_dir):
        for filename in sorted(files):
            file_path = os.path.join(subdir, filename)
            user_key = os.path.splitext(os.path.basename(filename))[0]
            if found_user_list and user_key not in user_list:
                continue

            user = OrderedDict()
            user_event_num = 0
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    event_data = [str(k) for k in line[:-1].split('\t')]
                    if len(event_data) != 6:
                        print('invalid row:', len(event_data), 'line:', line[:-1])
                        continue
                    user_event_num += 1
                    year_week = event_data[0] + '_' + event_data[1]
                    datum = ['u_' + user_key,   # user
                             year_week,           # week
                             int(event_data[3]),  # duration
                             event_data[4],  # title
                             int(event_data[5]),  # slot
                             event_data[2]]       # sequence

                    user_yw = user.get(year_week)
                    if user_yw is None:
                        user_yw = list()
                        user_yw.append(datum)
                        user[year_week] = user_yw
                    else:
                        user_yw.append(datum)

                    if max_year_week <= len(user[year_week]):
                        max_year_week = len(user[year_week])

            user2idx['u_' + user_key] = len(user_infos)

            total_week += len(user)
            total_event += user_event_num
            user_infos.append((user_key, user, len(user), user_event_num))  # id, user data, #week, #event
    print("Done!")
    print("\tMax Week:", max_year_week, "Total Week:", total_week, 'Total Event', total_event)
    return user_infos, user2idx, total_event


def get_data(params):
    # get parameters
    data_dir = params['calendar_data_dir']
    user_list_path = params['user_list_path']
    cal2vec_path = params['cal2vec_path']
    user2vec_path = params['user2vec_path']
    avg2vec_path = params['avg2vec_path']
    shuffle_data = params['shuffle_data']
    cold_start = params['cold_start']

    # word vectors
    cal2vec_set = load_cal2vec(cal2vec_path)
    print('Loading word vectors...', cal2vec_path, 'Done!')

    # user vectors
    _, _, user2vec = pickle.load(open(user2vec_path, "rb"))
    print('Loading user vectors...', user2vec_path, 'Done!')

    # get calendar data
    users, user2idx, n_event = get_calendar_data(data_dir, user_list_path)
    user_cnt = len(users)
    params['user_size'] = user_cnt
    if user_cnt == 0:
        print('Not found users')
        return

    # return only corresponding user's vector
    user_idx2vec = list()
    if cold_start:
        avg2vec = pickle.load(open(avg2vec_path, "rb"))
        print('Loading average user vectors...', avg2vec_path, 'Done!')
        for _ in user2idx:
            user_idx2vec.append(avg2vec)
    else:
        for user_id in user2idx:
            user_idx2vec.append(user2vec.get(user_id))

    user2vec_set = [user_idx2vec, user2idx]

    train_data = list()
    valid_data = list()

    common_data = list()
    unseen_data = list()
    recent_data = list()
    test_data = list()

    if user_cnt > 1:
        event_accumulator = 0
        seen_ratio = 0.605
        past_ratio = 0.650
        for user_idx, (user_key, user_data, _, _) in enumerate(sorted(users, key=itemgetter(3))):
            user_week_cnt = len(user_data)
            for data_idx, (week_idx, week_data) in enumerate(user_data.items()):
                event_accumulator += len(week_data)
                if data_idx < user_week_cnt * past_ratio:
                    if event_accumulator < n_event * seen_ratio:
                        common_data.append(week_data)
                    else:
                        unseen_data.append(week_data)
                else:
                    if event_accumulator < n_event * seen_ratio:
                        recent_data.append(week_data)
                    else:
                        test_data.append(week_data)

        train_data = np.concatenate((common_data, (recent_data if cold_start else unseen_data)), axis=0)
        valid_data = unseen_data if cold_start else recent_data

        print('common_data', '#week', len(common_data), '#event', sum([len(wk) for wk in common_data]))
        print('unseen_data', '#week', len(unseen_data), '#event', sum([len(wk) for wk in unseen_data]))
        print('recent_data', '#week', len(recent_data), '#event', sum([len(wk) for wk in recent_data]))
    else:  # single user
        #  #week of train:valid:test = 6:2:2
        train_ratio = 0.6
        valid_ratio = 0.2
        # test_ratio = 1 - train_ratio - valid_ratio

        assert train_ratio + valid_ratio < 1., '%f %f' % (train_ratio, valid_ratio)

        user0 = users[0]
        user_week_cnt = len(user0[1])
        for data_idx, (week_idx, week_data) in enumerate(user0[1].items()):
            if data_idx < user_week_cnt * train_ratio:
                train_data.append(week_data)
            elif data_idx < user_week_cnt * (train_ratio + valid_ratio):
                valid_data.append(week_data)
            else:
                test_data.append(week_data)

    if shuffle_data:
        shuffle(train_data)
        shuffle(valid_data)
        shuffle(test_data)

    print('\t%s start task' % ('Cold' if cold_start else 'Warm'))
    print('\tTrain: #week %d, #event %d, %.1f%%' % (len(train_data), sum([len(wk) for wk in train_data]),
                                                    sum([len(wk) for wk in train_data])*100./n_event))
    print('\tValid: #week %d, #event %d, %.1f%%' % (len(valid_data), sum([len(wk) for wk in valid_data]),
                                                    sum([len(wk) for wk in valid_data])*100./n_event))
    print('\tTest : #week %d, #event %d, %.1f%%' % (len(test_data), sum([len(wk) for wk in test_data]),
                                                    sum([len(wk) for wk in test_data])*100./n_event))

    return train_data, valid_data, test_data, cal2vec_set, user2vec_set


def word_to_idx(title, max_length, dictionary):
    idxes = list()
    for cnt, word in enumerate(nltk.word_tokenize(title)):
        if cnt >= max_length:
            break
        idxes.append(dictionary[word if word in dictionary else 'UNK'])

    while len(idxes) != max_length:
        idxes.append(-1)
        # break
        
    return idxes


def get_num_class(raw_class, num_input, num_week_slot):
        return raw_class // (num_week_slot // num_input)


def get_duration_slots(event_features, before_slot, num_input, num_slot_raw, max_slot_num):
    slots = list()
    duration_slot_num = math.ceil(event_features[2] / (30 * num_slot_raw // num_input))
    if duration_slot_num == 1:
        slots.append(before_slot)
        while len(slots) < max_slot_num:
            slots.append(-1)
    elif duration_slot_num > 1:
        for delta in range(duration_slot_num):
            slots.append(before_slot + delta)
            if len(slots) == max_slot_num:
                break
        while len(slots) < max_slot_num:
            slots.append(-1)
    else:
        print('invalid duration_slot_num', duration_slot_num, event_features)
    assert len(slots) == max_slot_num
    return slots


def get_next_batch(week_set, user_dict, batch_sz, max_time_steps, word_dict, num_input, params):
    global week_itr
    _batch_x = list()
    _batch_x_title = list()
    _batch_y = list()
    _batch_y_title = list()
    _batch_seq_len = list()
    _batch_user_idx = list()
    eou_ = False

    max_slot_num = params['max_slot_num']
    dim_input = params['dim_input']
    num_week_slot = params['num_week_slot']
    max_title_length = params['max_title_len']
    
    while len(_batch_x) < batch_sz:
        for w_idx, events in enumerate(week_set):
            if w_idx < week_itr:
                continue
            x_pop_slot = list()
            y_pop_slot = list()
            y_pop_title = list()
            step_user_idx = list()
            for idx in range(max_time_steps):  # for zero padding
                if idx < len(events) - 1:
                    # print(idx, events[idx])
                    before_slot = get_num_class(events[idx][4], num_input, num_week_slot)
                    x_pop_slot.append(get_duration_slots(events[idx], before_slot, dim_input, num_week_slot,
                                                         max_slot_num))
                    current_slot = get_num_class(events[idx + 1][4], num_input, num_week_slot)
                    current_title = events[idx + 1][3]
                    y_pop_slot.append(current_slot)
                    y_pop_title.append(word_to_idx(current_title, max_title_length, word_dict))
                    step_user_idx.append(user_dict[events[idx][0]])

                    assert before_slot >= 0
                    assert events[idx][0] == events[idx+1][0]
                    assert events[idx][1] == events[idx+1][1]
                else:
                    x_pop_slot.append([-1] * max_slot_num)
                    y_pop_slot.append(0)
                    y_pop_title.append([-1] * max_title_length)
                    step_user_idx.append(-1)

            if x_pop_slot[0][0] > -1:
                _batch_x.append(x_pop_slot)
                _batch_y.append(y_pop_slot)
                _batch_y_title.append(y_pop_title)
                _batch_seq_len.append(len(events) - 1)
                _batch_user_idx.append(step_user_idx)
            else:
                assert True, "no batch generated"

            week_itr = w_idx + 1
            if len(_batch_x) >= batch_sz:
                break

            # code for batch debugging
            ''' 
            if len(_batch_x) == 10:
                for b_idx, (snapshots, labels, titles) in enumerate(zip(_batch_x, _batch_y, _batch_y_title)):
                    print('batch', b_idx)
                    for snapshot, label, title in zip(snapshots, labels, titles):
                        print(snapshot, '=>', label, ':', title)

                print('batch length', _batch_seq_len)
                sys.exit()
            '''

        if week_itr == len(week_set):
            week_itr = 0
            eou_ = True
            break

    return _batch_x, _batch_x_title, _batch_y, _batch_y_title, _batch_seq_len, _batch_user_idx, eou_
