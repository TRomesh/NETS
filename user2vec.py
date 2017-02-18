import pickle
import numpy as np
import os
from datetime import datetime
import nltk
import operator
# from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def concat(user_slot_freq_vectors_pkl_path, user_title_word_freq_vectors_pkl_path, _output_dir, dim):
    print('\nConcatenating..')
    output_path = _output_dir + '/user_vectors (concat_sf_tf).%d' % dim + 'd.pkl'

    name_sf, dim_sf, user_slot_freq_dict = pickle.load(open(user_slot_freq_vectors_pkl_path, "rb"))
    name_tf, dim_tf, user_title_freq_dict = pickle.load(open(user_title_word_freq_vectors_pkl_path, "rb"))

    concat_dict = dict()
    # assume both dictionaries have same user list
    for user_id in user_slot_freq_dict:
        concat_dict[user_id] = np.concatenate((user_slot_freq_dict.get(user_id), user_title_freq_dict.get(user_id)),
                                              axis=0)

    # 0: name, 1: #slot, # 2: user vector dictionary
    user_digest = ('user vector: concat of slot/title freq', dim, concat_dict)

    pickle.dump(user_digest, open(output_path, "wb"))
    print('DONE', output_path)


def write_user2vec_slot_freq(_input_path, _output_dir, _output_slot_n):
    if not os.path.exists(_output_dir):
        os.makedirs(_output_dir)
    raw_slot_n = 336  # constant

    user_vectors_output_file = _output_dir + '/user_vectors (slot) (freq).%d' % _output_slot_n + 'd.pkl'

    user_count = 0
    total_event_count = 0
    total_week_count = 0

    user_dict = dict()

    file_names = os.listdir(_input_path)
    for filename in file_names:
        full_filename = os.path.join(_input_path, filename)
        user_count += 1
        if user_count % 1000 == 0:
            print(datetime.now(), 'user', user_count)

        count_dict = dict()
        event_num = 0
        week_dict = set()
        with open(full_filename, 'r', encoding='utf-8') as x_file:
            for line in x_file:
                event_features = line.split('\t')
                if len(event_features) != 6:
                    print('invalid line:', filename, line)
                    continue
                event_num += 1
                slot = int(event_features[5]) // (raw_slot_n // _output_slot_n)
                slot_count = count_dict.get(slot)
                count_dict[slot] = 1 if slot_count is None else slot_count + 1
                week_dict.add((event_features[0], event_features[1]))

        total_event_count += event_num
        total_week_count += len(week_dict)
        sorted_count_dict = sorted(count_dict.items(), key=operator.itemgetter(1), reverse=True)
        # print(filename, event_num, sorted_count_dict)

        slot_weight_vector = [0.] * _output_slot_n
        for _slot, cnt in sorted_count_dict:
            slot_weight_vector[_slot] = cnt / event_num
        assert 1. - sum(slot_weight_vector) < 0.000001
        # print(filename, event_num, sum(slot_weight_vector), slot_weight_vector)

        u_id = 'u_' + os.path.splitext(os.path.basename(filename))[0]
        user_dict[u_id] = slot_weight_vector

    print('#user', user_count)
    print('#event', total_event_count)
    print('#week', total_week_count)

    # 0: name, 1: #slot, 2: user vector dictionary
    user_digest = ('user slot frequency vectors', _output_slot_n, user_dict)
    pickle.dump(user_digest, open(user_vectors_output_file, "wb"))
    print('Saved', user_vectors_output_file)
    return user_vectors_output_file


def write_title_word_freq(_input_path, _output_dir, _output_dim):
    if not os.path.exists(_output_dir):
        os.makedirs(_output_dir)

    cal2vec_g_path = os.path.expanduser('~') + '/.secret/embedding/model/glove_init (2-300) (special).pickle'

    user_count = 0
    total_event_count = 0
    total_week_count = 0

    u_id_list = list()
    user_vector_list = list()

    word_set = set()

    def load_cal2vec(_cal2vec_path):
        print('Loading Cal2vec...', _cal2vec_path, end=' ')
        with open(_cal2vec_path, 'rb') as f:
            _final_embeddings, _reverse_dictionary, _dictionary, _count = pickle.load(f)
            print('Done!', len(_dictionary))
            return _final_embeddings, _reverse_dictionary, _dictionary, _count

    _, __, dictionary, ___ = load_cal2vec(cal2vec_g_path)

    file_names = os.listdir(_input_path)
    for filename in file_names:
        full_filename = os.path.join(_input_path, filename)
        user_count += 1
        if user_count % 500 == 0:
            print(datetime.now(), 'user', user_count)

        word_freq_dict = dict()
        event_num = 0
        user_word_count = 0
        week_set = set()
        with open(full_filename, 'r', encoding='utf-8') as x_file:
            for line in x_file:
                event_features = line.split('\t')
                if len(event_features) != 6:
                    print('invalid line:', filename, line)
                    continue
                event_num += 1

                title_words = nltk.word_tokenize(event_features[4].replace('\t', ' '))
                for word in title_words:
                    word_freq = word_freq_dict.get(word)
                    word_freq_dict[word] = 1 if word_freq is None else word_freq + 1

                week_set.add((event_features[0], event_features[1]))

        total_event_count += event_num
        total_week_count += len(week_set)

        cal2vec_word_id_freq_dict = dict()
        for word in word_freq_dict:
            if word in dictionary:
                word_id = dictionary[word]
                word_set.add(word)
            else:
                word_id = dictionary['UNK']

            word_count = word_freq_dict.get(word)

            user_word_count += word_count

            cal2vec_word_id_freq = cal2vec_word_id_freq_dict.get(word_id)
            cal2vec_word_id_freq_dict[word_id] = word_count if cal2vec_word_id_freq is None else \
                cal2vec_word_id_freq + word_count

        # print(filename, event_num, len(cal2vec_word_id_freq_dict))

        word_weight_vector = [0.] * len(dictionary)
        for word_id in cal2vec_word_id_freq_dict:
            word_weight_vector[word_id] = cal2vec_word_id_freq_dict.get(word_id) / user_word_count
        # import numpy as np
        # word_weight_vector = np.random.uniform(-0.1, 0.1, 5000)  # temp
        # print(sum(slot_weight_vector))
        assert 1. - sum(word_weight_vector) < 0.000001
        # print(filename, event_num, sum(slot_weight_vector), slot_weight_vector)
        u_id = 'u_' + os.path.splitext(os.path.basename(filename))[0]
        u_id_list.append(u_id)
        user_vector_list.append(word_weight_vector)

    print('#word_in_dictionary_set', len(word_set))

    print('\nDimensionality reduction..')
    '''
    # http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    tsne = TSNE(n_components=output_dim, perplexity=30.0, early_exaggeration=4.0, learning_rate=1000.0, n_iter=1000,
                n_iter_without_progress=30, min_grad_norm=1e-07, metric='euclidean', init='pca', verbose=0,
                random_state=None, method='barnes_hut', angle=0.5)
    dim_reduced_user_vector_list = tsne.fit_transform(user_vector_list)
    '''

    # http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    pca = PCA(n_components=_output_dim, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto',
              random_state=None)
    dim_reduced_user_vector_list = pca.fit_transform(user_vector_list)

    user_dict = dict()
    for uid, user2vec in zip(u_id_list, dim_reduced_user_vector_list):
        user_dict[uid] = user2vec

    user_digest = ('user title word frequency vectors', _output_dim, user_dict)  # 0: name, 1: #slot,
    # 2: user vector dictionary
    user_vectors_output_file = output_dir + '/user_vectors (title) (freq).%d' % _output_dim + 'd.pkl'
    pickle.dump(user_digest, open(user_vectors_output_file, "wb"))
    print('Saved', user_vectors_output_file)
    return user_vectors_output_file


if __name__ == "__main__":
    output_dim = 336
    input_path = './data/inputs'
    output_dir = './data/embedding'
    user_vectors_slot_freq_output_file = write_user2vec_slot_freq(input_path, output_dir, output_dim // 2)
    user_vectors_title_word_freq_output_file = write_title_word_freq(input_path, output_dir, output_dim // 2)
    concat(user_vectors_slot_freq_output_file, user_vectors_title_word_freq_output_file, output_dir, output_dim)
