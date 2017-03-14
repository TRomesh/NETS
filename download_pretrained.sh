#!/usr/bin/env bash


echo 'Downloading... It may take several minutes to receive files'

wget -P LSTM_TSU/data/embedding/ https://s3-us-west-1.amazonaws.com/ml-man/user_vectors\ \(avg\).336d.pkl
wget -P LSTM_TSU/data/embedding/ https://s3-us-west-1.amazonaws.com/ml-man/glove_init_special.300d.pkl
wget -P LSTM_TSU/result/pretrained/ https://s3-us-west-1.amazonaws.com/ml-man/model_pretrained/checkpoint
wget -P LSTM_TSU/result/pretrained/ https://s3-us-west-1.amazonaws.com/ml-man/model_pretrained/model.ckpt-1809.data-00000-of-00001
wget -P LSTM_TSU/result/pretrained/ https://s3-us-west-1.amazonaws.com/ml-man/model_pretrained/model.ckpt-1809.index
wget -P LSTM_TSU/result/pretrained/ https://s3-us-west-1.amazonaws.com/ml-man/model_pretrained/model.ckpt-1809.meta
