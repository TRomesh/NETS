# NETS: Neural Event Time Scheduler with Title, Slot, and User Embedding
Donghyeon Kim and Jinhyuk Lee et al.

# Prerequisites
* GPU
* Python
    * Linux or Mac OS: [Python 3.4 or greater](https://www.python.org/downloads/)
    * Windows: [Python 3.5 or greater](https://www.python.org/downloads/)
* TensorFlow r1.0 or greater, GPU enabled
* A Google account, Google Calendar API client
* Gensim, nltk, numpy, python-dateutil, scikit-learn

# Prerequisites Installation Guide
* Linux: Ubuntu 16.04 LTS
```
$ sudo apt install python3-pip
$ sudo pip3 install tensorflow-gpu
$ sudo pip3 install --upgrade google-api-python-client
$ sudo pip3 install gensim nltk numpy python-dateutil scikit-learn
```

* Mac OS
```
$ sudo pip install tensorflow-gpu
$ sudo pip install --upgrade google-api-python-client
$ sudo pip install gensim nltk numpy python-dateutil scikit-learn
```

* Windows
    * "TensorFlow only supports version 3.5.x of Python on Windows." [[TF home](https://www.tensorflow.org/install/install_windows)] 2017-2-23
        * [Download python-3.5.3](https://www.python.org/ftp/python/3.5.3/python-3.5.3-amd64.exe) and install
    * Download and install pre-built binaries:
[numpy+mkl](http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy)
[scipy](http://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy)
[gensim](http://www.lfd.uci.edu/~gohlke/pythonlibs/#gensim)
[scikit-learn](http://www.lfd.uci.edu/~gohlke/pythonlibs/#scikit-learn)
```
# python 3.5.x 64-bit
> pip install tensorflow-gpu
> pip install nltk python-dateutil
> pip install .\Downloads\numpy-1.12.0+mkl-cp35-cp35m-win_amd64.whl
> pip install .\Downloads\scipy-0.18.1-cp35-cp35m-win_amd64.whl
> pip install .\Downloads\gensim-0.13.4.1-cp35-cp35m-win_amd64.whl
> pip install .\Downloads\scikit_learn-0.18.1-cp35-cp35m-win_amd64.whl
```

# Clone this repository
```
$ cd
$ git clone https://github.com/donghyeonk/NETS.git
```

* Move to the project directory.
```
$ cd NETS
```

# Create your user vector
* Get dataset from your Google calendar
    * Important: Download client_secret.json to the project folder before running get_google_calendar_events.py.
    * Please see [python quick start document](https://developers.google.com/google-apps/calendar/quickstart/python#step_1_turn_on_the_api_name).
```
# If your browser is on local machine
$ python3 get_google_calendar_events.py
```
```
# If your browser is on different machine
$ python3 get_google_calendar_events.py --noauth_local_webserver
```
* Check if &lt;primary_calendar_id>_events.txt file is in **LSTM_TSU/data/inputs** directory.
    * ex. primary@gmail.com_events.txt
* Event fields
    * Six columns: `[year, week, sequence in a week, duration (min), title, start time slot (0~335)]`
        * e.g., 2017\t4\t0\t60\tMeeting with Jake\t25
    * Sorted by year, week and sequence in a week
    * Time slot unit: 30 minutes

* Finally, you can create your user vector
```
$ python3 user2vec.py
```

# Get pre-trained word vectors, an averaged user vector, and a pre-trained model
* [GloVe (Jeffrey Pennington et al.)](http://nlp.stanford.edu/projects/glove/) based word (+special characters) vectors (300-dim, 59 MB)
* An averaged user vector for cold start (336-dim, 3 KB) 
* A pre-trained TensorFlow model (138 MB)
```
$ sh download_pretrained.sh
```

# Run LSTM_TSU
* Move to LSTM_TSU directory.
```
$ cd LSTM_TSU
```

* Run main.py
    * Default: One-hour based class (168), cold start, an averaged user vector
    * Results of the model could be different for each dataset.
```
$ python3 main.py
```

* An Example of a result of LSTM_TSU

TSU model

Input tensor Tensor("Reshape:0", shape=(?, 936), dtype=float32) dimension 936

pre-concat version

Word embedding initialized `[ 0.00974513 -0.122094    0.00709154  0.10331635  0.03429483]`

Slot embedding initialized `[-0.01252203  0.04094937  0.03608277  0.03158236 -0.051265  ]`

User embedding initialized `[ 0.00049178  0.00017338  0.00033416  0.00013555  0.00031524]` prefix_prob_sum 1.00000000202

Resuming! => Model restored from /home/&lt;YOUR_HOME>/NETS/LSTM_TSU/result/pretrained/model.ckpt-1809

Input 168 output 168

`[Model ready!]`

\### TESTING ###

Model restored from /home/&lt;YOUR_HOME>/NETS/LSTM_TSU/result/pretrained/model.ckpt-1809

Testing loss: 3.916 Top1 Accuracy: 7.609% Top5 Accuracy: 26.087% step: 1809

Elapsed Time: 0.00 Minutes &lt;END_TIME>

\### END OF TESTING ###
