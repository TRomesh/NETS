# NETS: Neural Event Time Scheduler with Title, Slot, and User Embedding
Donghyeon Kim and Jinhyuk Lee et al.

# Prerequisites
* GPU
* [Python 3.5 or greater](https://www.python.org/downloads/)
* TensorFlow r1.0 or greater, GPU enabled
* A Google account, Google Calendar API client
* Gensim, nltk, numpy, python-dateutil, scikit-learn
    
```
# Ubuntu 16.04 LTS
$ sudo apt install python3-pip
$ sudo pip3 install tensorflow-gpu
$ sudo pip3 install --upgrade google-api-python-client
$ sudo pip3 install gensim nltk numpy python-dateutil scikit-learn
```
```
# Mac and Windows
$ sudo pip install tensorflow-gpu
$ sudo pip install --upgrade google-api-python-client
$ sudo pip install gensim nltk numpy python-dateutil scikit-learn
```

# Clone this repository
```
$ git clone https://github.com/donghyeonk/NETS.git
```

* Move to the project directory
```
$ cd NETS
```

# Get your calendar events from your Google calendar
* Important: Download client_secret.json to the project folder before running google calendar events getter
* Please see [python quick start document](https://developers.google.com/google-apps/calendar/quickstart/python#step_1_turn_on_the_api_name)

```
# If your browser is on local machine
$ python3 get_google_caleandar_events.py
```
```
# If your browser is on different machine
$ python3 get_google_caleandar_events.py --noauth_local_webserver
```
* Check if <primary_calendar_id>_events.txt file is in **LSTM_TSU/data/inputs** directory
    * ex. primary@gmail.com_events.txt
* Event fields
    * Six columns: [year, week, sequence in a week, duration (min), title, start time slot (0~335)]
        * e.g., 2017\t4\t60\tMeeting with Jake\t25
    * Sorted by year, week and sequence in a week
    * Time slot unit: 30 minutes


# Get pre-trained word vectors
* Download a .pickle file to **LSTM_TSU/data/embedding** directory 
    * [GloVe (Jeffrey Pennington et al.)](http://nlp.stanford.edu/projects/glove/) based word (+special characters) vectors (300-dim, 59 MB)
```
$ wget -P data/embedding/ https://s3-us-west-1.amazonaws.com/ml-man/glove_init_special.300d.pkl
```

# Create your user vector
* Run after getting your calendar events
```
$ python3 user2vec.py
```

# Run NETS
* LSTM_TSU, 1 hour based class (168), warm start
```
$ python3 LSTM_TSU/main.py
```
