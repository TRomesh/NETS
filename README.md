# NETS: Neural Event Time Scheduler with Title, Slot, and User Embedding
Donghyeon Kim and Jinhyuk Lee et al.

# Pre-requisites
* GPU
* Python 3.5 or greater
* TensorFlow r0.12.1 or greater, GPU enabled
* Google Calendar API client
* Numpy, gensim, nltk
```
# Ubuntu 16.04 LTS
$ sudo apt install python3-pip
$ sudo pip3 install tensorflow-gpu
$ sudo pip3 install --upgrade google-api-python-client
$ sudo pip3 install numpy gensim nltk
```

# Clone this repository
```
$ git clone https://github.com/donghyeonk/NETS.git
$ cd NETS
```

# Get your calendar events
* From your Google calendar
```  
# Important: Download client_secret.json to the project folder before running google caleandar events getter
# (see https://developers.google.com/google-apps/calendar/quickstart/python)
$ python3 get_google_caleandar_events.py
```
* Check if <primary_calendar_id>_events.txt file is in __LSTM_TSU/data/inputs__ directory
    * ex. primary@gmail.com_events.txt
* Event fields
    * Example: 2017\t4\t60\tMeeting with Jake\t25
    * Length 6, [year, week, sequence in a week, duration (min), title, start time slot (0~335)]
    * Sorted by year, week and sequence
    * Time slot unit: 30 min.


# Get pre-trained word vectors
* Download a .pickle file to __LSTM_TSU/data/embedding__ directory 
    * [Download](https://drive.google.com/uc?export=download&confirm=oasY&id=0B3aQGyHHaGQCZVJkUnh0eEZPQm8) GloVe (Jeffrey Pennington et al.) based word vectors (+ special characters, 300-dim, 59 MB, see [GloVe home](http://nlp.stanford.edu/projects/glove/))
  

# Create your user vector
```
# Run after getting your events
$ python3 user2vec.py
```

# Run NETS
* LSTM_TSU, output 168-class, warm start
```
$ python3 LSTM_TSU/main.py
```
