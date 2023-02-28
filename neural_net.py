from optparse import NO_DEFAULT
import tensorflow as tf
import numpy as np
import keras
import os
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from process_data import clean_csv, get_data_targets, get_file_list_from_dir
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences


class OurRNN():
    def __init__(self, hidden_units=1, input_shape=(None, 15), output_shape=2):
        model = Sequential()
        model.add(SimpleRNN(hidden_units, input_shape=input_shape, 
                        activation="relu"))
        model.add(Dense(units=output_shape, activation="linear")) # double regression problem
        model.compile(loss='mean_squared_error', optimizer='adam')
        self.model = model
        self.hidden_units = hidden_units
        self.input_shape = input_shape
        self.output_shape = output_shape
    
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train, batch_size=2, epochs=2)
        return self

    def predict(self, X_test):
        preds = []
        for x in X_test:
            t = x.shape[0]
            p = x.shape[1]
            # Reshape the input to the required sample_size x time_steps x features 
            x_input = np.reshape(x,(1, t, p))
            y_pred_model = self.model.predict(x_input)
            preds.append(y_pred_model)
        return preds



if __name__ == "__main__":

    VAR_NAMES = ['play_djoko', 'posx_djoko', 'posy_djoko', 'posx_opponent',
       'posy_opponent', 'targetx_ball', 'targety_ball', 'serve_djoko',
       'set_djoko', 'set_opponent', 'games_djoko', 'games_opponent',
       'duration', 'time', 'forehand']

    # get the data
    MAX_SEQUENCE_LENGTH = 10
    n_sequences = 0

    targets_all = []
    sequences_all = []
    to_remove = ["./data/loic_seq_1.csv", "./data/loic_seq_2.csv",
     "./data/loic_seq_3.csv", "./data/loic_seq_4.csv", "./data/loic_seq_5.csv"] # sequences to be treated later

    for file_path in get_file_list_from_dir(path=".", datadir="data"):
        if file_path not in to_remove: 
            df = clean_csv(file_path)
            df = df[VAR_NAMES] # make sure the columns are in the same order
            print("Shape of {}: {}".format(file_path, df.shape))
            print("try")
            sub_sequences, targets = get_data_targets(file_path)
            print("ok")
            for (sub_sequence, target) in zip(sub_sequences, targets):
                targets_all.append(target)
                sequences_all.append(sub_sequence)
                n_sequences += 1

    print("Total number of sequences:", n_sequences)

    # split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(sequences_all, targets_all,
                                                        train_size=0.8, random_state=0)
                            
    x_train = pad_sequences(x_train, maxlen=MAX_SEQUENCE_LENGTH, value=-100)
    x_test = pad_sequences(x_test, maxlen=MAX_SEQUENCE_LENGTH, value=-100)
    y_train = np.array(y_train)
    print(y_train.shape)
    y_test = np.array(y_test)
    print(x_test.shape)

    n_features = x_test.shape[2]

    # fit the RNN
    rnn = OurRNN(input_shape=(None, n_features), output_shape=2) 
    rnn.model.fit(x=x_train, y=y_train, batch_size=10, epochs=3)

    # evaluate the RNN
    #pred = rnn.predict([sequences_all[-1]])
    pred = rnn.predict(x_test)
    print(pred[0])
    print(y_test[0])



    

