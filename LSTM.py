# LSTM neural network to generate text from South Park dialogue

# Data from here https://www.kaggle.com/tovarischsukhov/southparklines
# Loosely following the structure from 
# https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py
# and 
# https://towardsdatascience.com/generating-text-using-a-recurrent-neural-network-1c3bfee27a5e

from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau

import pandas as pd
import numpy as np
import random
import sys
import io
import re

"""
THINGS TO TRY:
 - Use wordembeddings as input https://adventuresinmachinelearning.com/keras-lstm-tutorial/
 - BIGGER MODLES ON GPU
"""

def containsAny(str, set):
    """ Check whether sequence str contains ANY of the items in set. """
    return 1 in [c in str for c in set]


# A couple of helper functions for training
def sample(preds, temperature=1.0):
    """
    Helper function to sample an index from a probability array
    Temperature is the amount of freedom the model has when
    generating text
    """
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, _):
    """
    Function invoked at end of each epoch. Prints generated text.
    """
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

##### ----------------- CLEANING DATA
"""
# Reading csv and combining to a single string of text
df = pd.read_csv('All-seasons.csv')
# Checking contents of the dataframe
df.describe()

# Only keeping data from the first 7 seasons to not spend an eternity fitting models
keep = ['1','2','3','4','5','6','7']
df = df[df['Season'].isin(keep)]
df = df.reset_index()


# Making newlines their own entitiy
df['Line'] = [line.replace('\n', ' \n') for line in df['Line']]

# Adding the speaker to the line
df['Line'] = [character + ": " + line for line, character in zip(df['Line'], df['Character'])]

# Getting all the lines into a single list
lines = df['Line'].str.cat().lower()

chars = sorted(list(set(lines)))
len(chars) # lots of strange characters.. let's remove some of the weird ones

remove = chars[61:]
remove = ''.join(remove)

bad_indices = [idx for idx, line in enumerate(df['Line']) 
    if containsAny(line, remove) == True]
# only 219 lines contains the bad indices, removing them
df = df.drop(bad_indices)

lines = df['Line'].str.cat().lower()
# checking how many characters/tokens/types there are
len(lines) # 2,256,997
len(lines.split()) # 396,412
len(set(lines.split())) # 33,984
# Over 2 million characters, that's more than enough!

# Saving to txt
with open('south_park_text.txt', 'w') as f:
    f.write(lines)
"""


##### ------------ STARTING ANALYSIS

### Reading the text and starting the analysis
with io.open('south_park_text.txt', encoding='utf-8') as f:
    text = f.read()


chars = sorted(list(set(text)))
print('total chars:', len(chars))
# down to 61 different characters, that's fine

# Creating a mapping from character to integer and vice versa
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
# This is necesarry since the neural network expects numbers
# as input, but we need to be able to recreate the output as
# normal text

# cut the text in semi-redundant sequences of maxlen characters
# Essentially, cutting the text into 40 character slices,
# with a stride of 3. Each sentence serves as an input to the network
# Having a stride/step of 3 allows for more inputs
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences)) # 7,523,219



# Turning the data into a boolean array which can be fed to the model
print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

print(x.shape) # 3 dimensional, the first being the number of sentences,
# the second being the length of sentences, the last being the total number 
# of different characters)
print(y.shape) # y is the character following each sentence
# so first dimension is the same, but second dimension is the total
# number of characters

# build the model: a single LSTM
print('Build model...')

#M1 and M2

#model = Sequential()
#model.add(LSTM(128, input_shape=(maxlen, len(chars)), return_sequences = True))
#model.add(LSTM(128))
#model.add(Dense(len(chars), activation='softmax'))


model = Sequential()
model.add(LSTM(128*2, input_shape=(maxlen, len(chars)), return_sequences = True))
model.add(LSTM(128*2))
model.add(Dense(len(chars), activation='softmax'))

#optimizer = RMSprop(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Making it print a sample text everytime it finishes an epoch
print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

# Making it save the weights each time the model improves
filepath = "weights_4_lstm.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss',
                             verbose=1, save_best_only=True,
                             mode='min')

# Making it reduce the learning rate if it plateous
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,
                              patience=1, min_lr=0.001)
                              
callbacks = [print_callback, checkpoint, reduce_lr]

model.load_weights('weights_4_lstm.hdf5')
# Fitting the model
model.fit(x, y,
          batch_size=128,
          epochs=0,
          callbacks=callbacks)

# m1 epochs: 17 + 17
# m2 epochs: 13 +  17 + 4
# m3 epochs: 1 +


# Creating a helper function to generate text using the model
def generate_text(length, diversity, seed = None):
    """
    Uses the LSTM model to generate a random text of the specified
    length. Diversity controls how much the model should stick
    to the training data. 
    Seed = None takes a random sentence from the training data
    Otherwise specify own seed with length maxlen
    """
    if seed == None:
        start_index = random.randint(0, len(text) - maxlen - 1)
        sentence = text[start_index: start_index + maxlen]

    else:
        if len(seed) != maxlen:
            raise ValueError(f'Length of seed sentence is {len(seed)} not {maxlen}')
        sentence = seed

    generated = ''
    generated += sentence
    for i in range(length):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char
    return generated


t1 = "cartman: screw you guys, i'm going home."
print(generate_text(600, 0.8, t1))


