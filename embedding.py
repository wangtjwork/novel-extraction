import os
import json
import numpy as np

MAX_NB_WORDS = 10000
VALIDATION_SPLIT = 0.1
EMBEDDING_DIM = 100
MAX_SEQUENCE_LENGTH = 40000
HIDDEN_SIZE = 600
NOVEL_DIR = 'data/novel/'
LABEL_DIR = 'data/label/'
TRUE_SEQUENCE_LENGTH = 39194

def changeTrain(x_train, y_train):
    x_tra = []
    for i in range(len(x_train)):
        x_tra.append(np.append(x_train[i], y_train[i]))
    return np.asarray(x_tra)

def changeLabel(labels):
    label = []
    for each_label in labels:
        toadd = []
        for element in each_label:
            addarray = np.zeros(word_num + 1)
            addarray[element] = 1
            toadd.append(addarray)
        label.append(toadd)

    return np.asarray(label)

texts = []  # list of text samples
labels = []  # list of label ids


for name in sorted(os.listdir(NOVEL_DIR)):
    novel = os.path.join(NOVEL_DIR, name)
    label = os.path.join(LABEL_DIR, name)
    if os.path.exists(label):
        f = open(novel)
        texts.append(f.read())
        f.close()
        f = open(label)
        labels.append(f.read())
        f.close
    else:
        print "not found %s" % label

"""
with open('data/signalmedia-1m.jsonl', 'r') as f:
    for line in f:
        jfile = json.loads(line)
        texts.append(jfile["content"])
        labels.append(jfile["title"])

print('Found %s texts.' % len(texts))
exit()
"""

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts + labels)
data_sequences = tokenizer.texts_to_sequences(texts)
label_sequences = tokenizer.texts_to_sequences(labels)

word_index = tokenizer.word_index
word_num = len(word_index)
print('Found %s unique tokens.' % word_num)

data = pad_sequences(data_sequences)
label = pad_sequences(label_sequences)


print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', label.shape)


# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
label = label[indices]
label1 = pad_sequences(label_sequences, maxlen = 39937)
label1 = label1[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = label[:-nb_validation_samples]
x_train = changeTrain(x_train, y_train)
y_train = label1[:-nb_validation_samples]
y_train = changeLabel(y_train)
x_val = data[-nb_validation_samples:]
y_val = label1[-nb_validation_samples:]
y_val = changeLabel(y_val)

embeddings_index = {}
f = open('glove.6B.100d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()


print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

from keras.models import Sequential
from keras.layers.core import Lambda
from keras.layers.embeddings import Embedding
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector
from keras.layers.recurrent import LSTM

model = Sequential()
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=TRUE_SEQUENCE_LENGTH + 742,
                            trainable=False)
model.add(embedding_layer)
"""
print embedding_layer.output_shape

model.add(LSTM(HIDDEN_SIZE))
model.add(RepeatVector(TRUE_SEQUENCE_LENGTH + 742 + 1))
model.add(LSTM(HIDDEN_SIZE, return_sequences = True))
model.add(LSTM(HIDDEN_SIZE, return_sequences = True))
model.add(LSTM(HIDDEN_SIZE, return_sequences = True))
model.add(LSTM(HIDDEN_SIZE, return_sequences = True))
model.add(TimeDistributed(Dense(len(word_index) + 1)))
layer1 = Activation('softmax')
model.add(layer1)
print layer1.output_shape

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train, validation_data=(x_val, y_val),
          nb_epoch=10, batch_size=10)

#sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
#embedded_sequences = embedding_layer(sequence_input)
"""


import seq2seq
from seq2seq.models import Seq2Seq

model.add(Seq2Seq(input_shape=(TRUE_SEQUENCE_LENGTH, EMBEDDING_DIM), hidden_dim=HIDDEN_SIZE, output_length=742, output_dim=19, depth=4))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train, validation_data=(x_val, y_val),
          nb_epoch=10, batch_size=10)
