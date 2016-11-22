"""
from FileProcess import FileProcess

f = FileProcess('data/001_to_kill_a_mockingbird.txt')
f.getMainCharacter(5)
f.filterFile('Jem', 'newfile')
"""
MAX_NB_WORDS = 10000
MAX_SEQUENCE_LENGTH = 300
VALIDATION_SPLIT = 0.1
EMBEDDING_DIM = 100
HIDDEN_SIZE = 600

import numpy as np

def changeLabel(labels, word_num):
    label = []
    for each_label in labels:
        toadd = []
        for element in each_label:
            addarray = np.zeros(MAX_NB_WORDS + 1)
            addarray[element] = 1
            toadd.append(addarray)
        label.append(toadd)

    return np.asarray(label)

import os, json

texts = []
labels = []

f = open('data/signalmedia-1m.jsonl', 'r')
for _ in range(10000):
#for line in f:
    line = f.readline()
    jfile= json.loads(line)

    if len(jfile['content'].encode("utf-8").split(' ')) <= 300:
        texts.append(jfile["content"].encode("utf-8"))
        labels.append(jfile["title"].encode("utf-8"))

f.close()
print len(texts)
print len(labels)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts + labels)
data_sequences = tokenizer.texts_to_sequences(texts)
label_sequences = tokenizer.texts_to_sequences(labels)

word_index = tokenizer.word_index
word_num = len(word_index) + 1
print('Found %s unique tokens.' % word_num)

data = pad_sequences(data_sequences, maxlen = MAX_SEQUENCE_LENGTH)
label = pad_sequences(label_sequences)


print('Shape of data tensor:', data.shape)
label_shape = label.shape
print('Shape of label tensor:', label_shape)


# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
label = label[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = label[:-nb_validation_samples]
y_train = changeLabel(y_train, word_num)
x_val = data[-nb_validation_samples:]
y_val = label[-nb_validation_samples:]
y_val = changeLabel(y_val, word_num)

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

from keras.models import Sequential, Model
from keras.layers.core import Reshape
from keras.layers.embeddings import Embedding
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, Dropout, Input
from keras.layers.recurrent import LSTM
from recurrentshop import LSTMCell, RecurrentContainer
from seq2seq.cells import LSTMDecoderCell

model = Sequential()
embedding_layer = Embedding(word_num,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)


model.add(embedding_layer)
#print embedding_layer.output_shape
encode_layer = LSTM(HIDDEN_SIZE)
model.add(encode_layer)
#print encode_layer.output_shape
state_multiply = RepeatVector(label_shape[-1])
model.add(state_multiply)
#print state_multiply.output_shape
decode_layer = LSTM(HIDDEN_SIZE, return_sequences = True)
model.add(decode_layer)
#print decode_layer.output_shape
time_layer = TimeDistributed(Dense(MAX_NB_WORDS + 1))
model.add(time_layer)
#print time_layer.output_shape
layer1 = Activation('softmax')
model.add(layer1)
#print layer1.output_shape


"""
import seq2seq
from seq2seq.models import Seq2Seq

#model2 = Seq2Seq(input_length=MAX_SEQUENCE_LENGTH,input_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_SIZE, output_length=44, output_dim=MAX_NB_WORDS + 1)
model2 = Seq2Seq(batch_input_shape=(16, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM), hidden_dim=10, output_length=8, output_dim=20, depth=4, unroll = True)
model.add(model2)
print model2.output_shape
exit()
"""

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train, validation_data=(x_val, y_val),
          nb_epoch=4, batch_size=1000)

