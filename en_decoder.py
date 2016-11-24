import numpy as np
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

MAX_NB_WORDS = 10000
TEXT_DIR = 'data/texts.txt'
LABEL_DIR = 'data/labels.txt'
BATCH_SIZE = 1000
MAX_TEXT_LENGTH = 300
MAX_LABEL_LENGTH
VALIDATION_SPLIT = 0.1
EMBEDDING_DIM = 100
HIDDEN_SIZE = 600
NB_EPOCH = 1

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

def tokenGet(textDir, labelDir):
    texts = []
    labels = []
    ftext = open(textDir, 'r')
    for line in ftext:
        texts.append(line)
    ftext.close()

    flabel = open(labelDir, 'r')
    for line in flabel:
        labels.append(line)
    flabel.close()
    
    file_size = len(texts)
    
    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts + labels)
    return tokenizer, file_size

def dataSlice(textDir, labelDir, file_size, pointer_now):
    #input file_size is the total length of lines in the file
    #input pointer_now is the line we would like to start with
    i = pointer_now
    texts = []
    labels = []
    
    with open(textDir, 'r') as ftext:
        with open(labelDir, 'r') as flabel:
            for _ in range(i):
                ftext.readline()
                flabel.readline()
            for _ in range(BATCH_SIZE):
                text = ftext.readline()
                label = flabel.readline()
                texts.append(text)
                labels.append(label)
                i += 1
                if i >= file_size:
                    break
    
    return texts, labels

def dataShuffle(texts, labels, tokenizer):
    text_sequences = tokenizer.texts_to_sequences(texts)
    label_sequences = tokenizer.texts_to_sequences(labels)
    
    textList = pad_sequences(text_sequences, maxlen = MAX_TEXT_LENGTH)
    labelList = pad_sequences(label_sequences, maxlen = MAX_LABEL_LENGTH, padding = 'post')
    
    # split the data into a training set and a validation set
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    textList = data[indices]
    labelList = label[indices]
    nb_validation_samples = int(VALIDATION_SPLIT * textList.shape[0])

    x_train = textList[:-nb_validation_samples]
    y_train = labelList[:-nb_validation_samples]
    y_train = changeLabel(y_train)
    x_val = textList[-nb_validation_samples:]
    y_val = labelList[-nb_validation_samples:]
    y_val = changeLabel(y_val)
    
    return x_train, y_train, x_val, y_val

tokenizer, file_size = tokenGet(TEXT_DIR, LABEL_DIR)
word_index = tokenizer.word_index

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


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

spiltNum = (int) (file_size / BATCH_SIZE)
for _ in range(NB_EPOCH):
    for i in range(spiltNum):
        texts, labels = dataSlice(TEXT_DIR, LABEL_DIR, file_size, i * BATCH_SIZE)
        x_train, y_train, x_val, y_val = dataShuffle(texts, labels, tokenizer)

        model.fit(x_train, y_train, validation_data=(x_val, y_val),
              nb_epoch=1, batch_size=250)

model.save_weights('summary.h5')