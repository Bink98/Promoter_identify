import os
import numpy as np
from Bio import SeqIO
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import tensorflow as tf
from keras.models import Model
from keras.models import Sequential
from keras.models import model_from_yaml
from keras.layers.convolutional import Conv1D # Keras 2 syntax
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.initializers import Constant
from keras.callbacks import EarlyStopping

#from keras.callbacks import TensorBoard

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt

import time
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
# with tf.device('/cpu:0'):
os.chdir(r"E:\Users\Bink\Documents\iGEM\panda\Promoter_identify")
print(os.getcwd())

oneHotDict = {'A': np.array([1, 0, 0, 0]), 'G': np.array([0, 1, 0, 0]), 'C': np.array([0, 0, 1, 0]),
              'T': np.array([0, 0, 0, 1]), 'N': np.array([0, 0, 0, 0])}


def vectorizeseq(seq):
    return np.array(list(map(lambda letter: oneHotDict[letter], seq)))


# This is why we use one hot encoder to perform “binarization”
# of the category and include it as a feature to train the model.
def onehot_fasta_as_numpy(fname):
    allseq = []
    for record in SeqIO.parse("data/{0}.fa".format(fname), "fasta"):
        one_hot_seq = vectorizeseq(str(record.seq))
        allseq.append(one_hot_seq)
    return np.array(allseq)

# Create function returning a compiled network
def create_network(nb_filters,kernel_size,length,dimensions,pool_size):
    # Initialize network
    classifier = Sequential()
    # Layers 1 and 2: Add two convolutional layers
    classifier.add(
        Conv1D(filters=nb_filters, kernel_size=kernel_size, input_shape=(length, dimensions), activation="relu"))

    # Layer 3: Max-Pool
    classifier.add(MaxPooling1D(pool_size=pool_size))

    # Layer 7: Flatten to be fed into fully-connected layers
    classifier.add(Flatten())

    # Layers 8: fully connected layer with 256 nodes
    classifier.add(Dense(units=128, activation='relu'))

    # Layer 11: final layer for binary prediction using softmax activation function
    classifier.add(Dense(units=2, activation='softmax'))

    print(classifier.summary())
    # Return compiled network
    return classifier

import matplotlib.pyplot as plt

def plot_model_history(model_history):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # summarize history for training accuracy, validation accuracy, and loss
    axs[0].plot(range(1, len(model_history.history['acc']) + 1), model_history.history['acc'])
    axs[0].plot(range(1, len(model_history.history['val_acc']) + 1), model_history.history['val_acc'])

    # Set titles/labels, labels, etc.
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1, len(model_history.history['acc']) + 1), len(model_history.history['acc']) / 10)
    axs[0].legend(['train', 'val'], loc='best')

    # summarize history for loss
    axs[1].plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'])
    axs[1].plot(range(1, len(model_history.history['val_loss']) + 1), model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1, len(model_history.history['loss']) + 1), len(model_history.history['loss']) / 10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()


def train_network(prom, nonprom, nb_filters, kernel_size, length, pool_size, save_file='', plot=1):
    prom = onehot_fasta_as_numpy("{0}".format(prom))
    nonprom = onehot_fasta_as_numpy("{0}".format(nonprom))
    y = np.array([1, 0])
    y = np.repeat(y, [prom.shape[0], nonprom.shape[0]])
    y = to_categorical(y)
    X = np.concatenate((prom, nonprom), axis=0)
    indices = np.arange(X.shape[0])
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(X, y, indices, test_size=0.2,
                                                                             random_state=15)
    # open("Model_score_l61f.txt", 'a')
    # # for nb_filters in range(100, 201, 50):
    # #     for kernel_size in range(3, 32, 2):
    # # for pool_size in range(1, length-kernel_size+1, 2):
    # pool_size = length - kernel_size + 1
    classifier = create_network(nb_filters, kernel_size, length, 4, pool_size)
    print(classifier)
    from keras.callbacks import EarlyStopping
    earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=5,
                              verbose=1, mode='auto')

    callbacks_list = [earlystop]

    # Compile neural network
    classifier.compile(loss='binary_crossentropy',  # Cross-entropy
                       optimizer='adam',
                       metrics=['accuracy'])  # Accuracy performance metric

    # train the model
    start = time.time()
    ISOTIMEFORMAT = '%Y-%m-%d-%X'
    current_time = time.strftime(ISOTIMEFORMAT, time.localtime(time.time()))
    model_info = classifier.fit(X_train, y_train, verbose=1,
                                validation_steps=0,
                                validation_split=0.25,
                                batch_size=16,
                                callbacks=callbacks_list,
                                epochs=30)
    classifier.fit(X_train, y_train, verbose=1,
                   validation_steps=0,
                   validation_split=0.25,
                   batch_size=16,
                   callbacks=callbacks_list,
                   epochs=30)
    end = time.time()
    if plot == 1:
        plot_model_history(model_info)

    # compute test accuracy
    print("Model took {0} seconds to train".format(end - start))

    # compute test accuracy
    score = classifier.evaluate(X_test, y_test)
    print("Accuracy on test data is: {0}".format(score))
    if save_file != '':
        with open("Model_Score_All.txt", 'a') as new_file:
            new_file.write(
                str(current_time)+' '+str(save_file)+' parameters:'+' '+str(length)+' '+str(nb_filters)+' '
                + str(kernel_size)+' '+str(pool_size)+ '  score:'+str(score)+'\n')
        classifier.save("{0}.h5".format(save_file))

    # y_pred = classifier.predict(X_test)
    # rounded = [round(val[1]) for val in y_pred]

    # print(y_pred[:500,:])
    # print(rounded[:500])


for p in range(2, 251-13+2, 4):
    train_network('Mouse_prom_n20000', 'Random_Exon_Mouse_Cat251_n20000',
                  200, 13, 251, p, 'Mouse_Prom_Exon', 0)



