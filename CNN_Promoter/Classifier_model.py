import os
import numpy as np
from Bio import SeqIO
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import tensorflow as tf
from keras.models import Model
from keras.models import Sequential
from keras.models import load_model
from keras.layers.convolutional import Conv1D # Keras 2 syntax
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.layers import Dense

os.chdir(r"E:/Users/Bink/Documents/iGEM/panda")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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


X = onehot_fasta_as_numpy("Ecoli_genome_trim")
print(X)
Classifier = load_model("CNN_Ecoli_Promoter.h5")
y_pred = Classifier.predict(X)
rounded = [round(val[1]) for val in y_pred]

print(y_pred[:580, :])
print(rounded[:580])