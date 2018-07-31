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

os.chdir(r"E:\Users\Bink\Documents\iGEM\panda\Promoter_identify")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print(os.getcwd())

oneHotDict = {'A': np.array([1, 0, 0, 0]), 'G': np.array([0, 1, 0, 0]), 'C': np.array([0, 0, 1, 0]),
              'T': np.array([0, 0, 0, 1]), 'N': np.array([0, 0, 0, 0]), 'M': np.array([0, 0, 0, 0]),
              'K': np.array([0, 0, 0, 0]), 'D': np.array([0, 0, 0, 0]), 'R': np.array([0, 0, 0, 0]),
              'Y': np.array([0, 0, 0, 0]), 'S': np.array([0, 0, 0, 0])}


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


with tf.device('/cpu:0'):
    X_pre = onehot_fasta_as_numpy(r"PromoterSet")
    # X_pre_f = onehot_fasta_as_numpy(r"PromoterSigma70Set_Sub61_First")
    # X_pre_l = onehot_fasta_as_numpy(r"PromoterSigma70Set_Sub61_Last")
    # print(X_pre)
    Classifier = load_model("CNN_Ecoli_Promoter_Pub.h5")
    # Classifier_f = load_model("CNN_Ecoli_Promoter_Sub61_First.h5")
    # Classifier_l = load_model("CNN_Ecoli_Promoter_Sub61_Last.h5")
    # y_pred_f = Classifier_f.predict(X_pre_f)
    # y_pred_l = Classifier_l.predict(X_pre_l)
    y_pred = Classifier.predict(X_pre)
    print("Prediction completed!")
    # rounded = [round(val[1]) for val in y_pred]
    y_final = []
    tp = 0
    fn = 0
    for val in y_pred:
        if val[1] > 0.5:
            y_final.append('11')
            tp += 1
        else:
            y_final.append('00')
            fn += 1
    # for i in range(len(y_pred_f)):
    #     if y_pred_f[i][1] > 0.5 or y_pred_l[i][1] > 0.5:
    #         y_final.append('11')
    #         tp += 1
    #     else:
    #         y_final.append('00')
    #         np += 1
    # result_f = y_pred_f[:, :]
    # result_l = y_pred_l[:, :]
    pre_promoter = y_final[:]
    mapList = []
    for i in range(len(y_pred)):
        mapList.append([0, 0, 0, 0])
    i = 0
    for proto_seq in SeqIO.parse(r"data/PromoterSet.fa", "fasta"):
        mapList[i][0] = str(proto_seq.seq)
        i += 1
    for j in range(len(y_pred)):
        mapList[j][1] = y_final[j]
        mapList[j][3] = str(y_pred[j][1]) # +'  '+str(y_pred_l[j][1])
    # for k in range(len(y_pred)):
    #     if k < 5000:
    #         pos_list = str(k + 1) + '---' + str(k + 43)
    #         mapList[k][2] = pos_list
    #
    #     else:
    #         k = k - 5000
    #         pos_list = str(k + 1) + '---' + str(k + 81)
    #         k = k + 5000
    #         mapList[k][2] = pos_list
    for k in range(len(y_pred)):
            pos_list = str(k + 1) + '---' + str(k + 81)
            mapList[k][2] = pos_list


result = open("result1-position.txt", 'w')
for i in range(len(mapList)):
    result.write(str(mapList[i]) + '\n')
result.write(str(tp)+' '+str(fn)+' '+str(tp/(tp+fn))+'\n')

