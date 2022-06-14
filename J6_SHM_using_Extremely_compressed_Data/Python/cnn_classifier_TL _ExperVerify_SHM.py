import tensorflow as tf
import keras
from keras import Sequential
from keras.models import load_model
from livelossplot.keras import PlotLossesCallback

from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.normalization import BatchNormalization
from keras.layers import  Dense, Flatten, Dropout, Conv1D, MaxPool1D, AvgPool1D,Conv2D, MaxPool2D, Activation
from keras.utils.vis_utils import plot_model
from utils import convert_matlab_file, load_dataset, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import StratifiedKFold

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from itertools import cycle
import hdf5storage
from keras.utils import np_utils

import numpy
import math
import scipy.io
import time
################################## for plot confusion imports
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


class History(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []
        self.val_acc = []
        self.loss = []
        self.val_loss = []

    def on_epoch_end(self, epoch, logs={}):
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))


def create_CNN_model():
    model = Sequential()
    # ................................................................................................................
    model.add(Conv1D(filters=16, kernel_size=3, strides=1, padding='same', activation='relu',use_bias=False,  input_shape=(lenSignal, nSensors)))
    # model.add(Conv1D(filters=16, kernel_size=3, strides=1, padding='same', activation='relu',use_bias=False,  input_shape=(lenSignal, nSensors)))
    # model.add(Conv1D(filters=16, kernel_size=3, strides=1, padding='same', activation='relu',use_bias=False,  input_shape=(lenSignal, nSensors)))
    # model.add(Conv1D(filters=16, kernel_size=3, strides=1, padding='same', activation='relu',use_bias=False,  input_shape=(lenSignal, nSensors)))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None))  # Batch Normalization
    model.add(LeakyReLU(alpha=.01))  # advanced activation layer
    model.add(MaxPool1D(pool_size=2, strides=None, padding='valid'))
    # ................................................................................................................
    model.add(Conv1D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu',use_bias=False,  input_shape=(lenSignal, nSensors)))
    # model.add(Conv1D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu',use_bias=False,  input_shape=(lenSignal, nSensors)))
    # model.add(Conv1D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu',use_bias=False,  input_shape=(lenSignal, nSensors)))
    # model.add(Conv1D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu',use_bias=False,  input_shape=(lenSignal, nSensors)))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None))  # Batch Normalization
    model.add(LeakyReLU(alpha=.01))  # advanced activation layer
    model.add(MaxPool1D(pool_size=2, strides=None, padding='valid'))

    # ................................................................................................................
    model.add(Flatten())
    #
    # model.add(Dense(nClasses*4))
    # model.add(Dropout(0.2))
    # model.add(Dense(nClasses*2))
    # model.add(Dropout(0.2))
    model.add(Dense(nClasses, activation='softmax'))


    ###################################################################################################################


#   ...................................................................................................................
    return model


# ...................................................................................................................
# #####################################################################################################################
loadMAT = 'allTHData_shm_a_freq'      # 01a-09a
# loadMAT = 'Avci_B'      # Avci
# loadMAT = 'Avci_B_freq'      # Avci

# loadModel = 'A_SHM8_TH'
saveMAT = loadMAT
saveModel = loadMAT

# #####################################################################################################################
def shuffle(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

# #####################################################################################################################
matlab_file = 'C:/MOHSEN/Research/Benchmarks/ERA Identification SHM Benchmark/NEES Files/files_bldg_shm_exp2/CD of UBC.experiment 2002 2/data/Ambient/Ambient/'+loadMAT+'.mat'
# matlab_file = 'C:/MOHSEN/Research/Temp/03_CNN/Dataset/Ovci/'+loadMAT+'.mat'
mat = hdf5storage.loadmat(matlab_file)

InputData = np.transpose(mat['InputData2'], (2, 0, 1))
TargetData = mat['TargetData']
nSamples = int(np.asscalar(mat['nSamples']))
lenSignal = int(np.asscalar(mat['lenSignal']))
nSensors = int(np.asscalar(mat['nSensors']))
nClasses = int(np.asscalar(mat['nClasses']))

print(nSamples)
X = InputData
Y = np_utils.to_categorical(TargetData)

X, Y = shuffle(X, Y)
# Y_vec = np.argmax(Y, axis=1)
# -----------------------------------------------------------------------------------------------------------------
# #####################################################################################################################
# -----------------------------------------------------------------------------------------------------------------

# indices = np.arange(nSamples)
# X_train, X_test, Y_train, Y_test, train_index, test_index= train_test_split(X, Y, indices, test_size=0.15, shuffle='False')   #



# ############ CNN ###############
with tf.device("/gpu:1"):

    nFolds = 5
    print("Training is started")
    t0 = time.time()  # t0
    skfold = StratifiedKFold(numpy.argmax(Y, axis=-1), n_folds=nFolds, shuffle=True, random_state=None)
    # print("skfold:",skfold)

    fold = 1
    for train_index, test_index in skfold:

        # # I: Create a CNN
        model = create_CNN_model()
        # for layer in model.layers:
        #     layer.trainable = True

        learnignRate = 0.00001

        # # II:Load a pre-trained model
        # model = load_model('saveModels/'+loadModel+'_fold' + str(fold) + '.h5')
        # # model.summary()
        #
        # # # remove last layers and add new layers
        # # model.pop() # remove last layer: model.add(Dense(nClasses, activation='softmax'))
        # # model.pop() # remove  layer: model.add(LeakyReLU(alpha=.01))  # advanced activation layer
        # # model.pop() # remove  layer: model.add(Dense(nClasses*10))
        # # model.pop() # remove  layer: model.add(Dense(nClasses*100))
        #             # now we reached Flattern Layer
        #
        # for i, layer in enumerate(model.layers):
        #     print(i, layer.name)
        #
        # # # # Add new layers
        # # model.add(Dense(nClasses*10))
        # # model.add(Dense(nClasses*10))
        # # # model.add(Dropout(0.2))
        # # model.add(Dense(nClasses, activation='softmax'))
        #
        # # for i, layer in enumerate(model.layers):
        # #     print(i, layer.name)
        #
        # # Freeze the layers except the last 4 layers
        #
        # for layer in model.layers:
        #     layer.trainable = True
        #
        # # for layer in model.layers[:-4]:
        # #     layer.trainable = False
        #
        # # Check the trainable status of the individual layers
        # for layer in model.layers:
        #     print(layer, layer.trainable)
        #
        # learnignRate = 0.00005
        # #   End of Load a CNN

        # Do not forget to compile
        opt = keras.optimizers.Adam(lr=learnignRate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(loss='categorical_crossentropy', optimizer= opt, metrics=['acc'])
        model.summary()

        # nepochs = 500     #avci
        # batchsize = 128   #avci
        nepochs = 500
        batchsize = 1024   #64

        history = History()
        es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, mode='auto')

        model.fit(X[train_index], Y[train_index], epochs=nepochs, batch_size=batchsize, callbacks=[history, es_callback],
                  verbose=1, validation_split=0.1)

        scores = model.evaluate(X[test_index], Y[test_index], verbose=1)

        print("*****  Fold {} *****".format(fold))
        print(scores)

        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        Y_testpred = model.predict_classes(X[test_index])
        Y_testpredScores = model.predict(X[test_index])

        # Compute confusion matrix
        Y_testtrue = np.argmax(Y[test_index], axis=1)
        cnf_matrix = confusion_matrix(Y_testtrue, Y_testpred)

        X_test2MAT = np.transpose(X[test_index], (1, 2, 0))

        model.save('saveModels/Expr_' + saveModel + '_fold' + str(fold) + '.h5')

        scipy.io.savemat('saveMATs/Expr_' + saveMAT + '_fold' + str(fold) + '.mat', {
            # 'X_test2MAT': X_test2MAT,
            'Y_test2MAT': Y[test_index],
            'scores': scores,
            'cnf_matrix': cnf_matrix,
            'Y_true': Y_testtrue,
            'Y_pred': Y_testpred,
            'Y_predScores': Y_testpredScores,
            'AccuracyTH': history.acc,
            'AccuracyTH_val': history.val_acc,   # only if we have validaiton_split
            'LossTH': history.loss,
            'LossTH_val': history.val_loss,
            'nClasses': nClasses,
            'nepochs': nepochs,
            'nFolds': nFolds,
            'batchsize': batchsize,
            'fold': fold,
            'learnignRate': learnignRate
        })




        # np.set_printoptions(precision=2)
        #
        # Plot non-normalized confusion matrix
        # plt.figure()
        # # class_names = np.array(["Intact", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"],dtype='<U10')
        # class_names = np.array(["P1", "P2", "P3", "P4", "P5", "P6", "P7","P8", "P9", "P10", "P11", "P12", "P13", "P14","P15", "P16", "P17", "P18", "P19", "P20", "P21","P22", "P23", "P24", "P25", "P26", "P27", "P28", "P29", "P30"],dtype='<U10')
        # plot_confusion_matrix(cnf_matrix, classes=class_names,
        #                       title='Confusion matrix, without normalization')
        # #
        # # Plot normalized confusion matrix
        # # plt.figure()
        # # plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
        # #                       title='Normalized confusion matrix')
        # # #
        # plt.show()

        # plt.figure()
        # plt.plot(range(0, nepochs), history.val_acc,
        #          'b--')  # plt.plot(range(0, nepochs*math.ceil(0.9*nSamples/batchsize)), history.acc, 'b--')
        # plt.xlabel('Epochs')
        # plt.ylabel('val_Accuracy')
        # plt.show()
        # plt.pause(0.05)

        # plt.figure()
        # plt.plot(range(0, nepochs), history.loss, 'r--')
        # plt.xlabel('Epochs')
        # plt.ylabel('Loss')
        # plt.show()
        # plt.pause(0.05)

        t1 = time.time()  # t1 at the end
        print("Total Run Time: ", int(t1 - t0), " seconds")


        fold += 1











