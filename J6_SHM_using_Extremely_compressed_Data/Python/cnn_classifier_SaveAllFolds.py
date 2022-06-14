import tensorflow as tf
import keras
from keras import Sequential
from keras.models import load_model

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
        self.loss = []

    def on_epoch_end(self, epoch, logs={}):
        self.acc.append(logs.get('acc'))
        self.loss.append(logs.get('loss'))



def create_CNN_model():
    model = Sequential()
    # activation: softmax,linear, hard_sigmoid, sigmoid, tanh, relu, softsign, softplus, selu, elu,
    #
    # model.add(Conv2D(filters=28, kernel_size=(3, 3), strides=(1, 1), input_shape=(lenSignal, nSensors, 1), activation='relu'))
    # model.add(MaxPool2D(pool_size=(2, 2)))
    #
    # model.add(Conv2D(filters=28, kernel_size=(3, 3), strides=(1, 1), input_shape=(lenSignal, nSensors, 1), activation='relu'))
    # model.add(MaxPool2D(pool_size=(2, 2)))


    # ................................................................................................................
    # ................................................................................................................
    model.add(Conv1D(filters=16, kernel_size=3, strides=1, padding='same', activation='relu', use_bias=False, input_shape=(lenSignal, nSensors)))
    model.add(Conv1D(filters=16, kernel_size=3, strides=1, padding='same', activation='relu',use_bias=False,  input_shape=(lenSignal, nSensors)))
    # model.add(Conv1D(filters=16, kernel_size=3, strides=1, padding='same', activation='relu', use_bias=False, input_shape=(lenSignal, nSensors)))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None))  # Batch Normalization
    model.add(LeakyReLU(alpha=.01))  # advanced activation layer
    model.add(MaxPool1D(pool_size=2, strides=None, padding='valid'))

    # ................................................................................................................
    model.add(Conv1D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu',use_bias=False,  input_shape=(lenSignal, nSensors)))
    model.add(Conv1D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu',use_bias=False,  input_shape=(lenSignal, nSensors)))
    # model.add(Conv1D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu',use_bias=False,  input_shape=(lenSignal, nSensors)))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None))  # Batch Normalization
    model.add(LeakyReLU(alpha=.01))  # advanced activation layer
    model.add(MaxPool1D(pool_size=2, strides=None, padding='valid'))

    # ................................................................................................................
    model.add(Conv1D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', use_bias=False, input_shape=(lenSignal, nSensors)))
    model.add(Conv1D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', use_bias=False, input_shape=(lenSignal, nSensors)))
    # model.add(Conv1D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', use_bias=False, input_shape=(lenSignal, nSensors)))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None))  # Batch Normalization
    model.add(LeakyReLU(alpha=.01))  # advanced activation layer
    model.add(MaxPool1D(pool_size=2, strides=None, padding='valid'))

    # ................................................................................................................
    model.add(Flatten())
    model.add(BatchNormalization())
    #
    model.add(Dense(nClasses*1))
    model.add(Dense(nClasses*1))
    model.add(Dense(nClasses, activation='softmax'))

    ###################################################################################################################
    learnignRate = 0.00005
    opt = keras.optimizers.Adam(lr=learnignRate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc', 'mse', 'mae','categorical_crossentropy'])
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    model.summary()


#   ...................................................................................................................
    return model


# ...................................................................................................................
# #####################################################################################################################

loadMAT = 'A_SHM3_freq'   # 1,2,   4,5,6,7
# loadMAT = 'A_SHM1'   # 1,2,   4,5,6,7
saveMAT = loadMAT
saveModel = loadMAT

# #####################################################################################################################
def shuffle(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

# #####################################################################################################################

matlab_file = 'C:/MOHSEN/Research/Temp/03_CNN/AISC_BenchMark/BenchMark/Outputs/'+loadMAT+'.mat'
mat = hdf5storage.loadmat(matlab_file)

InputData = np.transpose(mat['InputData'], (2, 0, 1))
TargetData = mat['TargetData']
nSamples = int(np.asscalar(mat['nSamples']))
lenSignal = int(np.asscalar(mat['lenSignal']))
nSensors = int(np.asscalar(mat['nSensors']))
nClasses = int(np.asscalar(mat['nClasses']))

X = InputData
Y = np_utils.to_categorical(TargetData)

X, Y = shuffle(X, Y)

# Y_vec = np.argmax(Y, axis=1)

# -----------------------------------------------------------------------------------------------------------------
# #####################################################################################################################
# -----------------------------------------------------------------------------------------------------------------




# ############ CNN ###############
# ################################
# ###############################

with tf.device("/gpu:1"):
    print("Training is started")
    t0 = time.time()  # t0

    nFolds = 10
    skfold = StratifiedKFold(numpy.argmax(Y, axis=-1), n_folds=nFolds, shuffle=True, random_state=None)
    print("skfold:",skfold)

    fold = 1
    for train_index, test_index  in skfold:

        model = create_CNN_model()
        # model = create_SVC_model()

        history = History()
        # nepochs = 100
        nepochs = 500
        batchsize = 256

        model.fit(X[train_index], Y[train_index], epochs=nepochs, batch_size=batchsize, callbacks=[history], verbose=1)
        scores = model.evaluate(X[test_index], Y[test_index], verbose=1)


        print(scores)
        print("%s: %.2f" % (model.metrics_names[0], scores[0] ))
        print("%s: %.2f" % (model.metrics_names[1], scores[1] ))
        print("%s: %.2f" % (model.metrics_names[2], scores[2] ))
        print("%s: %.2f" % (model.metrics_names[3], scores[3] ))
        print("%s: %.2f" % (model.metrics_names[4], scores[4] ))



        Y_testpred = model.predict_classes(X[test_index])
        Y_testpredScores = model.predict(X[test_index])




        # Compute confusion matrix
        Y_testtrue = np.argmax(Y[test_index], axis=1)
        cnf_matrix = confusion_matrix(Y_testtrue, Y_testpred)

        # X_test2MAT = np.transpose(X[test_index], (1, 2, 0))

        # TH:
        # model.save('saveModels/' + saveModel + '_TH_fold' + str(fold) +  '.h5')

        # Freq:
        model.save('saveModels/' + saveModel + '_fold' + str(fold) +  '.h5')

        # TH:
        # scipy.io.savemat('saveMATs/' + saveMAT + '_TH_fold' + str(fold) + '.mat', {

        # Freq:
        scipy.io.savemat('saveMATs/' + saveMAT + '_fold' + str(fold) + '.mat', {
                # 'X_test2MAT': X_test2MAT,
            # 'Y_test2MAT': Y[test_index],
            'Y_true': Y_testtrue,
            'Y_pred': Y_testpred,
            'Y_predScores': Y_testpredScores,
            'AccuracyTH': history.acc,
            'LossTH': history.loss,
            'nClasses': nClasses,
            'nepochs': nepochs,
            'batchsize':  batchsize,
            'scores': scores,
            'cnf_matrix': cnf_matrix,
            'nFolds': nFolds,
            'fold': fold,
        })
        #
        # np.set_printoptions(precision=2)
        #
        # # Plot non-normalized confusion matrix
        # plt.figure()
        # class_names = np.array(["Intact", "Pattern1", "Pattern2"],dtype='<U10')
        # plot_confusion_matrix(cnf_matrix, classes=class_names,
        #                       title='Confusion matrix, without normalization')
        #
        # # Plot normalized confusion matrix
        # plt.figure()
        # plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
        #                       title='Normalized confusion matrix')
        #
        # plt.show()


        #
        # plt.figure()
        # plt.plot(range(0, nepochs), history.acc, 'b--') # plt.plot(range(0, nepochs*math.ceil(0.9*nSamples/batchsize)), history.acc, 'b--')
        # plt.xlabel('Epochs')
        # plt.ylabel('Accuracy')
        # plt.show()
        # plt.pause(0.05)


        # plt.figure()
        # plt.plot(range(0, nepochs), history.loss, 'r--')
        # plt.xlabel('Epochs')
        # plt.ylabel('Loss')
        # plt.show()
        # plt.pause(0.05)

        t1 = time.time()  # t1 at the end
        print("Total Run Time: ", int(t1-t0), " seconds")
        fold += 1

    # ####################################################################################


    ################################
    ############ SVM ###############
    #################################

    # with tf.device("/gpu:1"):
    #     print("Training is started")
    #
    #     Xx = np.reshape(X, [nSamples, lenSignal * nSensors])
    #
    #     # shuffle and split training and test sets
    #     X_train, X_test, y_train, y_test = train_test_split(Xx, Y_bin, test_size=.1,
    #                                                         random_state=0)
    #
    #     # Learn to predict each class against the other
    #     classifier = OneVsRestClassifier(svm.SVC( C=0.1, kernel='rbf', probability=True, verbose=True, max_iter=100000, random_state=0))
    #     # classifier = OneVsRestClassifier(svm.NuSVC())
    #     y_score = classifier.fit(X_train, y_train).decision_function(X_test)
    #
    #     # model.fit(X[train_index], Y[train_index])
    #     # scores = model.evaluate(X[test_index], Y[test_index], verbose=1)
    #
    #
    #
    #     #
    #
    #     # Compute ROC curve and ROC area for each class
    #     fpr = dict()
    #     tpr = dict()
    #     roc_auc = dict()
    #     for i in range(nClasses):
    #         fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    #         roc_auc[i] = auc(fpr[i], tpr[i])
    #
    #     # Compute micro-average ROC curve and ROC area
    #     fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    #     roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    #
    #
    #
    #
    #
    #     scipy.io.savemat('saveMATs\A_SHM1_SVC1.mat', {
    #         'y_score': y_score,
    #         'fpr': fpr,
    #         'tpr': tpr,
    #         'roc_auc': roc_auc,
    #     })
    #
    #
    #     # Plot of a ROC curve for a specific class
    #     plt.figure()
    #     lw = 2
    #     cls = 0
    #     plt.plot(fpr[cls], tpr[cls], color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[cls])
    #
    #     cls = 1
    #     plt.plot(fpr[cls], tpr[cls], color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[cls])
    #
    #
    #     cls = 2
    #     plt.plot(fpr[cls], tpr[cls], color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[cls])
    #
    #
    #
    #     plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    #     plt.xlim([0.0, 1.0])
    #     plt.ylim([0.0, 1.05])
    #     plt.xlabel('False Positive Rate')
    #     plt.ylabel('True Positive Rate')
    #     plt.title('Receiver operating characteristic example')
    #     plt.legend(loc="lower right")
    #     plt.show()
    #     # #############################################################################################################
    #     # #############################################################################################################
    #     # Compute macro-average ROC curve and ROC area
    #
    #     # First aggregate all false positive rates
    #     all_fpr = np.unique(np.concatenate([fpr[i] for i in range(nClasses)]))
    #
    #     # Then interpolate all ROC curves at this points
    #     mean_tpr = np.zeros_like(all_fpr)
    #     for i in range(nClasses):
    #         mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    #
    #     # Finally average it and compute AUC
    #     mean_tpr /= nClasses
    #
    #     fpr["macro"] = all_fpr
    #     tpr["macro"] = mean_tpr
    #     roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    #
    #     # Plot all ROC curves
    #     plt.figure()
    #     plt.plot(fpr["micro"], tpr["micro"],
    #              label='micro-average ROC curve (area = {0:0.2f})'
    #                    ''.format(roc_auc["micro"]),
    #              color='deeppink', linestyle=':', linewidth=4)
    #
    #     plt.plot(fpr["macro"], tpr["macro"],
    #              label='macro-average ROC curve (area = {0:0.2f})'
    #                    ''.format(roc_auc["macro"]),
    #              color='navy', linestyle=':', linewidth=4)
    #
    #     colors = cycle(['aqua', 'darkorange', 'cornflowerblue','aqua', 'darkorange', 'cornflowerblue','aqua', 'darkorange', 'cornflowerblue'])
    #     for i, color in zip(range(nClasses), colors):
    #         plt.plot(fpr[i], tpr[i], color=color, lw=lw,
    #                  label='ROC curve of class {0} (area = {1:0.2f})'
    #                        ''.format(i, roc_auc[i]))
    #
    #     plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    #     plt.xlim([0.0, 1.0])
    #     plt.ylim([0.0, 1.05])
    #     plt.xlabel('False Positive Rate')
    #     plt.ylabel('True Positive Rate')
    #     plt.title('Some extension of Receiver operating characteristic to multi-class')
    #     plt.legend(loc="lower right")
    #     plt.show()
    #
    #

    import winsound
    duration = 1000  # milliseconds
    freq = 440  # Hz
    winsound.Beep(freq, duration)
