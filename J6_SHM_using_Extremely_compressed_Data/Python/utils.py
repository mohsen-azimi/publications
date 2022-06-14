import numpy as np
import scipy.io
import h5py
import hdf5storage
from keras.utils import np_utils


import itertools
import numpy as np
import matplotlib.pyplot as plt




def shuffle(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')




def convert_matlab_file(matlab_file, save_path):
    #mat = scipy.io.loadmat(matlab_file)
    # mat = h5py.File(matlab_file, 'r')
    mat = hdf5storage.loadmat(matlab_file)

    InputData = np.transpose(mat['InputData'], (2, 0, 1))
    TargetData = mat['TargetData']

    nSamples = int(np.asscalar(mat['nSamples']))
    lenSignal= int(np.asscalar(mat['lenSignal']))
    nSensors = int(np.asscalar(mat['nSensors']))
    nClasses = int(np.asscalar(mat['nClasses']))

    X = InputData
    Y = np_utils.to_categorical(TargetData)

    X, Y = shuffle(X, Y)
    # np.savez(save_path, X, Y, nSamples, lenSignal, nSensors, nClasses)
    return X, Y, Y_vec, nSamples, lenSignal, nSensors, nClasses



def load_dataset(path):
    dataset = np.load(path + ".npz")
    return dataset['X'], dataset['Y'],dataset['Y_vect'], dataset['nSamples'], dataset['lenSignal'], dataset['nSensors'], dataset['nClasses']
    #return dataset['arr_0'], dataset['arr_1'], dataset['arr_2'], dataset['arr_3'], dataset['arr_4'], dataset['arr_5']