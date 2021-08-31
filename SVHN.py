import numpy as np
import scipy as sp
import scipy.io as sio
from scipy.misc import *
from tflearn.data_utils import shuffle, to_categorical


########################################

#           DATA LOADING               # 

########################################




#Dataset location
train_location = 'train_32x32.mat'
test_location = 'test_32x32.mat'


def load_train_data():
    train_dict = sio.loadmat(train_location)
    X = np.asarray(train_dict['X'])

    X_train = []
    for i in range(X.shape[3]):
        X_train.append(X[:,:,:,i])
    X_train = np.asarray(X_train)

    Y_train = train_dict['y']
    for i in range(len(Y_train)):
        if Y_train[i]%10 == 0:
            Y_train[i] = 0
    Y_train = to_categorical(Y_train[:, 0],10)
    print("Y_train shape :",Y_train.shape)
    return (X_train,Y_train)

def load_test_data():
    test_dict = sio.loadmat(test_location)
    X = np.asarray(test_dict['X'])

    X_test = []
    for i in range(X.shape[3]):
        X_test.append(X[:,:,:,i])
    X_test = np.asarray(X_test)

    Y_test = test_dict['y']
    for i in range(len(Y_test)):
        if Y_test[i]%10 == 0:
            Y_test[i] = 0
    Y_test = to_categorical(Y_test,10)
    print("Y_test shape :",Y_test.shape)

    return (X_test,Y_test)
    
    
##########################################

# Modeling , Validation and Architecture # 

##########################################



import tflearn
#from keras_tqdm import TQDMNotebookCallback
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, global_avg_pool
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import batch_normalization
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import numpy as np

# initialize tqdm callback with default parameters

# from load_input import load_train_data


X_train, Y_train = load_train_data()
X_train, Y_train = shuffle(X_train, Y_train)
print('shuffle done')

X_val = X_train[2000:4000]
Y_val = Y_train[2000:4000]
network = input_data(shape=[None, 32, 32, 3])

network = conv_2d(network, 16, 3, activation='relu', weights_init='xavier')
network = batch_normalization(network)

network = conv_2d(network, 16, 3, activation='relu', weights_init='xavier')      
network = max_pool_2d(network, 2)
network = batch_normalization(network)

network = conv_2d(network, 32, 3, activation='relu', weights_init='xavier')    
network = max_pool_2d(network, 2)
network = batch_normalization(network)

network = conv_2d(network, 32, 3, activation='relu', weights_init='xavier')
network = max_pool_2d(network, 2)
network = batch_normalization(network)


network = conv_2d(network, 64, 3, activation='relu', weights_init='xavier')
network = max_pool_2d(network, 2)
network = batch_normalization(network)

network = fully_connected(network, 256, activation='relu', weights_init='xavier')
network = dropout(network, 0.25)


network = fully_connected(network, 10, activation='softmax', weights_init='xavier')


network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

model = tflearn.DNN(network, tensorboard_verbose=0)

model.fit(X_train, Y_train, n_epoch=3, shuffle=True, validation_set=(X_val, Y_val),
          show_metric=True, batch_size=100,
          snapshot_epoch=True,
          run_id='svhn_2')

model.save("svhn_2.tfl")
print("Done")
    
    
    
