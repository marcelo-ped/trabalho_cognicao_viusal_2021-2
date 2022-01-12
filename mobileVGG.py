import numpy as np
import os
import pickle
import pandas as pd
import time
import warnings
import cv2
warnings.filterwarnings("ignore")
import random
from numpy.random import permutation
np.random.seed(2016)
from tensorflow.keras.models import Sequential

from keras.layers.core import Dense, Activation, Flatten # GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.models import model_from_json
from sklearn.metrics import log_loss, confusion_matrix
from keras import regularizers
import h5py


from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import LeakyReLU
from sklearn.metrics import confusion_matrix
import keras
from keras_applications.imagenet_utils import _obtain_input_shape
from tensorflow.python.keras import backend as K
from tensorflow.keras.layers import Input, Convolution2D, \
    GlobalAveragePooling2D, Dense, BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.utils import get_source_inputs
from depthwise_conv2d import DepthwiseConvolution2D
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
import tensorflow as tf
import argparse


use_cache = 0
####ORIGINALLLL#####
"""
def load_train():
    '''Give path of the dataset .csv file of training data below'''
    df = pd.read_csv(r'/home/marcelo/Documentos/Distracted-Driver-Detection/auc.distracted.driver.train.csv')
    x = df.iloc[:,0] 
    y = df.iloc[:,1]
    X_train = []
    Y_train = []
    print('Read test images')
    for i in range (0,len(x)):
        fl=x[i]
        img = get_im_cv2(fl)
        X_train.append(img)
        Y_train.appen d(y[i])
    return X_train, Y_train
"""
x_test = []
y_test = []

def get_im_cv2(path):
    img = cv2.imread(path)
    resized = cv2.resize(src=img, dsize=(128, 128), interpolation=cv2.INTER_LINEAR)
    return resized

def load_train():
    '''Give path of the dataset .csv file of training data below'''
    folder_root_name = '/home/marcelo/Downloads/Frames_29k_organizado_previamente_motorista_fora_do_treino_5/train'
    folder_root = os.listdir(folder_root_name)
    x = []
    y = []
    cont = 0
    for dir in folder_root:
        images_folder = os.listdir(folder_root_name + "/" + dir)
        for i in range(0 ,  int(len(images_folder))):
            #random.shuffle(images_folder)
            if images_folder[i].endswith('jpg'): #and images_folder[i].find("01_") != -1:
                
                print(folder_root_name + "/" + dir + "/" + images_folder[i])
                #x.append(folder_root_name + "/" + dir + "/" + images_folder[i])
                x.append(folder_root_name + "/" + dir + "/" + images_folder[i])
                y.append(dir[1])
                """ print(dir[1])
                if cont >= 40:
                    cont = 0
                    break
                cont += 1
                """
    X_train = []
    Y_train = []
    print('Read train images')
    for i in range (0, int(len(x))):
        fl=x[i]
        img = get_im_cv2(fl)
        X_train.append(img)
        Y_train.append(y[i])
    return X_train, Y_train
"""

def get_im_cv2(path):
    img = cv2.imread(path)
    resized = cv2.resize(src=img, dsize=(128, 128), interpolation=cv2.INTER_LINEAR)
    return resized


def load_train():
    '''Give path of the dataset .csv file of training data below'''
    folder_root_name = '/home/marcelo/Downloads/Frames_reduzidos'
    folder_root = os.listdir(folder_root_name)
    x = []
    y = []
    cont = 0
    X_train = []
    Y_train = []
    for dir in folder_root:
        images_folder = os.listdir(folder_root_name + "/" + dir)
        for i in range(0 ,  int(len(images_folder))):
            #random.shuffle(images_folder)
            if images_folder[i].endswith('jpg') and images_folder[i].find("01_") != -1:
                
                print(folder_root_name + "/" + dir + "/" + images_folder[i])
                if cont < 30:
                    img = get_im_cv2(folder_root_name + "/" + dir + "/" + images_folder[i])
                    X_train.append(img)
                    Y_train.append(dir[1])
                else:
                    img = get_im_cv2(folder_root_name + "/" + dir + "/" + images_folder[i])
                    x_test.append(img)
                    y_test.append(dir[1])
                #x.append(folder_root_name + "/" + dir + "/" + images_folder[i])
                x.append(folder_root_name + "/" + dir + "/" + images_folder[i])
                y.append(dir[1])
                print(dir[1])
                if cont >= 40:
                    cont = 0
                    break
                cont += 1
    return X_train, Y_train

"""

def load_train_1(path):
    '''Give path of the dataset .csv file of training data below'''
    '''Give path of the dataset .csv file of training data below'''
    #df = pd.read_csv(r'/home/marcelo/Downloads/v1_cam1_no_split/Train_data_list.csv')
    df = pd.read_csv(os.path.join(path, 'train.csv'))
    x = df.iloc[:,0]
    y = df.iloc[:,1]
    x1 = []
    for i in x:
        aux = str(i).replace("/distracted.driver", "/home/marcelo/Downloads/v1_cam1_no_split")
        x1.append(aux)
    X_train = []
    Y_train = []
    print('Read train images')
    for i in range (0,len(x)):
        fl=x1[i]
        img = get_im_cv2(fl)
        X_train.append(img)
        Y_train.append(y[i])
    return X_train, Y_train



def load_valid():
    '''Give path of .csv file of test data below'''
    #df = pd.read_csv(r'/home/marcelo/Documentos/Distracted-Driver-Detection/auc.distracted.driver.test.csv')
    '''Give path of the dataset .csv file of training data below'''
    folder_root_name = '/home/marcelo/Downloads/Frames_29k_organizado_previamente_motorista_fora_do_treino_5/test'
    folder_root = os.listdir(folder_root_name)
    x = []
    y = []
    cont = 0
    for dir in folder_root:
        images_folder = os.listdir(folder_root_name + "/" + dir)
        random.shuffle(images_folder)
        for i in range(0 , int(len(images_folder))):
            if images_folder[i].endswith('jpg'):
                print(folder_root_name + "/" + dir + "/" + images_folder[i])
                x.append(folder_root_name + "/" + dir + "/" + images_folder[i])
                y.append(dir[1])
    X_train = []
    Y_train = []
    print('Read test images')
    for i in range (0,len(x)):
        fl=x[i]
        img = get_im_cv2(fl)
        X_train.append(img)
        Y_train.append(y[i])
    return X_train, Y_train

def load_valid_1(path):
    '''Give path of .csv file of test data below'''
    #df = pd.read_csv(r'/home/marcelo/Downloads/v1_cam1_no_split/Test_data_list.csv')
    df = pd.read_csv(os.path.join(path, 'valid.csv'))
    x = df.iloc[:,0]
    y = df.iloc[:,1]
    x1 = []
    for i in x:
        aux = str(i).replace("/distracted.driver", "/home/marcelo/Downloads/v1_cam1_no_split")
        x1.append(aux)
    X_valid = []
    Y_valid = []
    print('Read test images')
    for i in range (0,len(x)):
        fl=x1[i] 
        img = get_im_cv2(fl)
        X_valid.append(img)
        Y_valid.append(y[i])
    return X_valid, Y_valid

def load_test_1(path):
    '''Give path of .csv file of test data below'''
    #df = pd.read_csv(r'/home/marcelo/Downloads/v1_cam1_no_split/Test_data_list.csv')
    df = pd.read_csv(os.path.join(path, 'test.csv'))
    x = df.iloc[:,0]
    y = df.iloc[:,1]
    x1 = []
    for i in x:
        aux = str(i).replace("/distracted.driver", "/home/marcelo/Downloads/v1_cam1_no_split")
        x1.append(aux)
    X_valid = []
    Y_valid = []
    print('Read test images')
    for i in range (0,len(x)):
        fl=x1[i] 
        img = get_im_cv2(fl)
        X_valid.append(img)
        Y_valid.append(y[i])
    return X_valid, Y_valid

def get_im_cv2(path):
    img = cv2.imread(path)
    resized = cv2.resize(src=img, dsize=(128, 128), interpolation=cv2.INTER_LINEAR)
    return resized

def cache_data(data, path):
    if os.path.isdir(os.path.dirname(path)):
        file = open(path, 'wb')
        pickle.dump(data, file)
        file.close()
    else:
        print('Directory doesnt exists')


def restore_data(path):
    data = dict()
    if os.path.isfile(path):
        file = open(path, 'rb')
        data = pickle.load(file)
    return data
    



def read_and_normalize_train_data(path):
    cache_path = os.path.join('/home/marcelo/Documentos/Distracted-Driver-Detection','cache', 'train_r_' + str(128) + '_c_' + str(128) + '_t_' + str(3) + '.dat')
    if not os.path.isfile(cache_path) or use_cache == 0:
        train_data, train_target= load_train_1(path)
        #cache_data((train_data, train_target), cache_path)
    else:
        print('Restore train from cache!')
        #(train_data, train_target) = restore_data(cache_path)
    
    print('Convert to numpy...')
    train_data = np.array(train_data, dtype=np.uint8)
    train_target = np.array(train_target, dtype=np.uint8)
    
    print('Reshape...')
    train_data = train_data.transpose((0, 1, 2, 3))

    # Normalise the train data
    print('Convert to float...')
    train_data = train_data.astype('float16')
    mean_pixel = [80.857, 81.106, 82.928]
    
    print('Substract 0...')
    train_data[:, :, :, 0] -= mean_pixel[0]
    
    print('Substract 1...')
    train_data[:, :, :, 1] -= mean_pixel[1]

    print('Substract 2...')
    train_data[:, :, :, 2] -= mean_pixel[2]

    train_target = np_utils.to_categorical(train_target, 10)
    
    # Shuffle experiment START !!
    perm = permutation(len(train_target))
    train_data = train_data[perm]
    train_target = train_target[perm]
    # Shuffle experiment END !!
    
    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data, train_target

def read_and_normalize_valid_data(path):
    start_time = time.time()
    #os.mkdir(str(os.path.join('/home/marcelo/Documentos/Distracted-Driver-Detection','cache')))
    cache_path = os.path.join('/home/marcelo/Documentos/Distracted-Driver-Detection','cache', 'test_r_' + str(128) + '_c_' + str(128) + '_t_' + str(3) + '.dat')

    if not os.path.isfile(cache_path) or use_cache == 0:
        test_data, test_target = load_valid_1(path) #x_test, y_test
        #cache_data((test_data, test_target ), cache_path)
    else:
        print('Restore test from cache [{}]!')
        #(test_data, test_target) = restore_data(cache_path)

    test_data = np.array(test_data, dtype=np.uint8)
    test_data = test_data.transpose((0, 1, 2, 3))

    # Normalise the test data data

    test_data = test_data.astype('float16')
    mean_pixel = [80.857, 81.106, 82.928]

    test_data[:, :, :, 0] -= mean_pixel[0]

    test_data[:, :, :, 1] -= mean_pixel[1]

    test_data[:, :, :, 2] -= mean_pixel[2]

    test_target = np_utils.to_categorical(test_target, 10)
    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    print('Read and process test data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return test_data, test_target

def read_and_normalize_test_data(path):
    start_time = time.time()
    #os.mkdir(str(os.path.join('/home/marcelo/Documentos/Distracted-Driver-Detection','cache')))
    cache_path = os.path.join('/home/marcelo/Documentos/Distracted-Driver-Detection','cache', 'test_r_' + str(128) + '_c_' + str(128) + '_t_' + str(3) + '.dat')

    if not os.path.isfile(cache_path) or use_cache == 0:
        test_data, test_target = load_test_1(path) #x_test, y_test
        #cache_data((test_data, test_target ), cache_path)
    else:
        print('Restore test from cache [{}]!')
        #(test_data, test_target) = restore_data(cache_path)

    test_data = np.array(test_data, dtype=np.uint8)
    test_data = test_data.transpose((0, 1, 2, 3))

    # Normalise the test data data

    test_data = test_data.astype('float16')
    mean_pixel = [80.857, 81.106, 82.928]

    test_data[:, :, :, 0] -= mean_pixel[0]

    test_data[:, :, :, 1] -= mean_pixel[1]

    test_data[:, :, :, 2] -= mean_pixel[2]

    test_target = np_utils.to_categorical(test_target, 10)
    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    print('Read and process test data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return test_data, test_target


def VGG_with_MobileNet(input_tensor=None, input_shape=None, alpha=1, shallow=False, classes=10):


    
    include_top = True
    input_shape = _obtain_input_shape(input_shape,
                                        default_size=128,
                                        min_size=96,
                                        data_format=K.image_data_format(),
                                        require_flatten=include_top)
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    """ Input and 3x3 conv 64 filters"""
    x = Convolution2D(int(64 * alpha), (3, 3), strides=(1,1), padding='same',kernel_regularizer=l2(0.001), use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x= LeakyReLU(alpha=0.01)(x)

    """ 3x3 conv 64 filters and maxpooling by 2"""
    x = DepthwiseConvolution2D(int(64 * alpha), (3, 3), strides=(2, 2), padding='same', kernel_regularizer=l2(0.001), use_bias=False)(x)
    x = BatchNormalization()(x)
    x= LeakyReLU(alpha=0.01)(x)
    
    x = Convolution2D(int(64 * alpha), (1, 1), strides=(1, 1), padding='same',kernel_regularizer=l2(0.001),  use_bias=False)(x)
    x = BatchNormalization()(x)
    x= LeakyReLU(alpha=0.01)(x)
   
    """ 3x3 conv 128 filters"""   
    x = DepthwiseConvolution2D(int(64 * alpha), (3, 3), strides=(1, 1), padding='same',kernel_regularizer=l2(0.001), use_bias=False)(x)
    x = BatchNormalization()(x)
    x= LeakyReLU(alpha=0.01)(x)
    
    x = Convolution2D(int(128 * alpha), (1, 1), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001), use_bias=False)(x)
    x = BatchNormalization()(x)
    x= LeakyReLU(alpha=0.01)(x)
 
    """ 3x3 conv 128 filters and maxpooling by 2"""    
    x = DepthwiseConvolution2D(int(128 * alpha), (3, 3), strides=(2, 2), padding='same', kernel_regularizer=l2(0.001), use_bias=False)(x)
    x = BatchNormalization()(x)
    x= LeakyReLU(alpha=0.01)(x)
    
    x = Convolution2D(int(128 * alpha), (1, 1), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001),use_bias=False)(x)
    x = BatchNormalization()(x)
    x= LeakyReLU(alpha=0.01)(x)

    """ 3x3 conv 256 filters"""   
    x = DepthwiseConvolution2D(int(128 * alpha), (3, 3), strides=(1, 1), padding='same',kernel_regularizer=l2(0.001), use_bias=False)(x)
    x = BatchNormalization()(x)
    x= LeakyReLU(alpha=0.01)(x)
    x = Convolution2D(int(256 * alpha), (1, 1), strides=(1, 1), padding='same',kernel_regularizer=l2(0.001), use_bias=False)(x)
    x = BatchNormalization()(x)
    x= LeakyReLU(alpha=0.01)(x)

    """ 3x3 conv 256 filters"""   
    x = DepthwiseConvolution2D(int(256 * alpha), (3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001),use_bias=False)(x)
    x = BatchNormalization()(x)
    x= LeakyReLU(alpha=0.01)(x)
    x = Convolution2D(int(256 * alpha), (1, 1), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001),use_bias=False)(x)
    x = BatchNormalization()(x)
    x= LeakyReLU(alpha=0.01)(x)

    """ 3x3 conv 256 filters and maxpooling by 2"""    
    x = DepthwiseConvolution2D(int(256 * alpha), (3, 3), strides=(2, 2), padding='same', kernel_regularizer=l2(0.001),use_bias=False)(x)
    x = BatchNormalization()(x)
    x= LeakyReLU(alpha=0.01)(x)
    x = Convolution2D(int(256 * alpha), (1, 1), strides=(1, 1), padding='same',kernel_regularizer=l2(0.001), use_bias=False)(x)
    x = BatchNormalization()(x)
    x= LeakyReLU(alpha=0.01)(x)
    
    x= Dropout(0.05)(x)

    
    """ 3x3 conv 512 filters"""       
    x = DepthwiseConvolution2D(int(256 * alpha), (3, 3), strides=(1, 1), padding='same',kernel_regularizer=l2(0.001), use_bias=False)(x)
    x = BatchNormalization()(x)
    x= LeakyReLU(alpha=0.01)(x)
    x = Convolution2D(int(512 * alpha), (1, 1), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001), use_bias=False)(x)
    x = BatchNormalization()(x)
    x= LeakyReLU(alpha=0.01)(x)
    
    """ 3x3 conv 512 filters"""       
    x = DepthwiseConvolution2D(int(512 * alpha), (3, 3), strides=(1, 1), padding='same',kernel_regularizer=l2(0.001), use_bias=False)(x)
    x = BatchNormalization()(x)
    x= LeakyReLU(alpha=0.01)(x)
    x = Convolution2D(int(512 * alpha), (1, 1), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001), use_bias=False)(x)
    x = BatchNormalization()(x)
    x= LeakyReLU(alpha=0.01)(x)
    
        
    """ 3x3 conv 512 filters and maxpooling by 2"""       
    x = DepthwiseConvolution2D(int(512 * alpha), (3, 3), strides=(2, 2), padding='same',kernel_regularizer=l2(0.001), use_bias=False)(x)
    x = BatchNormalization()(x)
    x= LeakyReLU(alpha=0.01)(x)
    x = Convolution2D(int(512 * alpha), (1, 1), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001), use_bias=False)(x)
    x = BatchNormalization()(x)
    x= LeakyReLU(alpha=0.01)(x)

    x= Dropout(0.1)(x)
    
    """ 3x3 conv 512 filters"""       
    x = DepthwiseConvolution2D(int(512 * alpha), (3, 3), strides=(1, 1), padding='same',kernel_regularizer=l2(0.001), use_bias=False)(x)
    x = BatchNormalization()(x)
    x= LeakyReLU(alpha=0.01)(x)
    x = Convolution2D(int(512 * alpha), (1, 1), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001), use_bias=False)(x)
    x = BatchNormalization()(x)
    x= LeakyReLU(alpha=0.01)(x)
    
    """ 3x3 conv 512 filters"""       
    x = DepthwiseConvolution2D(int(512 * alpha), (3, 3), strides=(1, 1), padding='same',kernel_regularizer=l2(0.001), use_bias=False)(x)
    x = BatchNormalization()(x)
    x= LeakyReLU(alpha=0.01)(x)
    x = Convolution2D(int(512 * alpha), (1, 1), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001), use_bias=False)(x)
    x = BatchNormalization()(x)
    x= LeakyReLU(alpha=0.01)(x)
    
        
    """ 3x3 conv 512 filters and maxpooling by 2"""       
    x = DepthwiseConvolution2D(int(512 * alpha), (3, 3), strides=(2, 2), padding='same',kernel_regularizer=l2(0.001), use_bias=False)(x)
    x = BatchNormalization()(x)
    x= LeakyReLU(alpha=0.01)(x)
    x = Convolution2D(int(512 * alpha), (1, 1), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001), use_bias=False)(x)
    x = BatchNormalization()(x)
    x= LeakyReLU(alpha=0.01)(x)

    x= Dropout(0.2)(x)
    
    """ 7x7 conv 512 filters"""       
    x = DepthwiseConvolution2D(int(512 * alpha), (7, 7), strides=(1, 1), padding='same',kernel_regularizer=l2(0.001), use_bias=False)(x)
    x = BatchNormalization()(x)
    x= LeakyReLU(alpha=0.01)(x)
    x = Convolution2D(int(512 * alpha), (1, 1), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001), use_bias=False)(x)
    x = BatchNormalization()(x)
    x= LeakyReLU(alpha=0.01)(x)
    
    x= Dropout(0.3)(x)
    
    """ 7x7 conv 512 filters"""       
    x = Convolution2D(int(512 * alpha), (1, 1), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001), use_bias=False)(x)
    x = BatchNormalization()(x)
    #x = Activation('relu')(x)
    x= LeakyReLU(alpha=0.01)(x)
    
    x= Dropout(0.4)(x)
    
  
    x = GlobalAveragePooling2D()(x)
    out = Dense(classes, activation='softmax')(x)

    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = Model(inputs, out, name='VGG_with_MobileNet')
    #model.load_weights('Checkpoint/weights.h5');
    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000001)

    model.compile(adam, loss='categorical_crossentropy',metrics=['accuracy'])
    return model



def train_model(path_to_train_dataset):
    batch_size = 64
    nb_epoch =256

    
    X_train, Y_train = read_and_normalize_train_data(path_to_train_dataset)
    X_valid, Y_valid = read_and_normalize_valid_data(path_to_train_dataset)
    print("SHAPE!!!!")
    print(X_valid.shape[0])
    print(X_train.shape[0])
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=9216)])
        except RuntimeError as e:
            print(e)
    """
    #Data augmentation
    
    datagen = ImageDataGenerator(
              width_shift_range=0.2,
              height_shift_range=0.2,
              zoom_range=0.2,
              shear_range=0.2
              )
    
    datagen.fit(X_train)
    
    #with tf.device('/GPU:0'):
    model = VGG_with_MobileNet()
    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000001)

    model.compile(adam, loss='categorical_crossentropy',metrics=['accuracy'])
    
    weights_path=os.path.join('/home/marcelo/Documentos/Distracted-Driver-Detection','Checkpoint','weights.h5')       
    callbacks = [ModelCheckpoint(weights_path, monitor='val_acc', save_weights_only=True, verbose=1)]

    #with tf.device('/GPU:0'):
    hist=model.fit(datagen.flow(X_train, Y_train, batch_size=batch_size),
                    steps_per_epoch=len(X_train) / batch_size, epochs=nb_epoch,
           verbose=1, validation_data=(X_valid, Y_valid), callbacks=callbacks)


    pd.DataFrame(hist.history).to_csv("/home/marcelo/Documentos/Distracted-Driver-Detection/cache/try_hist.csv")

    predictions_valid = model.predict(X_valid.astype('float32'), batch_size=batch_size, verbose=1)
    cm1=confusion_matrix(Y_valid.argmax(axis=1), predictions_valid.argmax(axis=1))
    #ss=cm1[0,0]+cm1[1,1]+cm1[2,2]+cm1[3,3]+cm1[4,4]+cm1[5,5]+cm1[6,6]+cm1[7,7]+cm1[8,8]+cm1[9,9];
    #test_accuracy=np.divide(ss,X_valid.shape[0]);
    #print('Test Accuracy:',test_accuracy)
    
    ppath=os.path.join('/home/marcelo/Documentos/Distracted-Driver-Detection','cache','confusion_mat.npy')
    np.save(ppath, cm1)

def test_model(path_weight, path_to_test_dataset):
    X_test, Y_test = read_and_normalize_test_data(path_to_test_dataset)
    new_model = VGG_with_MobileNet()
    new_model.load_weights(path_weight)
    loss, acc = new_model.evaluate(X_test, Y_test, verbose=2)
    print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
    
   

if __name__ == '__main__':
    #print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    prog_parser = argparse.ArgumentParser()
    prog_parser.add_argument('--dataset', action = 'store', type = str, required = True)
    prog_parser.add_argument('--test', action = 'store', type = str)
    args = prog_parser.parse_args()
    if not( args.dataset or args.test):
        prog_parser.error('No arguments provided')
    if (args.dataset) and not(args.test):
        train_model(args.dataset)
    else:
        test_model(args.test, args.dataset)