from keras.models import Sequential
from keras.models import load_model
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras.layers import Flatten, Dropout, Dense, BatchNormalization

from keras.layers.experimental.preprocessing import RandomCrop
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Adam, SGD
from keras.initializers import GlorotNormal, HeNormal

import tensorflow as tf
from kerastuner import HyperModel, HyperParameters

class ConcreteModel(HyperModel):
    """ VGG-B without last C512_C512_P block,
        with relu and glorot,
        extra dropout and BatchNormalization
        not sure about opt
    """

    def __init__(self, num_classes, input_shape):
        self.num_classes = num_classes
        self.input_shape = input_shape

    def build(self, hp):

        # The same initializer
        initializer = GlorotNormal()
        # The same activation for all hidden is relu
        activation = Activation('relu')

        model = Sequential()
        model.add(RandomCrop(48,48))    

        model.add(Conv2D(filters = 64, kernel_size = (3, 3), padding = 'same', input_shape = self.input_shape, kernel_initializer=initializer))
        model.add(BatchNormalization())
        model.add(activation)
        model.add(Conv2D(filters = 64, kernel_size = (3, 3), padding = 'same', kernel_initializer=initializer))
        model.add(BatchNormalization())
        model.add(activation)
        model.add(MaxPooling2D((2, 2)))

        model.add(Dropout(0.5))
        model.add(Conv2D(filters = 128, kernel_size = (3, 3), padding = 'same', kernel_initializer=initializer))
        model.add(BatchNormalization())
        model.add(activation)
        model.add(Conv2D(filters = 128, kernel_size = (3, 3), padding = 'same', kernel_initializer=initializer))
        model.add(BatchNormalization())
        model.add(activation)
        model.add(MaxPooling2D((2, 2)))


        model.add(Dropout(0.5))
        model.add(Conv2D(filters = 256, kernel_size = (3, 3), padding = 'same', kernel_initializer=initializer))
        model.add(BatchNormalization())
        model.add(activation)
        model.add(Conv2D(filters = 256, kernel_size = (3, 3), padding = 'same', kernel_initializer=initializer))
        model.add(BatchNormalization())
        model.add(activation)
        model.add(MaxPooling2D((2, 2)))


        model.add(Dropout(0.5))
        model.add(Conv2D(filters = 512, kernel_size = (3, 3), padding = 'same', kernel_initializer=initializer))
        model.add(BatchNormalization())
        model.add(activation)
        model.add(Conv2D(filters = 512, kernel_size = (3, 3), padding = 'same', kernel_initializer=initializer))
        model.add(BatchNormalization())
        model.add(activation)
        model.add(MaxPooling2D((2, 2)))
        
        model.add(Flatten())
        
        model.add(Dropout(0.5))
        model.add(Dense(1024))
        model.add(BatchNormalization())
        model.add(activation)
        
        model.add(Dense(self.num_classes, activation='softmax'))
       
        model.compile(optimizer=SGD(learning_rate=0.1, momentum=0.9, decay = 0.0001))
        return model

    def define_hp(hp_model = None):
        hp = ConcreateHyperParameters()
        return hp

    def generate_model_name(iterable, **kwarg): 
        hp = kwarg['hp']
        return f'VGG-B-{hp["name"]}'

class ConcreateHyperParameters():
    """!Important! Ensure that if min provided, then max > min is specified either.
    """
    def __init__(self):
        pass