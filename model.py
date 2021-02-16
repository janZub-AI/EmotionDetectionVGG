from keras.models import Sequential
from keras.models import load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dropout, Dense, BatchNormalization

from keras.layers.advanced_activations import PReLU
from keras.optimizers import Adam
from keras.initializers import HeNormal

from kerastuner import HyperModel, HyperParameters

class ConcreteModel(HyperModel):

    def __init__(self, num_classes, input_shape):
        self.num_classes = num_classes
        self.input_shape = input_shape

    def build(self, hp):
        model = Sequential()


        return model

    def define_hp(hp_model = None):
        hp = HyperParameters()

        return hp

    def generate_model_name(iterable, **kwarg):
        hp = kwarg['hp']    
            
        return name

class ConcreateHyperParameters():
    """!Important! Ensure that if min provided, then max > min is specified either.
    """
    def __init__(self):
        pass