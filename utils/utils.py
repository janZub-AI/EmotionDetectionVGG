import tensorflow as tf
import os
import pandas as pd
import numpy as np
import datetime
import time
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from utils.data_operations import DataOperations

dirname = os.path.join(os.path.dirname( __file__ ), os.pardir)
max_int = 99999

class Utils():
    
    def load_data_generator(subfolder, batch_size):
        path = os.path.join(dirname, f'norm_{subfolder}')

        generator = ImageDataGenerator(horizontal_flip = True,
                                        samplewise_center = True,
                                        samplewise_std_normalization = True)


        data = generator.flow_from_directory(directory = path,
                                      target_size = (48,48), 
                                      color_mode= 'grayscale',
                                      batch_size=batch_size)

        return data

    def load_dataset(subfolder, batch_size = 32, take_batches = max_int, aug_data = False):


        df = DataOperations.get_data(os.path.join(dirname, subfolder), take = batch_size * take_batches)

        if(aug_data):
            df = df.append(DataOperations.get_data(os.path.join(dirname, f'aug_{subfolder}'), take = batch_size * take_batches))

        labels_as_ints = Utils.get_labels_as_ints(df.get('class'))     
        labels = tf.convert_to_tensor(labels_as_ints)
        image_paths = tf.convert_to_tensor(df.get('filename'), dtype=tf.string)

        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        dataset = dataset.map(Utils.parse_function, num_parallel_calls = 10)
        dataset = dataset.batch(batch_size)
        dataset.shuffle(1000000)
        dataset = dataset.cache()

        return dataset

    def parse_function(filename, label):
        image_string = tf.io.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=1)
        image = tf.cast(image_decoded/255, tf.float32)
        image = tf.image.resize(image, [48, 48])
        return image, label

    def get_labels_as_ints(values):
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(values)
        #onehot_encoder = OneHotEncoder(sparse=False, dtype=np.int32)
        #integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        #return onehot_encoder.fit_transform(integer_encoded)
        return integer_encoded


    def plot_images_summary():
        def count_exp(path, set_):
            dict_ = {}
            for expression in os.listdir(path):
                dir_ = os.path.join(path, expression)
                dict_[expression] = len(os.listdir(dir_))
            df = pd.DataFrame(dict_, index=[set_])
            return df

        folders = ['train', 'aug_train', 'dev', 'aug_dev', 'test']
        data = None

        for f in folders:
            f_path = os.path.join(dirname, f)
            print(f_path)
            if(os.path.isdir(f_path) is not True):continue
            if(data is None):
                data = [count_exp(f_path, f)]
            else:
                data.append(count_exp(f_path, f))
            
        for d in data:
            print(d)
            d.transpose().plot(kind='bar')

        plt.show()