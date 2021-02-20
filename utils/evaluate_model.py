import os
from tensorflow import keras
import tensorflow as tf
from utils import Utils

if tf.config.list_physical_devices('GPU'):
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
    tf.config.experimental.set_virtual_device_configuration(physical_devices[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000)])

def evaluate_model(path, model_name):
    model = keras.models.load_model(os.path.join(path, model_name))

    print('-----------------------------------------')
    print(model_name) 
    print('------------------')
    print("Evaluate on test data")
    train_results = model.evaluate(train_dataset, batch_size=128)
    dev_results = model.evaluate(dev_dataset, batch_size=128)
    test_results = model.evaluate(test_dataset, batch_size=128)
    print('-----------------------------------------')
    print("train loss, train acc:", train_results)
    print("dev loss, dev acc:", dev_results)
    print("test loss, test acc:", test_results)
    print('-----------------------------------------')

models = '20210220-105303'

dirname = os.path.join(os.path.dirname( __file__ ), os.path.pardir, os.path.pardir)

train_dataset = Utils.load_data_generator('train', 128)
dev_dataset = Utils.load_data_generator('dev', 128)
test_dataset = Utils.load_data_generator('test', 128)


dir = os.path.join(dirname, 'models_checkpoint', models)
print(dir)
for k,j in enumerate(os.listdir(dir)):
    evaluate_model(dir, j)