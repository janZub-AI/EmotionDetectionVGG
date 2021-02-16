from kerastuner.tuners import RandomSearch
from keras.metrics import SparseTopKCategoricalAccuracy
import datetime
import time
from utils.utils import Utils
import tensorflow as tf
from kerastuner import HyperModel, HyperParameters
from model import ConcreteModel, ConcreateHyperParameters
from utils.rename_tensorboard import FileManager
from callbacks.callback_creator import CallbackCreator

# global for logging and unique name
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
project_name = 'vgg'

if tf.config.list_physical_devices('GPU'):
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
    tf.config.experimental.set_virtual_device_configuration(physical_devices[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8000)])

# helpers
def load_data():

    dev_dataset = Utils.load_dataset('dev', TUNER_SETTINGS['batch_size'], TUNER_SETTINGS['batches_for_validation'])
    train_dataset = Utils.load_dataset('train', TUNER_SETTINGS['batch_size'], TUNER_SETTINGS['batches_per_category'])
    return train_dataset, dev_dataset

# runner
def run_tuner(hypermodel, hp):   
    # load dataset
    train_dataset, dev_dataset = load_data()

    # init tensorboard here so each run will have folder, 
    # which we can rename based on trial_id
    tb_callback = CallbackCreator.get_tensorboard(TUNER_SETTINGS['log_dir'])

    tuner = RandomSearch(
        hypermodel,
        objective = TUNER_SETTINGS['objective'],
        max_trials = TUNER_SETTINGS['max_trials'],      
        metrics= ['accuracy'], 
        loss='sparse_categorical_crossentropy',
        hyperparameters = hp,
        executions_per_trial = TUNER_SETTINGS['executions_per_trial'],
        directory = TUNER_SETTINGS['log_dir'],     
        project_name = project_name)

    tuner.search(train_dataset, validation_data = dev_dataset,
        batch_size = TUNER_SETTINGS['batch_size'],
        callbacks = TUNER_SETTINGS['callbacks'] + [tb_callback],
        epochs = TUNER_SETTINGS['epochs']
        )

# callbacks
mc_callback = CallbackCreator.get_model_checkout(current_time)
lr_callback = CallbackCreator.get_lr_scheduler()
es_callback = CallbackCreator.get_early_stopping()

# params
TUNER_SETTINGS = {
    'log_dir' : f'logs/{current_time}',    
    'batch_size' : 128,  
    'batches_per_category' : 100000,
    'batches_for_validation' : 10000,
    'epochs' : 100,
    'max_trials' : 2,
    'executions_per_trial' : 1,
    'objective' : 'val_loss',
    'callbacks' : [lr_callback, mc_callback]
    }

# params

hp = HyperParameters()
hypermodel = ConcreteModel(num_classes = 7, input_shape = (48,48,1))

run_tuner(hypermodel, hp)

print(TUNER_SETTINGS['log_dir'])
FileManager.rename_files(TUNER_SETTINGS['log_dir'], hypermodel.generate_model_name, project_name)

