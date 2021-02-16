
from kerastuner.tuners import RandomSearch
from keras.metrics import SparseTopKCategoricalAccuracy
import datetime
import time
from utils.utils import Utils

from model import ConcreteModel, ConcreateHyperParameters
from rename_tensorboard import FileManager
from callbacks.callback_creator import CallbackCreator

# global for logging and unique name
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
project_name = 'template'
# callbacks

# helpers
def load_data():
    train_data_per_category = TUNER_SETTINGS['batches_per_category'] * TUNER_SETTINGS['batch_size']
    validation_data_per_category = TUNER_SETTINGS['batches_for_validation'] * TUNER_SETTINGS['batch_size']

    dev_dataset = Utils.load_data('dev', validation_data_per_category, TUNER_SETTINGS['batch_size'])
    train_dataset = Utils.load_data('train', train_data_per_category, TUNER_SETTINGS['batch_size'], True)

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
    'batch_size' : 32,  
    'batches_per_category' : 10,
    'batches_for_validation' : 2,
    'epochs' : 100,
    'max_trials' : 30,
    'executions_per_trial' : 1,
    'objective' : 'val_loss',
    'callbacks' : [es_callback, lr_callback, mc_callback]
    }

# params
hyperparameters = ConcreateHyperParameters()
hp = ConcreteModel.define_hp(hyperparameters)

num_classes = 0
input_shape = (1,1)
hypermodel = ConcreteModel(num_classes = num_classes, input_shape = input_shape)

run_tuner(hypermodel, hp)

time.sleep(5.)
FileManager.rename_files(TUNER_SETTINGS['log_dir'], hypermodel.generate_model_name, project_name)

