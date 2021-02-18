import os
from tensorflow import keras

from utils import Utils

def evaluate_model(path, model_name):
    model = keras.models.load_model(os.path.join(path, model_name))
    train_dataset = Utils.load_data_generator('train', 32)
    dev_dataset = Utils.load_data_generator('dev', 32)
    test_dataset = Utils.load_data_generator('test', 32)

    print('-----------------------------------------')
    print(model_name) 
    print('------------------')
    print("Evaluate on test data")
    train_results = model.evaluate(train_dataset, batch_size=32)
    dev_results = model.evaluate(dev_dataset, batch_size=32)
    test_results = model.evaluate(test_dataset, batch_size=32)
    print('-----------------------------------------')
    print("train loss, train acc:", train_results)
    print("dev loss, dev acc:", dev_results)
    print("test loss, test acc:", test_results)
    print('-----------------------------------------')

models = '20210218-230114'


dirname = os.path.join(os.path.dirname( __file__ ), os.path.pardir, os.path.pardir)
dir = os.path.join(dirname, 'models_checkpoint', models)
print(dir)
for k,j in enumerate(os.listdir(dir)):
    evaluate_model(dir,j)