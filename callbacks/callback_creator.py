from keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from callbacks.early_stopping import EarlyStoppingAt

class CallbackCreator():
    def get_model_checkout(current_time, monitor = 'val_accuracy'):
        return ModelCheckpoint(
                filepath = f'models_checkpoint/{current_time}/'+'{epoch:02d}-{' + monitor + ':.5f}.hdf5',
                save_weights_only=False,
                verbose = 1,
                monitor = monitor,
                mode='auto',
                period = 1,
                save_best_only=True)
                
    def get_tensorboard(log_dir):
        return TensorBoard(
                log_dir = log_dir,
                histogram_freq = 1,
                embeddings_freq = 1,
                update_freq = 'epoch')

    def get_lr_scheduler():
        dlearning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=10, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.000001)

        return dlearning_rate_reduction

    def get_early_stopping(stop_at = 'val_accuracy'):
        return EarlyStoppingAt(patience = 3, ignored_epoch = 5, stop_at = stop_at)


