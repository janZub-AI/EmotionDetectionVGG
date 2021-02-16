from keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler

from early_stopping import EarlyStoppingAt

class CallbackCreator():
    def get_model_checkout(current_time, monitor = 'val_loss'):
        return ModelCheckpoint(
                filepath = f'models_checkpoint/{current_time}/'+'{epoch:02d}-{' + monitor + ':.5f}.hdf5',
                save_weights_only=False,
                verbose = 1,
                monitor= monitor,
                mode='auto',
                period = 5,
                save_best_only=True)
                
    def get_tensorboard(log_dir):
        return TensorBoard(
                log_dir = log_dir,
                histogram_freq = 1,
                embeddings_freq = 1,
                update_freq = 'epoch')

    def get_lr_scheduler():
        def scheduler(epoch, lr):
            if epoch % 5 == 0:
                lr = lr*0.8
            return lr

        return LearningRateScheduler(scheduler)

    def get_early_stopping(stop_at = 'val_loss'):
        return EarlyStoppingAt(patience = 3, ignored_epoch = 5, stop_at = stop_at)


