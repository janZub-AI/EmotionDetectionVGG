from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
import os
import tensorflow as tf

from data_operations import DataOperations

def generate_aug_data(df,class_name, datagen, aug_class_name_path):
	created = 0
	limit = len(df.get('filename'))
	# Custom logic per class
	if(class_name == 'name'): limit = limit * 2

	for batch in datagen.flow_from_dataframe(df, 
											target_size = (48, 48), 
											color_mode = 'grayscale',
											batch_size=64,
											save_to_dir=aug_class_name_path, 
											save_prefix='aug_', 
											save_format='jpeg'):
		created = created + 64
		if(created > limit):
			return

def view_batch(batch):
	batch_size = len(batch[0])
	for i in range(0, batch_size):
		pyplot.subplot(10 + (lambdabatch_size % 10 > 0), batch_size//10, i + 1)
		pyplot.imshow(batch[0][i].reshape(48, 48), cmap=pyplot.get_cmap('gray'))
	pyplot.show()

def process_folder(folder):

	folder_path = os.path.join(dirname, folder)	
	aug_path = os.path.join(dirname, f'aug_{folder}')
	if not os.path.exists(aug_path):
		os.makedirs(aug_path)

	for class_name in os.listdir(folder_path):
		
		aug_class_name_path = os.path.join(aug_path, class_name)
		if not os.path.exists(aug_class_name_path):
			os.makedirs(aug_class_name_path)
	
		folder_class_name_path = os.path.join(folder_path, class_name)
		data = DataOperations.get_data_for_category(folder_class_name_path)

		generate_aug_data(data, class_name, datagen, aug_class_name_path)



dirname = os.path.join(os.path.dirname( __file__ ), os.path.pardir)

# in case of zca_whitening 
'''files = DataOperations.get_data(os.path.join(dirname, 'train'))

imgs = []

for f in files.get('file'):
	image = tf.keras.preprocessing.image.load_img(
            f, color_mode='grayscale', target_size=(48,48),
            interpolation='nearest'
        )
	img_array = tf.keras.preprocessing.image.img_to_array(image)
	img_array = tf.cast(img_array ,tf.float32)
	imgs.append(img_array)
'''
shift = 0.1
datagen = ImageDataGenerator(rotation_range=45, 
							 vertical_flip=True, 
							 horizontal_flip=True, 
							 width_shift_range=shift, 
							 height_shift_range=shift)
#datagen.fit(imgs)



folders = ['train', 'dev', 'test']

process_folder('dev')