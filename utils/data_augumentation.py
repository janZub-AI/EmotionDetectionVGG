from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img, img_to_array, save_img
from matplotlib import pyplot
import os
import numpy as np
import tensorflow as tf
from data_operations import DataOperations

from skimage import exposure
dirname = os.path.join(os.path.dirname( __file__ ), os.path.pardir)


files = DataOperations.get_data(os.path.join(dirname, 'train')).get('filename')

dataset = np.ndarray(shape=(len(files), 48, 48, 1),
				dtype=np.float32)
i = 0
for f in files:
	image = load_img(f, 
	color_mode='grayscale', target_size=(48,48),
	interpolation='nearest')

	img_array = img_to_array(image)
	dataset[i] = tf.cast(img_array/255. ,tf.float32)
	i += 1
	
mean = dataset.mean(axis=(0,1,2))
std = dataset.std(axis=(0,1,2))

print(mean,std)

def EH(img):
	img_adapteq = exposure.equalize_hist(img)
	return img_adapteq


def generate_aug_data(df,class_name, aug_class_name_path):
	       
	datagen = ImageDataGenerator(rotation_range=45, 
								vertical_flip=True, 
								horizontal_flip=True, 
								width_shift_range=shift, 
								height_shift_range=shift)
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

def generate_norm_data(df,class_name, aug_class_name_path):

	local_files = df.get('filename')
	local_dataset = np.ndarray(shape=(len(local_files), 48, 48, 1), dtype=np.float32)
	i = 0
	for f in local_files:
		image = load_img(f, 
		color_mode='grayscale', target_size=(48,48),
		interpolation='nearest')

		img_array = img_to_array(image)
		local_dataset[i] = tf.cast(img_array/255. ,tf.float32)
		i += 1

	print(local_dataset[0,0,0])
	local_dataset[..., 0] -= mean[0]
	local_dataset[..., 0] /= std[0]
	print('==============')
	print(local_dataset[0,0,0])
	i=0
	for f in local_files:
		_, fn = os.path.split(f)
		local_dataset[i] = EH(local_dataset[i])
		save_img(os.path.join(aug_class_name_path, fn), local_dataset[i])
		i += 1

def view_batch(batch):
	batch_size = len(batch[0])
	for i in range(0, batch_size):
		pyplot.subplot(10 + (lambdabatch_size % 10 > 0), batch_size//10, i + 1)
		pyplot.imshow(batch[0][i].reshape(48, 48), cmap=pyplot.get_cmap('gray'))
	pyplot.show()

def process_folder(folder):

	folder_path = os.path.join(dirname, folder)	
	aug_path = os.path.join(dirname, f'norm_{folder}')
	if not os.path.exists(aug_path):
		os.makedirs(aug_path)

	for class_name in os.listdir(folder_path):
		
		aug_class_name_path = os.path.join(aug_path, class_name)
		if not os.path.exists(aug_class_name_path):
			os.makedirs(aug_class_name_path)
	
		folder_class_name_path = os.path.join(folder_path, class_name)
		data = DataOperations.get_data_for_category(folder_class_name_path)

		generate_norm_data(data, class_name, aug_class_name_path)
		#generate_aug_data(data, class_name, aug_class_name_path)



	# add std and mean normalization, not sample-wise but training wise




folders = ['train', 'dev', 'test']
for f in folders:
	process_folder(f)