
from FeatureExtractorBase import FeatureExtractorBase
import logging 	# logging 
from tqdm import tqdm
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv3D, MaxPooling3D, ZeroPadding3D
from keras.optimizers import SGD


class FeatureExtractorC3D(FeatureExtractorBase):


	def __init__( self ):
		logging.info( 'FeatureExtractorC3D Initialization' )
		logging.debug( 'create the C3D network from Keras or whatever' )
		self.TFResource = ''
		self.featureName = 'C3D'
		self.model = self.C3D_model(weights_path='weights_C3D_sports1M_tf.h5', summary=True )
		return
		
			
	def elaborateVideo(self, videoPath, game_start_time, game_end_time):
		""" Elaborates the video by first extracting the frames then converting them to the desired elaboration

		arguments:
			videoPath  (str): string of the video Path ie "D:\\2014-2015\\03 - 12 -2014 P$G vs Monaco\\1.mkv" for windows
			start_time (int): half start time in seconds
			end_time   (int): half end time in seconds
	
		return: 
			C3D features_array of the video containing C3D representations of the halfime seperated by the specified stride and obtained from the video at a downsample of 25 fps 
		"""
		logging.debug( 'infer the C3D Model on the videos in gamePath' )
		logging.info('Working in C3D Model')
		
		fps = self.getFPS(videoPath, handler = '')
		fps_downsample_ratio=fps/25
		logging.debug(str(fps) + ' fps for video "' + videoPath + '"')		

		logging.debug('From index ' + str(game_start_time * fps) + ' to index ' + str(game_end_time * fps))

		vid = self.getVideoFrames(videoPath = videoPath,
									game_start_index = int(game_start_time * fps), 
									game_end_index = int((game_end_time + game_start_time) * fps), 
									height = 112, 
									width = 112, 
									handler = '')

		logging.debug( 'Converting the Video into C3D' )

		features_array = []

		for index_frame in tqdm(np.arange(0, (len(vid)-16)), desc='Converting the Video into C3D', unit='batch'):
			if (int((index_frame))%(fps*self.intervalStrideTime) < 1):

				# interval_index = interval_time * fps

				frames = []
				for i in range(0,16):
					index = index_frame + i * fps_downsample_ratio
					# logging.info(index)
					frames.append(vid[int(index)])
					
				C3D_rep = self.model.predict_on_batch(np.array(frames).reshape (1, 16, 112, 112, 3))

				features_array.append( C3D_rep )


		return features_array
		
		





	def C3D_model(self, weights_path=None, summary=False):
		""" Return the Keras model of the network

		arguments:
			weights_path (str):  (default=None)
			summary (bool):  (default=False)
	
		return: 
			Keras model of the network		
		"""

		model = Sequential()

		# 1st layer group
		model.add(Conv3D(64, (3, 3, 3), activation="relu",name="conv1", 
						 input_shape=(16,112,112,3),
						 strides=(1, 1, 1), padding="same"))  
		model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name="pool1", padding="valid"))

		# 2nd layer group  
		model.add(Conv3D(128, (3, 3, 3), activation="relu",name="conv2", 
						 strides=(1, 1, 1), padding="same"))
		model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name="pool2", padding="valid"))

		# 3rd layer group   
		model.add(Conv3D(256, (3, 3, 3), activation="relu",name="conv3a", 
						 strides=(1, 1, 1), padding="same"))
		model.add(Conv3D(256, (3, 3, 3), activation="relu",name="conv3b", 
						 strides=(1, 1, 1), padding="same"))	
		model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name="pool3", padding="valid"))

		# 4th layer group  
		model.add(Conv3D(512, (3, 3, 3), activation="relu",name="conv4a", 
						 strides=(1, 1, 1), padding="same"))   
		model.add(Conv3D(512, (3, 3, 3), activation="relu",name="conv4b", 
						 strides=(1, 1, 1), padding="same"))
		model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name="pool4", padding="valid"))

		# 5th layer group  
		model.add(Conv3D(512, (3, 3, 3), activation="relu",name="conv5a", 
						 strides=(1, 1, 1), padding="same"))   
		model.add(Conv3D(512, (3, 3, 3), activation="relu",name="conv5b",
						  strides=(1, 1, 1), padding="same"))
		model.add(ZeroPadding3D(padding=(0, 1, 1)))	
		model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name="pool5", padding="valid"))
		model.add(Flatten())
						 
		# FC layers group
		model.add(Dense(4096, activation='relu', name='fc6'))
		model.add(Dropout(.5))
		model.add(Dense(4096, activation='relu', name='fc7'))
		model.add(Dropout(.5))
		model.add(Dense(487, activation='softmax', name='fc8'))

		if summary:
			logging.info('Before Popping :' )
			logging.info(model.summary())

		logging.info('Loading Model Weights...')
		model.trainable_weights

		if weights_path:
			model.load_weights(weights_path)
		# model.load_weights('weights_C3D_sports1M_tf.h5')

		logging.info('Popping classification layer')

		model.pop()
		model.pop()

		if summary:
			logging.info('After Popping :')
			logging.info(model.summary())
			
		model.compile(loss='mean_squared_error', optimizer='sgd')	

		return model

