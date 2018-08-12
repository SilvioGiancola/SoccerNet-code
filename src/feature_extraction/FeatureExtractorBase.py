
import logging 	# logging 
import os
import configparser
import numpy as np
import json
from tqdm import tqdm
import time
import re
import cv2
import platform
import traceback

class FeatureExtractorBase():

	#Main variables involved which are initialzied to default values. 

	intervalStrideTime = 1 # initial value for interval stride set to 1 s
	overwrite = False # Bypass elaborating the video if there exists a .npy with the name that matches the featureFilename
	featureFilename = 'data' #Generic Name, preferable to call it into something specific to the type of FeatureExtraction and include info like strideTime

	def __init__(self):
		logging.info('__init__')
		return
		


	def elaborateVideo(self, videoPath, game_start_time, game_end_time):
		""" Elaborates the video by first extracting the frames then converting them to the desired elaboration
			
		Arguments:
			videoPath : string of the video Path ie "D:\\2014-2015\\03 - 12 -2014 P$G vs Monaco\\1.mkv" for windows
			start_time : half start time
			end_time : half end time
		
		Return: 
			This function is overwritten when called into the Specific Feature Extractor code like FeatureExtractorC3D where the output is the array of the data specific for the video
			and which is saved later.
		"""
		logging.info( 'elaborateVideo using FeatureExtractorBase' )
		logging.warning( 'Please select a type of feature, you are using the Base class' )
		return

		

	def elaborateGame(self, gameDir):
		""" Iterates over the two videos in the directory by using Video extentions and the game-specific info from ini from the readINI function below
			
		Arguments:
			gameDir : string of the game Directory ie "D:\\2014-2015\\03 - 12 -2014 P$G vs Monaco" for windows

		Return: 
			None, but it creates two .npy files, one for each half in the gamedirectory depending on the type of extraction	
		"""
		logging.info('='*50)
		logging.info('Elaborating Game : {0}'.format(gameDir))
		
		start_times_of_halves, end_times_of_halves, video_formats = self.readINI(gameDir)

		logging.debug('Half 1 (' + video_formats[0] + '): From time ' + str(start_times_of_halves[0]) + ' to time ' + str(end_times_of_halves[0]))
		logging.debug('Half 2 (' + video_formats[1] + '): From time ' + str(start_times_of_halves[1]) + ' to time' + str(end_times_of_halves[1]))

		start_time_elaboration = time.time()



		# Elaborate Half 1		
		logging.debug('Grabbing Video Frames for half 1')
		npy_FileName = '1_' + self.featureName

		logging.info('Searching for data : {0}.npy'.format(npy_FileName))

		if (npy_FileName + '.npy' in os.listdir(gameDir)) and (self.overwrite == False ): 

			logging.info('Data Already Extracted For First Half in The Path {0} , skipping '.format(gameDir))

		else: 
			if (npy_FileName + '.npy' in os.listdir(gameDir)): logging.info('Overwriting Existing Data Half 1')

			try:
				features_array = self.elaborateVideo(videoPath=os.path.join(gameDir,video_formats[0]), 
													 game_start_time = start_times_of_halves[0], 
													 game_end_time = end_times_of_halves[0])

				np.save(os.path.join(gameDir,npy_FileName) , np.array(features_array))

			except Exception as e:

				logging.info('\n'+'!'*20 + '  error  ' + '!'*20)
				logging.info(traceback.format_exc()+'!'*20 + '  error  in Half 1   ' + '!'*20+'\n')
				logging.info(traceback.format_exc()+'!'*20 + '  error  ' + '!'*20+'\n')




		# Elaborate Half 2
		logging.debug('Grabbing Video Frames for half 2')
		npy_FileName = '2_' + self.featureName

		logging.info('Searching for data : {0}.npy'.format(npy_FileName))
		
		if (npy_FileName + '.npy' in os.listdir(gameDir)) and (self.overwrite == False ): 
			logging.info('Data Already Extracted For Second Half in The Path {0} , skipping '.format(gameDir))

		else : 
			if (npy_FileName + '.npy' in os.listdir(gameDir)): logging.info('Overwriting Existing Data Half 2')

			try:
				features_array = self.elaborateVideo(videoPath = os.path.join(gameDir,video_formats[1]), 
													 game_start_time = start_times_of_halves[1], 
													 game_end_time = end_times_of_halves[1])
				
				np.save(os.path.join(gameDir,npy_FileName) , np.array(features_array))

			except Exception as e:

				logging.info('\n'+'!'*20 + '  error  ' + '!'*20)
				logging.info('We have a problem in the directory {0} Half 2'.format(gameDir))
				logging.info(traceback.format_exc()+'!'*20 + '  error  ' + '!'*20+'\n')


		logging.debug('Total Elaboration time for the game ' + gameDir + ' : ' + str((time.time() - start_time_elaboration)/60) + ' min' )
		logging.info('Total Elaboration time for the game ' + gameDir + ' : ' + str((time.time() - start_time_elaboration)/60) + ' min' )
		return
				




	def elaborateDataset(self, rootDir):
		""" iterates over the entire dataset by using a list of strings of directories obtained by the get_directories_and_end_times_OpenCV function below
			
		Arguments:
			rootDir : string of the root Directory ie "D:\\" for windows

		Return: 
			None, but it creates two .npy files, one for each half in the gamedirectory depending on the type of extraction
		"""
		logging.info('elaborateDataset FeatureExtractorBase' )
		logging.debug('Get the list of Games inside the rootDir from Json file')



		games_directories = []
		for Championship in next(os.walk(rootDir))[1]:
			for Year in next(os.walk(os.path.join(rootDir, Championship)))[1]:
				for Game in next(os.walk(os.path.join(rootDir, Championship, Year)))[1]:
					Game_FullPath = os.path.join(rootDir, Championship, Year, Game)
					
					games_directories.append( Game_FullPath )

		
		for j,gameDir in enumerate(games_directories):

			logging.info('='*50)
			logging.info (' ||  Game {0}/{1}  ||'.format(j+1,len(games_directories)))

			self.elaborateGame(gameDir)
		
		return


############################################################################################################################################################################################################################################################
#=============================================JSON FILES OPERATIONS==========================================================##=============================================JSON FILES OPERATIONS==========================================================#
############################################################################################################################################################################################################################################################
	

	def readINI(self, gameDir):
		""" get the start time and the preferred present extension of the video in the directory based on the INI files
			
		Arguments:
			gameDir : string of the game Directory ie "D:\\2014-2015\\03 - 12 -2014 P$G vs Monaco" for windows which contains the INI file

		Return: 
			start_times_of_halves and end_times_of_halves: returns list of integers for each start and end [end/start first half , end/start second half] in seconds
			video_formats : returns list of strings the preferred format of each half ie : ['1.mkv' , '2.mkv' ]		
		"""
		
		filename1 = "1.mkv"
		filename2 = "2.mkv"
		
		logging.debug( 'File extensions are respectively {0} and {1}'.format(filename1, filename2) )
		
		Start1 = 0 #min1*60 + sec1
		Start2 = 0 #min2*60 + sec2

		import skvideo.io
		from tqdm import tqdm
		import numpy as np
		import cv2
		import json


		videoPath = os.path.join(gameDir, filename1)
		metadata = skvideo.io.ffprobe(videoPath)
		try:
			for key in metadata["video"]["tag"]:
				if (key["@key"] == "DURATION"):
					time_str = key["@value"]
					h, m, s = time_str.split(".")[0].split(":")
					End1 = int(h)*60*60 + int(m)*60 + int(s)
					break;
				
		except Exception as e:
			End1 = 45*60
					
		

		videoPath = os.path.join(gameDir, filename2)
		metadata = skvideo.io.ffprobe(videoPath)
		try:
			for key in metadata["video"]["tag"]:
				if (key["@key"] == "DURATION"):
					time_str = key["@value"]
					h, m, s = time_str.split(".")[0].split(":")
					End2 = int(h)*60*60 + int(m)*60 + int(s)
					break;
					
		except Exception as e:
			End2 = 45*60
					

		start_times_of_halves = [Start1, Start2] 
		end_times_of_halves = [End1 , End2]
		video_formats = [filename1, filename2]

		return start_times_of_halves, end_times_of_halves, video_formats

############################################################################################################################################################################################################################################################
#=================================================VIDEO HANDLING OPERATIONS===================================================#=================================================VIDEO HANDLING OPERATIONS==================================================#
############################################################################################################################################################################################################################################################

	def getFPS(self, videoPath, handler = ''):
		""" Return the frame per second (fps) of a video reading the metadata
			
		Arguments:
			videoPath (str): Path of the video to analyse
			handler   (str): select the tool to read the metadata: {'OpenCV', 'Scikit-Video', ''} (default = '')

		Return: 
			(float) Return a the frame per second (fps) of a video reading the metadata
		"""
		# Select default handler
		if ((handler == '') and (platform.system() == 'Windows')):
			# On Windows, prefer OpenCV
			handler = 'OpenCV'			
		elif ((handler == '') and (platform.system() == 'Linux')):
			# On Linux, prefer Scikit-Video
			handler = 'Scikit-Video'

		fps = 0

		if (handler == 'OpenCV'):
			# OpenCV
			vidcap = cv2.VideoCapture(videoPath)
			fps = vidcap.get(cv2.CAP_PROP_FPS)

		elif (handler == 'Scikit-Video'):
			# Scikit-Video
			import skvideo.io
			import json
			metadata = skvideo.io.ffprobe(videoPath)
			fps_meta = json.dumps(metadata["video"]['@avg_frame_rate'], indent=4).replace('"','')
			fps = int(fps_meta.split('/')[0])/int(fps_meta.split('/')[1])

		return 25



	def getVideoFrames(self, videoPath, game_start_index, game_end_index, height, width, handler = ''):
		""" Return a 4D numpy array frame per second (fps) of a video reading the metadata

		Arguments:
			videoPath		(str): Path of the video to analyse
			game_start_index (int): starting index for the video
			game_end_index   (int): ending index for the video
			height		   (int): height of the frames (frame downsampling)
			width			(int): width of the frames (frame downsampling)
			handler		  (str): select the tool to read the metadata: {'OpenCV', 'Scikit-Video', ''} (default = '')

		Return: 
			(ndarray) Return a 4D numpy array of a video
		"""
		# Select default handler
		if ((handler == '') and (platform.system() == 'Windows')):
			# On Windows, prefer OpenCV
			handler = 'OpenCV'			
		elif ((handler == '') and (platform.system() == 'Linux')):
			# On Linux, prefer Scikit-Video
			handler = 'Scikit-Video'

		logging.debug('getVideoFrames for ' + videoPath + \
			' from ' + str(game_start_index) + ' to ' + str(game_end_index) + \
			' with size ' + str(height) + 'x' + str(width) + \
			' using handler : ' + handler)
		
		vid = []

		if (handler == 'OpenCV'):

			vidcap = cv2.VideoCapture(videoPath)
			vidcap.set(1, game_start_index)

			for i in tqdm(range(int(game_start_index),int(game_end_index)), desc='Grabbing Video Frames', unit='frame'):
				ret, frame = vidcap.read()

				if not ret:
					logging.info('No more frames to grab, breaking')
					logging.warning('No more frames to grab, breaking')
					break

				vid.append(cv2.resize(frame, (height, width)))


		elif (handler == 'Scikit-Video'):

			import skvideo.io

			videogen = skvideo.io.vreader(videoPath, num_frames=int(game_end_index), backend='ffmpeg')

			pbar = tqdm(total=game_end_index,
						desc='Grabbing Video Frames', 
						unit='frame')
			index_frame = 0

			for frame in videogen:

				if (index_frame >= game_start_index):
					vid.append(cv2.resize(frame, (height, width)))

				index_frame = index_frame + 1  
				pbar.update(1)
			
				if (index_frame >= game_end_index):			
					break


			pbar.close()
		
		return vid