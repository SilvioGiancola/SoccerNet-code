############################################################################################################################################################################################################################################################
# THIS MAIN WILL CREATE A SINGLE PAIR of .npy DATA AND LABELS FOR EACH GAME 
#
#
import logging 	# logging purposes
import time		# timing puposes
from datetime import date # date purposes


# basic config
logFileName = "log/" + date.today().isoformat() + time.strftime('_%H-%M-%S')
logging.basicConfig(filename=logFileName + '_datalog.log',	# filename for logging
					level=logging.DEBUG, 		# level of logging (default = DEBUG)
					format='[%(asctime)s][%(levelname)s]: %(message)s')	# format

# create new stream handler
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# set a format which is simpler for console use
formatter = logging.Formatter('[%(asctime)s][%(levelname)s]: %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)


# create new stream handler
error = logging.FileHandler(logFileName + '_issuegame.log')
error.setLevel(logging.ERROR)
# set a format which is simpler for console use
formatter = logging.Formatter('[%(asctime)s][%(levelname)s]: %(message)s')
# tell the handler to use this format
error.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(error)

# fisrt log
logging.info('Start of the main.py file')



import argparse
parser = argparse.ArgumentParser(description='Extract Features for certain games')
parser.add_argument('--dataset', 	help='path of the dataset',		required=False,	type=str, 	default='../../data')
parser.add_argument('--stride', 	help='stride between features',	required=False,	type=float, default=0.5)
parser.add_argument('--GPU', 		help='ID of the GPU to use' , 	required=False, type=int, 	default=0)
parser.add_argument('--overwrite', 	help='overwrite the features',	required=False,	action="store_true")
parser.add_argument('--C3D', 		help='compute C3D features',	required=False,	action="store_true")
parser.add_argument('--ResNET', 	help='compute ResNET features',	required=False,	action="store_true")
args = parser.parse_args()



import os 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)


#Used to estimate operation time
start_time = time.time()

# import tensorflow as tf
# with tf.device("/gpu:" + str(args.GPU)):

if (args.C3D):
	from FeatureExtractorC3D import FeatureExtractorC3D
	FeatureExtractor = FeatureExtractorC3D()
	FeatureExtractor.intervalStrideTime = args.stride
	FeatureExtractor.overwrite = args.overwrite
	# FeatureExtractor.elaborateDataset(args.dataset)
	FeatureExtractor.elaborateGame(args.dataset + '/france_ligue-1/2015-2016/2015-09-19 - 18-30 Reims 1 - 1 Paris SG')




if (args.ResNET):
	from FeatureExtractorResNet import FeatureExtractorResNet
	FeatureExtractor = FeatureExtractorResNet()
	FeatureExtractor.intervalStrideTime = args.stride
	FeatureExtractor.overwrite = args.overwrite
	# FeatureExtractor.elaborateDataset(args.dataset)
	FeatureExtractor.elaborateGame(args.dataset + '/france_ligue-1/2015-2016/2015-09-19 - 18-30 Reims 1 - 1 Paris SG')



#Execution Time
TotExecution = (time.time() - start_time)

ExecutionTimeHr = int(TotExecution//3600)
ExecutionTimeMin = int(TotExecution%3600)//60
ExecutionTimeSec = int(TotExecution%3600)%60
logging.info("Done with net execution time of {0} hrs : {1} min : {2} sec \n".format(ExecutionTimeHr,ExecutionTimeMin,ExecutionTimeSec))
# print("Done with net execution time of {0} hrs : {1} min : {2} sec \n".format(ExecutionTimeHr,ExecutionTimeMin,ExecutionTimeSec))


#################################################             END             ##################################################################################################             END             #################################################

