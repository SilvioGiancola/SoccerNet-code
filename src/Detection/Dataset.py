

import numpy as np
import random
# import h5py
import os
import random
import json

from operator import itemgetter
import heapq
import time
from tqdm import tqdm

from itertools import islice

class dataset():
  
    # Initialization, takes into
    def __init__(self, ):
        print("Init")
        # training
        self.num_classes = 4
        self.count_labels = np.array([0, 0, 0, 0])
        self.size_batch = 0
        self.nb_batch_training = 0
        self.nb_epoch_per_batch = 1


    def loadTestingDataset(self, path_data, featureName, PCA=True, window_size_sec=60):

        print("Loading Testing Data:", path_data)
        path_data_dirname, path_data_basename = os.path.split(path_data)
        listGames = np.load(path_data)

        FeaturePerSecond = 2 # number of features per Time_Second
        number_frames_in_window = window_size_sec * FeaturePerSecond # 120 by default
        self.number_frames_in_window = number_frames_in_window
        print(window_size_sec)

        i = 0
        self.testing_Labels_onehot = {}
        self.testing_features = {}
        self.testing_GamesKeys = []

        for gamePath in tqdm(listGames):
            gamePath = os.path.join(path_data_dirname, gamePath)
            i += 1
            for featureFileName in os.listdir(gamePath):
                if (featureName in featureFileName and ( (PCA and "PCA" in featureFileName) or (not PCA and "PCA" not in featureFileName) ) ):
                    featureFullPath = os.path.join(gamePath, featureFileName)
                    if "1_" in featureFileName: key = os.path.join(gamePath,"Half_1")
                    elif "2_" in featureFileName: key = os.path.join(gamePath,"Half_2")
                    self.testing_GamesKeys.append(key)
                    self.testing_features_cont = np.load(featureFullPath)


                    # print ()
                    # print ("shape", self.testing_features_cont.shape)
                    # print ("strides", self.testing_features_cont.strides)

                    # ciculent business
                    strides = self.testing_features_cont.strides
                    nb_frames = self.testing_features_cont.shape[0]
                    size_feature = self.testing_features_cont.shape[1]
                    sliding_window_seconds = window_size_sec # in seconds
                    feature_per_second = 2 # number of features per Time_Second
                    sliding_window_frame = sliding_window_seconds * feature_per_second # number of features per Time_Second

                    self.testing_features_cont = np.append([self.testing_features_cont[0,:]]*sliding_window_seconds, self.testing_features_cont, axis=0)
                    self.testing_features_cont = np.append(self.testing_features_cont, [self.testing_features_cont[-1,:]]*sliding_window_seconds, axis=0)
                    self.testing_features[key] = np.lib.stride_tricks.as_strided(self.testing_features_cont, shape=(int(nb_frames/2), sliding_window_frame, size_feature), strides=(strides[0]*2,strides[0],strides[1]))

# 
                    # print(self.testing_features_cont.shape)
                    # print(self.testing_features[key].shape)
                  

                    for pl in range(20,40):
                        # print()
                        for pk in range(10):
                            # print(pl,pk,self.testing_features[key][pl,pk,:5])
                            assert((self.testing_features[key][pl,pk,:] - self.testing_features_cont[pl*2+pk,:]).sum() == 0)


                    # time.sleep(1)

                    # reverse does not works if impair number of features 
                    # for pl in range(-50,-70,-1):
                    #     print(pl,self.testing_features_cont[pl,:5])
                    # for pl in range(-20,-40,-1):
                    #     print()
                    #     for pk in range(-1,-10,-1):
                    #         print(pl,pk,self.testing_features[key][pl,pk,:5])
                    #         assert((self.testing_features[key][pl,pk,:] - self.testing_features_cont[pl*2+pk,:]).sum() == 0)





                    # FeaturePerTimeSecond = 2 # number of features per TimeSecond

                    labelFullPath = os.path.join(gamePath, "Labels.json")
                    with open(labelFullPath) as labelFile :
                        jsonLabel = json.loads(labelFile.read())

                    Labels = np.zeros((self.testing_features[key].shape[0],4), dtype=int)
                    Labels[:,0] = 1
                    
                    
                    for event in jsonLabel["annotations"]:
                        Half = int(event["gameTime"][0])
                        Time_Minute = int(event["gameTime"][-5:-3])
                        Time_Second = int(event["gameTime"][-2:])

                        if ("card" in event["label"]): label = 1
                        elif ("subs" in event["label"]): label = 2
                        elif ("soccer" in event["label"]): label = 3
                        # else: print("err in event Label", event["label"])

                        

                        if ("1_" in featureFileName and Half == 1) or ("2_" in featureFileName and Half == 2):
                            aroundValue = min(Time_Minute*60+Time_Second, Labels.shape[0]-1)
                            # print(aroundValue)
                            Labels[(aroundValue-int(sliding_window_seconds/2)):(aroundValue+int(sliding_window_seconds/2)), 0] = 0
                            Labels[(aroundValue-int(sliding_window_seconds/2)):(aroundValue+int(sliding_window_seconds/2)), label] = 1


                    # print(Labels[(aroundValue-int(sliding_window_seconds)):(aroundValue+int(sliding_window_seconds)), :])
                    # print(np.sum(Labels, axis=1))
                    # print(Labels[np.sum(Labels, axis=1)>1, :])

                    self.testing_Labels_onehot[key] = Labels

        self.nb_batch_testing = len(self.testing_GamesKeys)
        self.weights = [1, 1, 1, 1]
        print("nb_batch_testing", self.nb_batch_testing)

        




    def getTestingBatch(self, i):
        # print(self.testing_GamesKeys[i])
        testing_batch_features = self.testing_features     [self.testing_GamesKeys[i]]
        testing_batch_labels   = self.testing_Labels_onehot[self.testing_GamesKeys[i]]
        # print(testing_batch_features.shape)
        # print(testing_batch_labels.  shape)

        self.count_labels = np.sum(testing_batch_labels, axis=0)
        self._current_testing_batch_index = i  
        return testing_batch_features, testing_batch_labels, self.testing_GamesKeys[i]


    def getNextValidationBatch(self,):
        return getValidationBatch(self, self._current_validation_batch_index + 1)
       






