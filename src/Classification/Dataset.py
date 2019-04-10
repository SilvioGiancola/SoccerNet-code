

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



    def loadTrainingDataset(self, path_data, featureName, PCA=True, imbalance="", batch_size=60, window_size_sec=60):

        self.imbalance = imbalance
        self.size_batch = batch_size
        self.PCA = PCA
        self.featureName = featureName
        self.window_size_sec = window_size_sec

        FeaturePerSecond = 2 # number of features per Time_Second
        number_frames_in_window = window_size_sec * FeaturePerSecond # 120 by default
        self.number_frames_in_window = number_frames_in_window


        path_data_dirname, path_data_basename = os.path.split(path_data)
        # path_data_basename = os.path.basename(path_data)
        listGames = np.load(path_data)

        i = 0
        self.training_Labels_onehot = {}
        self.training_features = {}
        self.training_GamesKeys=[]
        self.training_indices_back = []
        self.training_indices_card = []
        self.training_indices_subs = []
        self.training_indices_goal = []
        print("Reading listGames")
        for gamePath in tqdm(listGames):
            gamePath = os.path.join(path_data_dirname, gamePath)
            i += 1
            for featureFileName in os.listdir(gamePath):
                if (featureName in featureFileName and ( (PCA and "PCA" in featureFileName) or (not PCA and "PCA" not in featureFileName) ) ):
                    featureFullPath = os.path.join(gamePath, featureFileName)
                    if "1_" in featureFileName: key = os.path.join(gamePath,"Half_1")
                    elif "2_" in featureFileName: key = os.path.join(gamePath,"Half_2")
                    self.training_GamesKeys.append(key)
                    self.training_features_cont = np.load(featureFullPath)


                    labelFullPath = os.path.join(gamePath, "Labels.json")
                    with open(labelFullPath) as labelFile :
                        jsonLabel = json.loads(labelFile.read())

                    # count for data augmentation
                    cnt_data_augmentation = 0 #len(self.training_features[key])
                    for event in jsonLabel["annotations"]:
                        Time_Half = int(event["gameTime"][0])
                        Time_Minute = int(event["gameTime"][-5:-3])
                        Time_Second = int(event["gameTime"][-2:])

                        if ("card" in event["label"]): label = 1
                        elif ("subs" in event["label"]): label = 2
                        elif ("soccer" in event["label"]): label = 3
                        # else: print("err in event Label", event["label"])

                        if (("1_" in featureFileName and Time_Half == 1) or ("2_" in featureFileName and Time_Half == 2)):                           
                            if ("DataAugmentation" in self.imbalance):
                                # ind = (Time_Minute*60 + Time_Second) * FeaturePerSecond
                                # for t in range(ind-40, ind+40, 4):                                
                                t = (Time_Minute*60 + Time_Second)
                                t_ini = t - int(window_size_sec*0.667/2.0) 
                                f_ini = t_ini * FeaturePerSecond
                                t_end = t + int(window_size_sec*0.667/2.0)
                                f_end = t_end * FeaturePerSecond
                                # print("t", t, "from", t_ini, "to", t_end)
                                for f in range(f_ini, f_end, 1):
                                    # print(Time_Minute, Time_Second, self.training_features_cont.shape)
                                    if (f+window_size_sec < len(self.training_features_cont)) and (f-window_size_sec  >0):
                                        cnt_data_augmentation += 1

                    # print (cnt_data_augmentation)       
                    l = self.training_features_cont.shape[0] - self.training_features_cont.shape[0]%number_frames_in_window
                    self.training_features[key] = np.zeros((cnt_data_augmentation + int(l/number_frames_in_window), number_frames_in_window, 512))
                    cnt_data_augmentation = 0
                    for minframe in np.reshape(self.training_features_cont[0:l,:], (-1, number_frames_in_window, 512)):
                        self.training_features[key][cnt_data_augmentation] = minframe
                        cnt_data_augmentation += 1


                    Labels = np.zeros((self.training_features[key].shape[0],4), dtype=int)
                    Labels[:,0] = 1
                    # Labels = np.zeros(self.training_features[key].shape[0], dtype=int)

                    # print(key)
                    for event in jsonLabel["annotations"]:
                        Time_Half = int(event["gameTime"][0])
                        Time_Minute = int(event["gameTime"][-5:-3])
                        Time_Second = int(event["gameTime"][-2:])

                        if ("card" in event["label"]): label = 1
                        elif ("subs" in event["label"]): label = 2
                        elif ("soccer" in event["label"]): label = 3
                        # else: print("err in event Label", event["label"])

                        if (("1_" in featureFileName and Time_Half == 1) or ("2_" in featureFileName and Time_Half == 2)):
                            index = min(Time_Minute,Labels.shape[0]-1)
                            Labels[index,0] = 0 # remove backgroun annotation 
                            Labels[index,label] = 1 # Add annotation
                            
                            if ("DataAugmentation" in self.imbalance):
                                t = (Time_Minute*60 + Time_Second)
                                t_ini = t - int(window_size_sec/2)+10 
                                f_ini = t_ini * FeaturePerSecond
                                t_end = t + int(window_size_sec/2)-10
                                f_end = t_end * FeaturePerSecond
                                for f in range(f_ini, f_end, 1):
                                    # print(Time_Minute, Time_Second, self.training_features_cont.shape, cnt_data_augmentation)
                                    if (f+window_size_sec < len(self.training_features_cont)) and (f-window_size_sec  >0):
                                        self.training_features[key][cnt_data_augmentation] = np.reshape(self.training_features_cont[f-window_size_sec:f+window_size_sec ,:], (1, number_frames_in_window, 512))

                                        Labels[cnt_data_augmentation,0] = 0 # remove backgroun annotation 
                                        Labels[cnt_data_augmentation,label] = 1 # Add annotation    #np.reshape(label,(1))
                                        cnt_data_augmentation += 1


                    
                    # print(cnt_data_augmentation)
                    # print(self.training_features[key].shape)
                    # print(Labels.shape)
                              

                    # print(Labels[np.sum(Labels, axis=1)>1, :])
                    self.training_Labels_onehot[key] = Labels #np.eye(4)[Labels]
                    # if "1_" in featureFileName:
                    for frame in range(0,len(self.training_Labels_onehot[key])):
                        if  ((Labels[frame,0] == 1)): self.training_indices_back.append([key, frame, 0])
                        if  ((Labels[frame,1] == 1)): self.training_indices_card.append([key, frame, 0])
                        if  ((Labels[frame,2] == 1)): self.training_indices_subs.append([key, frame, 0])
                        if  ((Labels[frame,3] == 1)): self.training_indices_goal.append([key, frame, 0])


        self.count_labels = [len(self.training_indices_back), len(self.training_indices_card), len(self.training_indices_subs), len(self.training_indices_goal)]
        print("count:", self.count_labels)
        self.ratio_labels = self.count_labels / np.sum(self.count_labels)

        self.weights = [1, 1, 1, 1]
        if ("Wratio1"       in self.imbalance): self.weights =           self.ratio_labels
        if ("Wratioinv1"    in self.imbalance): self.weights = np.power( self.ratio_labels, -1)
        if ("Wratio2"       in self.imbalance): self.weights = np.power( self.ratio_labels,  2)
        if ("Wratioinv2"    in self.imbalance): self.weights = np.power( self.ratio_labels, -2)
        if ("Wratiosoftmax" in self.imbalance): self.weights =  np.exp(  self.ratio_labels  ) 
        if ("Wratiodiff"    in self.imbalance): self.weights =     1  -  self.ratio_labels   

        if ("Wratio"        in self.imbalance): self.weights = self.weights / np.sum(self.weights)
        print("weights:", self.weights)

        if ("DataAugmentation" in self.imbalance):
            self.size_batch = 20
  

        # print(self.training_features[key].shape)
        # print(self.training_features[key].dtype)
        # print("nb_batch_training", self.nb_batch_training)
       

    def loadValidationDataset(self, path_data, featureName, PCA=True, window_size_sec=60):
        self.window_size_sec = window_size_sec

        FeaturePerSecond = 2 # number of features per Time_Second
        number_frames_in_window = window_size_sec * FeaturePerSecond # 120 by default
        self.number_frames_in_window = number_frames_in_window


        print("Loading Validation Data:", path_data)
        path_data_dirname, path_data_basename = os.path.split(path_data)
        listGames = np.load(path_data)

        i = 0
        self.validation_Labels_onehot = {}
        self.validation_features = {}
        self.validation_GamesKeys = []

        for gamePath in (listGames):
            gamePath = os.path.join(path_data_dirname, gamePath)
            i += 1
            for featureFileName in os.listdir(gamePath):
                if (featureName in featureFileName and ( (PCA and "PCA" in featureFileName) or (not PCA and "PCA" not in featureFileName) ) ):
                    featureFullPath = os.path.join(gamePath, featureFileName)
                    if "1_" in featureFileName: key = os.path.join(gamePath,"Half_1")
                    elif "2_" in featureFileName: key = os.path.join(gamePath,"Half_2")
                    self.validation_GamesKeys.append(key)
                    self.validation_features[key] = np.load(featureFullPath)
                    l = self.validation_features[key].shape[0] - self.validation_features[key].shape[0]%number_frames_in_window
                    self.validation_features[key] = np.reshape(self.validation_features[key][0:l,:], (-1, number_frames_in_window, 512))

                    FeaturePerTimeSecond = 2 # number of features per TimeSecond

                    labelFullPath = os.path.join(gamePath, "Labels.json")
                    with open(labelFullPath) as labelFile :
                        jsonLabel = json.loads(labelFile.read())

                    Labels = np.zeros((self.validation_features[key].shape[0],4), dtype=int)
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
                            index = min((Time_Minute*60+Time_Second)//window_size_sec, Labels.shape[0]-1)
                            Labels[index,0] = 0 # remove background annotation 
                            Labels[index,label] = 1 # Add annotation
                            # Labels[index] = label

                        # elif ("2_" in featureFileName and Half == 2):
                        #     index = min(Time_Minute,Labels.shape[0]-1)
                        #     Labels[index] = label

                    self.validation_Labels_onehot[key] = Labels # = np.eye(4)[Labels]

        self.nb_batch_validation = len(self.validation_GamesKeys)
        print("nb_batch_validation", self.nb_batch_validation)


    def loadTestingDataset(self, path_data, featureName, PCA=True, window_size_sec=60):

        self.window_size_sec = window_size_sec

        FeaturePerSecond = 2 # number of features per Time_Second
        number_frames_in_window = window_size_sec * FeaturePerSecond # 120 by default
        self.number_frames_in_window = number_frames_in_window

        print("Loading Testing Data:", path_data)
        path_data_dirname, path_data_basename = os.path.split(path_data)
        listGames = np.load(path_data)

        i = 0
        self.testing_Labels_onehot = {}
        self.testing_features = {}
        self.testing_GamesKeys = []

        for gamePath in (listGames):
            gamePath = os.path.join(path_data_dirname, gamePath)
            i += 1
            for featureFileName in os.listdir(gamePath):
                if (featureName in featureFileName and ( (PCA and "PCA" in featureFileName) or (not PCA and "PCA" not in featureFileName) ) ):
                    featureFullPath = os.path.join(gamePath, featureFileName)
                    if "1_" in featureFileName: key = os.path.join(gamePath,"Half_1")
                    elif "2_" in featureFileName: key = os.path.join(gamePath,"Half_2")
                    self.testing_GamesKeys.append(key)
                    self.testing_features[key] = np.load(featureFullPath)
                    l = self.testing_features[key].shape[0] - self.testing_features[key].shape[0]%number_frames_in_window
                    self.testing_features[key] = np.reshape(self.testing_features[key][0:l,:], (-1, number_frames_in_window, 512))

                    FeaturePerTimeSecond = 2 # number of features per TimeSecond

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
                            index = min((Time_Minute*60+Time_Second)//window_size_sec, Labels.shape[0]-1)
                            Labels[index,0] = 0 # remove background annotation 
                            Labels[index,label] = 1 # Add annotation

                        # if ("1_" in featureFileName and Half == 1):
                        #     index = min(Time_Minute,Labels.shape[0]-1)
                        #     Labels[index] = label

                        # elif ("2_" in featureFileName and Half == 2):
                        #     index = min(Time_Minute,Labels.shape[0]-1)
                        #     Labels[index] = label

                    self.testing_Labels_onehot[key] = Labels #= np.eye(4)[Labels]

        self.nb_batch_testing = len(self.testing_GamesKeys)
        self.weights = [1, 1, 1, 1]
        print("nb_batch_testing", self.nb_batch_testing)

        



    def prepareNewEpoch(self):
        

        if ("HNM" in self.imbalance):

            self.nb_epoch_per_batch = 20
            self.nb_batch_training = 1
            # self.nb_batch_training = 30
            # self.nb_batch_training = int( np.ceil(nb_halves/self.size_batch) ) # batchs (floor)
            # random.shuffle(self.training_GamesKeys)
            self._current_training_batch_index = -1
            self._current_validation_batch_index = -1

            self.nb_label= min([len(self.training_indices_back), len(self.training_indices_card), len(self.training_indices_subs), len(self.training_indices_goal)])
            # print(self.nb_label)
            self.size_batch = self.nb_label*4

            if ("rand" in self.imbalance):
                self.randomSample(self.nb_label)

            if ("small" in self.imbalance):
                self.smallestSample(self.nb_label)


        else:
            


            random.shuffle(self.training_GamesKeys)

            nb_halves = len(self.training_GamesKeys)
            print("size_batch:", self.size_batch)
            print("nb_halves:", nb_halves)
            # self.size_batch = 60 #halves
            self.nb_batch_training = int( np.ceil(nb_halves/self.size_batch) ) # batchs (floor)
            # self.nb_batch_training = 5
            # random.shuffle(self.training_GamesKeys)

        self._current_training_batch_index = -1
        self._current_validation_batch_index = -1

        return



    def smallestSample(self, nb_label):

        self.train_sample_features = []
        self.train_sample_labels   = []
        self.train_sample_indices = []
        self.i_sample_goal = []
        self.i_sample_subs = []
        self.i_sample_card = []
        self.i_sample_back = []
        start_time = time.time()

        for i in heapq.nsmallest(nb_label, self.training_indices_goal, key=itemgetter(2)) : self.train_sample_indices.append(i)
        for i in heapq.nsmallest(nb_label, self.training_indices_subs, key=itemgetter(2)) : self.train_sample_indices.append(i)
        for i in heapq.nsmallest(nb_label, self.training_indices_card, key=itemgetter(2)) : self.train_sample_indices.append(i)
        for i in heapq.nsmallest(nb_label, self.training_indices_back, key=itemgetter(2)) : self.train_sample_indices.append(i)



        for index in self.train_sample_indices:
            self.train_sample_features.append( self.training_features[index[0]][index[1]])
            self.train_sample_labels.append(  self.training_Labels_onehot[index[0]][index[1]])

        self.train_sample_features = np.array(self.train_sample_features)
        self.train_sample_labels = np.array(self.train_sample_labels)
        print("Elab Time for Sampling:", time.time() - start_time, "s")



    def randomSample(self, nb_label):

        self.train_sample_features = []
        self.train_sample_labels   = []
        self.train_sample_indices = []
        self.i_sample_goal = []
        self.i_sample_subs = []
        self.i_sample_card = []
        self.i_sample_back = []
        start_time = time.time()
 

        for i in random.sample(self.training_indices_goal, nb_label) : self.train_sample_indices.append(i)
        for i in random.sample(self.training_indices_subs, nb_label) : self.train_sample_indices.append(i)
        for i in random.sample(self.training_indices_card, nb_label) : self.train_sample_indices.append(i)
        for i in random.sample(self.training_indices_back, nb_label) : self.train_sample_indices.append(i)



        for index in self.train_sample_indices:
            self.train_sample_features.append( self.training_features[index[0]][index[1]])
            self.train_sample_labels.append(  self.training_Labels_onehot[index[0]][index[1]])

        self.train_sample_features = np.array(self.train_sample_features)
        self.train_sample_labels = np.array(self.train_sample_labels)
        print("Elab Time for Sampling:", time.time() - start_time, "s")





    def getTrainingBatch(self, i):  
        self._current_training_batch_index = i  

        if ("HNM" in self.imbalance):
            return self.train_sample_features, self.train_sample_labels, self.train_sample_indices

            # train_batch_features = self.train_sample_features
            # train_batch_labels = self.train_sample_labels
            # train_batch_indices = self.train_sample_indices

        else:
            init_games = i*self.size_batch
            end_games = min((i+1)*self.size_batch, len(self.training_GamesKeys))
            return self.getGamesBatch(init_games, end_games)
            # init_games = i*self.size_batch
            # end_games = min((i+1)*self.size_batch, len(self.training_GamesKeys))
            # train_batch_features = self.training_features[self.training_GamesKeys[init_games]]
            # train_batch_labels   = self.training_Labels_onehot[self.training_GamesKeys[init_games]]
            # train_batch_indices  = []
            # # print(self.train_batch_features.shape)
            # print("from", init_games, "to", end_games)
            # for gameKey in self.training_GamesKeys[init_games+1:end_games]:
            #     # print(gameKey)
            #     train_batch_features = np.concatenate((train_batch_features, self.training_features[gameKey]))
            #     train_batch_labels   = np.concatenate((train_batch_labels,   self.training_Labels_onehot[gameKey]))
            # # train_batch_features = np.array({gameKey: self.training_features[gameKey]      for gameKey in self.training_GamesKeys[init_games:end_games]})
            # # train_batch_labels   = np.array({gameKey: self.training_Labels_onehot[gameKey] for gameKey in self.training_GamesKeys[init_games:end_games]})

            #     # print(train_batch_features.shape)
            #     # train_batch_indices .append(self.training_indices [gameKey])
            # # train_batch_features  = np.stack(self.training_features [self.training_GamesKeys[i_games:i_games+self.size_batch]])
            # # train_batch_labels   = np.stack(self.training_Labels_onehot  [self.training_GamesKeys[i_games:i_games+self.size_batch]])
            # # self.count_labels = sum(train_batch_labels)
            # # self.count_labels = sum(self.count_labels)/self.count_labels

        return train_batch_features, train_batch_labels, train_batch_indices


    def getGamesBatch(self, init_games, end_games):
        train_batch_features = self.training_features[self.training_GamesKeys[init_games]]
        train_batch_labels   = self.training_Labels_onehot[self.training_GamesKeys[init_games]]
        train_batch_indices  = []
        # print(self.train_batch_features.shape)
        print("from", init_games, "to", end_games)
        for gameKey in self.training_GamesKeys[init_games+1:end_games]:
            # print(gameKey)
            train_batch_features = np.concatenate((train_batch_features, self.training_features[gameKey]))
            train_batch_labels   = np.concatenate((train_batch_labels,   self.training_Labels_onehot[gameKey]))
        return train_batch_features, train_batch_labels, train_batch_indices
    


    def updateResults(self, predictions, labels, indexes):
        print("indexes to update:",len(indexes))
        start_time = time.time()
        training_indices_list = [self.training_indices_back, self.training_indices_card, self.training_indices_subs, self.training_indices_goal]

        # print("maxgoal:", max(l[2] for l in self.training_indices_goal ))

        for i in range(len(labels)):
            prediction = predictions[i][:]
            label = labels[i][:]


            if (label[0] == 1): self.train_sample_indices[i][2] = prediction[0]
            if (label[1] == 1): self.train_sample_indices[i][2] = prediction[1]
            if (label[2] == 1): self.train_sample_indices[i][2] = prediction[2]
            if (label[3] == 1): self.train_sample_indices[i][2] = prediction[3]
 
        # print("maxgoal:", max(l[2] for l in self.training_indices_goal ))

        print("Elab Time for updating resutls:", time.time() - start_time, "s")



    def getNextTrainingBatch(self):
        return getTrainingBatch(self, self._current_training_batch_index + 1)


    def getValidationBatch(self, i):
        self.valid_batch_features = self.validation_features     [self.validation_GamesKeys[i]]
        self.valid_batch_labels   = self.validation_Labels_onehot[self.validation_GamesKeys[i]]


        # n_smallest = heapq.nsmallest(nb_label, self.training_indices_goal, key=itemgetter(2))       
        # for i in n_smallest : self.train_sample_indices.append(i)
        # n_smallest = heapq.nsmallest(nb_label, self.training_indices_subs, key=itemgetter(2))
        # for i in n_smallest : self.train_sample_indices.append(i)
        # n_smallest = heapq.nsmallest(nb_label, self.training_indices_card, key=itemgetter(2))
        # for i in n_smallest : self.train_sample_indices.append(i)
        # n_smallest = heapq.nsmallest(nb_label, self.training_indices_back, key=itemgetter(2))
        # for i in n_smallest : self.train_sample_indices.append(i)


        self.count_labels = np.sum(self.valid_batch_labels, axis=0)
        self._current_validation_batch_index = i  
        return self.valid_batch_features, self.valid_batch_labels

    def getTestingBatch(self, i):
        testing_batch_features = self.testing_features     [self.testing_GamesKeys[i]]
        testing_batch_labels   = self.testing_Labels_onehot[self.testing_GamesKeys[i]]

        self.count_labels = np.sum(testing_batch_labels, axis=0)
        self._current_testing_batch_index = i  
        return testing_batch_features, testing_batch_labels


    def getNextValidationBatch(self,):
        return getValidationBatch(self, self._current_validation_batch_index + 1)
       






