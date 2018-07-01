import os
import numpy as np
import json
from pprint import pprint
import argparse

	
def ReadFeatures(game_folder):
	data = {}       
			
	data["C3D_Half1"] = np.load(os.path.join(game_folder,"1_C3D.npy"))
	data["C3D_Half2"] = np.load(os.path.join(game_folder,"2_C3D.npy"))    
	data["I3D_Half1"] = np.load(os.path.join(game_folder,"1_I3D.npy"))
	data["I3D_Half2"] = np.load(os.path.join(game_folder,"2_I3D.npy"))
	data["ResNET_Half1"] = np.load(os.path.join(game_folder,"1_ResNET.npy"))
	data["ResNET_Half2"] = np.load(os.path.join(game_folder,"2_ResNET.npy"))


	data["C3D_Half1_PCA512"] = np.load(os.path.join(game_folder,"1_C3D_PCA512.npy"))
	data["C3D_Half2_PCA512"] = np.load(os.path.join(game_folder,"2_C3D_PCA512.npy"))    
	data["I3D_Half1_PCA512"] = np.load(os.path.join(game_folder,"1_I3D_PCA512.npy"))
	data["I3D_Half2_PCA512"] = np.load(os.path.join(game_folder,"2_I3D_PCA512.npy"))
	data["ResNET_Half1_PCA512"] = np.load(os.path.join(game_folder,"1_ResNET_PCA512.npy"))
	data["ResNET_Half2_PCA512"] = np.load(os.path.join(game_folder,"2_ResNET_PCA512.npy"))
	
	return data
	
def ReadLabels(game_folder):
	
	return json.load(open(os.path.join(game_folder,"Labels.json")))
		




if __name__ == "__main__": 

	description = 'Read the features and labels'
	p = argparse.ArgumentParser(description=description)
	p.add_argument('input_dir', type=str, default='data',
		help='Folder containing the championship folders.')

	args = p.parse_args()



	# example of game folder
	print(args.input_dir)


	features = ReadFeatures(args.input_dir) 
	print("FEATURES")
	print("The features are stored in .npy files")
	print("The number of features may differ because of additional time and/or the temporal context of the type of features")

	print("ResNET:")
	print(" - 2 feat. per sec.")
	print(" - temporal context = 1 frame")
	print(" - 45 min = 5400 features")
	print(" - 1st Half (shape) : ", features["ResNET_Half1"].shape)
	print(" - 2nd Half (shape) : ", features["ResNET_Half2"].shape) 
	print(" - PCA512")		
	print(" - 1st Half (shape) : ", features["ResNET_Half1_PCA512"].shape)
	print(" - 2nd Half (shape) : ", features["ResNET_Half2_PCA512"].shape) 	



	print("C3D:")
	print(" - 2 feat. per sec.")
	print(" - temporal context = 16 frames")
	print(" - 45 min = 5399 features")
	print(" - 1st Half (shape) : ", features["C3D_Half1"].shape)
	print(" - 2nd Half (shape) : ", features["C3D_Half2"].shape) 
	print(" - PCA512")		
	print(" - 1st Half (shape) : ", features["C3D_Half1_PCA512"].shape)
	print(" - 2nd Half (shape) : ", features["C3D_Half2_PCA512"].shape) 	



	print("I3D:")
	print(" - 2 feat. per sec.")
	print(" - temporal context = 64 frames")
	print(" - 45 min = 5395 features")
	print(" - 1st Half (shape) : ", features["I3D_Half1"].shape)
	print(" - 2nd Half (shape) : ", features["I3D_Half2"].shape) 
	print(" - PCA512")	
	print(" - 1st Half (shape) : ", features["I3D_Half1_PCA512"].shape)
	print(" - 2nd /Half (shape) : ", features["I3D_Half2_PCA512"].shape) 	




	print("LABELS")
	print("The labels are stored in .json files")
	labels = ReadLabels(args.input_dir)
	pprint(labels, )


