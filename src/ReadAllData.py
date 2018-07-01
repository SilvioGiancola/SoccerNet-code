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
    
    return data
    
def ReadLabels(game_folder):
    
    return json.load(open(os.path.join(game_folder,"Labels.json")))
        



if __name__ == "__main__":
 
	description = 'Read the features and labels'
	p = argparse.ArgumentParser(description=description)
	p.add_argument('input_dir', type=str, default='data',
		help='Folder containing the championship folders.')

	args = p.parse_args()



	# For each Championship
	for championship in os.listdir(args.input_dir):
	    
	    # For each Season
	    for season in os.listdir(os.path.join(args.input_dir, championship)):
	        
	        # For each Game
	        for game in os.listdir(os.path.join(args.input_dir, championship, season)):
	            
	            # Path ofthe game
	            print(game)
	            game_folder = os.path.join(args.input_dir, championship, season, game)
	            
	            features = ReadFeatures(game_folder)            
	            label = ReadLabels(game_folder)
