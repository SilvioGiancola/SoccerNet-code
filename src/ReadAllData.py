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
		
	
def ReadCommentaries(input_dir, championship, season, gameAwayTeam, gameHomeTeam):
	
	data = json.load(open(os.path.join(input_dir, championship, season, "commentaries.json")))
	for game in (data["Championship"]):
		if (data["Championship"][game]["gameAwayTeam"] == gameAwayTeam) and (data["Championship"][game]["gameHomeTeam"] == gameHomeTeam): 
			return data["Championship"][game]["comments"]

	return []



if __name__ == "__main__":
 
	description = 'Read the features and labels'
	p = argparse.ArgumentParser(description=description)
	p.add_argument('input_dir', type=str, default='data',
		help='Folder containing the championship folders.')

	args = p.parse_args()



	# For each Championship
	for championship in os.listdir(args.input_dir):
		if os.path.isdir(os.path.join(args.input_dir, championship)):

			# For each Season
			for season in os.listdir(os.path.join(args.input_dir, championship)):
				if os.path.isdir(os.path.join(args.input_dir, championship, season)):

					# For each Game
					for game in os.listdir(os.path.join(args.input_dir, championship, season)):
						if os.path.isdir(os.path.join(args.input_dir, championship, season, game)):


							# Path ofthe game
							print(game)
							game_folder = os.path.join(args.input_dir, championship, season, game)
							
							features = ReadFeatures(game_folder)            
							label = ReadLabels(game_folder)
							awayTeam = game.split("-")[-1][3:]
							homeTeam = game.split("-")[-2][3:-3]
							commentaries = ReadCommentaries(args.input_dir, championship, season, homeTeam, awayTeam)
