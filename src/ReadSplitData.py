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
	p.add_argument('split_set', type=str, default='listgame_Train_300.npy',
		help='Split set used for training (listGame_Train_300.npy), validation (listGame_Valid_100.npy) or testing (listGame_Test_100.npy).')

	args = p.parse_args()



	listGames = np.load(args.split_set)
	# Load all games        
	for gamePath in listGames:

		game = gamePath.split("/")[-1]
		season = gamePath.split("/")[-2]
		championship = gamePath.split("/")[-3]

		game_folder = os.path.join(args.input_dir, gamePath)

		if os.path.isdir(game_folder):

			# Path ofthe game
			print(championship, season, game)
				  
			features = ReadFeatures(game_folder)  
			label = ReadLabels(game_folder)
			awayTeam = game.split("-")[-1][3:]
			homeTeam = game.split("-")[-2][3:-3]
			commentaries = ReadCommentaries(args.input_dir, championship, season, homeTeam, awayTeam)

