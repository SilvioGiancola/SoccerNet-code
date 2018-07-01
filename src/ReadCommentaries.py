import os
import numpy as np
import json
from pprint import pprint
import argparse

	
def ReadCommentaries(input_dir, league, season, gameAwayTeam, gameHomeTeam):
	
	data = json.load(open(os.path.join(input_dir, league, season, "commentaries.json")))
	for game in (data["Championship"]):
		if (data["Championship"][game]["gameAwayTeam"] == gameAwayTeam) and (data["Championship"][game]["gameHomeTeam"] == gameHomeTeam): 
			return data["Championship"][game]["comments"]

	return []



if __name__ == "__main__":
 
	description = 'Read the features and labels'
	p = argparse.ArgumentParser(description=description)
	p.add_argument('input_dir', type=str, default='data',
		help='Folder containing the championship folders.')
	p.add_argument('league', type=str, default='data',
		help='League of interest.')
	p.add_argument('season', type=str, default='data',
		help='Season of interest.')
	p.add_argument('homeTeam', type=str, default='data',
		help='homeTeam.')
	p.add_argument('awayTeam', type=str, default='data',
		help='awayTeam.')

	args = p.parse_args()



	
	commentaries = ReadCommentaries(args.input_dir, 
		args.league, args.season, 
		args.homeTeam, args.awayTeam)   

	for commentary in commentaries:
		print (commentary["gameTime"]+" - ["+commentary["label"]+"] - "+commentary["description"])
