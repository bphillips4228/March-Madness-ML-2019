import pandas as pd
import math
from sklearn import linear_model
from sklearn.model_selection import cross_val_score, ShuffleSplit
import csv
import random
import numpy as np
from numpy import newaxis
import pickle

NAME = "march_madness_2019_model_1"
team_elos = pickle.load(open("team_elos.pickle", "rb"))
team_stats = pickle.load(open("team_stats.pickle", "rb"))
team_seeds = {}
final_data = []
folder = 'data'
prediction_year = 2018
prediction_range =[2014, 2015, 2016, 2017, 2018]

def init_data():
	for i in range(prediction_range[0], prediction_year+1):
		team_seeds[i] = {}

def get_elo(season, team):
	try:
		return team_elos[season][team]
	except:
		try:
			team_elos[season][team] = team_elos[season-1][team]
			return team_elos[season][team]
		except:
			team_elos[season][team] = base_elo
			return team_elos[season][team]

def get_stats(season, team, field):
	try:
		l = team_stats[season][team][field]
		return sum(l) / float(len(l))
	except:
		return 0

def predict_winner(team_1, team_2, model, season, stat_fields):
	team_1_features = []
	team_2_features = []

	team_1_features.append(get_elo(season, team_1) + get_seed_bonus(get_seed(season, team_1)))
	for stat in stat_fields:
		team_1_features.append(get_stats(season, team_1, stat))

	team_2_features.append(get_elo(season, team_2) + get_seed_bonus(get_seed(season, team_2)))
	for stat in stat_fields:
		team_2_features.append(get_stats(season, team_2, stat))

	matchup_features = [a - b for a, b in zip(team_1_features, team_2_features)]

	return model.predict(prepare_data(matchup_features))

def get_seed(team, season):
	try:
		return team_seeds[season][team]
	except:
		return 'a16'

def get_seed_bonus(seed):
	seed = seed[1:]
	if seed == 1:
		bonus = 185
	elif seed == 2:
		bonus = 67
	elif seed == 3:
		bonus = 50
	elif seed == 4:
		bonus = 22
	elif seed == 5:
		bonus = 15
	elif seed == 6:
		bonus = 8
	elif seed == 7:
		bonus = 6
	elif seed == 8:
		bonus = 14
	elif seed == 9 or seed == 10:
		bonus = 1
	elif seed == 11:
		bonus = 4
	else:
		bonus = 0

	return bonus 

def prepare_data(features):
	features = np.array(features)
	features = features[newaxis, :]
	return features

def build_team_dict():
	team_ids = pd.read_csv(folder + '/Teams.csv')
	team_id_map = {}
	for index, row in team_ids.iterrows():
		team_id_map[row['TeamID']] = row['TeamName']
	return team_id_map

def get_teams(team_list, year):
	for i in range(len(team_list)):
			for j in range(i + 1, len(team_list)):
				if team_list[i] < team_list[j]:
					prediction = predict_winner(team_list[i], team_list[j], model, year, stat_fields)
					label = str(year) + '_' + str(team_list[i]) + '_' + str(team_list[j])
					final_data.append([label, prediction[0]])

if __name__ == "__main__":
	init_data()
	stat_fields = ['score', 'fga', 'fgp', 'fga3', '3pp', 'ftp', 'or', 'dr', 'ast', 'to', 'stl', 'blk', 'pf']

	model = pickle.load(open("model.sav", "rb"))

	print("Getting teams")
	print("Predicting matchups")

	seeds = pd.read_csv(folder + '/NCAATourneySeeds.csv')
	tourney_teams = []
	for year in prediction_range:
		for index, row in seeds.iterrows():
			if row['Season'] == year:
				team_seeds[year][row['TeamID']] = row['Seed']
				tourney_teams.append(row['TeamID'])
		tourney_teams.sort()

		get_teams(tourney_teams, year)
		tourney_teams.clear()

	print(f"Writing {len(final_data)} results")
	with open(folder + f'/submission_1.csv', 'w', newline='') as f:
		writer = csv.writer(f)
		writer.writerow(['ID', 'Pred'])
		writer.writerows(final_data)

	print("Outputting readable results")
	team_id_map = build_team_dict()
	readable = []
	less_readable = []
	for pred in final_data:
		parts = pred[0].split('_')
		less_readable.append(
			[team_id_map[int(parts[1])], team_id_map[int(parts[2])], pred[1]])
		if pred[1] > 0.5:
			winning = int(parts[1])
			losing = int(parts[2])
			probability = pred[1]
		else:
			winning = int(parts[2])
			losing = int(parts[1])
			probability = 1 - pred[1]

		readable.append(
			[
				f"{team_id_map[winning]} beats {team_id_map[losing]}: {probability}"
			]
		)

	with open(folder + '/readable-predictions.csv', 'w') as f:
		writer = csv.writer(f)
		writer.writerows(readable)
	with open(folder + '/less-readable-predictions.csv', 'w') as f:
		writer = csv.writer(f)
		writer.writerows(less_readable)