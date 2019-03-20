import pandas as pd
import math
import csv
import random
import numpy as np
from numpy import newaxis
import pickle
from bracketeer import build_bracket 

NAME = "march_madness_2019_model_1"
team_elos = pickle.load(open("team_elos.pickle", "rb"))
team_stats = pickle.load(open("team_stats.pickle", "rb"))
base_elo = 1600
team_seeds = {}
final_data = []
folder = 'data'
prediction_year = 2019
prediction_range =[2019]

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

	team_1_features.append(get_elo(season, team_1) + get_seed_bonus(team_1, team_2, season))
	for stat in stat_fields:
		team_1_features.append(get_stats(season, team_1, stat))

	team_2_features.append(get_elo(season, team_2) + get_seed_bonus(team_2, team_1, season))
	for stat in stat_fields:
		team_2_features.append(get_stats(season, team_2, stat))

	matchup_features = [a - b for a, b in zip(team_1_features, team_2_features)]

	return model.predict(prepare_data(matchup_features))

def get_seed(team, season):
	try:
		return int(team_seeds[season][team][1:])
	except:
		return 16

def get_elo_difference(team_1, team_2, season):
	try:
		if get_elo(season, team_1) > get_elo(season, team_2):
			return get_elo(season, team_1) - get_elo(season, team_2) + 50
		elif get_elo(season, team_2) > get_elo(season, team_1):
			return get_elo(season, team_2) - get_elo(season, team_1) + 50
	except:
		return 0


def get_seed_bonus(team_1, team_2, season):
	seed1 = get_seed(team_1, season)
	seed2 = get_seed(team_2, season)

	bonus = 100 * random.random()

	if seed1 == 1 and seed2 == 2:
		bonus = 0
	elif seed1 == 10 and seed2 == 7:
		if random.random() < 0.382:
			bonus += get_elo_difference(team_1, team_2, season)
	elif seed1 == 11 and seed2 == 6:
		if random.random() < 0.375:
			bonus += get_elo_difference(team_1, team_2, season)
	elif seed1 == 12 and seed2 == 5:
		if random.random() < 0.346:
			bonus += get_elo_difference(team_1, team_2, season)
	elif seed1 == 13 and seed2 == 4:
		if random.random() < 0.201:
			bonus += get_elo_difference(team_1, team_2, season)
	elif seed1 == 14 and seed2 == 3:
		if random.random() < 0.154:
			bonus+= get_elo_difference(team_1, team_2, season)
	elif seed1 == 15 and seed2 == 2:
		if random.random() < 0.059:
			bonus += get_elo_difference(team_1, team_2, season)

	if seed1 == 6 and seed2 == 3:
		if random.random() < 0.201:
			bonus += get_elo_difference(team_1, team_2, season)
	elif seed1 == 7 and seed2 == 2:
		if random.random() < 0.185:
			bonus += get_elo_difference(team_1, team_2, season)
	elif seed1 == 10 and seed2 == 2:
		if random.random() < 0.133:
			bonus += get_elo_difference(team_1, team_2, season)
	elif seed1 == 11 and seed2 == 3:
		if random.random() < 0.125:
			bonus += get_elo_difference(team_1, team_2, season)
	elif seed1 == 8 and seed2 == 1:
		if random.random() < 0.0962:
			bonus+= get_elo_difference(team_1, team_2, season)
	elif seed1 == 12 and seed2 == 4:
		if random.random() < 0.0888:
			bonus += get_elo_difference(team_1, team_2, season)
	elif seed1 == 9 and seed2 == 1:
		if random.random() < 0.0444:
			bonus += get_elo_difference(team_1, team_2, season)
	elif seed1 == 13 and seed2 == 5:
		if random.random() < 0.0222:
			bonus += get_elo_difference(team_1, team_2, season)
	elif seed1 == 14 and seed2 == 6:
		if random.random() < 0.0148:
			bonus += get_elo_difference(team_1, team_2, season)
	elif seed1 == 15 and seed2 == 7:
		if random.random() < 0.0074:
			bonus += get_elo_difference(team_1, team_2, season)

	if seed1 is not 16:
		if seed2 == 1 and seed1 is not 1:
			if random.random() < .05:
				bonus = get_elo_difference(team_1, team_2, season)

	return bonus

def prepare_data(features):
	features = np.array(features)
	features = features[newaxis, :]
	return features

def get_teams(team_list, year):
	for i in range(len(team_list)):
			for j in range(i + 1, len(team_list)):
				if team_list[i] < team_list[j]:
					prediction = predict_winner(team_list[i], team_list[j], model, year, stat_fields)
					label = str(year) + '_' + str(team_list[i]) + '_' + str(team_list[j])
					final_data.append([label, prediction[0]])

init_data()
stat_fields = ['score', 'fga', 'fgp', 'fga3', '3pp', 'ftp', 'or', 'dr', 'ast', 'to', 'stl', 'blk', 'pf']

model = pickle.load(open("models/modelv2.sav", "rb"))

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

prediction_path = 'predictions/submission_1.csv'

print(f"Writing {len(final_data)} results")
with open(prediction_path, 'w', newline='') as f:
	writer = csv.writer(f)
	writer.writerow(['ID', 'Pred'])
	writer.writerows(final_data)

m = build_bracket(
	outputPath='output.png',
	teamsPath='Data/Teams.csv',
	seedsPath='Data/NCAATourneySeeds.csv',
	submissionPath=prediction_path,
	slotsPath='Data/NCAATourneySlots.csv',
	year=prediction_year
	)