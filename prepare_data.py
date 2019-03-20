import numpy as np
import os
import math
import pandas as pd
import random
import pickle
import csv

training_data = []
folder = 'data'
base_elo = 1600
team_elos = {}
team_stats = {}
prediction_year = 2019

def init_data():
	for i in range(2003, prediction_year+1):
		team_elos[i] = {}
		team_stats[i] = {}

def calc_elo(win_team, lose_team, season):
	winner_rank = get_elo(season, win_team)
	loser_rank = get_elo(season, lose_team)

	rank_diff = winner_rank - loser_rank
	exp = rank_diff * -1/400
	odds = 1 / (1 + math.pow(10, exp))

	if winner_rank < 2100:
		k = 32
	elif winner_rank >= 2100 and winner_rank < 2400:
		k = 24
	else:
		k = 16

	new_winner_rank = round(winner_rank + (k*(1 - odds)))
	new_rank_diff = new_winner_rank - winner_rank
	new_loser_rank = loser_rank - new_rank_diff

	return new_winner_rank, new_loser_rank

def get_elo(season, team):
	try:
		return team_elos[season][team]
	except:
		try:
			team_elo = team_elos[season-1][team]
			team_elo = (.65*team_elo)+(.35*base_elo)
			team_elos[season][team] = team_elo
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

def update_stats(season, team, fields):
	if team not in team_stats[season]:
		team_stats[season][team] = {}

	for key, value in fields.items():
		if key not in team_stats[season][team]:
			team_stats[season][team][key] = []

		if len(team_stats[season][team][key]) >= 9:
			team_stats[season][team][key].pop()

		team_stats[season][team][key].append(value)

def get_location_win(loc):
	if loc == 'H':
		return 2
	elif loc == 'A':
		return 1
	else:
		return 0

def get_location_lose(loc):
	if loc == 'H':
		return 1
	elif loc == 'A':
		return 2
	else:
		return 0

def build_season_data(all_data):
	print("Building season data.")

	for index, row in all_data.iterrows():
		skip = 0

		team_1_elo = get_elo(row['Season'], row['WTeamID'])
		team_2_elo = get_elo(row['Season'], row['LTeamID'])

		if row['WLoc'] == 'H':
			team_1_elo += 100
		elif row['WLoc'] == 'A':
			team_2_elo += 100

		team_1_features = [team_1_elo]
		team_2_features = [team_2_elo]

		for field in stat_fields:
			team_1_stat = get_stats(row['Season'], row['WTeamID'], field)
			team_2_stat = get_stats(row['Season'], row['LTeamID'], field)

			if team_1_stat is not 0 and team_2_stat is not 0:
				team_1_features.append(team_1_stat)
				team_2_features.append(team_2_stat)
			else:
				skip = 1

		if skip == 0:
			if random.random() > 0.5:
				matchup_features = []
				for i in range(14):
					matchup_features.append(team_1_features[i] - team_2_features[i])
				training_data.append([matchup_features, 1])
			else:
				matchup_features = []
				for i in range(14):
					matchup_features.append(team_2_features[i] - team_1_features[i])
				training_data.append([matchup_features, 0])

		if row['WFTA'] != 0 and row['LFTA'] != 0:
			stat_1_fields = {
				'score': row['WScore'],
				'fgp': row['WFGM'] / row['WFGA'],
				'fga': row['WFGA'],
				'fga3': row['WFGA3'],
				'3pp': row['WFGM3'] / row['WFGA3'],
				'ftp': row['WFTM'] / row['WFTA'],
				'or': row['WOR'],
				'dr': row['WDR'],
				'ast': row['WAst'],
				'to': row['WTO'],
				'stl': row['WStl'],
				'blk': row['WBlk'],
				'pf': row['WPF']
			}
			stat_2_fields = {
				'score': row['LScore'],
				'fgp': row['LFGM'] / row['LFGA'],
				'fga': row['LFGA'],
				'fga3': row['LFGA3'],
				'3pp': row['LFGM3'] / row['LFGA3'],
				'ftp': row['LFTM'] / row['LFTA'],
				'or': row['LOR'],
				'dr': row['LDR'],
				'ast': row['LAst'],
				'to': row['LTO'],
				'stl': row['LStl'],
				'blk': row['LBlk'],
				'pf': row['LPF']
			}
			update_stats(row['Season'], row['WTeamID'], stat_1_fields)
			update_stats(row['Season'], row['LTeamID'], stat_2_fields)

		new_winner_rank, new_loser_rank = calc_elo(row['WTeamID'], row['LTeamID'], row['Season'])
		team_elos[row['Season']][row['WTeamID']] = new_winner_rank
		team_elos[row['Season']][row['LTeamID']] = new_loser_rank

stat_fields = ['score', 'fga', 'fgp', 'fga3', '3pp', 'ftp', 'or', 'dr', 'ast', 'to', 'stl', 'blk', 'pf']
init_data()
season_data = pd.read_csv(folder + '/RegularSeasonDetailedResults.csv')
tourney_data = pd.read_csv(folder + '/NCAATourneyDetailedResults.csv')
frames = [season_data, tourney_data]
all_data = pd.concat(frames)

build_season_data(all_data)
random.shuffle(training_data)

with open(folder + '/training_data.csv', 'w', newline='') as f:
	writer = csv.writer(f)
	writer.writerows(training_data)

train_X = []
train_y = []

for features, label in training_data:
	train_X.append(features)
	train_y.append(label)

train_X = np.array(train_X)

pickle_out = open("X.pickle", "wb")
pickle.dump(train_X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(train_y, pickle_out)
pickle_out.close()

pickle_out = open("team_elos.pickle", "wb")
pickle.dump(team_elos, pickle_out)
pickle_out.close()

pickle_out = open("team_stats.pickle", "wb")
pickle.dump(team_stats, pickle_out)
pickle_out.close()

