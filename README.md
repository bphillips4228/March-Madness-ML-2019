# March-Madness-ML-2019
Using machine learning to predict the outcome of 2019 March Madness 

Model - Gradient Boosting Regression
(Estimators: 10000, Max Depth: 8, Learning Rate: 0.001, Validation Split: 0.2)

Features - Team's Season Averages for: Elo Score, Field goal percentage, Field goal attempts, 3pt attempts, 3pt average, 
Free throw average, Offensive rebounds, Defensive rebounds, Assists, Turnovers, Steals, Blocks, Fouls

Note - Factored in historical chances of upsets depending on seed match ups, e.i. 11 seeds upsets 3 seeds 12.5% of the time, rarely do all 1 seeds make it to the Final Four, etc.

Data-set used:
https://www.kaggle.com/c/mens-machine-learning-competition-2019/data
