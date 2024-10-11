import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv(r"C:\Users\shany\OneDrive\Desktop\NBA_2023_Shots.csv")

# Display the first few rows of the dataset
print(df.head())

# Step 1: Data Cleaning
df = df.dropna(subset=['SHOT_MADE', 'LOC_X', 'LOC_Y', 'PLAYER_NAME', 'TEAM_NAME', 'QUARTER', 'MINS_LEFT', 'SECS_LEFT'])

# Step 2: Feature Engineering
df['time_remaining'] = df['MINS_LEFT'] * 60 + df['SECS_LEFT']
df['SHOT_DISTANCE'] = np.sqrt(df['LOC_X']**2 + df['LOC_Y']**2)
df['SHOT_ZONE'] = pd.cut(df['SHOT_DISTANCE'], bins=[0, 8, 23, 40], labels=['Paint', 'Mid-Range', '3PT'])

# Step 3: Define Pressure Situations
# Assuming clutch time is 4th quarter or later (or less than 5 minutes remaining in the game)
df['is_clutch'] = np.where((df['QUARTER'] >= 4) & (df['time_remaining'] <= 300), 1, 0)

# Filter shots taken under pressure (clutch situations)
clutch_shots = df[df['is_clutch'] == 1]

# Step 4: Calculate Shooting Efficiency in Pressure Situations
# Calculate shooting efficiency by player during clutch situations
clutch_efficiency_by_player = clutch_shots.groupby('PLAYER_NAME')['SHOT_MADE'].mean().reset_index()
clutch_efficiency_by_player.columns = ['PLAYER_NAME', 'clutch_shot_percentage']

# Include team name
clutch_efficiency_by_player['TEAM_NAME'] = clutch_shots.groupby('PLAYER_NAME')['TEAM_NAME'].first().values

# Step 5: Identify the Best Player on Each Team During Pressure Situations
best_clutch_players = clutch_efficiency_by_player.loc[clutch_efficiency_by_player.groupby('TEAM_NAME')['clutch_shot_percentage'].idxmax()]

# Step 6: Visualize the Best Players Under Pressure
plt.figure(figsize=(12, 8))
sns.barplot(x='PLAYER_NAME', y='clutch_shot_percentage', data=best_clutch_players)
plt.xticks(rotation=90)
plt.title('Best Players Under Pressure (Clutch Situations) by Team')
plt.xlabel('Player Name')
plt.ylabel('Clutch Shooting Percentage')
plt.show()

print("Data analysis complete.")
