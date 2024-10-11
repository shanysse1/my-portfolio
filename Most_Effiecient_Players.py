import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv(r"C:\Users\shany\OneDrive\Desktop\NBA_2023_Shots.csv")

# Display the first few rows of the dataset
print(df.head())

# Step 1: Data Cleaning
df = df.dropna(subset=['SHOT_MADE', 'LOC_X', 'LOC_Y', 'PLAYER_NAME', 'QUARTER', 'MINS_LEFT', 'SECS_LEFT'])

# Step 2: Feature Engineering
df['time_remaining'] = df['MINS_LEFT'] * 60 + df['SECS_LEFT']
df['SHOT_DISTANCE'] = np.sqrt(df['LOC_X']**2 + df['LOC_Y']**2)
df['SHOT_ZONE'] = pd.cut(df['SHOT_DISTANCE'], bins=[0, 8, 23, 40], labels=['Paint', 'Mid-Range', '3PT'])

# Step 3: Calculating Shot Efficiency
# Overall shooting percentage per player
player_shot_efficiency = df.groupby('PLAYER_NAME')['SHOT_MADE'].mean().reset_index()
player_shot_efficiency.columns = ['PLAYER_NAME', 'shot_percentage']

# Assuming your DataFrame has a 'TEAM_NAME' column
player_shot_efficiency['TEAM_NAME'] = df.groupby('PLAYER_NAME')['TEAM_NAME'].first().values

# Find the most efficient player on each team
most_efficient_players = player_shot_efficiency.loc[player_shot_efficiency.groupby('TEAM_NAME')['shot_percentage'].idxmax()]

# Step 4: Visualizing the Most Efficient Players
plt.figure(figsize=(12, 8))
sns.barplot(x='PLAYER_NAME', y='shot_percentage', data=most_efficient_players)
plt.xticks(rotation=90)
plt.title('Most Efficient Players by Team')
plt.xlabel('Player Name')
plt.ylabel('Shooting Percentage')
plt.show()

print("Data wrangling and feature engineering complete.")
