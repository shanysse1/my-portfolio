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
# Calculate shooting percentage by shot zone
zone_efficiency = df.groupby('SHOT_ZONE')['SHOT_MADE'].mean().reset_index()
print(zone_efficiency)

# Step 4: Visualizing High-Percentage Shot Zones
plt.figure(figsize=(8, 6))
sns.barplot(x='SHOT_ZONE', y='SHOT_MADE', data=zone_efficiency, palette='coolwarm')
plt.title('Shooting Percentage by Shot Zone')
plt.xlabel('Shot Zone')
plt.ylabel('Shooting Percentage')
plt.ylim(0, 1)  # Percentage scale from 0 to 100%
plt.show()

# Step 5: Visualizing High-Percentage Shot Zones by Player
zone_efficiency_by_player = df.groupby(['PLAYER_NAME', 'SHOT_ZONE'])['SHOT_MADE'].mean().unstack().fillna(0)

plt.figure(figsize=(12, 8))
sns.heatmap(zone_efficiency_by_player, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Shooting Efficiency by Player and Shot Zone')
plt.xlabel('Shot Zone')
plt.ylabel('Player Name')
plt.show()
