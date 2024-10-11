import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv(r"C:\Users\shany\OneDrive\Desktop\NBA_2023_Shots.csv")




# Display the first few rows of the dataset
print(df.head())

# Step 1: Data Cleaning
# Remove rows with missing essential data (like shot result, shot location)
df = df.dropna(subset=['SHOT_MADE', 'LOC_X', 'LOC_Y', 'PLAYER_NAME', 'QUARTER', 'MINS_LEFT', 'SECS_LEFT'])

# Step 2: Feature Engineering
# 2.1. Combine minutes and seconds remaining to create a 'time_remaining' column in seconds
df['time_remaining'] = df['MINS_LEFT'] * 60 + df['SECS_LEFT']

# 2.2. Create a new feature for shot distance (if available as loc_x, loc_y)
# Assuming loc_x, loc_y are shot locations in a court coordinate system
df['SHOT_DISTANCE'] = np.sqrt(df['LOC_X']**2 + df['LOC_Y']**2)

# 2.3. Classify shots into different zones on the court
# Example: zone classification based on distance (adjust for specific court geometry)
df['SHOT_ZONE'] = pd.cut(df['SHOT_DISTANCE'],
                         bins=[0, 8, 23, 40],
                         labels=['Paint', 'Mid-Range', '3PT'])

# Step 3: Calculating Shot Efficiency
# 3.1. Calculate overall shooting percentage per player
player_shot_efficiency = df.groupby('PLAYER_NAME')['SHOT_MADE'].mean().reset_index()
player_shot_efficiency.columns = ['PLAYER_NAME', 'shot_percentage']

# 3.2. Calculate shooting efficiency by shot zone for each player
zone_efficiency = df.groupby(['PLAYER_NAME', 'SHOT_ZONE'], observed=False)['SHOT_MADE'].mean().unstack().fillna(0)
print(zone_efficiency)


# Step 4: Aggregating Offensive Strategy Insights
# 4.1. Determine high-percentage shot zones for each player
df['is_high_percentage_zone'] = df.groupby('PLAYER_NAME')['SHOT_MADE'].transform(lambda x: x.mean() > 0.5)

# Step 5: Shot Frequency per Zone
# 5.1. Calculate the number of shots taken per zone for each player
shot_count_by_zone = df.groupby(['PLAYER_NAME', 'SHOT_ZONE']).size().unstack().fillna(0)
print(shot_count_by_zone)

# Step 6: Time-based Analysis
# 6.1. Categorize shots based on game period (early vs late game) for clutch analysis
df['game_phase'] = np.where(df['QUARTER'] >= 4, 'Late', 'Early')

# Step 7: Visualization of Shooting Efficiency
# 7.1. Visualize shot efficiency by zone using a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(zone_efficiency, annot=True, cmap='Blues', fmt='.2f')
plt.title('Shooting Efficiency by Zone for Each Player')
plt.xlabel('Shot Zone')
plt.ylabel('Player Name')
plt.show()

# 7.2. Visualizing overall shot percentage by player
plt.figure(figsize=(12, 8))
sns.barplot(x='PLAYER_NAME', y='shot_percentage', data=player_shot_efficiency)
plt.xticks(rotation=90)
plt.title('Overall Shooting Percentage by Player')
plt.show()



print("Data wrangling and feature engineering complete. Cleaned data saved to 'nba_shots_cleaned.csv'.")
