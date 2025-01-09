# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
 Data transformation
"""


file_path = r"C:\Users\ralaz\UJI\0-MASTER\BIG DATA ANALYTICS-SJK006\CHESS\chess_games.csv"
# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

#Rename columns to match the other datasets
df.rename(columns={'AN': 'Moves'}, inplace=True)


# add id column so that when we do the embeddings in the new dataset, we can match it to this dataset
df['ID'] = df.index

nan = df.isnull().sum()
print(nan)

# prepare dataset for embeddings
#moves = df[['ID', 'Moves']]
#moves

#moves.to_csv('moves.csv', index=False)
#from google.colab import drive
#drive.mount('/content/drive')
#from google.colab import files
#files.download('moves.csv')

# Drop columns that we do not want
# They do not provide information relevant to our objective
df_wanted = df.drop(columns=['UTCDate', 'UTCTime', 'WhiteRatingDiff', 'BlackRatingDiff', 'TimeControl', 'Moves'])
# we drop moves because it is being processed in another dataset so this is not relevant here
#WhiteRatingDiff: White's rating points difference after the game
#BlackRatingDiff: Blacks's rating points difference after the game.


df_wanted.info()

nan_c = df_wanted.isnull().sum()
print(nan_c)


"""There are not NaN values"""

# See what values we have in the columns
for col in df_wanted.columns:
    # less than 20 because we have non-categorical columns that have many unique values
    if(len(df_wanted[col].unique())<=20):
        print("Categories in the column:", col)
        print(df_wanted[col].unique())

''' OUTPUT OF THE CATEGORIES:
Categories in the column: Event
[' Classical ' ' Blitz ' ' Blitz tournament ' ' Correspondence '
 ' Classical tournament ' ' Bullet tournament ' ' Bullet '
 'Blitz tournament ' 'Bullet ' 'Classical ' 'Blitz ' 'Bullet tournament '
 'Classical tournament ' 'Correspondence ']
 
Categories in the column: Result
['1-0' '0-1' '1/2-1/2' '*']
Categories in the column: Termination
['Time forfeit' 'Normal' 'Abandoned' 'Rules infraction' 'Unterminated']

'''

"""Event variables normalization"""
# trim spaces
df_wanted['Event'] = df_wanted['Event'].str.strip()

# standardize names
# Define a mapping of variations to standard names
standardization_map = {
    'Classical': ['Classical', 'Classical tournament'],
    'Blitz': ['Blitz', 'Blitz tournament'],
    'Bullet': ['Bullet', 'Bullet tournament'],
    'Correspondence': ['Correspondence']
}

# Replace the variations with the standard name
for standard, variations in standardization_map.items():
    df_wanted['Event'] = df_wanted['Event'].replace(variations, standard)

# Display the normalized DataFrame
 # less than 20 because we have non-categorical columns that have many unique values
if(len(df_wanted['Event'].unique())<=20):
    print("Categories in the column: Event")
    print(df_wanted['Event'].unique())


''' eliminate the rows that have * in the result column
win_loss_counts = df_wanted['Result'].value_counts()

From previous execution we got:
Result
1-0        3113572
0-1        2902394
1/2-1/2     238875
*             1343
Name: count, dtype: int64
'''

chess = df_wanted[df_wanted['Result'] != '*']
chess.info()
df_wanted.info()
# Save the processed data
chess.to_csv('chess_final.csv', index=False)



'''
 Now we analyze our data
'''
file_path = r".\chess_final.csv"
# Read the CSV file into a DataFrame
chess = pd.read_csv(file_path)

'''1- GAME RESULTS'''
#win or loss distribution
import matplotlib.pyplot as plt
win_loss_counts = chess['Result'].value_counts()


ax = win_loss_counts.plot(kind='bar')

plt.title('Win/Loss Distribution for White and Black')
plt.xlabel('Result')
plt.ylabel('Number of Games')
# Adding the count above each bar
for i in ax.patches:
    ax.text(i.get_x() + i.get_width()/2, i.get_height() + 5,
            str(i.get_height()), ha='center', va='bottom')

plt.show()
#by elo

# Filter data based on the result
white_wins = chess[chess['Result'] == '1-0']
black_wins = chess[chess['Result'] == '0-1']
draws = chess[chess['Result'] == '1/2-1/2']

# Create scatter plots
plt.figure(figsize=(10, 8))
#plt.scatter(white_wins['WhiteElo'], white_wins['BlackElo'], color='blue', label='White wins')
plt.scatter(black_wins['WhiteElo'], black_wins['BlackElo'], color='red', label='Black wins')
#plt.scatter(draws['WhiteElo'], draws['BlackElo'], color='green', label='Draw')

# Adding titles and labels
plt.title('Chess Games Outcomes by Elo Rating')
plt.xlabel('White Player Elo')
plt.ylabel('Black Player Elo')
plt.legend()

# Save plot
plt.savefig('scatter-black.png')

##Percentage
win_loss_counts = chess['Result'].value_counts()

# Calculate percentage values
total_games = len(chess['Result'])
white_percentage = (win_loss_counts['1-0'] / total_games) * 100
black_percentage = (win_loss_counts['0-1'] / total_games) * 100
draw_percentage = (win_loss_counts['1/2-1/2'] / total_games) * 100

# Plotting
ax = win_loss_counts.plot(kind='bar')

# Adding the count above each bar
for i in ax.patches:
    ax.text(i.get_x() + i.get_width()/2, i.get_height() + 1,
            f'{i.get_height() / total_games * 100:.2f}%', ha='center', va='bottom')

plt.title('Win/Loss Distribution for White and Black')
plt.xlabel('Result')
plt.ylabel('Percentage of Games')
plt.legend()
plt.show()

''' 2- TERMINATIONS'''
# Termination types
termination_counts = chess['Termination'].value_counts()

# Plotting
ax = termination_counts.plot(kind='bar')
plt.title('Game Termination Types')
plt.xlabel('Termination Type')
plt.ylabel('Count')
# Adding the count above each bar
for i in ax.patches:
    ax.text(i.get_x() + i.get_width()/2, i.get_height() + 5,
            str(i.get_height()), ha='center', va='bottom')

plt.show()

'''2- CHECK THE TYPES OF TERMINATIONS'''
term = chess['Termination'].value_counts()
print(term)

'''
Normal              4230089
Time forfeit        2011336
Abandoned             13288
Rules infraction        128
'''
# see if the abandonement are win or loose
filtered_df = chess[chess['Termination'] == 'Abandoned']
result = filtered_df[['Result', 'Termination']]
print(result)
abandoned_winloose = result['Result'].value_counts()
print(abandoned_winloose)
'''
    1-0    13281 more whites win because of abandonement
       0-1        7
        Name: count, dtype
'''


# see if the Rules infraction are win or loose
filtered_df2 = chess[chess['Termination'] == 'Rules infraction']
result2 = filtered_df2[['Result', 'Termination']]
print(result2)
infractions_winloose = result2['Result'].value_counts()
print(infractions_winloose)

'''
     1-0    128 all that have done infractions are black players'''


# see if the normal terminations are win or loose
filtered_df3 = chess[chess['Termination'] == 'Normal']
result3 = filtered_df3[['Result', 'Termination']]
print(result3)

normal_winloose = result3['Result'].value_counts()
print(normal_winloose)
'''1-0        2116730 more whites win
0-1        1926100
1/2-1/2     187259
    Name: count, dtype'''

# see if the time forfeit terminations are win or loose
filtered_df4 = chess[chess['Termination'] == 'Time forfeit']
result4 = filtered_df4[['Result', 'Termination']]
print(result4)

time_winloose = result4['Result'].value_counts()
print(time_winloose)
'''1-0        983433
0-1        976287
1/2-1/2     51616
    Name: count, dtype'''

'''3- DIFFERENCE OF ELO POINTS IN EACH GAME'''
# do not abs this because we want to identify which has the higher value so black would be negative
chess['ELO_diff'] = chess['WhiteElo'] - chess['BlackElo']

elo_diff_stats = abs(chess['ELO_diff']).describe()
pd.options.display.float_format = '{:.2f}'.format
# Print the summary statistics
print(elo_diff_stats)
'''
count   6254841.00
mean        147.18
std         139.33
min           0.00
25%          46.00
50%         106.00
75%         204.00
max        1702.00 very big difference in elo level
'''
# Elo rating difference threshold
elo_threshold = 500

# Filter the dataset to get games with Elo rating difference greater than the threshold
#abs because we want all of the values for white and for black that are better
big_elo_difference_games = chess[abs(chess['ELO_diff']) > elo_threshold]

# Print the filtered dataset
print(big_elo_difference_games)

# Create a new column to categorize Elo difference into 'Positive', 'Zero', or 'Negative'
chess['ELO_diff_category'] = 'Elo difference less than 500'
chess.loc[chess['ELO_diff'] > 500, 'ELO_diff_category'] = 'White higher Elo'
chess.loc[chess['ELO_diff'] < -500, 'ELO_diff_category'] = 'Black higher Elo'

# Group the DataFrame by 'ELO_diff_category' and 'Result' and count occurrences
result_counts = chess.groupby(['ELO_diff_category', 'Result']).size().unstack(fill_value=0)

# Display the result counts
print(result_counts)

# Plot
ax = result_counts.plot(kind='bar', stacked=False, figsize=(10, 6))

# Set plot labels and title
plt.xlabel('Elo Difference Category')
plt.ylabel('Counts')
plt.title('Distribution of Game Results by Elo Difference Category')

plt.legend(title='Result', loc='upper right')
for p in ax.patches:
    x = p.get_x() + p.get_width() / 2
    y = p.get_y() + p.get_height() + 50000  # Adjust the y-coordinate for label placement
    value = int(p.get_height())
    ax.annotate(value, (x, y), ha='center')

plt.show()

#elo diff 25

# do not abs this because we want to identify which has the higher value so black would be negative
chess['ELO_diff'] = chess['WhiteElo'] - chess['BlackElo']

elo_diff_stats = abs(chess['ELO_diff']).describe()
pd.options.display.float_format = '{:.2f}'.format
# Print the summary statistics
print(elo_diff_stats)
'''
count   6254841.00
mean        147.18
std         139.33
min           0.00
25%          46.00
50%         106.00
75%         204.00
max        1702.00 very big difference in elo level
'''
# Elo rating difference threshold
elo_threshold = 25

# Filter the dataset to get games with Elo rating difference greater than the threshold
#abs because we want all of the values for white and for black that are better
big_elo_difference_games = chess[abs(chess['ELO_diff']) > elo_threshold]

# Print the filtered dataset
print(big_elo_difference_games)

# Create a new column to categorize Elo difference into 'Positive', 'Zero', or 'Negative'
chess['ELO_diff_category'] = 'Elo diff < 25'
chess.loc[chess['ELO_diff'] > 25, 'ELO_diff_category'] = 'White higher Elo'
chess.loc[chess['ELO_diff'] < -25, 'ELO_diff_category'] = 'Black higher Elo'

# Group the DataFrame by 'ELO_diff_category' and 'Result' and count occurrences
result_counts = chess.groupby(['ELO_diff_category', 'Result']).size().unstack(fill_value=0)

# Display the result counts
print(result_counts)

# Plot
ax = result_counts.plot(kind='bar', stacked=False, figsize=(10, 6))

# Set plot labels and title
plt.xlabel('Elo Difference Category')
plt.ylabel('Counts')
plt.title('Distribution of Game Results by Elo Difference Category')
plt.ylim(0, 2e6)
plt.legend(title='Result', bbox_to_anchor=(1, 1))
for p in ax.patches:
    x = p.get_x() + p.get_width() / 2
    y = p.get_y() + p.get_height() + 50000  # Adjust the y-coordinate for label placement
    value = int(p.get_height())
    ax.annotate(value, (x, y), ha='center')

plt.show()

'''3.1- DIFFERENCE OF ELO to check baseline'''
# do not abs this because we want to identify which has the higher value so black would be negative
chess['diff'] = chess['WhiteElo'] - chess['BlackElo']

elo_diff_stats = abs(chess['diff']).describe()
pd.options.display.float_format = '{:.2f}'.format
# Print the summary statistics
print(elo_diff_stats)
'''
count   6254841.00
mean        147.18
std         139.33
min           0.00
25%          46.00
50%         106.00
75%         204.00
max        1702.00 very big difference in elo level
'''

white_bigger = chess[chess['diff'] >0]
print(len(white_bigger))
print(white_bigger['Result'].value_counts())
"""
white elo bigger: 3138470 (y)
Result
1-0        2028335 (x)
0-1         992683
1/2-1/2     117452

(x/y) *100 = 64.63%
"""

black_bigger = chess[chess['diff'] <0]
print(len(black_bigger))
print(black_bigger['Result'].value_counts())
'''
black elo bigger: 3096197 (y)
Result
0-1        1900495 (x)
1-0        1075249
1/2-1/2     120453

(x/y) *100 = 61.38%
'''

white_bigger_50 = chess[chess['diff'] >50]
print(len(white_bigger_50))
print(white_bigger['Result'].value_counts())
"""
white elo bigger: 2300728 (y)
Result
1-0        2028335 (x)
0-1         992683
1/2-1/2     117452

(x/y) *100 = 81.16%
"""

black_bigger_50 = chess[chess['diff'] <-50]
print(len(black_bigger_50))
print(black_bigger['Result'].value_counts())
'''
black elo bigger: 2263057 (y)
Result
0-1        1900495 (x)
1-0        1075249
1/2-1/2     120453

(x/y) *100 = 83.98%
'''

'''4- ELO CORRELATIONS, converting results to numeric (not useful)'''

# WE NEED TO CONVERT THE RESULT TO NUMERIC TO CHECK CORRELATIONS
def result_to_numeric(result):
    if result == '1-0':
        return 1  # White wins
    elif result == '0-1':
        return 0  # Black wins
    elif result == '1/2-1/2':
        return 0.5  # Draw
    else:
        return None

chess['numeric_result'] = chess['Result'].apply(result_to_numeric)
chess.info()
elo_corr = chess[['WhiteElo', 'BlackElo', 'numeric_result']].corr()
print(elo_corr)
'''
               WhiteElo  BlackElo  numeric_result
WhiteElo        1.000000  0.710445        0.140026
BlackElo        0.710445  1.000000       -0.145405
numeric_result  0.140026 -0.145405        1.000000

No correlation for the numeric result, maybe that whites win more
'''

'''5- DIVISION BY ELO LEVEL, USED LATER FOR ANALYSES'''

# Elo rating distributions to see how to partition them
#plt.hist(chess['WhiteElo'], alpha=0.5, label='White Elo')
plt.hist(chess['BlackElo'], alpha=0.5, label='Black Elo', color='orange')
plt.legend()
plt.title('Elo Rating Distribution')
plt.xlabel('Elo Rating')
plt.ylabel('Frequency')
plt.show()

# Calculate the 25th and 75th percentiles for both White and Black Elo ratings
white_25th_percentile = chess['WhiteElo'].quantile(0.25)
white_75th_percentile = chess['WhiteElo'].quantile(0.75)
black_25th_percentile = chess['BlackElo'].quantile(0.25)
black_75th_percentile = chess['BlackElo'].quantile(0.75)

# Categorizing players
high_rated_whites = chess[chess["WhiteElo"] >= white_75th_percentile]
high_rated_blacks = chess[chess["BlackElo"] >= black_75th_percentile]
low_rated_whites = chess[chess["WhiteElo"] < white_25th_percentile]
low_rated_blacks = chess[chess["BlackElo"] < black_25th_percentile]
avg_rated_whites = chess[(chess["WhiteElo"] >= white_25th_percentile) & (chess["WhiteElo"] < white_75th_percentile)]
avg_rated_blacks = chess[(chess["BlackElo"] >= black_25th_percentile) & (chess["BlackElo"] < black_75th_percentile)]

# Explore our players level
categories = ['All', 'High rated whites', 'High rated blacks', 'Average rated whites', 'Average rated blacks', 'Low rated whites', 'Low rated blacks']
data_size = [len(chess), len(high_rated_whites), len(high_rated_blacks), len(avg_rated_whites), len(avg_rated_blacks), len(low_rated_whites), len(low_rated_blacks)]

# Create the figure and axes objects
fig, ax = plt.subplots(figsize=(8, 4))

# Plot the histogram
barlist = ax.bar(categories, data_size, color='skyblue')

# Set the x-axis label and title
ax.set_xlabel('Player Category', fontsize=12)
ax.set_ylabel('Counts of rows', fontsize=12)
ax.set_title('Distribution of rows based on player rating', fontsize=14, fontweight='bold')

# Set the tick label font size
ax.tick_params(axis='both', which='major', labelsize=10)

# Add value labels to the bars
for i, v in enumerate(data_size):
    ax.text(i - 0.1, v + 100, str(v), color='black', fontsize=10)

# Rotate the x-axis labels
plt.xticks(rotation=45)

# Show the plot
plt.show()

'''6- RESULTS BY ELO LEVEL'''

# Function to calculate Win/Loss/Draw ratios
# I needed to order the categories for it to work because the plots were showing mixed otherwise
def get_win_loss_draw_ratios(dataframe):
    return dataframe['Result'].value_counts(normalize=True)

# Calculate ratios for each category
ratios_high_white = get_win_loss_draw_ratios(high_rated_whites)
ratios_avg_white = get_win_loss_draw_ratios(avg_rated_whites)
ratios_low_white = get_win_loss_draw_ratios(low_rated_whites)

ratios_high_black = get_win_loss_draw_ratios(high_rated_blacks)
ratios_avg_black = get_win_loss_draw_ratios(avg_rated_blacks)
ratios_low_black = get_win_loss_draw_ratios(low_rated_blacks)

#REORDER THE SERIES
expected_order = ['1-0', '0-1', '1/2-1/2']
# Reindexing the series to match the expected order
ratios_high_white = ratios_high_white.reindex(expected_order, fill_value=0)
ratios_avg_white = ratios_avg_white.reindex(expected_order, fill_value=0)
ratios_low_white = ratios_low_white.reindex(expected_order, fill_value=0)

ratios_high_black = ratios_high_black.reindex(expected_order, fill_value=0)
ratios_avg_black = ratios_avg_black.reindex(expected_order, fill_value=0)
ratios_low_black = ratios_low_black.reindex(expected_order, fill_value=0)

#first plot

# Define labels and locations for the bars
labels = ['1-0 (white)', '0-1 (black)', '1/2-1/2 (draw)']
x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
def add_value_labels(ax, bars, spacing=5):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.0%}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, spacing),  # how far above the bar the text is
                    textcoords="offset points",
                    ha='center', va='bottom')

# Plot for White players
bar1 = axs[0].bar(x - width, ratios_high_white, width, label='High Rated', alpha=0.6)
bar2 = axs[0].bar(x, ratios_avg_white, width, label='Average Rated', alpha=0.6)
bar3 = axs[0].bar(x + width, ratios_low_white, width, label='Low Rated', alpha=0.6)

add_value_labels(axs[0], bar1)
add_value_labels(axs[0], bar2)
add_value_labels(axs[0], bar3)

# Add labels
axs[0].set_ylabel('Ratio')
axs[0].set_title('Win/Loss/Draw Ratios for White Players')
axs[0].set_xticks(x)
axs[0].set_xticklabels(labels)
axs[0].legend()

# Plot for Black players
bar4 = axs[1].bar(x - width, ratios_high_black, width, label='High Rated', alpha=0.6)
bar5 = axs[1].bar(x, ratios_avg_black, width, label='Average Rated', alpha=0.6)
bar6 = axs[1].bar(x + width, ratios_low_black, width, label='Low Rated', alpha=0.6)

add_value_labels(axs[1], bar4)
add_value_labels(axs[1], bar5)
add_value_labels(axs[1], bar6)

# Add labels
axs[1].set_ylabel('Ratio')
axs[1].set_title('Win/Loss/Draw Ratios for Black Players')
axs[1].set_xticks(x)
axs[1].set_xticklabels(labels)
axs[1].legend()

plt.tight_layout()
plt.show()

#second plot
#White

data = {
    'High Rated White': ratios_high_white,
    'Average Rated White': ratios_avg_white,
    'Low Rated White': ratios_low_white
}
ratios_df = pd.DataFrame(data)

# New labels
expected_order = ['Win', 'Loss', 'Draw']
ratios_df.index = expected_order

# Define labels and locations for the bars
labels = ['High rated whites', 'AVG rated whites', 'Low rated whites']
x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

# Create subplots
fig, ax = plt.subplots(figsize=(10, 6))

# Plotting each row as a separate bar
bar_width = width / len(ratios_df.columns)  # Adjust bar width based on number of columns
for i, (label, row) in enumerate(ratios_df.iterrows()):
    ax.bar(x + i * bar_width, row, bar_width, label=label)

# Add value labels
def add_value_labels(ax, spacing=5):
    for i in ax.patches:
        height = i.get_height()
        ax.annotate(f'{height:.0%}',
                    xy=(i.get_x() + i.get_width() / 2, height),
                    xytext=(0, spacing),  # how far above the bar the text is
                    textcoords="offset points",
                    ha='center', va='bottom')

add_value_labels(ax)

# Add labels
ax.set_ylabel('Ratio')
ax.set_title('Win/Loss/Draw Ratios by White Rating')
ax.set_xticks(x + bar_width)
ax.set_xticklabels(labels)
ax.legend(title="Player Group", bbox_to_anchor=(1, 1))

plt.show()

#black

data = {
    'High Rated Black': ratios_high_black,
    'Average Rated Black': ratios_avg_black,
    'Low Rated Black': ratios_low_black
}
ratios_df = pd.DataFrame(data)

# New labels
expected_order = ['Loss', 'Win', 'Draw']
ratios_df.index = expected_order

# Define labels and locations for the bars
labels = ['High rated blacks', 'AVG rated blacks', 'Low rated blacks']
x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

# Create subplots
fig, ax = plt.subplots(figsize=(10, 6))

# Plotting each row as a separate bar
bar_width = width / len(ratios_df.columns)  # Adjust bar width based on number of columns
for i, (label, row) in enumerate(ratios_df.iterrows()):
    ax.bar(x + i * bar_width, row, bar_width, label=label)

# Add value labels
def add_value_labels(ax, spacing=5):
    for i in ax.patches:
        height = i.get_height()
        ax.annotate(f'{height:.0%}',
                    xy=(i.get_x() + i.get_width() / 2, height),
                    xytext=(0, spacing),  # how far above the bar the text is
                    textcoords="offset points",
                    ha='center', va='bottom')

add_value_labels(ax)

# Add labels
ax.set_ylabel('Ratio')
ax.set_title('Win/Loss/Draw Ratios by Black Rating')
ax.set_xticks(x + bar_width)
ax.set_xticklabels(labels)
ax.legend(title="Player Group", bbox_to_anchor=(1, 1))

plt.show()


'''7- Analyzing AVERAGE game results for each category (NOT USEFUL)
(redone in the section above)'''
# not interesting because we miss the wins and the looses

high_rated_whites_mean = high_rated_whites['numeric_result'].mean()
high_rated_blacks_mean = high_rated_blacks['numeric_result'].mean()
low_rated_whites_mean = low_rated_whites['numeric_result'].mean()
low_rated_blacks_mean = low_rated_blacks['numeric_result'].mean()
avg_rated_whites_mean = avg_rated_whites['numeric_result'].mean()
avg_rated_blacks_mean = avg_rated_blacks['numeric_result'].mean()

# Plotting
plt.figure(figsize=(12, 6))

# For White Players
plt.subplot(1, 2, 1)
plt.bar(['High Rated', 'Average Rated', 'Low Rated'], [high_rated_whites_mean, avg_rated_whites_mean, low_rated_whites_mean], color=['blue', 'green', 'red'])
plt.title('Average Game Results for White Players')
plt.xlabel('Elo Rating Category')
plt.ylabel('Average Game Result')
plt.ylim(0, 1)

# For Black Players
plt.subplot(1, 2, 2)
plt.bar(['High Rated', 'Average Rated', 'Low Rated'], [high_rated_blacks_mean, avg_rated_blacks_mean, low_rated_blacks_mean], color=['blue', 'green', 'red'])
plt.title('Average Game Results for Black Players')
plt.xlabel('Elo Rating Category')
plt.ylabel('Average Game Result')
plt.ylim(0, 1)

plt.tight_layout()
plt.show()



''' 8/9- OPENINGS effectiveness'''
#9- TOP 10 OPENINGS
opening_counts = chess['Opening'].value_counts().head(10)  # Top 10 openings

# Plotting
opening_counts.plot(kind='bar')
plt.title('Top 10 Most Popular Openings')
plt.xlabel('Opening')
plt.ylabel('Frequency')
plt.show()

#8- OPENINGS BY color and win rate this plots openings that were used only once and won the game

#White
# Group by opening and result
opening_effectiveness = chess.groupby(['Opening', 'Result']).size().unstack().fillna(0)

# Calculating win rate for White
opening_effectiveness['sum'] = opening_effectiveness.sum(axis=1)

opening_effectiveness['WhiteWinRate'] = opening_effectiveness['1-0'] / opening_effectiveness['sum']

# Sorting and selecting top 10 openings for White's win rate
top_openings = opening_effectiveness['WhiteWinRate'].sort_values(ascending=False).head(10)

# Plotting top 10 openings
plt.figure(figsize=(10, 6))
top_openings.plot(kind='barh', color='skyblue')
plt.title('Top 10 Openings with Highest Win Rate for White')
plt.xlabel('White Win Rate')
plt.ylabel('Opening')
plt.xticks()
plt.show()

#BLACK
opening_effectiveness['BlackWinRate'] = opening_effectiveness['0-1'] / (opening_effectiveness.sum(axis=1))

# Sorting and selecting top 10 openings for White's win rate
top_openings = opening_effectiveness['BlackWinRate'].sort_values(ascending=False).head(10)

# Plotting top 10 openings
plt.figure(figsize=(10, 6))
top_openings.plot(kind='barh', color='skyblue')
plt.title('Top 10 Openings with Highest Win Rate for Black')
plt.xlabel('Black Win Rate')
plt.ylabel('Opening')
plt.xticks()
plt.show()



'''10- OPENINGS BY ELO LEVEL (frequency)'''
# Function to get top N openings
def get_top_openings(dataframe, n=5):
    return dataframe['Opening'].value_counts().head(n)

# Getting top openings for each category
top_openings_high_white = get_top_openings(high_rated_whites)
top_openings_avg_white = get_top_openings(avg_rated_whites)
top_openings_low_white = get_top_openings(low_rated_whites)

top_openings_high_black = get_top_openings(high_rated_blacks)
top_openings_avg_black = get_top_openings(avg_rated_blacks)
top_openings_low_black = get_top_openings(low_rated_blacks)

# Top Openings for White players
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
fig.suptitle('Top 5 Openings for White Players by Rating Category')

axes[0].bar(top_openings_high_white.index, top_openings_high_white.values, color='blue')
axes[0].set_title('High Rated')
axes[0].tick_params(labelrotation=90)

axes[1].bar(top_openings_avg_white.index, top_openings_avg_white.values, color='orange')
axes[1].set_title('Average Rated')
axes[1].tick_params(labelrotation=90)

axes[2].bar(top_openings_low_white.index, top_openings_low_white.values, color='green')
axes[2].set_title('Low Rated')
axes[2].tick_params(labelrotation=90)

plt.tight_layout()
plt.show()

# Top Openings for Black players
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
fig.suptitle('Top 5 Openings for Black Players by Rating Category')

axes[0].bar(top_openings_high_black.index, top_openings_high_black.values, color='blue')
axes[0].set_title('High Rated')
axes[0].tick_params(labelrotation=90)

axes[1].bar(top_openings_avg_black.index, top_openings_avg_black.values, color='orange')
axes[1].set_title('Average Rated')
axes[1].tick_params(labelrotation=90)

axes[2].bar(top_openings_low_black.index, top_openings_low_black.values, color='green')
axes[2].set_title('Low Rated')
axes[2].tick_params(labelrotation=90)

plt.tight_layout()
plt.show()

'''11- MOST FREQUENT OPENINGS EFFECTIVENESS (BY COLOR AND WINRATE)'''

#Top 46 openings (for winrates)
#series containing all of the frequencies of the openings
opening_counts_total = chess['Opening'].value_counts()
opening_counts_total.info()

popular_openings = opening_counts_total[opening_counts_total > 25000] #there are 46 openings with more than 25000 occurences
popular_openings.info()

# Group by opening and result for the most used openings
most_used_openings = chess[chess['Opening'].isin(popular_openings.index)]
opening_effectiveness2 = most_used_openings.groupby(['Opening', 'Result']).size().unstack().fillna(0)

#White
# Calculating win rate for White
opening_effectiveness2['sum'] = opening_effectiveness.sum(axis=1)

opening_effectiveness2['WhiteWinRate'] = opening_effectiveness2['1-0'] / opening_effectiveness2['sum']

# Sorting and selecting top 10 openings for White's win rate
top_openings = opening_effectiveness2['WhiteWinRate'].sort_values(ascending=False).head(5)

'''
Philidor Defense                                0.30
Italian Game                                    0.29
Queen's Gambit Refused: Marshall Defense        0.29
Philidor Defense #3                             0.28
Scotch Game                                     0.28
French Defense: Normal Variation                0.28
Scandinavian Defense: Mieses-Kotroc Variation   0.28
Italian Game: Anti-Fried Liver Defense          0.28
Bishop's Opening                                0.27
Queen's Pawn                                    0.27
'''
# Plotting top 10 openings
plt.figure(figsize=(10, 6))
top_openings.plot(kind='barh', color='skyblue')
plt.title('Top 5 Openings with Highest Win Rate for White')
plt.ylabel('Opening')
plt.xlabel('White Win Rate')
plt.xticks()
plt.show()

#BLACK
opening_effectiveness2['BlackWinRate'] = opening_effectiveness2['0-1'] / (opening_effectiveness2.sum(axis=1))

# Sorting and selecting top openings for White's win rate
top_openings = opening_effectiveness2['BlackWinRate'].sort_values(ascending=False).head(5)

'''
French Defense #2                       0.18
Sicilian Defense: Bowdler Attack        0.18
Sicilian Defense                        0.18
Van't Kruijs Opening                    0.18
Sicilian Defense: French Variation      0.18
Indian Game                             0.17
King's Pawn Game: Leonardis Variation   0.17
Caro-Kann Defense                       0.17
Old Benoni Defense                      0.17
Sicilian Defense: Old Sicilian          0.17
'''
# Plotting top 10 openings
plt.figure(figsize=(10, 6))
top_openings.plot(kind='barh', color='skyblue')
plt.title('Top 5 Openings with Highest Win Rate for Black')
plt.ylabel('Opening')
plt.xlabel('Black Win Rate')
plt.xticks()
plt.show()


'''12- OPENINGS BY ELO LEVEL AND EFFECTIVENESS (BEST)'''

def calculate_opening_effectiveness_white(df):
    most_used_openings = df[df['Opening'].isin(popular_openings.index)]
    openings_effectiveness = most_used_openings.groupby(['Opening', 'Result']).size().unstack().fillna(0)
    openings_effectiveness['WinRate'] = openings_effectiveness['1-0'] / openings_effectiveness.sum(axis=1)
    return openings_effectiveness.sort_values(by='WinRate', ascending=False)

def calculate_opening_effectiveness_black(df):
    most_used_openings = df[df['Opening'].isin(popular_openings.index)]
    openings_effectiveness = most_used_openings.groupby(['Opening', 'Result']).size().unstack().fillna(0)
    openings_effectiveness['WinRate'] = openings_effectiveness['0-1'] / openings_effectiveness.sum(axis=1)
    return openings_effectiveness.sort_values(by='WinRate', ascending=False)


#performance of white
high_white_OE = calculate_opening_effectiveness_white(high_rated_whites).head(5)
avg_white_OE = calculate_opening_effectiveness_white(avg_rated_whites).head(5)
low_white_OE = calculate_opening_effectiveness_white(low_rated_whites).head(5)

#performance of blacks
high_black_OE = calculate_opening_effectiveness_black(high_rated_blacks).head(5)
avg_black_OE = calculate_opening_effectiveness_black(avg_rated_blacks).head(5)
low_black_OE = calculate_opening_effectiveness_black(low_rated_blacks).head(5)

# Separator line
separator = '-' * 40

# Performance of White Players
print("Performance of White Players:")
print(separator)

# High Rated White Players
print("High Rated White Players:")
print(high_white_OE)
print(separator)

# Average Rated White Players
print("Average Rated White Players:")
print(avg_white_OE)
print(separator)

# Low Rated White Players
print("Low Rated White Players:")
print(low_white_OE)
print(separator)

# Performance of Black Players
print("Performance of Black Players:")
print(separator)

# High Rated Black Players
print("High Rated Black Players:")
print(high_black_OE)
print(separator)

# Average Rated Black Players
print("Average Rated Black Players:")
print(avg_black_OE)
print(separator)

# Low Rated Black Players
print("Low Rated Black Players:")
print(low_black_OE)
print(separator)
'''
Performance of White Players:
----------------------------------------
High Rated White Players:
Result                                   0-1   1-0  1/2-1/2  WinRate
Opening                                                             
Philidor Defense                         827  2602      137     0.73
Italian Game: Anti-Fried Liver Defense   772  2326      122     0.72
Italian Game                             667  1928       93     0.72
Philidor Defense #3                     1589  4391      214     0.71
Scotch Game                             1643  4067      238     0.68
----------------------------------------
Average Rated White Players:
Result                                    0-1    1-0  1/2-1/2  WinRate
Opening                                                               
Philidor Defense                         4269   8130      501     0.63
Italian Game                             4834   8147      461     0.61
Philidor Defense #3                     13207  20599     1372     0.59
Italian Game: Anti-Fried Liver Defense   7761  11718      788     0.58
Scotch Game                              9966  15199     1144     0.58
----------------------------------------
Low Rated White Players:
Result                                      0-1    1-0  1/2-1/2  WinRate
Opening                                                                 
Italian Game                               4282   5096      279     0.53
Philidor Defense                           3860   4510      362     0.52
Queen's Gambit Refused: Marshall Defense   3559   3619      242     0.49
Scotch Game                                8370   8575      690     0.49
Philidor Defense #3                       10721  10707      744     0.48
----------------------------------------
Performance of Black Players:
----------------------------------------
High Rated Black Players:
Result                                    0-1   1-0  1/2-1/2  WinRate
Opening                                                              
King's Pawn Game: Wayward Queen Attack    660   201       28     0.74
French Defense #2                        2994  1102      167     0.70
Sicilian Defense: Bowdler Attack         8149  3296      426     0.69
King's Pawn Game: Leonardis Variation    2385  1042      132     0.67
Sicilian Defense                        11519  5641      844     0.64
----------------------------------------
Average Rated Black Players:
Result                                    0-1    1-0  1/2-1/2  WinRate
Opening                                                               
King's Pawn Game: Wayward Queen Attack   7190   4707      446     0.58
French Defense #2                        9394   6405      543     0.57
King's Pawn Game: Leonardis Variation   11881   8792      847     0.55
Sicilian Defense: Bowdler Attack        24012  17996     1535     0.55
Van't Kruijs Opening                    36678  28235     2131     0.55
----------------------------------------
Low Rated Black Players:
Result                                   0-1    1-0  1/2-1/2  WinRate
Opening                                                              
Sicilian Defense                       10137  11127      691     0.46
French Defense #2                       4689   5168      332     0.46
King's Pawn Game: Leonardis Variation   9256  10708      708     0.45
Scandinavian Defense                   13912  16176     1063     0.45
Sicilian Defense: Bowdler Attack        7591   8925      514     0.45
----------------------------------------
'''

'''13- event TYPES'''
# summary
event_counts = chess['Event'].value_counts()

# Plotting
ax = event_counts.plot(kind='bar')
plt.title('Summary of event Types')
plt.xlabel('Event Type')
plt.ylabel('Count')
# Adding the count above each bar
for i in ax.patches:
    ax.text(i.get_x() + i.get_width()/2, i.get_height() + 5,
            str(i.get_height()), ha='center', va='bottom')

plt.show()

# by elo

# Function to get the sum of each type of game for a given DataFrame
def get_event_counts(dataframe):
    return dataframe['Event'].value_counts()

# Calculate event counts for each category
event_counts_high_white = get_event_counts(high_rated_whites)
event_counts_avg_white = get_event_counts(avg_rated_whites)
event_counts_low_white = get_event_counts(low_rated_whites)

event_counts_high_black = get_event_counts(high_rated_blacks)
event_counts_avg_black = get_event_counts(avg_rated_blacks)
event_counts_low_black = get_event_counts(low_rated_blacks)

separator = '-' * 40

# Performance of White Players
print("Event types of White Players:")
print(separator)

# High Rated White Players
print("High Rated White Players:")
print(event_counts_high_white)
print(separator)

# Average Rated White Players
print("Average Rated White Players:")
print(event_counts_avg_white)
print(separator)

# Low Rated White Players
print("Low Rated White Players:")
print(event_counts_low_white)
print(separator)

# Performance of Black Players
print("Event types of Black Players:")
print(separator)

# High Rated Black Players
print("High Rated Black Players:")
print(event_counts_high_black)
print(separator)

# Average Rated Black Players
print("Average Rated Black Players:")
print(event_counts_avg_black)
print(separator)

# Low Rated Black Players
print("Low Rated Black Players:")
print(event_counts_avg_black)
print(separator)
'''
Event types of White Players:
----------------------------------------
High Rated White Players:
Event
Blitz             709424
Bullet            555025
Classical         300675
Correspondence      2960
Name: count, dtype: int64
----------------------------------------
Average Rated White Players:
Event
Blitz             1381718
Classical          890740
Bullet             843554
Correspondence      10143
Name: count, dtype: int64
----------------------------------------
Low Rated White Players:
Event
Blitz             720102
Classical         484774
Bullet            345892
Correspondence      9834
Name: count, dtype: int64
----------------------------------------
Event types of Black Players:
----------------------------------------
High Rated Black Players:
Event
Blitz             708692
Bullet            554282
Classical         298350
Correspondence      3035
Name: count, dtype: int64
----------------------------------------
Average Rated Black Players:
Event
Blitz             1384428
Classical          891877
Bullet             842138
Correspondence      10236
Name: count, dtype: int64
----------------------------------------
Low Rated Black Players:
Event
Blitz             1384428
Classical          891877
Bullet             842138
Correspondence      10236
Name: count, dtype: int64
----------------------------------------
'''

'''14- CLUSTERING BY ELO LEVEL'''
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5, n_init=10)
features = chess[['WhiteElo', 'BlackElo']]
kmeans.fit(features)

chess['cluster'] = kmeans.labels_

plt.scatter(chess['WhiteElo'], chess['BlackElo'], c=chess['cluster'], cmap='viridis')
plt.title('Cluster Analysis of Chess Games by Player Elo Ratings')
plt.xlabel('White Elo Rating')
plt.ylabel('Black Elo Rating')
plt.savefig('myplot.png')
plt.show()
plt.close()  # Close the window programmatically

