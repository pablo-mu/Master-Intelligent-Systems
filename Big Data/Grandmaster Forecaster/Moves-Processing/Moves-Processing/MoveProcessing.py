import pandas as pd
import re

loc_file = "./moves2.csv"

'''
Moves DataSet PreProcessing.

Basically, some games have stockfish evaluations inside the movement cell. However,
we don't want to delete that rows. So, we remove the stockfish evaluations of that games,
preserving the moves.
'''


def stockfish_viewing(df):
    # Create a boolean mask to filter rows with Stockfish evaluations
    mask = df['Moves'].str.contains(r'\[%eval [^)]+\]')

    # Apply the boolean mask to filter the rows
    filtered_df = df[mask]

    # Display the resulting DataFrame
    filtered_df.info()
    return print(filtered_df.head(5))

def stockfish_delete(df):
    df['Moves'] = df['Moves'].str.replace(r'\[%eval [^)]+\]', '')
    return df
def stockfish_filtering(row):
    #check only the evaluated_stockfish.
    row = re.sub(r'\{(.*?)\}', '', row)
    row = re.sub(r'\?', '', row)
    row = re.sub(r'\!', '', row)
    row = re.sub(r'\!\!', '', row)
    row = re.sub(r'\?\?', '', row)
    row = re.sub(r'\!\?', '', row)
    row = re.sub(r'\?\!', '', row)
    row = re.sub(r'\d+\.\.\.', '', row)
    row = re.sub('  ', ' ', row)
    return row

#no need
def count_moves(row):
    return len(re.findall(r'\d+\.',row))

def splitting(row):
    #delete the column that indicate which movement is.
    moves = re.sub(r'\d+\.', '', row)
    # delete movements with only 3 movements (because is impossible to win with only with one movement).
    #the shortest win is in the second movement of blacks.
    moves = moves.split()
    #moves = moves[0:len(moves)//2]
    return moves

def half_game(row):
    return row[0:len(row)//2]

def del_short_games(row):
    return len(row) > 4

def return_to_string(row):
    return ' '.join(row)

    #if len... delete row.
if __name__ == "__main__":
    print("Reading")
    df = pd.read_csv(loc_file)
    # The movement column have Stockfish evaluations in some columns.
    #stockfish_viewing(df)
    # Delete that evaluations, because we want to preserve that columns.
    print("Stockfish processing")
    df['Moves'] = df['Moves'].apply(stockfish_filtering)
    #we extract the half of the match.
    print("splitting")
    df['Moves'] = df['Moves'].apply(splitting)
    # delete movements with only 3 movements (because is impossible to win with only with one movement).
    #we set plus 4, because one column is the result.
    df = df[df['Moves'].apply(del_short_games)]
    #we only want half of the game.
    df['Moves'] = df['Moves'].apply(half_game)
    #back into string
    df['Moves'] = df['Moves'].apply(return_to_string)
    df.to_csv("moves2.csv", index = False, columns = ['ID','Moves'])
