'''
Here, we merge the preprocess chess_finals dataset with the preprocess
moves dataset with the corresponent IDs.
'''

import pandas as pd

moves_df = pd.read_csv("./moves2.csv")
general_df = pd.read_csv("./chess_final.csv")

merge_df = pd.merge(general_df, moves_df, on = 'ID')
merge_df.drop(columns = ['White','Black','ECO', 'Termination','ID'], inplace = True)
merge_df.to_csv("merged.csv", index = False)