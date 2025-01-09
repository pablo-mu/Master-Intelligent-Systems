import pandas as pd
from transformers import DistilBertTokenizerFast, DistilBertModel

'''
In this python programme we perform the move embeddings using the large lingual model 'distil-bert-uncased'.
We have a lot of problems with this programme due to limitations of memory. It is too slow to create a significant
 amount of embeddings, which we can then use to train our XGboost.
'''

df = pd.read_parquet('merged_20')
df.rename(columns = {'index':'ID'}, inplace=True)

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

def get_embeddings(move):
    tokenized_moves = tokenizer.batch_encode_plus([move], padding=True, truncation=True, return_tensors = 'pt', add_special_tokens=True)
    input_ids = tokenized_moves['input_ids']
    attention_mask = tokenized_moves['attention_mask']
    with torch.no_grad():
      outputs = model(input_ids, attention_mask = attention_mask)
      word_embeddings= outputs.last_hidden_state
      return word_embeddings.mean(dim=1).numpy()

csv_file = 'embeddings.csv'

indexes =  df['index'].tolist()
column_names = [f'C{i}' for i in range(768)]
pd.DataFrame(columns=['ID']+column_names).to_csv(csv_file, mode='w', header=True, index=False)

# 11 hours this loop has been running in colab and has stopped :(
# It works well because I've tested it with a small dataframe,
# but because it's so much data, it takes a lot of time. I've done it this way because it's the only way it doesn't run out of ram memory.

for index, row in df.iterrows():
    result = get_embeddings(row['Moves'])
    # Convert the result to a DataFrame with a single row
    df_to_append = pd.DataFrame(columns=['ID'] + column_names)
    df_to_append['ID']=index
    df_to_append.loc[:,column_names] = result
    df_to_append.to_csv(csv_file, mode='a', header=False, index=False)
    del df_to_append