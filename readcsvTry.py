import pandas as pd
import os
from path import MODEL_PATH, DATA_PATH
readcsv = pd.read_csv(os.path.join(DATA_PATH, 'MedicalClass/train.csv'))
file_unique = readcsv['label'].unique()
print(file_unique,len(file_unique))