import pandas as pd
import os
from path import MODEL_PATH, DATA_PATH
import numpy as np
readcsv = pd.read_csv(os.path.join(DATA_PATH, 'MedicalClass/train.csv'))
print(readcsv.shape)
file_unique = readcsv['label'].unique()
list_unique = file_unique.tolist()
print(file_unique,len(file_unique))
print(list_unique)