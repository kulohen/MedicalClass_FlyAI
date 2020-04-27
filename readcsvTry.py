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

a = np.array(([1,2,3],[4,5,6]))
b= np.argmax(a,axis=1)
print(a)
print(b)

print(np.argmax(a[0]))