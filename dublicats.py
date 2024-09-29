import kadr
import pandas as pd
import cv2
import numpy as np

# name = 'mixed_images/ukbench09268.jpg'
# img = cv2.imread(name)
# evol = kadr.key_dedup_one(img)

db = pd.read_csv('out.csv')
k = db['vec'][1]
print(db['vec'][1])
# for i in range(len(db)):
#     # print(db['vec'][i])
#     for j in db['vec'][i]:
#         print (j)
#
#         # print(j)