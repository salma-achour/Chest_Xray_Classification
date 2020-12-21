import os
import glob
import shutil
import itertools
import numpy as np

# create paths
covid_path = 'COVID-19 Radiography Database/COVID-19'
normal_path = 'COVID-19 Radiography Database/NORMAL'
pneumonia_path = 'COVID-19 Radiography Database/Viral Pneumonia'

# Making directories
"""
os.makedirs('Data_organized/train')
os.mkdir('Data_organized/test')

os.mkdir('Data_organized/train/covid')
os.mkdir('Data_organized/test/covid')

os.mkdir('Data_organized/train/normal')
os.mkdir('Data_organized/test/normal')

os.mkdir('Data_organized/train/pneumonia')
os.mkdir('Data_organized/test/pneumonia')

"""
# The length of each training set for each class
covid_train_len = int(np.floor(len(os.listdir(covid_path))*0.8))
covid_len = len(os.listdir(covid_path))

normal_train_len = int(np.floor(len(os.listdir(normal_path))*0.8))
normal_len = len(os.listdir(normal_path))

pneumonia_train_len = int(np.floor(len(os.listdir(pneumonia_path))*0.8))
pneumonia_len = len(os.listdir(pneumonia_path))

# Moving the data to the directories
for trainimg in itertools.islice(glob.iglob(os.path.join(covid_path, '*.png')), covid_train_len):
    shutil.copy(trainimg, 'Data_organized/train/covid')
    
for trainimg in itertools.islice(glob.iglob(os.path.join(normal_path, '*.png')), normal_train_len):
    shutil.copy(trainimg, 'Data_organized/train/normal')
    
for trainimg in itertools.islice(glob.iglob(os.path.join(pneumonia_path, '*.png')), pneumonia_train_len):
    shutil.copy(trainimg, 'Data_organized/train/pneumonia')


for testimg in itertools.islice(glob.iglob(os.path.join(covid_path, '*.png')), covid_train_len, covid_len):
    shutil.copy(testimg, 'Data_organized/test/covid')

for testimg in itertools.islice(glob.iglob(os.path.join(normal_path, '*.png')), normal_train_len, normal_len):
    shutil.copy(testimg, 'Data_organized/test/normal')

for testimg in itertools.islice(glob.iglob(os.path.join(pneumonia_path, '*.png')), pneumonia_train_len, pneumonia_len):
    shutil.copy(testimg, 'Data_organized/test/pneumonia')