import numpy as np 
import pandas as pd
import os
import matplotlib.pyplot as plt 
# Load the data
my_data = pd.read_csv('us-counties.csv', sep=',')


def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
    return allFiles

files = getListOfFiles("staggered_virus_and_weather_data")
first_file = files[0]

def getData(fileName):
    file_data = pd.read_csv(fileName, sep=',')
    del file_data["Station ID"]
    del file_data["Virus Date"]
    del file_data["Weather Date"]
    del file_data["(file)name"]
    del file_data["County"]
    del file_data["State"]
    del file_data["Deaths"]
    del file_data["Temp. Attributes"]
    del file_data["Rain Attributes"]
    growth = np.zeros((file_data.shape[0]-7, 1))
    for i in range(file_data.shape[0]-7):
        g = (file_data["Total Cases"].iloc[i+7] - file_data["Total Cases"].iloc[i])/file_data["Total Cases"].iloc[i+7]
        growth[i] = (round(g, 4))
    #print(growth.shape)
    #print(file_data.shape)
    file_data = file_data[:-7]
    #print(file_data.shape)
    file_data.insert(7, "Growth %", growth, True) 
    return file_data


final_data = getData(files[0])
for i in range(1,len(files)-1):
    file_name = files[i]
    temp_data = getData(file_name)
    final_data = pd.concat([final_data, temp_data])

#print(final_data.shape)
del final_data["Total Cases"]
final_data.to_csv("final.csv")


'''
plt.scatter(final_data["Temperature"], final_data["Growth %"])
plt.xlabel("Temp")
plt.ylabel("Growth")
plt.show()'''

#plt.hist(final_data["Growth %"], bins = 40)
#plt.show()