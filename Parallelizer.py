#driver for parallelizing ISD and MERRA-2 data in the southwest. Same functions
#in same order as in the Middle East.
#Emily Faber Spring 2023

import applesForApples
import pandas as pd
import datetime as dt
import os


def parallelize(list_of_stations,sdate, edate):
    # list_of_stations = input('Path to or CSV of STATION_ID, BEGIN_DATE, and END_DATE: ')
    df = pd.read_csv(list_of_stations)
    print('recieved ', len(df), ' stations to parallelize')
    correct = input('Is this correct? Y/N')
    if correct == 'Y' or correct == 'y':
        df['BEGIN_DATE'] = pd.to_datetime(df['BEGIN_DATE'])
        df['END_DATE'] = pd.to_datetime(df['END_DATE'])
        '''These dates are hard coded, I would like to make them command line arguments'''
        df.drop(df[df['END_DATE'] < dt.datetime(2012,6,1)].index, inplace = True) 
        df.drop(df[df['BEGIN_DATE']> dt.datetime(2012,6,30)].index, inplace = True)

        # df.drop(df[df['END_DATE'] < dt.datetime(sdate)].index, inplace = True) 
        # df.drop(df[df['BEGIN_DATE']> dt.datetime(edate)].index, inplace = True)

        station_IDs = df['STATION_ID'].tolist()
        for station_ID in station_IDs:
            print(station_ID) #station number being processed
            ParentDirectory_ISD = "ISDWind/"
            # ParentDirectory = "/Users/emily/Documents/UMBC/Dr_LimaLab/ISDWind/Southwest" # hardcodded to the correct directory
            FolderName = str(station_ID) + '_DATA'
            directory = os.path.join(ParentDirectory_ISD, FolderName)
            dirlist = os.listdir(directory) #list of all files in directory
            processedString = 'intersection' #if this station has already been procesesed the directory will have a filename containing this
            flag = [i for i in dirlist if processedString in i] #search filenames for processedString. Returns list of filenames with the processedString
            if len(flag) == 0: #if no filenames with processed string
                applesForApples.parallelize(station_ID, sdate, edate) #process this data
                # print( station_ID + ' needs processing')
            else: #otherwise
                print(station_ID + ' Already processed!') #tell user this is processed and move on
    else:
        exit()
    return
