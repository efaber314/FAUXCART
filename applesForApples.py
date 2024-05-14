#This program cleans MERRA-2 10m monthly whole data to match
#ISD data. uses all time, not just daytime. outputs to a csv labeled
#for the station. one station at a time
import os
import pandas as pd
import MERRA2_Functions
import my_ISD_Functions
import netCDF4 as nc
import numpy as np


def parallelize(station_ID):
    # station_ID = input('Station ID: ')
    # station_ID = '40155099999'
    stationList, directory = my_ISD_Functions.getStationList(station_ID)
    # print(stationList)
    BIGdf_ISD = pd.DataFrame()
    for file in stationList:
        # print(file + '\n')
        print(file)
        df = pd.read_csv(directory + '/' + file, low_memory = False)
        df = my_ISD_Functions.CleanSpeed(df)
        BIGdf_ISD = BIGdf_ISD.append(df, ignore_index = True)

    # print(BIGdf['DATE'].iloc[1].dt.hour)
    # print(BIGdf['DATE'].iloc[1])
    # print(BIGdf_ISD.tail())
    BIGdf_ISD['hour'] = BIGdf_ISD['DATE'].dt.hour
    BIGdf_ISD['day'] = BIGdf_ISD['DATE'].dt.day
    BIGdf_ISD['month'] = BIGdf_ISD['DATE'].dt.month
    BIGdf_ISD['year'] = BIGdf_ISD['DATE'].dt.year
    BIGdf_ISD = BIGdf_ISD.rename(columns = {'Speed':'Speed_ISD'})
    lat = BIGdf_ISD['LATITUDE'].iloc[0]
    long = BIGdf_ISD['LONGITUDE'].iloc[0]
    # print(lat,long)
    print(BIGdf_ISD.tail())
    # BIGdf_ISD.to_csv('ISD.csv')

    BIGdf_MERRA2 = pd.DataFrame()
    os.chdir('/Users/emily/Documents/UMBC/Dr_LimaLab/Merra2_Wind/MERRA2_1hourly_whole_10m') #change directory to the external drive that has my data on it https://medium.com/analytics-vidhya/read-and-write-files-in-the-hard-disk-using-python-b89114fcd18f
    MERRA2directory = os.getcwd()
    filelist = MERRA2_Functions.filelist(MERRA2directory)
    print('getting MERRA-2 files...')
    for file in filelist:
        # print(MERRA2directory,file)
        myDirectory = MERRA2directory + '/' + str(file)
        # print(myDirectory)
        # print(file)
        ds = nc.Dataset(myDirectory.replace('._',''))
        # print(file[29:37])
        tempdf = pd.DataFrame()
        ws = MERRA2_Functions.wind_magnitude_10m(ds)
        ws = MERRA2_Functions.location_values(lat,long,ws)
        # print(ws)
        # print(len(ws))
        tempdf['speed_MERRA2'] = ws
        tempdf['year'] = int(file[-16:-12])
        # print(file[-16:-12])
        tempdf['month'] = int(file[-12:-10])
        # print(file[-12:-10])
        tempdf['day'] = int(file[-10:-8])
        # print(file[-10:-8])
        tempdf['hour'] = np.arange(0,24,1)
        # print(tempdf.head(1))
        BIGdf_MERRA2 = BIGdf_MERRA2.append(tempdf, ignore_index = True)

    print(BIGdf_MERRA2.head())
    # BIGdf_MERRA2.to_csv('MERRA2.csv')
    intersection_df = pd.merge(BIGdf_ISD, BIGdf_MERRA2, how = 'inner', on = ['hour', 'day', 'month', 'year'])
    # print(intersection_df.head())
    # print(intersection_df.tail())
    # print(intersection_df.columns)
    os.chdir(directory)
    #TODO: save this in the results directory if in ME and idk where if not in the ME
    print('intersection csv saved to ', os.getcwd())
    intersection_df.to_csv(directory + '/MERRA2_intersection_' + str(station_ID) + '.csv')
    return

