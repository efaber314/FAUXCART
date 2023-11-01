#histogram driver
#uses functions from ApplesToApplesFunctions.py
#Emily Faber
#Fall 2022

from ISDWind import ApplesToApplesFunctions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openpyxl
import datetime as dt
from pathlib import Path

'''UNBLOCK FOR MIDDLE EAST '''
Confirmed10MStations = pd.read_csv('ISDWind/MiddleEast/10m_confirmed_stations.csv', low_memory = False)
'''UNBLOCK FOR SOUTHWEST '''
# Confirmed10MStations = pd.read_csv('stations_southwest_20years_10mConfirmed.csv', low_memory = False)
#drop stations without 10 m confirmed
# Confirmed10MStations = Confirmed10MStations[Confirmed10MStations['WindTower ID'].notna()].reset_index()

# print(len(Confirmed10MStations['STATION_ID']))
# print(Confirmed10MStations['STATION_ID'])
# print(Confirmed10MStations.head())
col_names = ['Station_ID', 'begin date', 'end date', 'ISD total', 'MERRA-2 emissions_calculated','MERRA-2 emissions given','percent_diff_ISD_calculatedM2', 'percent diff ISD given M2', 'percent diff calculated M2 given M2', 'longitude', 'latitude', 'diurnalText', 'todays date','ISD_73_total','ISD_1_4_total','ISD_2_4_total','ISD_4_5_total','ISD_8_total','MERRA2_73_total', 'MERRA2_1_4_total', 'MERRA2_4_5_total','MERRA2_8_total', 'length of flux df', 'map used 1- topo, 2-ssm']

# start_dates = pd.date_range('2001-01-01','2020-12-01', freq='1M')-pd.offsets.MonthBegin(1)
# stop_dates = pd.date_range('2001-01-31','2020-12-31', freq='1M')
# #yearly for all years
# start_dates = pd.date_range('2001-01-01','2020-01-01', freq='1Y')-pd.offsets.YearBegin(1)
# stop_dates = pd.date_range('2020-12-31','2020-12-31', freq='1Y')
start_dates = [pd.to_datetime('2001-01-01')]
stop_dates = [pd.to_datetime('2020-12-31')]
print(start_dates)
print(stop_dates)
map = input('Which supply map - 1 for topographic, 2 for Sediment Supply ')
for i in range(0,len(Confirmed10MStations['Station_ID'])):
# for station_ID in Confirmed10MStations['Station_ID']:
    station_ID = Confirmed10MStations['Station_ID'][i]
    print(station_ID)
    ParentDirectory = '/Users/emily/Documents/UMBC/Dr_LimaLab/ISDWind/MiddleEast/'
    # ParentDirectory = '/Users/emily/Documents/UMBC/Dr_LimaLab/ISDWind/Southwest/Confirmed10M/'
    directory = ParentDirectory + str(station_ID) + '_DATA/' + 'MERRA2_intersection_' + str(station_ID) + '.csv'
    #for each month
    for j in range(0,len(start_dates)):
        begin_date = start_dates[j]
        end_date = stop_dates[j]
        print(begin_date)
        print(end_date)
        df = pd.read_csv(directory, low_memory = False)
        # print(df.head(20))
        df['DATE'] = pd.to_datetime(df['DATE'])
        df = df[df['DATE'] >= begin_date]
        df = df[df['DATE'] <= end_date]
        print(len(df))

        if len(df)< 2:
            data = [station_ID, str(begin_date),str(end_date),'Nan','Nan','Nan','Nan','Nan', 'Nan', 'Nan', 'Nan', 'All time', str(dt.date.today()),'nan','nan','nan','nan','nan','nan', 'nan', 'nan','nan',map]
            filename = ParentDirectory +'PercentDifference_files/'+'ME('+ str(begin_date)[0:10] + '-' + str(end_date)[0:10] + ')_withSizeBins_omega' + '.xlsx'
            path = Path(ParentDirectory +'PercentDifference_files/' +'ME('+ str(begin_date)[0:10] + '-' + str(end_date)[0:10] + ')_withSizeBins_omega' + '.xlsx')
            if path.is_file():
                #file exists, just append
                oldData = pd.read_excel(path, index_col = False)
                print('old data')
                print(oldData)
                newdata = pd.DataFrame([data], columns = col_names)
                print('newdata')
                print(newdata)
                updated_df = pd.concat([oldData,newdata], names = col_names, ignore_index = True)
                print('combined data')
                print(updated_df)
                updated_df.to_excel(path, index = False)
            else:
                #create file AND append
                newdata = pd.DataFrame([data], columns = col_names)
                print(newdata)
                newdata.to_excel(path, index = False)
            continue
        mylat = df['LATITUDE'].iloc[-1]
        mylong = df['LONGITUDE'].iloc[-1]
        print('(lat,long) ',mylat,mylong)
        # if df.empty == True:
            # print('Time range not in dataframe')
            # wb = openpyxl.load_workbook(ParentDirectory +'PercentDifference_files/PercentDifferences_MiddleEast' +'('+begin_date + '-' + end_date + ')' + '.csv')
            # ws = wb.active
            # ws.append([station_ID, begin_date,end_date,'No Measurements',mylong,mylat])
            # wb.save(ParentDirectory +'PercentDifference_files/PercentDifferences_MiddleEast' +'('+begin_date + '-' + end_date + ')' + '.csv')
            # wb.close()
            # continue

        ApplesToApplesFunctions.histogram(df, station_ID,begin_date,end_date,mylong,mylat, 'All Time',map)
'''
        daytime_df, nighttime_df = ApplesToApplesFunctions.diurnal_df(df)
        # print(daytime_df.head(30))
        # print(nighttime_df.head(30))
        # ApplesToApplesFunctions.Histogram_AllTime(daytime_df, station_ID, 'Daytime', begin_date, end_date)
        # ApplesToApplesFunctions.Histogram_AllTime(nighttime_df,station_ID, 'Night-time', begin_date, end_date)
        print('daytime')
        ApplesToApplesFunctions.histogram(daytime_df, station_ID,begin_date,end_date,mylong,mylat,'Daytime')
        print('nightime')
        ApplesToApplesFunctions.histogram(nighttime_df, station_ID,begin_date,end_date,mylong,mylat,'Night-time')
    '''        # ApplesToApplesFunctions.U_tvsR(df, begin_date,end_date)
