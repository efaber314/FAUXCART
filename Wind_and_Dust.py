#Wind AND dust analysis code
#takes in FAUCART generated co-located files

'''
WIND ANALYSIS:
1.histograms (after combine subhourly)
2.monthly means (240)
3.monthly means (12)
4.one:one plots
5.bubble plots

DUST ANALYSIS
6.20-year emission block plots
7.dust emission bar and line
8.emission change map
9.Monthly means (240)
10.Monthly means (12)

STATISTICAL ANALYSIS
11.ztest scores

''' 
import ApplesToApples
import pandas as pd
import os
import numpy as np

list_of_stations = '/Users/emily/Documents/UMBC/Dr_LimaLab/ISDWind/MiddleEast/10m_confirmed_stations.csv'
MasterList = pd.read_csv(list_of_stations)
stationIDs = MasterList['Station_ID']

for station_ID in stationIDs:
    # df = pd.read_csv('/Users/emily/Documents/UMBC/Dr_LimaLab/FAUXCART/FAUXCART SSM Run#1/' + str(station_ID) + '_FAUXCART_SSM.csv', low_memory=False)
    # ApplesToApples.taylorDiagram(df)
    df = pd.read_csv('/Users/emily/Documents/UMBC/Dr_LimaLab/FAUXCART/FAUXCART_Run1/' + str(station_ID) + '_FAUCART.csv', low_memory=False)
    # print(df.head())
    # ApplesToApples.Zscore(df)
    # df_hist = ApplesToApples.combineSubHourly(df)
    # ApplesToApples.Histogram_AllTime(df_hist, station_ID,'All Time')
    # ApplesToApples.allYearsPlot(df, station_ID, 'Speed_ISD', 'OG_winds_correct_box')
    # ApplesToApples.MonthlyMeans12(df, station_ID, 'Speed_ISD', 'OG_winds_correct_box', 'Wind Speed')
    # App|lesToApples.AverageScatter(df,station_ID)
    ApplesToApples.allYearsPlot(df,station_ID, 'ISD_flux', 'Recalculated_flux_correct_box')
    # ApplesToApples.MonthlyMeans12(df, station_ID, 'ISD_flux', 'Recalculated_flux_correct_box', 'Dust Flux SSM')


# #1.histograms (after combine subhourly)
# station_ID = '40400099999'
# df = pd.read_csv('/Users/emily/Documents/UMBC/Dr_LimaLab/FAUXCART/FAUXCART_Run1/40400099999_FAUCART.csv', low_memory=False)
# df_hist = ApplesToApples.combineSubHourly(df)
# ApplesToApples.Histogram_AllTime(df_hist, station_ID,'All Time')

#2.monthly means (240)
# ISDmeans,m2means,numpoints = ApplesToApples.allYearsPlot(df_hist,station_ID, 'Speed_ISD', 'OG_winds_correct_box')
# print(ISDmeans,m2means,numpoints)

# 3.monthly means (12)
# ApplesToApples.MonthlyMeans12(df_hist, station_ID, 'Speed_ISD', 'OG_winds_correct_box')

#4. one:one plots
# ApplesToApples.AverageScatter(df,station_ID)

#5.bubble plots
#| station ID | lat | long | Summer Average ISD | ...other seasons ... | Summer variance ISD | Summer Variance M2 | ... other seasons ... |
# BubblePlotDf = pd.DataFrame(columns=['StationID', 'Lat', 'Long', 'SummerAvgISD','FallAvgISD','WinterAvgISD','SpringAvgISD','SummerAvgM2','FallAvgM2','WinterAvgM2','SpringAvgM2','STD_ISD_Summer','STD_ISD_Fall','STD_ISD_Winter', 'STD_ISD_Spring','STD_M2_Summer','STD_M2_Fall','STD_M2_Winter','STD_M2_Spring'])
# print(BubblePlotDf)
# for file in os.scandir('/Users/emily/Documents/UMBC/Dr_LimaLab/FAUXCART/FAUXCART SSM Run#1/'):
#     if '.csv' in str(file.name):
#         df = pd.read_csv('/Users/emily/Documents/UMBC/Dr_LimaLab/FAUXCART/FAUXCART SSM Run#1/' + str(file.name), low_memory=False)
#         df_seasons = ApplesToApples.Seasons(df)
#         myRow = []
#         myRow = [df['STATION'][0],df['LATITUDE'][0],df['LONGITUDE'][0],df.loc[df['Season'] == 'Summer']['Speed_ISD'].mean(),\
#                       df.loc[df['Season'] == 'Fall']['Speed_ISD'].mean(),df.loc[df['Season'] == 'Winter']['Speed_ISD'].mean(),\
#                         df.loc[df['Season'] == 'Spring']['Speed_ISD'].mean(),df.loc[df['Season'] == 'Summer']['OG_winds_correct_box'].mean()\
#                             ,df.loc[df['Season'] == 'Fall']['OG_winds_correct_box'].mean(),df.loc[df['Season'] == 'Winter']['OG_winds_correct_box'].mean(),\
#                                 df.loc[df['Season'] == 'Spring']['OG_winds_correct_box'].mean(),np.nanstd(df.loc[df['Season'] == 'Summer']['Speed_ISD']),\
#                                     np.nanstd(df.loc[df['Season'] == 'Fall']['Speed_ISD']),np.nanstd(df.loc[df['Season'] == 'Winter']['Speed_ISD']),\
#                                         np.nanstd(df.loc[df['Season'] == 'Spring']['Speed_ISD']),np.nanstd(df.loc[df['Season'] == 'Summer']['OG_winds_correct_box'])\
#                                             ,np.nanstd(df.loc[df['Season'] == 'Fall']['OG_winds_correct_box']),np.nanstd(df.loc[df['Season'] == 'Winter']['OG_winds_correct_box']),\
#                                                 np.nanstd(df.loc[df['Season'] == 'Spring']['OG_winds_correct_box'])]
#         BubblePlotDf.loc[len(BubblePlotDf)] = myRow
#         print(BubblePlotDf.tail())
# ApplesToApples.SeasonMapPlot(BubblePlotDf['WinterAvgISD']-BubblePlotDf['WinterAvgM2'], BubblePlotDf['SummerAvgISD']-BubblePlotDf['SummerAvgM2'], BubblePlotDf['SpringAvgISD']-BubblePlotDf['SpringAvgM2'], BubblePlotDf['FallAvgISD']-BubblePlotDf['FallAvgM2'],\
#                              np.sqrt(np.abs(BubblePlotDf['STD_ISD_Winter']**2 - BubblePlotDf['STD_M2_Winter']**2)),\
#                              np.sqrt(np.abs(BubblePlotDf['STD_ISD_Summer']**2 - BubblePlotDf['STD_M2_Summer']**2)),\
#                              np.sqrt(np.abs(BubblePlotDf['STD_ISD_Spring']**2 - BubblePlotDf['STD_M2_Spring']**2)),\
#                              np.sqrt(np.abs(BubblePlotDf['STD_ISD_Fall']**2 - BubblePlotDf['STD_M2_Fall']**2)),\
#                              BubblePlotDf['Lat'],BubblePlotDf['Long'], 'All Time', 'bwr', -4, 4)

#6.20-year emission block plots
# ApplesToApples.BlockPlot('/Users/emily/Documents/UMBC/Dr_LimaLab/FAUXCART/FAUXCART SSM Run#1/')
# ApplesToApples.BlockPlot('/Users/emily/Documents/UMBC/Dr_LimaLab/FAUXCART/FAUXCART_Run1/', 'mass')
# ApplesToApples.BlockPlot('/Users/emily/Documents/UMBC/Dr_LimaLab/FAUXCART/FAUXCART_Run1/', 'percent')


#7.dust emission bar and line
#/Users/emily/Documents/UMBC/Dr_LimaLab/FAUXCART/FAUXCART_Run1/
# MassDifferenceTOPO, percentDifferenceTOPO = ApplesToApples.DustImpactPlot('/Users/emily/Documents/UMBC/Dr_LimaLab/FAUXCART/FAUXCART_Run1/', '_FAUCART.csv')
# print(percentDifferenceTOPO)
# MassDifferenceTOPO, PercentDifferenceTOPO = ApplesToApples.DustImpactPlot('/Users/emily/Documents/UMBC/Dr_LimaLab/FAUXCART/OG_flux_correct_box/', '_og_flux_correct_box_added.csv')
# MassDifferenceSSM = ApplesToApples.DustImpactPlot('/Users/emily/Documents/UMBC/Dr_LimaLab/FAUXCART/FAUXCART SSM Run#1/', '_FAUXCART_SSM.csv')
# print(MassDifference)

#double impact plot 
# ApplesToApples.ConfusingPlot(MassDifferenceTOPO)

#8.emission change map
# ApplesToApples.PercentChangeMap(np.array(MassDifferenceSSM) - np.array(MassDifferenceTOPO))
# ApplesToApples.PercentChangeMap(np.array(PercentDifferenceTOPO))
# ApplesToApples.PercentChangeMap(np.array(MassDifferenceSSM))

#9. Monthly means (240)
# ApplesToApples.allYearsPlot(df,station_ID, 'ISD_flux', 'Recalculated_flux_correct_box')

#10. Monthly means (12)
# ApplesToApples.MonthlyMeans12(df, station_ID, 'ISD_flux', 'Recalculated_flux_correct_box')

#11. Z-test scores 
# ApplesToApples.Zscore(df)
