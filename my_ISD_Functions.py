#Python functions for ISD station data processing

import os
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
# import cartopy.crs as ccrs, feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
from pathlib import Path



'''It is likely that stations of size 244 kb are actually empty files
in some cases its also imperative that files are imported
in chronological order, so this does that as well. returns the sorted file list
and the directory for this station'''
def getStationList(Station_ID):
    # correct = 'n'
    # while correct == 'n' or correct == 'N':
        # Station_ID = input('Station ID: ')
        # ParentDirectory = "/Users/emily/Documents/UMBC/Dr_LimaLab/ISDWind/MiddleEast"
        # where = str(input('Where is the station? 1 = Middle East, 2 = Sahara, 3 = American Southwest \n'))
        # if where == '1':
        #     ParentDirectory = "/Users/emily/Documents/UMBC/Dr_LimaLab/ISDWind/MiddleEast"
        # elif where == '2':
        #     ParentDirectory = "/Users/emily/Documents/UMBC/Dr_LimaLab/ISDWind/Sahara"
        # elif where == '3':
    ParentDirectory = "ISDWind/"
    FolderName = str(Station_ID) + '_DATA'
    directory = os.path.join(ParentDirectory, FolderName)
        # correct = input('Is this the correct directory? y/n \n' + directory + '\n')

    filelist = []
    for entry in os.scandir(directory):
        statinfo = os.stat(entry)
        if (statinfo.st_size != 244) and not entry.name.endswith(str(Station_ID)+'.csv') and entry.name.endswith('.csv'):
            filelist.append(str(entry.name))
    return sorted(filelist), directory
#returns a data frame of the daytime wind speed values. Daytime is hard coded
#for the middle east - Northern hemisphere SeasonSTD_ISD
def DayDataFrame(df):
    daytime = 15
    nightime = 3
    Daydf = df[(df['DATE'].dt.hour <= daytime) & (df['DATE'].dt.hour >= nightime)]
    return Daydf

#retuns a data frame with the speed column cleaned and scaled using the scaling
#factor. 0000 is a true calm, and is replaced as such. 9999 is missing, and
#is replaced with nans before dropping those rows from the data frame
def CleanSpeed(df):
    #Break up the wind (WND) column into columns for each piece of info stored
    df[['AngleFromNorth', 'QualityCode', 'TypeCode','Speed','SpeedQualityCode']] = df.WND.str.split(',', expand = True)
    # df['AngleFromNorth'] = df['AngleFromNorth'].str.lstrip('0')
    for i in range(0, len(df['Speed'])):
        if df['Speed'][i] == '0000':
            df['Speed'][i] = '0'
        else:
            df['Speed'][i] = df['Speed'][i].lstrip('0')
    #convert to floats
    df['Speed'] = pd.to_numeric(df['Speed'], downcast = 'float')
    #Convert date and time to datetime object
    df['DATE'] = pd.to_datetime(df['DATE'], format ='%Y-%m-%dT%H:%M:%S')
    #drop filled-values
    df['Speed'] = df['Speed'].replace(9999.0,np.nan)
    # df['AngleFromNorth'] = df['AngleFromNorth'].replace(999,np.nan)
    df.dropna(subset = ['Speed'], inplace = True )
    df['Speed'] = df['Speed']/10

    return df
#retuns the data frame with a column assigning the season
def CleanSeason(df):
    df['Season'] = df['DATE']
    df.loc[df['DATE'].dt.month == 1, 'Season'] = 'Winter'
    df.loc[df['DATE'].dt.month == 2, 'Season'] = 'Winter'
    df.loc[df['DATE'].dt.month == 12, 'Season'] = 'Winter'
    df.loc[df['DATE'].dt.month == 11, 'Season'] = 'Fall'
    df.loc[df['DATE'].dt.month == 10, 'Season'] = 'Fall'
    df.loc[df['DATE'].dt.month == 9, 'Season'] = 'Fall'
    df.loc[df['DATE'].dt.month == 8, 'Season'] = 'Summer'
    df.loc[df['DATE'].dt.month == 7, 'Season'] = 'Summer'
    df.loc[df['DATE'].dt.month == 6, 'Season'] = 'Summer'
    df.loc[df['DATE'].dt.month == 5, 'Season'] = 'Spring'
    df.loc[df['DATE'].dt.month == 4, 'Season'] = 'Spring'
    df.loc[df['DATE'].dt.month == 3, 'Season'] = 'Spring'

    return df

#using a list of the stations and the directory that the data is stored in,
# creates a data frame
def makeDF(station_list, directory):
    BIGdf_ISD = pd.DataFrame()
    for file in station_list:
        df = pd.read_csv(str(directory) + '/' + str(file), low_memory = False)
        df = CleanSpeed(df)
        BIGdf_ISD = BIGdf_ISD.append(df, ignore_index = True)

    return BIGdf_ISD

#makes the bubble plot
def SeasonMapPlot(winterAVG,summerAVG,springAVG,fallAVG,winterSTD,SummerSTD,SpringSTD,FallSTD,lat,long,titleString, color, min,max, station_IDS):
    crs = ccrs.PlateCarree()
    fig, ax = plt.subplots(2,2, subplot_kw = {'projection': crs}, figsize = (13,12))
    '''middle east extent'''
    # ax[0,0].set_extent([30, 60, 10, 40], ccrs.PlateCarree())
    '''southwest extent'''
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')
    ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor='black')
    # ax.add_feature(cf.BOARDERS)
    ax[0,0].set_extent([-126, -93, 25, 50], ccrs.PlateCarree())

    # ax[0,0].states(resolution = '10m')
    ax[0,0].coastlines(resolution='10m')
    winterScatter = ax[0,0].scatter(np.array(long), np.array(lat), s = winterSTD*100, c = winterAVG, cmap = color, vmin = min, vmax = max, alpha = .7, edgecolors = 'black')
    ax[0,0].title.set_text(titleString+' Winter 10m Wind Speed')
    # AVGlegend = ax[0,0].legend(*winterScatter.legend_elements(num = 6), loc = 'lower left', title = 'Average 10m \nWind Speed [m/s]', bbox_to_anchor = (-.35,-.2), frameon = False)
    # ax[0,0].add_artist(AVGlegend)
    # extraScatter = ax[0,0].scatter(np.array(np.append(long,long)), np.array(np.append(lat,lat)), s = np.append(winterSTD*100,SummerSTD*100), c = np.append(winterAVG, summerAVG), cmap = color, vmin = min, vmax = max, alpha = .7)
    handles, labels = winterScatter.legend_elements(prop = 'sizes', num = 6)
    labels = np.array(labels)
    labels = [int(''.join(char for char in string if char.isdigit())) for string in labels]
    labels = [labels/100 for labels in labels]
    # STDlegend = ax[0,0].legend(handles, labels, loc = 'upper right', title = 'STD [m/s]', bbox_to_anchor = (0,-.3))

    ax[0,1].set_extent([-126, -93, 25, 50], ccrs.PlateCarree())
    # ax[0,1].states(resolution = '10m')
    ax[0,1].coastlines(resolution='10m')

    SpringScatter = ax[0,1].scatter(np.array(long), np.array(lat), s = SpringSTD*100, c = springAVG, cmap = color, vmin = min, vmax = max, alpha = .7, edgecolors = 'black')
    ax[0,1].title.set_text(titleString+' Spring 10m Wind Speed')
    # AVGlegend = ax[0,1].legend(*SpringScatter.legend_elements(num = 6), loc = 'lower left', title = 'Average 10m \nWind Speed [m/s]', bbox_to_anchor = (-.25,0))
    # ax[0,1].add_artist(AVGlegend)
    handles, labels = SpringScatter.legend_elements(prop = 'sizes', num = 6)
    labels = np.array(labels)
    labels = [int(''.join(char for char in string if char.isdigit())) for string in labels]
    labels = [labels/100 for labels in labels]
    # STDlegend = ax[0,1].legend(handles, labels, loc = 'upper right', title = 'STD [m/s]', bbox_to_anchor = (1.15,1))

    ax[1,0].set_extent([-126, -93, 25, 50], ccrs.PlateCarree())
    ax[1,0].coastlines(resolution='10m')

    SummerScatter = ax[1,0].scatter(np.array(long), np.array(lat), s = SummerSTD*100, c = summerAVG, cmap = color, vmin = min, vmax = max, alpha = .7, edgecolors = 'black')
    ax[1,0].title.set_text(titleString+' Summer 10m Wind Speed')
    # AVGlegend = ax[1,0].legend(*SummerScatter.legend_elements(num = 6), loc = 'lower left', title = 'Average 10m \nWind Speed [m/s]', bbox_to_anchor = (-.25,0))
    # ax[1,0].add_artist(AVGlegend)
    handles, labels = SummerScatter.legend_elements(prop = 'sizes', num = 6)
    labels = np.array(labels)
    labels = [int(''.join(char for char in string if char.isdigit())) for string in labels]
    labels = [labels/100 for labels in labels]
    STDlegend = ax[1,0].legend(handles, labels, loc = 'upper right', title = 'STD [m/s]', bbox_to_anchor = (-.05,1), frameon = False)

    ax[1,1].set_extent([-126, -93, 25, 50], ccrs.PlateCarree())
    ax[1,1].coastlines(resolution='10m')

    FallScatter = ax[1,1].scatter(np.array(long), np.array(lat), s = FallSTD*100, c = fallAVG, cmap = color, vmin = min, vmax = max, alpha = .7, edgecolors = 'black')
    ax[1,1].title.set_text(titleString+' Fall 10m Wind Speed')
    AVGlegend = ax[1,1].legend(*winterScatter.legend_elements(num = 8), loc = 'lower left', title = 'Average 10m \nWind Speed [m/s]', bbox_to_anchor = (-1.55,1.2))
    ax[1,1].add_artist(AVGlegend)
    '''
    handles, labels = winterScatter.legend_elements(prop = 'sizes', num = 6)
    labels = np.array(labels)
    labels = [int(''.join(char for char in string if char.isdigit())) for string in labels]
    labels = [labels/100 for labels in labels]
    '''
    # STDlegend = ax[1,1].legend(handles, labels, loc = 'upper right', title = 'STD [m/s]', bbox_to_anchor = (1.15,1))
    # lat = np.array(lat)
    # long = np.array(long)
    # for i, ann in enumerate(station_IDS):
    #     plt.annotate(ann, long[i], lat[i])
    # plt.show()
    # print(os.getcwd())
    plt.savefig('TimeCorrelatedBubblePlot_Outline_OnlyAvglegend_SOUTHWEST' + titleString + '.png', dpi = 300)
    return

#plots the seasonal averages for all confirmed 10 meter stations in the middle East
#for each confirmed station, it averages the wind speed for the last 20 years and
#plots it. with 31 stations it should have 31*4 points
def Scatter_Plots_All_Station_Averages():
    Confirmed10MStations = pd.read_csv('ISDWind/MiddleEast/10m_confirmed_stations.csv', low_memory = False)
    # print(Confirmed10MStations)
    # lat = np.array(Confirmed10MStations['Lat'])
    # long = np.array(Confirmed10MStations['Long'])
    station_IDS = np.array(Confirmed10MStations['Station_ID'])
    ParentDirectory = '/Users/emily/Documents/UMBC/Dr_LimaLab/ISDWind/MiddleEast/'
    i = 1
    for station_ID in station_IDS:
        BubbleDf = pd.DataFrame()
        directory = ParentDirectory + str(station_ID) + '_DATA/' + 'MERRA2_intersection_' + str(station_ID) + '.csv'
        tempDF = pd.read_csv(directory, low_memory = False)
        tempDF['DATE'] = pd.to_datetime(tempDF['DATE'])
        # print(len(tempDF['DATE']))
        # print(tempDF.info())
        tempDF = CleanSeason(tempDF)
        # print(tempDF.head())
        SeasonMeans_ISD = tempDF.groupby(tempDF['Season'])['Speed_ISD'].mean()
        SeasonSTD_ISD = tempDF.groupby(tempDF['Season'])['Speed_ISD'].std()
        SeasonMeans_MERRA2 = tempDF.groupby(tempDF['Season'])['speed_MERRA2'].mean()
        SeasonSTD_MERRA2 = tempDF.groupby(tempDF['Season'])['speed_MERRA2'].std()
        BubbleDf.loc[i, 'WinterSTD_ISD'] = SeasonSTD_ISD[3]
        BubbleDf.loc[i, 'SummerSTD_ISD'] = SeasonSTD_ISD[2]
        BubbleDf.loc[i, 'SpringSTD_ISD'] = SeasonSTD_ISD[1]
        BubbleDf.loc[i, 'FallSTD_ISD'] = SeasonSTD_ISD[0]
        BubbleDf.loc[i, 'WinterMean_ISD'] = SeasonMeans_ISD[3]
        BubbleDf.loc[i, 'SummerMean_ISD'] = SeasonMeans_ISD[2]
        BubbleDf.loc[i, 'SpringMean_ISD'] = SeasonMeans_ISD[1]
        BubbleDf.loc[i, 'FallMean_ISD'] = SeasonMeans_ISD[0]
        BubbleDf.loc[i, 'WinterSTD_MERRA2'] = SeasonSTD_MERRA2[3]
        BubbleDf.loc[i, 'SummerSTD_MERRA2'] = SeasonSTD_MERRA2[2]
        BubbleDf.loc[i, 'SpringSTD_MERRA2'] = SeasonSTD_MERRA2[1]
        BubbleDf.loc[i, 'FallSTD_MERRA2'] = SeasonSTD_MERRA2[0]
        BubbleDf.loc[i, 'WinterMean_MERRA2'] = SeasonMeans_MERRA2[3]
        BubbleDf.loc[i, 'SummerMean_MERRA2'] = SeasonMeans_MERRA2[2]
        BubbleDf.loc[i, 'SpringMean_MERRA2'] = SeasonMeans_MERRA2[1]
        BubbleDf.loc[i, 'FallMean_MERRA2'] = SeasonMeans_MERRA2[0]
        # print(BubbleDf.head())
        i += 1
        # BubbleDf['Lat'] = lat
        # BubbleDf['Long'] = long
        # BubbleDf['Station_ID'] = station_IDS
        # print(BubbleDf.keys())
        x = np.linspace(0,10,1000)
        plt.scatter(BubbleDf['WinterMean_MERRA2'], BubbleDf['WinterMean_ISD'], label = 'Winter Average', marker = 'P', alpha = .7, s = 14)
        plt.scatter(BubbleDf['SpringMean_MERRA2'], BubbleDf['SpringMean_ISD'], label = 'Spring Average', marker = 'x', alpha = .7, s = 14)
        plt.scatter(BubbleDf['SummerMean_MERRA2'], BubbleDf['SummerMean_ISD'], label = 'Summer Average', marker = 'p', alpha = .7, s = 14)
        plt.scatter(BubbleDf['FallMean_MERRA2'], BubbleDf['FallMean_ISD'], label = 'Fall Average', marker = '*', alpha = .7, s = 14)
        plt.plot(x,x, label = '1:1')
        plt.show()
        # plt.savefig('/Users/emily/Documents/UMBC/Dr_LimaLab/ISDWind/MiddleEast/SeasonalScatterPlots_outputs/' + str(station_ID) + 'SeasonalScatterPlot')
    # print(BubbleDf)
    # x = np.linspace(0,10,1000)
    # plt.scatter(MERRA2_Data['WinterAVG'], ISD_Data['WinterAverage'], label = 'Winter Average', marker = 'P', alpha = .7)
    # plt.scatter(MERRA2_Data['SpringAVG'], ISD_Data['SpringAverage'], label = 'Spring Average', marker = 'x', alpha = .7)
    # plt.scatter(MERRA2_Data['SummerAVG'], ISD_Data['SummerAverage'], label = 'Summer Average', marker = 'p', alpha = .7)
    # plt.scatter(MERRA2_Data['FallAVG'], ISD_Data['FallAverage'], label = 'Fall Average', marker = '*', alpha = .7)
    # plt.plot(x,x, label = '1:1')
    return

def Single_station_scatter(ParentDirectory, Confirmed10MStations):
    BubbleDf = pd.DataFrame()
    i = 1
    j = 0
    # ParentDirectory = '/Users/emily/Documents/UMBC/Dr_LimaLab/ISDWind/MiddleEast/'
    # station_ID = input('Which station ID: ')
    zscore_output = pd.DataFrame(columns = ['Station_ID','ISD W', 'ISD Su', 'ISD Sp', 'ISD F', 'M2 W', 'M2 Su', 'M2 Sp', 'M2 F'])
    # Confirmed10MStations = pd.read_csv('ISDWind/MiddleEast/10m_confirmed_stations_copy copy.csv', low_memory = False)
    for station_ID in Confirmed10MStations:
        # station_ID = '40356099999'
        directory = ParentDirectory + str(station_ID) + '_DATA/' + 'MERRA2_intersection_' + str(station_ID) + '.csv'
        tempDF = pd.read_csv(directory, low_memory = False)
        tempDF['DATE'] = pd.to_datetime(tempDF['DATE'])
        tempDF = CleanSeason(tempDF)
        # print(tempDF.head())
        SeasonMeans_ISD = tempDF.groupby(['year', 'Season'])['Speed_ISD'].mean()
        SeasonMeans_ISD = SeasonMeans_ISD.reset_index()
        SeasonSTD_ISD = tempDF.groupby(['Season', 'year'])['Speed_ISD'].std()
        SeasonSTD_ISD = SeasonSTD_ISD.reset_index()

        SeasonMeans_MERRA2 = tempDF.groupby(['Season', 'year'])['speed_MERRA2'].mean()
        SeasonMeans_MERRA2 = SeasonMeans_MERRA2.reset_index()
        SeasonSTD_MERRA2 = tempDF.groupby(['Season', 'year'])['speed_MERRA2'].std()
        SeasonSTD_MERRA2 = SeasonSTD_MERRA2.reset_index()

        # PlotSTD = (SeasonSTD_MERRA2['speed_MERRA2'].mean() + SeasonSTD_ISD['Speed_ISD'].mean())/2

        # print('seasonSTD_MERRA2', SeasonSTD_MERRA2['speed_MERRA2'].mean())
        # print(SeasonMeans_ISD[SeasonMeans_ISD['Season'] == 'Fall']['Speed_ISD'])

        x = np.linspace(0,10,1000)
        plt.figure(figsize = (7,7))
        plt.ylim(0,7)
        plt.xlim(0,7)
        plt.scatter(np.array(SeasonMeans_ISD[SeasonMeans_ISD['Season'] == 'Winter']['Speed_ISD']), np.array(SeasonMeans_MERRA2[SeasonMeans_MERRA2['Season'] == 'Winter']['speed_MERRA2']), label = 'Winter Average', marker = '*', alpha = .9, s = 35)
        plt.scatter(np.array(SeasonMeans_ISD[SeasonMeans_ISD['Season'] == 'Summer']['Speed_ISD']), np.array(SeasonMeans_MERRA2[SeasonMeans_MERRA2['Season'] == 'Summer']['speed_MERRA2']), label = 'Summer Average', marker = 'x', alpha = .9, s = 35)
        plt.scatter(np.array(SeasonMeans_ISD[SeasonMeans_ISD['Season'] == 'Spring']['Speed_ISD']), np.array(SeasonMeans_MERRA2[SeasonMeans_MERRA2['Season'] == 'Spring']['speed_MERRA2']), label = 'Spring Average', marker = 'p', alpha = .9, s = 35)
        plt.scatter(np.array(SeasonMeans_ISD[SeasonMeans_ISD['Season'] == 'Fall']['Speed_ISD']), np.array(SeasonMeans_MERRA2[SeasonMeans_MERRA2['Season'] == 'Fall']['speed_MERRA2']), label = 'Fall Average', marker = 'P', alpha = .9, s = 35)
        plt.plot(x,x, label = '1:1')
        # plt.fill_between(x, (x-(PlotSTD/2)), (x+(PlotSTD/2)), color='black', alpha=.1, label = 'Average Standard Deviation of Wind Speeds')
        Zscore_ISD_winter = np.array(SeasonMeans_ISD[SeasonMeans_ISD['Season'] == 'Winter']['Speed_ISD']).std()/np.sqrt(len(np.array(SeasonMeans_ISD[SeasonMeans_ISD['Season'] == 'Winter']['Speed_ISD'])))
        Zscore_ISD_summer = np.array(SeasonMeans_ISD[SeasonMeans_ISD['Season'] == 'Summer']['Speed_ISD']).std()/np.sqrt(len(np.array(SeasonMeans_ISD[SeasonMeans_ISD['Season'] == 'Summer']['Speed_ISD'])))
        Zscore_ISD_spring = np.array(SeasonMeans_ISD[SeasonMeans_ISD['Season'] == 'Spring']['Speed_ISD']).std()/np.sqrt(len(np.array(SeasonMeans_ISD[SeasonMeans_ISD['Season'] == 'Spring']['Speed_ISD'])))
        Zscore_ISD_fall = np.array(SeasonMeans_ISD[SeasonMeans_ISD['Season'] == 'Fall']['Speed_ISD']).std()/np.sqrt(len(np.array(SeasonMeans_ISD[SeasonMeans_ISD['Season'] == 'Fall']['Speed_ISD'])))

        Zscore_M2_winter = np.array(SeasonMeans_MERRA2[SeasonMeans_MERRA2['Season'] == 'Winter']['speed_MERRA2']).std()/np.sqrt(len(np.array(SeasonMeans_MERRA2[SeasonMeans_MERRA2['Season'] == 'Winter']['speed_MERRA2'])))
        Zscore_M2_summer = np.array(SeasonMeans_MERRA2[SeasonMeans_MERRA2['Season'] == 'Summer']['speed_MERRA2']).std()/np.sqrt(len(np.array(SeasonMeans_MERRA2[SeasonMeans_MERRA2['Season'] == 'Summer']['speed_MERRA2'])))
        Zscore_M2_spring = np.array(SeasonMeans_MERRA2[SeasonMeans_MERRA2['Season'] == 'Spring']['speed_MERRA2']).std()/np.sqrt(len(np.array(SeasonMeans_MERRA2[SeasonMeans_MERRA2['Season'] == 'Spring']['speed_MERRA2'])))
        Zscore_M2_fall = np.array(SeasonMeans_MERRA2[SeasonMeans_MERRA2['Season'] == 'Fall']['speed_MERRA2']).std()/np.sqrt(len(np.array(SeasonMeans_MERRA2[SeasonMeans_MERRA2['Season'] == 'Fall']['speed_MERRA2'])))
        #save z-scores to output

        zscore_output.loc[j] = [station_ID,Zscore_ISD_winter,Zscore_ISD_summer,Zscore_ISD_spring,Zscore_ISD_fall,Zscore_M2_winter,Zscore_M2_summer,Zscore_M2_spring,Zscore_M2_fall]
        j += 1

        #error bar calculations
        W_ISD_std = np.array(SeasonMeans_ISD[SeasonMeans_ISD['Season'] == 'Winter']['Speed_ISD']).std()
        Su_ISD_std = np.array(SeasonMeans_ISD[SeasonMeans_ISD['Season'] == 'Summer']['Speed_ISD']).std()
        Sp_ISD_std = np.array(SeasonMeans_ISD[SeasonMeans_ISD['Season'] == 'Spring']['Speed_ISD']).std()
        F_ISD_std = np.array(SeasonMeans_ISD[SeasonMeans_ISD['Season'] == 'Fall']['Speed_ISD']).std()

        W_M2_std = np.array(SeasonMeans_MERRA2[SeasonMeans_MERRA2['Season'] == 'Winter']['speed_MERRA2']).std()
        Su_M2_std = np.array(SeasonMeans_MERRA2[SeasonMeans_MERRA2['Season'] == 'Summer']['speed_MERRA2']).std()
        Sp_M2_std = np.array(SeasonMeans_MERRA2[SeasonMeans_MERRA2['Season'] == 'Spring']['speed_MERRA2']).std()
        F_M2_std = np.array(SeasonMeans_MERRA2[SeasonMeans_MERRA2['Season'] == 'Fall']['speed_MERRA2']).std()
        # Zscore_Path = Path('/Users/emily/Documents/UMBC/Dr_LimaLab/ISDWind/MiddleEast/SeasonalScatterPlots_outputs/zscore_outputs_' + str(station_ID) + '.csv')
        # col_names = ['ISD W', 'ISD Su', 'ISD Sp', 'ISD F', 'M2 W', 'M2 Su', 'M2 Sp', 'M2 F']
        # if Zscore_Path.is_file():
        #     oldData = pd.read_csv(Zscore_Path, index_col = False)
        #     newdata = pd.DataFrame(columns = col_names)
        #     newdata['ISD W'] = Zscore_ISD_winter
        #     newdata['ISD Su'] = Zscore_ISD_summer
        #     newdata['ISD Sp'] = Zscore_ISD_spring
        #     newdata['ISD F'] = Zscore_ISD_fall
        #     newdata['M2 W'] = Zscore_M2_winter
        #     newdata['M2 Su'] = Zscore_M2_summer
        #     newdata['M2 Sp'] = Zscore_M2_spring
        #     newdata['M2 F'] = Zscore_M2_fall
        #     updated_df = pd.concat([oldData,newdata], names = col_names, ignore_index = True)
        #     updated_df.to_csv(Zscore_Path, index = False)
        # else:
        #     newdata = pd.DataFrame(columns = col_names)
        #     newdata['ISD W'] = Zscore_ISD_winter
        #     newdata['ISD Su'] = Zscore_ISD_summer
        #     newdata['ISD Sp'] = Zscore_ISD_spring
        #     newdata['ISD F'] = Zscore_ISD_fall
        #     newdata['M2 W'] = Zscore_M2_winter
        #     newdata['M2 Su'] = Zscore_M2_summer
        #     newdata['M2 Sp'] = Zscore_M2_spring
        #     newdata['M2 F'] = Zscore_M2_fall
        #
        #     newdata.to_csv(Zscore_Path, index = False)
        #https://pythonprogramminglanguage.com/kmeans-clustering-centroid/
        K = 1
        distances = []
        SeasonMeans_ISD = SeasonMeans_ISD.fillna(0)
        SeasonMeans_MERRA2 = SeasonMeans_MERRA2.fillna(0)
        #winter
        kmeans_model = KMeans(n_clusters = K).fit(np.array(list(zip(np.array(SeasonMeans_ISD[SeasonMeans_ISD['Season'] == 'Winter']['Speed_ISD']),np.array(SeasonMeans_MERRA2[SeasonMeans_MERRA2['Season'] == 'Winter']['speed_MERRA2'])))))
        centersW = np.array(kmeans_model.cluster_centers_)
        distances.append(np.abs(centersW[:,0]-centersW[:,1])/(np.sqrt(2)))
        #summer
        kmeans_model = KMeans(n_clusters = K).fit(np.array(list(zip(np.array(SeasonMeans_ISD[SeasonMeans_ISD['Season'] == 'Summer']['Speed_ISD']),np.array(SeasonMeans_MERRA2[SeasonMeans_MERRA2['Season'] == 'Summer']['speed_MERRA2'])))))
        centersSU = np.array(kmeans_model.cluster_centers_)
        distances.append(np.abs(centersSU[:,0]-centersSU[:,1])/(np.sqrt(2)))
        #spring
        kmeans_model = KMeans(n_clusters = K).fit(np.array(list(zip(np.array(SeasonMeans_ISD[SeasonMeans_ISD['Season'] == 'Spring']['Speed_ISD']),np.array(SeasonMeans_MERRA2[SeasonMeans_MERRA2['Season'] == 'Spring']['speed_MERRA2'])))))
        centersSP = np.array(kmeans_model.cluster_centers_)
        distances.append(np.abs(centersSP[:,0]-centersSP[:,1])/(np.sqrt(2)))
        #fall
        kmeans_model = KMeans(n_clusters = K).fit(np.array(list(zip(np.array(SeasonMeans_ISD[SeasonMeans_ISD['Season'] == 'Fall']['Speed_ISD']),np.array(SeasonMeans_MERRA2[SeasonMeans_MERRA2['Season'] == 'Fall']['speed_MERRA2'])))))
        centersF = np.array(kmeans_model.cluster_centers_)
        distances.append(np.abs(centersF[:,0]-centersF[:,1])/(np.sqrt(2)))

        distances = [np.round(i,3) for i in distances]
        plt.errorbar(centersW[:,0], centersW[:,1], yerr = W_M2_std, xerr = W_ISD_std, label = 'Winter Centroid r = ' + str(distances[0]) , ecolor = 'blue', fmt = 'none') #+ '(' + str(np.round(Zscore_ISD_winter,3)) + ',' + str(np.round(Zscore_M2_winter,3)) + ')'
        plt.errorbar(centersSU[:,0], centersSU[:,1], yerr = Su_M2_std, xerr = Su_ISD_std, label = 'Summer Centroid r = ' + str(distances[1]), ecolor = 'orange', fmt = 'none')
        plt.errorbar(centersSP[:,0], centersSP[:,1], yerr = Sp_M2_std, xerr = Sp_ISD_std, label = 'Spring Centroid r = ' + str(distances[2]), ecolor = 'green', fmt = 'none')
        plt.errorbar(centersF[:,0], centersF[:,1], yerr = F_M2_std, xerr = F_ISD_std, label = 'Fall Centroid r = ' + str(distances[3]), ecolor = 'red', fmt = 'none')

        plt.xlabel('ISD Average[m/s]', fontsize = 20)
        plt.xticks(fontsize=18)
        plt.ylabel('MERRA-2 Average[m/s]', fontsize = 20)
        plt.yticks(fontsize=18)
        plt.title('Measured (ISD) vs. Modeled (MERRA-2) \n Surface Wind Speed ' + str(station_ID), fontsize = 18)
        plt.legend(facecolor = '0.75')
        plt.legend(fontsize = 11)
        # plt.show()
        plt.savefig('/Users/emily/Documents/UMBC/Dr_LimaLab/ScatterPlot_Outputs_Southwest/' + str(station_ID) + 'SeasonalScatterPlot_publicationSize.png', dpi = 300)
        plt.close()
    # plt.savefig('/Users/emily/Documents/UMBC/Dr_LimaLab/ISDWind/MiddleEast/' + str(station_ID) + '_DATA/Merra2VsISD_Scatter_with_centroids_' + str(station_ID) + '.png', dpi = 300)
    # Zscore_Path = Path('/Users/emily/Documents/UMBC/Dr_LimaLab/ISDWind/MiddleEast/SeasonalScatterPlots_outputs/zscore_outputs_' + str(station_ID) + '.csv')
    # zscore_output.to_csv(Zscore_Path, index = False)

    return
