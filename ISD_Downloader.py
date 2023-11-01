#Beautiful soup ISD CSV grabber
import sys
import os
import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd

def ISD_Downloader():
    '''ISD URL: https://www.ncei.noaa.gov/data/global-hourly/access/ '''

    #Get info from user
    #StationID = input("Station ID from ISD: ")
    to_read = input('path to or name of list of STATION_IDs to download as a csv: ')
    df = pd.read_csv(to_read)
    print('recieved ', len(df), ' stations to download')
    correct = input('Is this correct? Y/N')
    if correct == 'Y' or correct == 'y':
        # df = pd.read_csv(r'C:\Users\Emily\Documents\UMBC\Dr_LimaLab\ISDWind\MiddleEast\StationsList_modified.csv')
        for i in range(0,len(df)):
            # if df['Data in the last two decades (01/01/2001, 12/31/2010)'][i] == 1:
            #print(df['STATION_ID'][i]
            StationID = df['STATION_ID'][i]
            StartYear = 2001
            EndYear = 2020
            # StartYear = input("Start Year: ")
            # EndYear = input("End Year: ")
            print('Getting data for ' + str(StationID) + " From " + str(StartYear) + " to " + str(EndYear))
            #Create folder
            #check to see if it exists already:
            #myPath = 'C:\Users\Emily\Documents\UMBC\Dr_LimaLab\ISDWind\MiddleEast\\' +  str(StationID) + '_DATA'
            if not os.path.exists('ISDWind\\' +  str(StationID) + '_DATA'):
                FolderName = str(StationID) + '_DATA'
                print('Folder made ' + str(StationID))
                ''' THIS DOES NOT CHECK TO SEE IF THERE IS PARTIAL DATA DOWNLOADED'''
            else:
                continue

            '''CHANGE ParentDirectory to where you want the new folder to be made!---------'''
            ParentDirectory = 'ISDWind/'

            path = os.path.join(ParentDirectory, FolderName)
            os.mkdir(path)
            print('Directory Made')
            years = np.arange(int(StartYear), int(EndYear)+1 , 1)
            #This makes the bot more 'real looking'
            headers = {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Max-Age': '3600',
                'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0'
                }
            #The base site url, this changes later depending on user input
            BASEurl = "https://www.ncei.noaa.gov/data/global-hourly/access/"
            print('Downloading data... this may take a few seconds')
            for year in years:
                url = BASEurl + '/' + str(year) + '/' + str(StationID) + '.csv'
                req = requests.get(url, headers)
                url_content = req.content
                path_filename = str(path)+ '/' + str(StationID) + '_' + str(year) + '.csv'
                csv_file = open(path_filename, 'wb')
                csv_file.write(url_content)
                csv_file.close()
    else:
        exit()
    return to_read
