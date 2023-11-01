#MERRA-2 Functions
import numpy as np
import os
import pandas as pd

#returns a pandas dataframe of only the daytime values of wind speed at a specific
#location. Hour cut-offs are hard coded for the middle east time zone. Times are UTC
def daytime_ws(long, lat, ws):
    '''Page 9 of MERRA2 ReadME includes how to go from coordinats to array index'''
    i = ((long + 180)/(5/8)) + 1
    j = ((lat + 90)/0.5) + 1
    Daytime_ws = ws[3:15,round(j),round(i)]

    return Daytime_ws

#returns a pandas dataframe of values of wind speed at a specific location.
# doesn't cut off any times
def location_values(lat,long,ws):
    i = ((float(long) + 180)/(5/8)) + 1
    j = ((float(lat) + 90)/0.5) + 1
    # print(ws.shape)
    ws_new = ws[0:24,round(j), round(i)]

    return ws_new

#returns a vector of the wind speeds magnitude for each data point, at 2 meters
#magnitude based off of MERRA-2 instructions on magnitude
def wind_magnitude(ds):
    U2meter = ds.variables['U2M'] #Eastward wind
    V2meter = ds.variables['V2M'] #Northward wind
    U2meter_NANS = U2meter[:]
    #_FillValue to replace the _FillValue with NANS
    temp = U2meter._FillValue
    U2meter_NANS[U2meter_NANS == temp] = np.nan
    #Northward direction
    V2meter_NANS = V2meter[:]
    temp = V2meter._FillValue
    V2meter_NANS[V2meter_NANS == temp] = np.nan
    ws = np.sqrt(U2meter_NANS**2+V2meter_NANS**2)

    return ws

#returns a vector of wind speed magnitudes for each data point, at 10 Meters
#based off of MERRA-2 instructions on magnitude
def wind_magnitude_10m(ds):
    U10meter = ds.variables['U10M'] #Eastward wind
    V10meter = ds.variables['V10M'] #Northward wind
    U10meter_NANS = U10meter[:]
    #_FillValue to replace the _FillValue with NANS
    temp = U10meter._FillValue
    U10meter_NANS[U10meter_NANS == temp] = np.nan
    #Northward direction
    V10meter_NANS = V10meter[:]
    temp = V10meter._FillValue
    V10meter_NANS[V10meter_NANS == temp] = np.nan
    ws = np.sqrt(U10meter_NANS**2+V10meter_NANS**2)
    return ws

#returns a list of the nc4 (should be MERRA-2) files in a given directory
def filelist(directory):
    filelist = []
    for entry in os.scandir(directory):
        if(entry.path.endswith(".nc4")):
            filelist.append(entry.name)
    return(sorted(filelist))
    # return sorted(filelist)

#returns a data frame of the daytime values and a season column 
def daytime_df(Daytime_ws,entry):
    df = pd.DataFrame()
    df['Speed'] = Daytime_ws
    df['Month'] = entry[31:33]
    df['Month'] = df['Month'].str.lstrip('0')
    df['Month'] = df['Month'].astype('int32')
    df['Season'] = df['Month']
    df.loc[df.Month == 1, 'Season'] = 'Winter'
    df.loc[df.Month == 2, 'Season'] = 'Winter'
    df.loc[df.Month == 3, 'Season'] = 'Spring'
    df.loc[df.Month == 4, 'Season'] = 'Spring'
    df.loc[df.Month == 5, 'Season'] = 'Spring'
    df.loc[df.Month == 6, 'Season'] = 'Summer'
    df.loc[df.Month == 7, 'Season'] = 'Summer'
    df.loc[df.Month == 8, 'Season'] = 'Summer'
    df.loc[df.Month == 9, 'Season'] = 'Fall'
    df.loc[df.Month == 10, 'Season'] = 'Fall'
    df.loc[df.Month == 11, 'Season'] = 'Fall'
    df.loc[df.Month == 12, 'Season'] = 'Winter'
    return df
