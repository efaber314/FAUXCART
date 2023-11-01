#apples to apples functions - python
#Emily Faber
#January 2022
import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from scipy.stats import linregress
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import r2_score
# from Merra2_Wind import MERRA2_Functions
import netCDF4 as nc
from matplotlib.offsetbox import AnchoredText
import openpyxl
import cartopy.crs as ccrs
import matplotlib
from pathlib import Path
import monet
import xarray as xr

#makes data frame from a single file in directory
def ReadData(directory, station_ID):
    for entry in os.scandir(directory):
        if entry.name.endswith('intersection_'+str(station_ID + '.csv')):
            print(entry)
            df = pd.read_csv(entry, low_memory = False)
    return df
#taking a property name from the nc file, can return a merra-2 data frame of the
#property and the time range wanted.
def ReadDataNC_MERRA2(df,directory,lat,long,begin_date,end_date,property):
    # print(df)
    print('lenght of df ', len(df))
    print(str(property))
    filelist = []
    datelist = []
    for entry in os.scandir(directory):
        # print(entry)
        if(entry.path.endswith(".nc")):
            dateString = entry.name[-15:-7] #This is a dangerous hard code
            dateTime = pd.to_datetime(dateString)
            if dateTime >= pd.to_datetime(begin_date) and dateTime <= pd.to_datetime(end_date):
                filelist.append(entry.name)
                datelist.append(dateTime)
                # print(entry.name)
            # print(entry.name)
    # Property = np.empty(shape = (30*24)) #30 days in june - hardcoded
    filelist = sorted(filelist)
    # print(filelist)
    Property = []
    df_property = pd.DataFrame()
    timevector = np.arange(pd.to_datetime(begin_date), pd.to_datetime(end_date)+datetime.timedelta(days=1), dtype = 'datetime64[h]')
    # print(timevector)
    print(len(timevector))
    # print('Whats missing: ', set(np.array(datelist)).difference(set(np.array(timevector))))
    print('length of file list: ', len(filelist), len(filelist)*24, str(property))
    for file in filelist:
        ds = nc.Dataset(directory + str(file))
        property_temp = ds.variables[str(property)]
        i = ((float(long) + 180)/(5/8)) + 1
        j = ((float(lat) + 90)/0.5) + 1
        Property.extend(property_temp[:,round(j),round(i)])
    df_property[str(property)] = Property
    # print(timevector)
    df_property['DateTime'] = timevector
    print('Length of property after being read in: ',len(Property))

    '''https://www.geeksforgeeks.org/python-find-missing-additional-values-two-lists/'''
    '''https://stackoverflow.com/questions/43269548/pandas-how-to-remove-rows-from-a-dataframe-based-on-a-list'''
    '''https://stackoverflow.com/questions/62543350/pandas-force-minute-and-seconds-to-be-zero'''

    # df['DATE'] = pd.to_datetime(df['DATE'])
    # df['DATE'] = df['DATE'].dt
    missing_values = (list(set(np.array(df_property['DateTime'].astype('datetime64[h]'))).difference(np.array((df['DATE'].astype('datetime64[h]'))))))
    print('length of missing values ', len(missing_values))
    df_property = df_property[~df_property.DateTime.isin(missing_values)]
    missing_values = (list(set(np.array((df['DATE'].astype('datetime64[h]')))).difference(np.array(df_property['DateTime'].astype('datetime64[h]')))))
    df = df[~df.DATE.isin(missing_values)]
    print(df.head())
    print('length of df after missing values removed ', len(df))
    print('length of missing vallues ', len(missing_values))
    print('length of df property', len(df_property))
    if len(df['DATE']) < 1:
        print('ERROR: datetime comparison not possible with current station selection. Usually a timestamp issue - ISD reported not at top of the hour.')
        df = pd.DataFrame()
        df_property = pd.DataFrame()
        return df_property,df
    # print(len(df_property))

    # print(len(df_property))
    # print(df_property.tail(7))
    # print(df.tail(7))
    # print(len(df))
    print(df_property.head(10))
    print('length of property: ', len(df_property))
    return df_property, df

def ReadNC(df,directory,lat,long,begin_date,end_date,property):
    print(str(property))
    filelist = []
    datelist = []
    for entry in os.scandir(directory):
        # print(entry)
        '''looks at each file's dates and sees if they are within the time range desired
        if so, that filename is saved for later '''
        if(entry.path.endswith(".nc")):
            ds = nc.Dataset(directory + str(entry.name))
            minutes_since = ds.variables['time']
            NC_dates = nc.num2date(minutes_since[:],minutes_since.units)
            if NC_dates[0] >= pd.to_datetime(begin_date) and NC_dates[0] <= pd.to_datetime(end_date)+datetime.timedelta(days=1):
                filelist.append(entry.name)
    # filelist = sorted(filelist)

    '''Full set time vector '''
    # timevector = np.arange(pd.to_datetime(begin_date), pd.to_datetime(end_date)+datetime.timedelta(days=1), dtype = 'datetime64[h]')

    '''Now we want to dig out the property desired from each file '''
    Property = []
    property_dates = []
    df_property = pd.DataFrame()
    '''DATE TIME | Property
    --------------------------'''

    for file in sorted(filelist):
        ds = nc.Dataset(directory + str(file))
        property_temp = ds.variables[str(property)]
        i = ((float(long) + 180)/(5/8)) + 1
        j = ((float(lat) + 90)/0.5) + 1
        Property.extend(property_temp[:,round(j),round(i)])
        property_dates.extend(nc.num2date(ds.variables['time'][:],ds.variables['time'].units, only_use_cftime_datetimes=False, only_use_python_datetimes = True))
    df_property[str(property)] = Property
    df_property['DATE'] = list(set(np.array(property_dates).astype('datetime64[h]')))

    print(df_property.head(15))
    print(df_property.tail(15))
    print(len(df_property))

    '''Now, what time stamps do we need to throw out since there isn't a measurement then '''
    missing_values = (list(set(np.array(df_property['DATE'].astype('datetime64[h]'))).difference(np.array((df['DATE'].astype('datetime64[h]'))))))
    print('length of missing values ', len(missing_values))
    df_property = df_property[~df_property.DATE.isin(missing_values)]
    print(len(df_property))

    df_combinedHourly = combineSubHourly(df)
    print('combined hourly df')
    print(len(df_combinedHourly))
    print('property df')
    print(len(df_property))

    # df[str(property)] = df_property[str(property)]
    # new_df = combineSubHourly(df)
    # print('length of combine subhourly ', len(new_df))
    # for measurement in df['DATE']:


    # measurement_times = measurement_times.replace(minute = 0, second = 0, microsecond = 0)
    # property_times = property_times.replace(minute = 0, second = 0, microsecond = 0)

    # synced = []
    # for element in list(set(np.array(measurement_times))):
    #     #create time delta between element and each measurement time
    #     timedelta = []
    #     timedelta = property_times - element
    #     for t in timedelta:
    #         if t <= pd.Timedelta(1, unit = 'h'):
    #             synced.append(t)
    #     # timedelta = [timedelta for t in timedelta if t < pd.Timedelta(1, unit = 'h')]
    #     # timedelta = timedelta < pd.Timedelta(1, unit = 'h')
    #     # print(synced)
    #     print(len(synced))
    #
    #
    #     #search all timedeltas and if each is less than an hour they are 'the same time'
    #     #otherwise, this is a missing time
    #     if element not in np.array(measurement_times):
    #
    #         missing_values.append(element)
    # # print(missing_values)
    # print(len(missing_values))

    return df_property, df_combinedHourly
#finds the monthly means of MERRA-2 and ISD and returns as a vector
def monthlyMeans(df):
    MERRA_2_Hourly_Means = df.groupby(by = ['year','month','day','hour'])['speed_MERRA2'].mean()
    ISD_Hourly_Means = df.groupby(by = ['year','month','day','hour'])['Speed_ISD'].mean()
    '''https://stackoverflow.com/questions/10373660/converting-a-pandas-groupby-output-from-series-to-dataframe '''
    MERRA_2_Hourly_Means = MERRA_2_Hourly_Means.reset_index()
    ISD_Hourly_Means = ISD_Hourly_Means.reset_index()
    #get monthly means for each and make the series a data frame
    MERRA_2_Monthly_Means = MERRA_2_Hourly_Means.groupby(by = ['year', 'month'])['speed_MERRA2'].mean()
    MERRA_2_Monthly_Means = MERRA_2_Monthly_Means.reset_index()
    num_points = ISD_Hourly_Means.groupby(by = ['year','month'])['Speed_ISD'].size()

    ISD_Monthly_means = ISD_Hourly_Means.groupby(by = ['year', 'month'])['Speed_ISD'].mean()
    ISD_Monthly_means = ISD_Monthly_means.reset_index()

    return (ISD_Monthly_means, MERRA_2_Monthly_Means, num_points)

#returns df with season column
def Seasons(df):
    df['Season'] = df['month']
    #winter = 1
    #Fall = 2
    #Summer = 3
    #Spring = 4
    df.loc[df['month'] == 1, 'Season'] = 'Winter'
    df.loc[df['month'] == 2, 'Season'] = 'Winter'
    df.loc[df['month'] == 12, 'Season'] = 'Winter'
    df.loc[df['month'] == 11, 'Season'] = 'Fall'
    df.loc[df['month'] == 10, 'Season'] = 'Fall'
    df.loc[df['month'] == 9, 'Season'] = 'Fall'
    df.loc[df['month'] == 8, 'Season'] = 'Summer'
    df.loc[df['month'] == 7, 'Season'] = 'Summer'
    df.loc[df['month'] == 6, 'Season'] = 'Summer'
    df.loc[df['month'] == 5, 'Season'] = 'Spring'
    df.loc[df['month'] == 4, 'Season'] = 'Spring'
    df.loc[df['month'] == 3, 'Season'] = 'Spring'

    return df

#makes plot of all years monthly means - should be 20 years * 12 months = 240 points
def allYearsPlot(df, station_ID):
    ISD_Monthly_means, MERRA_2_Monthly_Means, num_points = monthlyMeans(df)

    #sanity check
    '''https://www.w3schools.com/python/python_datetime.asp'''
    MERRA_2_Monthly_Means['DATE'] = MERRA_2_Monthly_Means['year']
    ISD_Monthly_means['DATE'] = ISD_Monthly_means['year']

    for i in range(0, len(MERRA_2_Monthly_Means)):
        MERRA_2_Monthly_Means['DATE'][i] = datetime.datetime(MERRA_2_Monthly_Means['year'][i],MERRA_2_Monthly_Means['month'][i],15)
        ISD_Monthly_means['DATE'][i] = datetime.datetime(ISD_Monthly_means['year'][i],ISD_Monthly_means['month'][i],15)

    # print(MERRA_2_Monthly_Means)
    # print(ISD_Monthly_means)
    fig,ax = plt.subplots(figsize = (24,12))
    # plt.figure(figsize = (12,6))
    plt.plot(ISD_Monthly_means['DATE'],ISD_Monthly_means['Speed_ISD'], label = "ISD Measurments", linewidth = 12)
    plt.plot(MERRA_2_Monthly_Means['DATE'], MERRA_2_Monthly_Means['speed_MERRA2'], label = 'MERRA-2 data', linewidth = 12)
    #trendlines
    slope_ISD, intercept_ISD = np.polyfit(np.arange(0,len(ISD_Monthly_means['DATE']),1), ISD_Monthly_means['Speed_ISD'],1)
    slope_M, intercept_M = np.polyfit(np.arange(0,len(MERRA_2_Monthly_Means['DATE']),1), MERRA_2_Monthly_Means['speed_MERRA2'],1)

    line_isd = [slope_ISD * i + intercept_ISD for i in np.arange(0,len(ISD_Monthly_means['DATE']),1)]
    line_M = [slope_M * i + intercept_M for i in  np.arange(0,len(MERRA_2_Monthly_Means['DATE']),1)]

    correlation_matrix_isd = np.corrcoef(ISD_Monthly_means['Speed_ISD'], line_isd)
    corr_isd = correlation_matrix_isd[0,1]
    R2_isd = corr_isd**2
    correlation_matrix_M = np.corrcoef(MERRA_2_Monthly_Means['speed_MERRA2'], line_M)
    corr_M = correlation_matrix_M[0,1]
    R2_M = corr_M**2

    plt.plot(ISD_Monthly_means['DATE'], line_isd, label = 'ISD trend ' + str(np.round(slope_ISD,5)) + ' R^2: ' +str(round(R2_isd,3)))
    plt.plot(MERRA_2_Monthly_Means['DATE'], line_M, label = 'MERRA-2 trend' + str(np.round(slope_M,5))+ ' R^2: ' +str(round(R2_M,3)))


    # slope, intercept, r_value, p_value, std_err = linregress(MERRA_2_Monthly_Means['DATE'], MERRA_2_Monthly_Means['speed_MERRA2'])

    # plt.plot(MERRA_2_Monthly_Means['DATE'], intercept + slope*MERRA_2_Monthly_Means['DATE'], 'r', label = 'MERRA-2 trend line')
    plt.legend(fontsize = 25)
    plt.xlabel('Time', fontsize = 35)
    plt.ylabel('Monthly Average Wind Speeds [m/s]', fontsize = 35)
    # print(ISD_Monthly_means)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.ylim(0,8)

    plt.title(str(station_ID) + ' All Monthly Averages', fontsize = 40)
    # ticks = np.arange(0,240,1)
    # print(num_points)
    # print(len(num_points))
    # plt.show()
    # for i in range(len(ISD_Monthly_means['DATE'])):
    #     ax.text(np.array(ISD_Monthly_means['DATE'])[i], np.array(ISD_Monthly_means['Speed_ISD'])[i], np.array(num_points)[i], size = 20)

    SaveDirectoryAll = '/Users/emily/Documents/UMBC/Dr_LimaLab/MERRA2VsISD_YearsTimeSeries/Southwest/'
    plt.savefig(SaveDirectoryAll  + str(station_ID) +'_timeCorrelated' + '20year_10meter_allTime_with_trendline.png', dpi = 300)
    plt.close()
    # plt.show()
def ScaleSourceFunction():
    S = nc.Dataset('/Users/emily/Documents/UMBC/Dr_LimaLab/Merra2_Wind/gocart.dust_source.v5a.x1152_y721.nc', low_memory = False)
    source_function_whole = np.array(S.variables['du_src'])
    #source function is 721, 1152 (lat,long)
    source_function = np.zeros((360,575))
    for i in range(0, 360):
        for j in range(0,575):
            # print(source_function_whole[0,2*i,2*j])
            source_function[i,j] = np.mean(source_function_whole[0,2*i:2*i+1,2*j:2*j+1])
    # print(source_function.shape)
    return source_function

'''Depricated '''
def RetrieveSourceFunction(latitude, longitude):
    ds = nc.Dataset('/Users/emily/Documents/UMBC/Dr_LimaLab/Merra2_Wind/gocart.dust_source.v5a.x1152_y721.nc', low_memory = False)
    source_function_whole = ds.variables['du_src']
    long_source = ds.variables['longitude'][:]
    lat_source = ds.variables['latitude'][:]
    #take difference between lat_source and latitude of station
    #find location of smallest difference between the two
    lat_source = np.array(lat_source)
    long_source = np.array(long_source)
    difference_lat = [abs(x - latitude) for x in lat_source]
    difference_long = [abs(x - longitude) for x in long_source]
    lat_index = np.argmin(difference_lat)
    long_index = np.argmin(difference_long)
    print('Source function retrieval lat,long: ', lat_source[lat_index], long_source[long_index])
    source_function_square = source_function_whole[0,lat_index-1:lat_index+2,long_index-1:long_index+2]
    source_function_square = np.array(source_function_square)
    '''todo: upgrade to exclude 0s for over ocean in average'''
    # where = np.where(source_function_square == 0)
    # print([where])
    # if len(where) != 0:
    #     source_function_square[where] = np.nan()
    source_function = np.mean(source_function_square.flatten())
    # print(source_function)
    # source_function = source_function_whole[0,lat_index,long_index]

    return source_function
'''Depricated '''
def Flux_MB(df,begin_date,end_date):
    #calculate a threshold friction velocity - based on Matricorena 1995, eq 6
    #assume a diameter
    effective_radii = [.73, 1.4, 2.4, 4.5, 8] #microns
    A = 6.5 #dimensionless tuning parameter
    density_air = 1230 #g/m^3 to match marticorena

    g = 9.81 #m/s^2
    C = 1 #microgram *s^2/m^5
    source_function = RetrieveSourceFunction(df['LATITUDE'].iat[0], df['LONGITUDE'].iat[0])
    FluxU = ReadDataNC_MERRA2(df, '/Users/emily/Documents/UMBC/Dr_LimaLab/MERRA-2_FluxAndTau/', df['LATITUDE'].iat[0],df['LONGITUDE'].iat[0],begin_date,end_date,'DUFLUXU')
    FluxV = ReadDataNC_MERRA2(df, '/Users/emily/Documents/UMBC/Dr_LimaLab/MERRA-2_FluxAndTau/', df['LATITUDE'].iat[0],df['LONGITUDE'].iat[0],begin_date,end_date,'DUFLUXV')
    FluxMagnitude = np.sqrt((FluxU['DUFLUXU']**2) + (FluxV['DUFLUXV']**2))
    FLUX_MERRA_2_GIVEN = sum(FluxMagnitude)/10000

    '''soil surface wetness'''
    soil_wetness = ReadDataNC_MERRA2(df,'/Users/emily/Documents/UMBC/Dr_LimaLab/MERRA-2_SurfaceWetness/', df['LATITUDE'].iat[0], df['LONGITUDE'].iat[0], begin_date,end_date, 'GWETTOP')

    a = 1331 #/cm^-x
    b = 0.38
    x = 1.56
    a *= 100**x #/m^3
    flux_ISD = 0
    flux_MERRA2 = 0
    colors = ['black','crimson','chartreuse','cyan','orchid']
    i = 0
    for r in effective_radii:
        R = r * 10**-5 #cm
        if r == 0.73:
            density_particle = 2.5 #g/cm^3 #this will throw the B calculation off from MARTICORENA, but i'm hoping not by too much - TODO: update this with a calculation for the B coefficients using this density
            density_particle*= 100**3 #g/m^3
        else:
            density_particle = 2.65 #g/cm^3
            density_particle*= 100**3 #g/m^3
        '''MB95'''
        k = (((density_particle*g*2*R)/(density_air))**(1/2))* ((1 + ((.006)/(density_particle*g*(2*R)**2.5)))**(1/2))
        U_t = (.129*k)/(((1.928*((a*((2*R)**x) + b)**0.092) - 1)**(1/2))) #for low reynolds number
        # U_t = 0.129*k*(1 - (0.0858*np.exp(-0.0617*((a*((2*R)**x)+b)-10)))) #for high reynolds number - turbulence
        f_isd = C*source_function*2*R*(df['Speed_ISD']**2)*(df['Speed_ISD'] - U_t)
        f_merra2 = C*source_function*2*R*(df['speed_MERRA2']**2)*(df['speed_MERRA2'] - U_t)
        flux_ISD += f_isd
        flux_MERRA2 += f_merra2
        #separate out size bin flux
        if r == 0.73:
            df['Flux0.73_isd'] = f_isd
            df['Flux0.73_merra2'] = f_merra2
        if r == 1.4:
            df['Flux1.4_isd'] = f_isd
            df['Flux1.4_merra2'] = f_merra2
        if r == 2.4:
            df['Flux2.4_isd'] = f_isd
            df['Flux2.4_merra2'] = f_merra2
        if r == 4.5:
            df['Flux4.5_isd'] = f_isd
            df['Flux4.5_merra2'] = f_merra2
        if r == 8.0:
            df['Flux8_isd'] = f_isd
            df['Flux8_merra2'] = f_merra2
        plt.axvline(np.mean(U_t), label = 'U_t, d = ' + str(np.round(2*r,3)) + 'um , U_t = ' + str(np.round(np.mean(U_t),3)), c = colors[i])
        i += 1
    df['ISD_total_flux'] = flux_ISD
    df['MERRA2_total_flux'] = flux_MERRA2
    return df

def U_tvsR(df, begin_date,end_date):
    '''knowns'''
    R = np.arange(0,50,.0695)#Microns
    # R = R*10**-6 #microns
    A = 6.5 #dimensionless tuning parameter
    C = 1 #microgram *s^2/m^5
    sp = 0.33
    a = 1331 #/cm^-x
    b = 0.38
    x = 1.56
    a *= 100**x #/m^3
    g = 9.81 #m/s^2
    '''source function'''
    source_function = RetrieveSourceFunction(df['LATITUDE'].iat[0], df['LONGITUDE'].iat[0])
    '''soil surface wetness'''
    soil_wetness = ReadDataNC_MERRA2(df,'/Users/emily/Documents/UMBC/Dr_LimaLab/MERRA-2_SurfaceWetness/', df['LATITUDE'].iat[0], df['LONGITUDE'].iat[0], begin_date,end_date, 'GWETTOP')
    #comes back as GWETTOP and DATETIME dataframe
    U_t_ginoux = []
    U_t_MB = []
    i = 0
    for r in R:
        r = r * 10**-5 #m
        '''ginoux 2001'''
        density_air = 0.00123 #g/cm^3
        density_particle = 2.65*10**-3 #g/cm^3
        Ut1 = A * ((g*2*r*((density_particle - density_air)/density_air))**(1/2))
        Ut1 = Ut1 * (1.2 + 0.2*np.log10(soil_wetness['GWETTOP'][i]))
        i += 1
        U_t_ginoux.append(Ut1)
        '''MB95'''
        density_air = 1230 #g/m^3 to match marticorena
        density_particle = 2.65 #g/cm^3
        density_particle*= 100**3 #g/m^3
        k = (((density_particle*g*2*r)/(density_air))**(1/2))* ((1 + ((.006)/(density_particle*g*(2*r)**2.5)))**(1/2))
        # U_t = (.129*k)/(((1.928*((a*((2*R)**x) + b)**0.092) - 1)**(1/2))) #for low reynolds number
        Ut2 = 0.129*k*(1 - (0.0858*np.exp(-0.0617*((a*((2*r)**x)+b)-10)))) #for high reynolds number - turbulence
        U_t_MB.append(Ut2)

    plt.plot(R, U_t_ginoux, label = 'Ginoux')
    plt.plot(R, U_t_MB, label = 'MB95' )
    plt.xlabel('Radius [um]')
    plt.ylabel('U_t [m/s]')
    plt.title('Threshold Wind Speed of MB95 and Ginoux 2001')
    plt.legend()
    plt.show()
    return

'''Depricated '''
def Flux_Ginoux(df, begin_date, end_date):
    effective_radii = [.73, 1.4, 2.4, 4.5, 8] #microns
    A = 6.5 #dimensionless tuning parameter
    density_air = 0.00123 #g/cm^3
    g = 9.81 #m/s^2
    C = 1 #microgram *s^2/m^5
    '''source function'''
    source_function = RetrieveSourceFunction(df['LATITUDE'].iat[0], df['LONGITUDE'].iat[0])
    '''flux components (given)'''
    FluxU = ReadDataNC_MERRA2(df,'/Users/emily/Documents/UMBC/Dr_LimaLab/MERRA-2_FluxAndTau/', df['LATITUDE'].iat[0],df['LONGITUDE'].iat[0],begin_date,end_date,'DUFLUXU')
    FluxV = ReadDataNC_MERRA2(df,'/Users/emily/Documents/UMBC/Dr_LimaLab/MERRA-2_FluxAndTau/', df['LATITUDE'].iat[0],df['LONGITUDE'].iat[0],begin_date,end_date,'DUFLUXV')
    FluxMagnitude = np.sqrt((FluxU['DUFLUXU']**2) + (FluxV['DUFLUXV']**2))
    FLUX_MERRA_2_GIVEN = sum(FluxMagnitude)/10000

    '''soil surface wetness'''
    soil_wetness = ReadDataNC_MERRA2(df,'/Users/emily/Documents/UMBC/Dr_LimaLab/MERRA-2_SurfaceWetness/', df['LATITUDE'].iat[0], df['LONGITUDE'].iat[0], begin_date,end_date, 'GWETTOP')
    #comes back as GWETTOP and DATETIME dataframe

    '''calculate threshold friction velocity'''
    flux_ISD = 0
    flux_MERRA2 = 0
    colors = ['black','crimson','chartreuse','cyan','orchid'] #for plotting Ut lines
    i = 0 #for choosing colors
    for r in effective_radii:
        R = r * 10**-5 #m
        if r == 0.73:
            density_particle = 2.5*10**-3 #g/cm^3 #this will throw the B calculation off from MARTICORENA, but i'm hoping not by too much - TODO: update this with a calculation for the B coefficients using this density
            sp = 0.1
        else:
            density_particle = 2.65*10**-3 #g/cm^3
            sp = 0.25 #updated in MERRA-2 from 0.33 given in Ginoux 2001
        '''ginoux 2001'''
        U_t = A * ((g*2*R*((density_particle - density_air)/density_air))**(1/2))
        U_t = U_t * (1.2 + 0.2*np.log(soil_wetness['GWETTOP'])) #equation 3 paul ginoux 2001
        f_isd = C*source_function*sp*(df['Speed_ISD']**2)*(df['Speed_ISD'] - U_t)  #ginoux CSspu^10(u_10 - Ut)
        print(f_isd)
        flux_ISD += f_isd
        f_merra2 = C*source_function*sp*(df['speed_MERRA2']**2)*(df['speed_MERRA2'] - U_t)  #ginoux CSspu^10(u_10 - Ut)
        flux_MERRA2 += f_merra2
        #separate out size bin flux
        if r == 0.73:
            df['Flux0.73_isd'] = f_isd
            df['Flux0.73_merra2'] = f_merra2
        if r == 1.4:
            df['Flux1.4_isd'] = f_isd
            df['Flux1.4_merra2'] = f_merra2
        if r == 2.4:
            df['Flux2.4_isd'] = f_isd
            df['Flux2.4_merra2'] = f_merra2
        if r == 4.5:
            df['Flux4.5_isd'] = f_isd
            df['Flux4.5_merra2'] = f_merra2
        if r == 8.0:
            df['Flux8_isd'] = f_isd
            df['Flux8_merra2'] = f_merra2
        plt.axvline(np.mean(U_t), label = 'U_t, d = ' + str(np.round(2*r,3)) + 'um , U_t = ' + str(np.round(np.mean(U_t),3)), c = colors[i])
        i += 1 #moves to the next color for Ut line

    df['ISD_total_flux'] = flux_ISD
    df['MERRA2_total_flux'] = flux_MERRA2
    return df

def U_tVector(df,begin_date,end_date, mylat, mylong):
    density_air = 1250 #g/m^3 to match marticorena
    a = 1331 #/cm^-x
    g = 9.81 #m/s^2
    b = 0.38
    x = 1.56
    a *= 100**x #/m^3
    print(len(df), 'length of df before soilwetness retrieval')
    soil_wetness, df = ReadNC(df,'/Users/emily/Documents/UMBC/Dr_LimaLab/MERRA-2_SurfaceWetness/', mylat, mylong, begin_date,end_date, 'GWETTOP')
    print('length of soil wetness: ', len(soil_wetness))
    print('length of df: ', len(df))
    effective_radii = [0.73, 1.4, 2.4, 4.5, 8] #microns - size bins of MERRA-2
    U_t_df = pd.DataFrame()

    for r in effective_radii:
        R = r * 10**-5 #m
        if r == 0.73:
            density_particle = 2.5*100**3  #this will throw the B calculation off from MARTICORENA, but i'm hoping not by too much - TODO: update this with a calculation for the B coefficients using this density
            sp = 0.1
        else:
            density_particle = 2.65*100**3
            sp = 0.25 #updated in MERRA-2 from 0.33 given in Ginoux 2001

        '''MB95'''
        k = (((density_particle*g*2*R)/(density_air))**(1/2))* ((1 + ((.006)/(density_particle*g*(2*R)**2.5)))**(1/2))
        U_t = (.129*k)/(((1.928*((a*((2*R)**x) + b)**0.092) - 1)**(1/2))) #for low reynolds number
        # U_t = 0.129*k*(1 - (0.0858*np.exp(-0.0617*((a*((2*R)**x)+b)-10)))) #for high reynolds number - turbulence
        ''' Ginoux 2001 correction to U_t'''
        print(soil_wetness.head())
        # print(soil_wetness.dtypes)
        # test = np.log(soil_wetness['GWETTOP'])
        # print(test)
        # print(test.dtypes)
        U_t_vector = U_t * (1.2 + 0.2*np.log(soil_wetness['GWETTOP']))#same dimensions as soil_wetness
        print('length of U_t after ginoux correction: ', len(U_t_vector))
        print(df.tail(5), U_t_vector.tail(5))
        # if len(U_t) > len(df):
        #     difference = len(df) - len(U_t)
        #     print('vectors not the same size, adjusting U_t_df by', difference)
        #     df = df[0:len(U_t)]
        # if len(U_t) < len(df):
        #     difference = len(U_t) - len(df)
        #     print('vectors not the same size, adjusting df by', difference)
        #     U_t = U_t[0:len(df)]

        U_t_df[str(r)] = U_t_vector #dimensions should be Nx5 at end
        string = 'u_t at ' + str(r)
        df[str(string)] = U_t_vector
        print(df.head(5))
        print(len(df), 'length of df at end of Ut Vector')

    return U_t_df, df

'''Depricated '''
def U_tVector_const_GWETTOP(df,begin_date,end_date):
    density_air = 1250 #g/m^3 to match marticorena
    a = 1331 #/cm^-x
    g = 9.81 #m/s^2
    b = 0.38
    x = 1.56
    a *= 100**x #/m^3
    # soil_wetness = 0.1
    soil_wetness, df = ReadNC(df,'/Users/emily/Documents/UMBC/Dr_LimaLab/MERRA-2_SurfaceWetness/', df['LATITUDE'].iat[0], df['LONGITUDE'].iat[0],begin_date,end_date,'GWETTOP')
    # df = SubHourly(df_return)
    # soil_wetness, hourly_df = ReadDataNC_MERRA2(df,'/Users/emily/Documents/UMBC/Dr_LimaLab/MERRA-2_SurfaceWetness/', df['LATITUDE'].iat[0], df['LONGITUDE'].iat[0],begin_date,end_date,'GWETTOP')
    print('length of soil wetness ', len(soil_wetness['GWETTOP']))
    print('length of df ', len(df['speed_MERRA2']))
    print(df.head())
    effective_radii = [.73, 1.4, 2.4, 4.5, 8] #microns - size bins of MERRA-2
    U_t_vector = []
    for r in effective_radii:
        R = r * 10**-5 #m
        if r == 0.73:
            density_particle = 2.5*100**3  #this will throw the B calculation off from MARTICORENA, but i'm hoping not by too much - TODO: update this with a calculation for the B coefficients using this density
            sp = 0.1
        else:
            density_particle = 2.65*100**3
            sp = 0.25 #updated in MERRA-2 from 0.33 given in Ginoux 2001

        '''MB95'''
        k = (((density_particle*g*2*R)/(density_air))**(1/2))* ((1 + ((.006)/(density_particle*g*(2*R)**2.5)))**(1/2))
        U_t = (.129*k)/(((1.928*((a*((2*R)**x) + b)**0.092) - 1)**(1/2))) #for low reynolds number
        # U_t = 0.129*k*(1 - (0.0858*np.exp(-0.0617*((a*((2*R)**x)+b)-10)))) #for high reynolds number - turbulence
        ''' Ginoux 2001 correction to U_t'''
        U_t = U_t * (1.2 + 0.2*np.log(soil_wetness['GWETTOP']))
        U_t_vector.append(U_t)
    print(U_t_vector)
    print(len(U_t_vector))
    print('LENGTH OF DF ', len(df))
    return U_t_vector, df

#makes a histogram of the wind speed values and data points. All points that have been time and space aligned
def CombinationFlux(df,flux001,flux002,flux003,flux004,flux005, begin_date, end_date, mylong,mylat,map):
    #calculate a threshold friction velocity - based on Matricorena 1995, eq 6 and correction from Ginoux 2001
    #assume a radius

    '''constants'''
    effective_radii = [0.73, 1.4, 2.4, 4.5, 8] #microns - size bins of MERRA-2
    A = 6.5 #dimensionless tuning parameter
    density_air = 1250 #g/m^3 to match marticorena
    g = 9.81 #m/s^2
    C = 1 #microgram *s^2/m^5
    C  = C *(1*10**-9) #kg*s^2/m^5
    C = C*0.08

    '''Source Function retrieval

    1 -- topographic source (OG)
    2 -- Sediment supply map (FENGSHA)

    '''
    if map == 1:
        source_function = ScaleSourceFunction()
        #returns 360x575 - the entire map
        i = round(((float(mylong) + 180)/(5/8)) + 1)
        j = round(((float(mylat) + 90)/0.5) + 1)
        source_function = source_function[j,i]
        print('source function', source_function)
    if map == 2:
        ds = xr.open_dataset('/Users/emily/Documents/UMBC/Dr_LimaLab/FENGSHA/FENGSHA_SOILGRIDS2017_GEFSv12_v1.2-2_M2gridsize.nc')
        source_function = ds.monet.nearest_latlon(lat = mylat, lon = mylong)
        print('Sediment Supply Map value', float(source_function.ssm[6]))
        source_function = float(source_function.ssm[6])
    # print('S lat,long: ', )
    # print('S: ', source_function)
    # FineSourceFunction = RetrieveSourceFunction(mylat,mylong)
    # print('S retrived: ', FineSourceFunction)

    '''soil surface wetness and threshold wind speed'''
    # soil_wetness,df = ReadDataNC_MERRA2(df,'/Users/emily/Documents/UMBC/Dr_LimaLab/MERRA-2_SurfaceWetness/', df['LATITUDE'].iat[0], df['LONGITUDE'].iat[0], begin_date,end_date, 'GWETTOP')
    # print('length of soil wetness: ', len(soil_wetness))
    # if len(soil_wetness) == 0:
    #     return
    flux_ISD = 0
    flux_MERRA2 = 0
    # colors = ['black','crimson','chartreuse','cyan','orchid']
    # hourly_df = SubHourly(df) #hourly average if sub-hourly is reported
    # hourly_df = df
    # Ut,hourly_df = U_tVector(hourly_df,begin_date,end_date, mylat, mylong) #average Ut for each size range for the whole time series
    # print(Ut['0.73'])
    print('length of df before UT vector ', len(df))
    Ut, df = U_tVector(df,begin_date,end_date,mylong,mylat )
    print('length of df after UT vector ', len(df))
    # df = combineSubHourly(my_df)
    # hourly_df = df
    # print('LENGTH OF df',len(df))
    # print('LENGTH OF HOURLY DF', len(df))
    # print(len(Ut))
    # print(len(df))
    s_p = [0.1, 0.25, 0.25, 0.25, 0.25]
    for j in range(0,len(effective_radii)):
        r = effective_radii[j]
        R = r * 10**-5 #m
        sp = s_p[j]
        string = 'u_t at ' + str(r)
        U_t = df[str(string)]
        print('length of Ut after retrieval in combination flux ', len(U_t))
        # U_t = Ut[str(r)] #should be the same length as soil wetness which should be the same as Speed_ISD
        # print(len(U_t), 'length of Ut')
        # print(U_t)
        # print(len(np.array(df['Speed_ISD'])), 'length of ISD windspeed before combine subhourly')
        # print(df.head(5))
        # for col in df.columns:
        #     print(col)
        '''Ginoux flux as in GOCART '''
        print('length of Ut', len(U_t))
        print('length of speed ', len(df['Speed_ISD']))

        # print(len(hourly_df['Speed_ISD']), ' ISD windspeed length after combining subhourly')
        f_isd = C*source_function*sp*(np.array(df['Speed_ISD'])**2)*(np.array(df['Speed_ISD']) - U_t)
        # print(len(f_isd), 'length of f_isd')
        f_merra2 = C*source_function*sp*(np.array(df['speed_MERRA2'])**2)*(np.array(df['speed_MERRA2']) - U_t)
        flux_ISD += f_isd
        flux_MERRA2 += f_merra2
        '''separate out size bin flux'''
        if r == 0.73:
            df['Flux0.73_isd'] = np.array(f_isd)
            df['Flux0.73_merra2'] = np.array(f_merra2)
        if r == 1.4:
            df['Flux1.4_isd'] = np.array(f_isd)
            df['Flux1.4_merra2'] = np.array(f_merra2)
        if r == 2.4:
            df['Flux2.4_isd'] = np.array(f_isd)
            df['Flux2.4_merra2'] = np.array(f_merra2)
        if r == 4.5:
            df['Flux4.5_isd'] = np.array(f_isd)
            df['Flux4.5_merra2'] = np.array(f_merra2)
        if r == 8.0:
            df['Flux8_isd'] = np.array(f_isd)
            df['Flux8_merra2'] = np.array(f_merra2)
        # plt.axvline(np.mean(U_t), label = 'd = ' + str(np.round(2*r,3)) + 'um , U_t = ' + str(np.round(np.mean(U_t),3)) + ' m/s', c = colors[j])
    df['ISD_total_flux'] = np.array(flux_ISD)
    df['MERRA2_total_flux'] = np.array(flux_MERRA2)
    return df

def myPercentDiff(v1,v2):
    top = np.abs(v1-v2)
    bottom = (np.abs(v2))
    Percent_diff = 100*(top/bottom)
    return Percent_diff

def combineSubHourly(df):
    '''https://towardsdatascience.com/pandas-dataframe-group-by-consecutive-same-values-128913875dba'''
    test_speed_group_ISD = df.groupby((df['hour'].shift() != df['hour']).cumsum())['Speed_ISD'].mean(numeric_only=True)

    print(len(test_speed_group_ISD))
    # print(test_speed_group_ISD.head(20))
    test_speed_group_M2 = df.groupby((df['hour'].shift() != df['hour']).cumsum())['speed_MERRA2'].mean(numeric_only=True)
    # print(test_speed_group_M2.head(20))
    new_df = pd.DataFrame()
    new_df['Speed_ISD'] = np.array(test_speed_group_ISD)
    new_df['speed_MERRA2'] = np.array(test_speed_group_M2)
    new_df['LATITUDE'] = df.groupby((df['hour'].shift() != df['hour']).cumsum())['LATITUDE'].mean(numeric_only=True)
    new_df['LONGITUDE'] = df.groupby((df['hour'].shift() != df['hour']).cumsum())['LONGITUDE'].mean(numeric_only=True)
    df['DATE'] = pd.to_datetime(df['DATE'])
    new_df['DATE'] = df.groupby((df['hour'].shift() != df['hour']).cumsum())['DATE'].agg(pd.Series.mean)
    new_df['hour'] = df.groupby((df['hour'].shift() != df['hour']).cumsum())['hour'].mean()
    # for k, v in df.groupby((df['hour'].shift() != df['hour']).cumsum()):
    #     print(f'[group {k}]')
    #     print(v)
    return new_df

def histogram(df,station_ID,begin_date,end_date,mylong,mylat,diurnalText,map):

    ParentDirectory = '/Users/emily/Documents/UMBC/Dr_LimaLab/ISDWind/MiddleEast/'
    # directory = ParentDirectory + str(station_ID) + '_DATA/' + 'MERRA2_intersection_' + str(station_ID) + '.csv'


    '''Flux bins (givens)'''
    flux001,df = ReadNC(df,'/Users/emily/Documents/UMBC/Dr_LimaLab/MERRA-2_FluxBins/', mylat, mylong,begin_date,end_date,'DUEM001')
    flux002,df = ReadNC(df,'/Users/emily/Documents/UMBC/Dr_LimaLab/MERRA-2_FluxBins/', mylat, mylong,begin_date,end_date,'DUEM002')
    flux003,df = ReadNC(df,'/Users/emily/Documents/UMBC/Dr_LimaLab/MERRA-2_FluxBins/', mylat, mylong,begin_date,end_date,'DUEM003')
    flux004,df = ReadNC(df,'/Users/emily/Documents/UMBC/Dr_LimaLab/MERRA-2_FluxBins/', mylat, mylong,begin_date,end_date,'DUEM004')
    flux005,df = ReadNC(df,'/Users/emily/Documents/UMBC/Dr_LimaLab/MERRA-2_FluxBins/', mylat, mylong,begin_date,end_date,'DUEM005')
    # print('length of flux bins: ', len(flux003))
    print('length of df ', len(df))
    '''Total EMISSIONS given'''
    total_flux = (sum(flux001['DUEM001']) + sum(flux002['DUEM002']) + sum(flux003['DUEM003']) + sum(flux004['DUEM004']) + sum(flux005['DUEM005']))
    total_flux /= len(flux001)
    '''AOD comparable flux '''
    # DUFLUXU, df= ReadDataNC_MERRA2(df,'/Users/emily/Documents/UMBC/Dr_LimaLab/MERRA-2_FluxAndTau/',df['LATITUDE'].iat[0], df['LONGITUDE'].iat[0],begin_date,end_date,'DUFLUXU')
    # DUFLUXV, df = ReadDataNC_MERRA2(df,'/Users/emily/Documents/UMBC/Dr_LimaLab/MERRA-2_FluxAndTau/',df['LATITUDE'].iat[0], df['LONGITUDE'].iat[0],begin_date,end_date,'DUFLUXV')
    # DUFLUX = np.sqrt((DUFLUXU['DUFLUXU']**2)+(DUFLUXV['DUFLUXV']**2))

    '''Total EMISSIONS calculated for both ISD and MERRA-2 '''
    df_flux = CombinationFlux(df,flux001,flux002,flux003,flux004,flux005,begin_date,end_date,mylong,mylat,map)
    print('length of df_flux: ', len(df_flux))
    df_flux['ISD_total_flux'] = df_flux['ISD_total_flux']/len(df_flux['ISD_total_flux'])
    df_flux['MERRA2_total_flux'] = df_flux['MERRA2_total_flux']/len(df_flux['MERRA2_total_flux'])
    print(df_flux.keys())
    print(df_flux.head(10))
    '''by size bin'''
    df_flux['Flux0.73_merra2'] = df_flux['Flux0.73_merra2']/len(df_flux['Flux0.73_merra2'])
    df_flux['Flux1.4_merra2'] = df_flux['Flux1.4_merra2']/len(df_flux['Flux1.4_merra2'])
    df_flux['Flux2.4_merra2'] = df_flux['Flux2.4_merra2']/len(df_flux['Flux2.4_merra2'])
    df_flux['Flux4.5_merra2'] = df_flux['Flux4.5_merra2']/len(df_flux['Flux4.5_merra2'])
    df_flux['Flux8_merra2'] = df_flux['Flux8_merra2']/len(df_flux['Flux8_merra2'])
    df_flux['Flux0.73_isd'] = df_flux['Flux0.73_isd']/len(df_flux['Flux0.73_isd'])
    df_flux['Flux1.4_isd'] = df_flux['Flux1.4_isd']/len(df_flux['Flux1.4_isd'])
    df_flux['Flux2.4_isd'] = df_flux['Flux2.4_isd']/len(df_flux['Flux2.4_isd'])
    df_flux['Flux4.5_isd'] = df_flux['Flux4.5_isd']/len(df_flux['Flux4.5_isd'])
    df_flux['Flux8_isd'] = df_flux['Flux8_isd']/len(df_flux['Flux8_isd'])
    # print(df_flux.head(20))

    # plt.legend()

    col_names = ['Station_ID', 'begin date', 'end date', 'ISD total', 'MERRA-2 emissions_calculated','MERRA-2 emissions given','percent_diff_ISD_calculatedM2', 'percent diff ISD given M2', 'percent diff calculated M2 given M2', 'longitude', 'latitude', 'diurnalText', 'todays date','ISD_73_total','ISD_1_4_total','ISD_2_4_total','ISD_4_5_total','ISD_8_total','MERRA2_73_total', 'MERRA2_1_4_total', 'MERRA2_4_5_total','MERRA2_8_total', 'length of flux df', 'map used 1- topo, 2-ssm']
    ISD_total = np.nansum(np.array(df_flux['ISD_total_flux']))
    nan_count = df_flux['ISD_total_flux'].isna().sum()
    print('nan count ', nan_count)
    print('totals----------------')
    print(ISD_total)
    MERRA2_calculated = np.nansum(np.array(df_flux['MERRA2_total_flux']))
    print(MERRA2_calculated)
    ISD_73_total = np.nansum(np.array(df_flux['Flux0.73_isd']))
    print(ISD_73_total)
    ISD_1_4_total = np.nansum(np.array(df_flux['Flux1.4_isd']))
    ISD_2_4_total = np.nansum(np.array(df_flux['Flux2.4_isd']))
    ISD_4_5_total = np.nansum(np.array(df_flux['Flux4.5_isd']))
    ISD_8_total = np.nansum(np.array(df_flux['Flux8_isd']))
    MERRA2_73_total = np.nansum(np.array(df_flux['Flux0.73_merra2']))
    MERRA2_1_4_total = np.nansum(np.array(df_flux['Flux1.4_merra2']))
    MERRA2_2_4_total = np.nansum(np.array(df_flux['Flux2.4_merra2']))
    MERRA2_4_5_total = np.nansum(np.array(df_flux['Flux4.5_merra2']))
    MERRA2_8_total = np.nansum(np.array(df_flux['Flux8_merra2']))
    data = [station_ID, str(begin_date),str(end_date),ISD_total,MERRA2_calculated,total_flux,myPercentDiff(ISD_total,MERRA2_calculated),myPercentDiff(ISD_total, total_flux), myPercentDiff(MERRA2_calculated, total_flux), mylong, mylat, diurnalText, str(datetime.date.today()),ISD_73_total,ISD_1_4_total,ISD_2_4_total,ISD_4_5_total,ISD_8_total,MERRA2_73_total, MERRA2_1_4_total, MERRA2_4_5_total,MERRA2_8_total, len(df_flux),map]
    filename = ParentDirectory +'PercentDifference_files/'+'ME('+ str(begin_date)[0:10] + '-' + str(end_date)[0:10] + ')_withSizeBins_omega_map_experiments' + '.xlsx'
    path = Path(ParentDirectory +'PercentDifference_files/' +'ME('+ str(begin_date)[0:10] + '-' + str(end_date)[0:10] + ')_withSizeBins_omega_map_experiments' + '.xlsx')
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
        # wb = openpyxl.load_workbook(path, 'rb')
        # ws = wb.active
        # ws.append(data)
        # wb.save(filename)
        # wb.close()
    else:
        #create file AND append
        newdata = pd.DataFrame([data], columns = col_names)
        print(newdata)
        newdata.to_excel(path, index = False)
        # path.parent.mkdir(parents = True, exist_ok = True)
        # newdata.to_csv(path,header=not os.path.exists(output_path))
        # with open(filename, 'w') as creating_new_csv_file:
        #     print(filename + ' created successfully')
        # pass
        # wb = openpyxl.load_workbook(path, 'rb')
        # ws = wb.active
        # ws.append(data)
        # wb.save(filename)
        # wb.close()
    '''#histogram
    plt.figure(figsize = (12,5))
    df['Speed_ISD'].hist(alpha = 0.5, bins = 10, label = 'ISD', range = [0,12])
    df['speed_MERRA2'].hist(alpha = 0.5, bins = 10, label = 'MERRA-2', range = [0,12])
    plt.title(str(station_ID) + ' ('+str(begin_date) + '-' + str(end_date) + ')', fontsize = 18)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.ylabel('Number of occurrences')
    plt.xlabel('Wind Speed [m/s]')'''
    # plt.annotate('Number of synchronized points: ' + str(len(df_flux['ISD_total_flux'])), xy = (0.75, 0.45), xycoords = 'axes fraction')
    # plt.annotate('(MERRA-2 - ISD): ' + str(np.round(sum(np.array(df_flux['MERRA2_total_flux'])) - sum(np.array(df_flux['ISD_total_flux'])),12)) + ' kg/s*m^2', xy = (0.25, 0.95), xycoords = 'axes fraction')
    # plt.annotate('r = 0.73 (MERRA-2 - ISD): ' + str(np.round(sum(np.array(df_flux['Flux0.73_merra2'])) - sum(np.array(df_flux['Flux0.73_isd'])),12))+ ' kg/s*m^2', xy = (0.75, 0.25), xycoords = 'axes fraction')
    # plt.annotate('r = 1.4 (MERRA-2 - ISD): ' + str(np.round(sum(np.array(df_flux['Flux1.4_merra2'])) - sum(np.array(df_flux['Flux1.4_isd'])),12))+ ' kg/s*m^2', xy = (0.75, 0.2), xycoords = 'axes fraction')
    # plt.annotate('r = 2.4 (MERRA-2 - ISD): ' + str(np.round(sum(np.array(df_flux['Flux2.4_merra2'])) - sum(np.array(df_flux['Flux2.4_isd'])),12))+ ' kg/s*m^2', xy = (0.75, 0.15), xycoords = 'axes fraction')
    # plt.annotate('r = 4.5 (MERRA-2 - ISD): ' + str(np.round(sum(np.array(df_flux['Flux4.5_merra2'])) - sum(np.array(df_flux['Flux4.5_isd'])),12))+ ' kg/s*m^2', xy = (0.75, 0.1), xycoords = 'axes fraction')
    # plt.annotate('r = 8.0 (MERRA-2 - ISD): ' + str(np.round(sum(np.array(df_flux['Flux8_merra2'])) - sum(np.array(df_flux['Flux8_isd'])),12))+ ' kg/s*m^2', xy = (0.75, 0.05), xycoords = 'axes fraction')
    # plt.annotate('MERRA-2 Given flux: '+ str(np.round(total_flux,12))+ ' kg/s*m^2', xy = (0.75, 0.30), xycoords = 'axes fraction')
    # plt.annotate('MERRA-2 calculated flux: ' + str(np.round(sum(np.array(df_flux['MERRA2_total_flux'])),12))+' kg/s*m^2', xy = (0.75, 0.35), xycoords = 'axes fraction')
    # plt.annotate('ISD calculated flux: ' + str(np.round(sum(np.array(df_flux['ISD_total_flux'])),12)) +' kg/s*m^2', xy = (0.75, 0.40), xycoords = 'axes fraction')

    print('MERRA-2 given - calculated: ' + str(total_flux - np.nansum(np.array(df_flux['MERRA2_total_flux']))))
    # plt.savefig(ParentDirectory + 'HistogramOutputs/' + str(station_ID) + 'LowB_dividebylength_Histogram_Ut_sized'+ ' ('+begin_date + '-' + end_date + ')'+ diurnalText +'.png', dpi = 200)
    plt.close()
    return

def Histogram_AllTime(df,station_ID, diurnalText, begin_date, end_date, mylat, mylong):
    plt.figure(figsize = (12,5))
    df = combineSubHourly(df)
    df['Speed_ISD'].hist(alpha = 0.5, bins = 10, label = 'ISD', range = [0,12])
    df['speed_MERRA2'].hist(alpha = 0.5, bins = 10, label = 'MERRA-2', range = [0,12])
    '''Threshold Velocity for size bins'''
    # U_t = U_tVector_const_GWETTOP(df,begin_date,end_date)
    U_t, df = U_tVector(df,begin_date,end_date, mylat, mylong)
    print(U_t.keys())
    print(U_t.head())
    colors = ['black','crimson','chartreuse','cyan','orchid']
    r = [0.73, 1.4, 2.4, 4.5, 8] #microns - size bins of MERRA-2
    # ut_keys = 'u_t at 0.73'
    plt.axvline(np.nanmean(U_t['0.73']), label = 'd = ' + str(np.round(2*r[0],3)) + 'um , U_t = ' + str(np.round(np.nanmean(U_t['0.73']),3)) + ' m/s', c = colors[0])
    plt.axvline(np.nanmean(U_t['1.4']), label = 'd = ' + str(np.round(2*r[1],3)) + 'um , U_t = ' + str(np.round(np.nanmean(U_t['1.4']),3)) + ' m/s', c = colors[1])
    plt.axvline(np.nanmean(U_t['2.4']), label = 'd = ' + str(np.round(2*r[2],3)) + 'um , U_t = ' + str(np.round(np.nanmean(U_t['2.4']),3)) + ' m/s', c = colors[2])
    plt.axvline(np.nanmean(U_t['4.5']), label = 'd = ' + str(np.round(2*r[3],3)) + 'um , U_t = ' + str(np.round(np.nanmean(U_t['4.5']),3)) + ' m/s', c = colors[3])
    plt.axvline(np.nanmean(U_t['8']), label = 'd = ' + str(np.round(2*r[4],3)) + 'um , U_t = ' + str(np.round(np.nanmean(U_t['8']),3)) + ' m/s', c = colors[4])

    plt.title(str(station_ID) + ' 2001-2020 10m Wind Speed ' + diurnalText, fontsize = 18)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.ylabel('Number of occurrences')
    plt.xlabel('Wind Speed [m/s]')
    plt.legend()
    plt.savefig('/Users/emily/Documents/UMBC/Dr_LimaLab/HistogramsISDvsMERRA2/' + str(station_ID) + diurnalText +'allTime_10bins_with_omega.png', dpi = 300)
    plt.close()

    return
#makes scatter plot of the seasonal means
def AverageScatter(directory, station_ID):
    station_folder = station_ID + '_DATA'
    df = ReadData(directory,station_ID)
    df = Seasons(df)
    df['Season'] = df['Season']
    plt.figure(figsize = (12,12))
    SeasonMeans_ISD = df.groupby( by = ['Season', 'year'])['Speed_ISD'].mean()
    SeasonSTD_ISD = df.groupby(by = ['Season', 'year'])['Speed_ISD'].std()
    SeasonMeans_MERRA2 = df.groupby( by = ['Season', 'year'])['speed_MERRA2'].mean()
    SeasonSTD_MERRA2 = df.groupby(by = ['Season', 'year'])['speed_MERRA2'].std()
    plt.scatter(SeasonMeans_ISD['Winter'],SeasonMeans_MERRA2['Winter'], label = 'Winter', marker = 'P', s = 100)
    plt.scatter(SeasonMeans_ISD['Summer'],SeasonMeans_MERRA2['Summer'], label = 'Summer', marker = 'p', s = 100)
    plt.scatter(SeasonMeans_ISD['Spring'],SeasonMeans_MERRA2['Spring'], label = 'Spring', marker = 'x', s = 100)
    plt.scatter(SeasonMeans_ISD['Fall'],SeasonMeans_MERRA2['Fall'], label = 'Fall', marker = '*', s = 100)

    #winter
    K = 1
    distances = []
    kmeans_model = KMeans(n_clusters = K).fit(np.array(list(zip(SeasonMeans_ISD['Winter'],SeasonMeans_MERRA2['Winter']))))
    centers = np.array(kmeans_model.cluster_centers_)
    distances.append(np.abs(centers[:,0]-centers[:,1])/(np.sqrt(2)))
    plt.scatter(centers[:,0], centers[:,1], label = 'Winter Centroid ' + str(distances[0]), marker = 's', s = 100, c = 'blue', edgecolors = 'black')
    #to do: Find the distance between the centroids and the 1:1 line
    #summer
    kmeans_model = KMeans(n_clusters = K).fit(np.array(list(zip(SeasonMeans_ISD['Summer'],SeasonMeans_MERRA2['Summer']))))
    centers1 = np.array(kmeans_model.cluster_centers_)
    distances.append(np.abs(centers1[:,0]-centers1[:,1])/(np.sqrt(2)))
    plt.scatter(centers1[:,0], centers1[:,1], label = 'Summer Centroid ' + str(distances[1]), marker = 's', s = 100, c = 'orange', edgecolors = 'black')
    #spring
    kmeans_model = KMeans(n_clusters = K).fit(np.array(list(zip(SeasonMeans_ISD['Spring'],SeasonMeans_MERRA2['Spring']))))
    centers2 = np.array(kmeans_model.cluster_centers_)
    distances.append(np.abs(centers2[:,0]-centers2[:,1])/(np.sqrt(2)))
    plt.scatter(centers2[:,0], centers2[:,1], label = 'Spring Centroid ' + str(distances[2]), marker = 's', s = 100, c = 'green', edgecolors = 'black')
    #fall
    kmeans_model = KMeans(n_clusters = K).fit(np.array(list(zip(SeasonMeans_ISD['Fall'],SeasonMeans_MERRA2['Fall']))))
    centers3 = np.array(kmeans_model.cluster_centers_)
    distances.append(np.abs(centers3[:,0]-centers3[:,1])/(np.sqrt(2)))
    plt.scatter(centers3[:,0], centers3[:,1], label = 'Fall Centroid ' + str(distances[3]), marker = 's', s = 100,c = 'red', edgecolors = 'black')

    xx = np.linspace(1,7,100)
    plt.plot(xx,xx, label = '1:1')

    plt.ylabel('MERRA-2 Average[m/s]', fontsize = 30)
    plt.xlabel('ISD Average[m/s]', fontsize = 30)
    plt.title('Modeled (MERRA-2) vs. Measured (ISD) Surface \n Wind Speed for station #' + station_ID, fontsize = 35)
    plt.legend(facecolor = '0.75', fontsize = 17)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    # plt.show()
    plt.savefig(directory + '/' + station_folder + 'MERRA_2vsISD_seasonal.png', dpi = 200)
    return

def diurnal_df(df):
    daytime_on = 3
    daytime_off = 15

    daytime_df = df[(df['DATE'].dt.hour >= daytime_on) & (df['DATE'].dt.hour <= daytime_off)]
    nighttime_df = df[(df['DATE'].dt.hour <= daytime_on) | (df['DATE'].dt.hour >= daytime_off)]
    return daytime_df, nighttime_df

def GeneralBubblePlot(df, sizeName, titleString):
    crs = ccrs.PlateCarree()
    fig, ax = plt.subplots(1,1, subplot_kw = {'projection': crs}, figsize = (9,9))
    # print(sorted(df['MERRA-2_emissions_calculated']*(10**9)-df['ISD_emissions']*(10**9)))
    df['MERRA-2_emissions_calculated']  = df['MERRA-2 emissions_calculated']*(10**9)
    df['ISD_emissions'] = df['ISD total']*(10**9)
    print(df.head())
    print(df['MERRA-2_emissions_calculated'])
      # print(np.min(df['MERRA-2_emissions_calculated']-df['ISD_emissions']))
    # print(np.max(df['MERRA-2_emissions_calculated']-df['ISD_emissions']))
    ax.set_extent([30, 60, 10, 40], ccrs.PlateCarree()) #hardcoded middle east
    ax.coastlines(resolution='10m')
    # colormap = plt.cm.RdBu
    # normalize = matplotlib.colors.Normalize(vmin = -1*10e-10, vmax = 1*10e-10)
    # print(normalize)
    Scatter = ax.scatter(np.array(df['longitude']), np.array(df['latitude']), c = (df['MERRA-2_emissions_calculated']-df['ISD_emissions']), vmin= -8, vmax = 8, s = df[str(sizeName)].fillna(0), cmap = 'RdBu', alpha = .7, edgecolors = 'black')
    ax.title.set_text(titleString)

    color_legend = ax.legend(*Scatter.legend_elements(), title = 'Difference in Emissions' + "\n" +"[kg/m^2s] x10^-9", bbox_to_anchor = (1.05,.9), facecolor='white', framealpha=1)
    ax.add_artist(color_legend)

    handles, labels = Scatter.legend_elements(prop = 'sizes', alpha = 0.6)
    size_legend = ax.legend(handles,labels, title = '% Difference ISD vs ' + "\n" + "M2 Re-calculated", bbox_to_anchor = (1.05,.5),facecolor='white', framealpha=1)

    plt.savefig('ISDWind/MiddleEast/BubblePlot_outputs/2001-2020/' + titleString + '.png', dpi = 300)
    plt.close()

    return
# def diurnal_plot():
#     station_number = input('Which ISD station number would you like: ')
#     # station_number = '40199099999'
#     Begin_date = input('Begining date (dd/mm/yyyy): ')
#     End_date = input('Begining date (dd/mm/yyyy): ')
#
#     # Begin_date = '01/09/2015'
#     # End_date = '15/09/2015'
#     ParentDirectory_ISD = '/Users/emily/Documents/UMBC/Dr_LimaLab/ISDWind/MiddleEast/' + str(station_number) + '_DATA/'
#     ParentDirectory_MERRA2 = ''
#
#     # End_date = input('End date (dd-mm-yyyy): ')
#     return

'''def bubblePlot():


    return
'''
'''
def AERONET_PLOT():
    # station_number = input('Which ISD station number would you like: ')
    station_number = '40199099999'
    # AERONET_station = input('Which AERONET station name?: ')
    AERONET_station = 'Eilat'
    # Begin_date = input('Begining date (dd-mm-yyyy): ')
    Begin_date = '01/08/2015'
    End_date = '15/09/2015'
    # End_date = input('End date (dd-mm-yyyy): ')

    ParentDirectory_ISD_MERRA2 = '/Users/emily/Documents/UMBC/Dr_LimaLab/ISDWind/MiddleEast/' + str(station_number) + '_DATA/'
    ParentDirectory_AERONET = '/Users/Emily/Documents/UMBC/Dr_LimaLab/AERONET/' + str(AERONET_station) +'/'
    ISD_MERRA2_df = ReadData(ParentDirectory_ISD_MERRA2, station_number)
    # ISD_MERRA2_df = pd.read_csv(ParentDirectory_ISD_MERRA2 + 'MERRA2_intersection_' + station_number + '.csv', low_memory = False)
    for entry in os.scandir(ParentDirectory_AERONET):
        filename_string = entry.name
    AERONET_df = pd.read_csv(ParentDirectory_AERONET + filename_string, skiprows = 6, low_memory = False)
    #datetime object formatting
    AERONET_df['Date(dd:mm:yyyy)'] = AERONET_df['Date(dd:mm:yyyy)'].str.replace(':', '/')
    AERONET_df['Date(dd:mm:yyyy)'] = pd.to_datetime(AERONET_df['Date(dd:mm:yyyy)'], format ='%d/%m/%Y')
    ISD_MERRA2_df['DATE'] = pd.to_datetime(ISD_MERRA2_df['DATE'], format = '%Y-%m-%d %H:%M:%S')
    #https://sparkbyexamples.com/pandas/pandas-select-dataframe-rows-between-two-dates/
    #choose dates from user input
    AERONET_df_myDates = AERONET_df[AERONET_df['Date(dd:mm:yyyy)'].between(Begin_date, End_date)]
    ISD_MERRA2_df_myDates = ISD_MERRA2_df[ISD_MERRA2_df['DATE'].between(Begin_date, End_date)]

    # ISD_MERRA2_df = ISD_MERRA2_df['DATE'].between(Begin_date, End_date)
    # ISD_MERRA2_df_myDates = ISD_MERRA2_df_myDates.to_frame()
    print(ISD_MERRA2_df_myDates.head())
    print(AERONET_df_myDates.head())
    print(ISD_MERRA2_df_myDates.tail())
    print(AERONET_df_myDates.tail())
    # print(ISD_MERRA2_df_myDates.info())
    # print(AERONET_df_myDates.info())
    # print(AERONET_df_myDates.tail())
    # print(ISD_MERRA2_df_myDates.head())
    # print(ISD_MERRA2_df_myDates.tail())
    # print(ISD_MERRA2_df_myDates.info())
    # print(AERONET_df_myDates['AOD_675nm'])

    #Plot AERONET AOD
    plt.figure(figsize = (16,9))
    plt.subplot(2,1,1)
    plt.plot(AERONET_df_myDates['Date(dd:mm:yyyy)'],AERONET_df_myDates['AOD_675nm'], label = 'AERONET AOD @ 675nm')
    plt.xlabel('Time')
    plt.ylabel('AOD 675nm')
    plt.title('AERONET - ' + str(AERONET_station) + ' AOD at 674nm')
    plt.legend()

    #plot ISD and MERRA-2 10 meter winds
    plt.subplot(2,1,2)
    plt.plot(ISD_MERRA2_df_myDates['DATE'], ISD_MERRA2_df_myDates['Speed_ISD'], label = 'ISD 10m wind speed')
    plt.plot(ISD_MERRA2_df_myDates['DATE'], ISD_MERRA2_df_myDates['speed_MERRA2'], label = 'MERRA-2 10m wind speed')
    plt.ylabel('10 meter wind speed [m/s]')
    plt.xlabel('Time')
    plt.title('ISD and MERRA-2 10m wind speed @' + str(station_number))
    plt.legend()
    # plt.show()
    plt.savefig(ParentDirectory_AERONET + '/Comparison_with_ISD_' + str(station_number) + '_.png', dpi = 200)
    return
'''
