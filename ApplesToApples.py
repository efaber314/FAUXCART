#apples to apples functions - python
#Emily Faber
#January 2022
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import datetime
from datetime import date, timedelta
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
# import monet
import xarray as xr
import scipy 
import geocat.viz as gv

#makes data frame from a single file in directory
def ReadData(directory, station_ID):
    for entry in os.scandir(directory):
        if entry.name.endswith('intersection_'+str(station_ID + '.csv')):
            print(entry)
            df = pd.read_csv(entry, low_memory = False)
    return df

'''Takes in a latitude and longitude and returns the index of the associated grid
box to retrieve from in MERRA-2 '''
def colocationIndicies(lat,long):
    i = int((long + 180)/(5/8))+1
    j = int((lat + 90)/0.5)+1
    longIndex = i - 1
    latIndex = j - 1
    return longIndex, latIndex

#taking a property name from the nc file, can return a merra-2 data frame of the
#property and the time range wanted.
def ReadDataNC_MERRA2(df,directory,lat,long,begin_date,end_date,property):
    print('lenght of df ', len(df))
    print(str(property))
    filelist = []
    datelist = []
    for entry in os.scandir(directory):
        if(entry.path.endswith(".nc")):
            dateString = entry.name[-15:-7] #This is a dangerous hard code
            dateTime = pd.to_datetime(dateString)
            if dateTime >= pd.to_datetime(begin_date) and dateTime <= pd.to_datetime(end_date):
                filelist.append(entry.name)
                datelist.append(dateTime)

    filelist = sorted(filelist)
    Property = []
    df_property = pd.DataFrame()
    timevector = np.arange(pd.to_datetime(begin_date), pd.to_datetime(end_date)+datetime.timedelta(days=1), dtype = 'datetime64[h]')
    print(len(timevector))
    print('length of file list: ', len(filelist), len(filelist)*24, str(property))
    for file in filelist:
        ds = nc.Dataset(directory + str(file))
        property_temp = ds.variables[str(property)]
        i,j = colocationIndicies(lat,long)
        Property.extend(property_temp[:,j,i])
    df_property[str(property)] = Property
    df_property['DateTime'] = timevector
    print('Length of property after being read in: ',len(Property))

    '''https://www.geeksforgeeks.org/python-find-missing-additional-values-two-lists/'''
    '''https://stackoverflow.com/questions/43269548/pandas-how-to-remove-rows-from-a-dataframe-based-on-a-list'''
    '''https://stackoverflow.com/questions/62543350/pandas-force-minute-and-seconds-to-be-zero'''

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
            ds.close()
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
        i,j = colocationIndicies(lat,long)
        Property.extend(property_temp[:,j,i])
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

    return df_property, df_combinedHourly
#finds the monthly means of MERRA-2 and ISD and returns as a vector
def monthlyMeans240(df, ISDColumnName, M2ColumnName):
    MERRA_2_Hourly_Means = df.groupby(by = ['year','month','day','hour'])[M2ColumnName].mean()
    ISD_Hourly_Means = df.groupby(by = ['year','month','day','hour'])[ISDColumnName].mean()
    '''https://stackoverflow.com/questions/10373660/converting-a-pandas-groupby-output-from-series-to-dataframe '''
    MERRA_2_Hourly_Means = MERRA_2_Hourly_Means.reset_index()
    ISD_Hourly_Means = ISD_Hourly_Means.reset_index()
    #get monthly means for each and make the series a data frame
    MERRA_2_Monthly_Means = MERRA_2_Hourly_Means.groupby(by = ['year', 'month'])[M2ColumnName].mean()
    MERRA_2_Monthly_Means = MERRA_2_Monthly_Means.reset_index()
    num_points = ISD_Hourly_Means.groupby(by = ['year','month'])[ISDColumnName].size()
    num_points = num_points.reset_index()
    ISD_Monthly_means = ISD_Hourly_Means.groupby(by = ['year', 'month'])[ISDColumnName].mean()
    ISD_Monthly_means = ISD_Monthly_means.reset_index()
    # dateDF = pd.to_datetime(df['DATE'])
    # print(dateDF['DATE'])
    # DaysInMonths = dateDF.groupby(by = [dateDF['DATE'].year, dateDF['DATE'].month])['DATE'].datetime.daysinmonth
    # print('Number of days in months with measurements')
    # print(DaysInMonths)

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

def DaysInMonth(df):
    days_list = []
    for j in range(0, len(df)):
        month = df['month'][j]
        year = df['year'][j]

        if((month==2) and ((year%4==0)  or ((year%100==0) and (year%400==0)))):
            days_list.append(29)

        elif(month==2):
            days_list.append(28)

        elif(month==1 or month==3 or month==5 or month==7 or month==8 or month==10 or month==12):
            days_list.append(31)

        else:
            days_list.append(30)
    return days_list

#makes plot of all years monthly means - should be 20 years * 12 months = 240 points
def allYearsPlot(df, station_ID, ISDColumnName, M2ColumnName):
    ISD_Monthly_means, MERRA_2_Monthly_Means, num_points = monthlyMeans240(df,ISDColumnName, M2ColumnName)
    days_list = DaysInMonth(MERRA_2_Monthly_Means)
    print(ISD_Monthly_means)
    # print(days_list)
    # ISD_Monthly_means[ISDColumnName] = ISD_Monthly_means[ISDColumnName]*60*60*24*days_list
    # MERRA_2_Monthly_Means[M2ColumnName] = MERRA_2_Monthly_Means[M2ColumnName]*60*60*24*days_list
    # print(num_points.head(20)) #year month Speed_ISD, where Speed_ISD is number of points in the month
    #sanity check
    '''https://www.w3schools.com/python/python_datetime.asp'''
    MERRA_2_Monthly_Means['DATE'] = MERRA_2_Monthly_Means['year']
    ISD_Monthly_means['DATE'] = ISD_Monthly_means['year']
    Source_Value = df['source_used'][0]

    for i in range(0, len(MERRA_2_Monthly_Means)):
        MERRA_2_Monthly_Means['DATE'][i] = datetime.datetime(MERRA_2_Monthly_Means['year'][i],MERRA_2_Monthly_Means['month'][i],15)
        ISD_Monthly_means['DATE'][i] = datetime.datetime(ISD_Monthly_means['year'][i],ISD_Monthly_means['month'][i],15)

    #plot M2 original
    # RecalculatedMonthlyMeansM2Location = ''
    # M2_original_monthly_means = pd.read_csv(RecalculatedMonthlyMeansM2Location+str(station_ID)+'.csv')
    # print('Days in months')
    # print(ISD_Monthly_means['DATE'].dt.daysinmonth)
    fig,ax = plt.subplots(figsize = (12,6))
    plt.plot(ISD_Monthly_means['DATE'],ISD_Monthly_means[ISDColumnName], label = "ISD Measurements", linewidth = 8)
    plt.plot(MERRA_2_Monthly_Means['DATE'], MERRA_2_Monthly_Means[M2ColumnName], label = 'MERRA-2 data (time-coordinated values)', linewidth = 8)
    dates = pd.date_range('2001-01-01','2020-12-31',freq = 'MS')
    # plt.plot(dates, M2_original_monthly_means['w10'], label = 'Original MERRA-2 data (all values)', linewidth = 12)

    #trendlines
    slope_ISD, intercept_ISD = np.polyfit(np.arange(0,len(ISD_Monthly_means['DATE']),1), ISD_Monthly_means[ISDColumnName],1)
    slope_M, intercept_M = np.polyfit(np.arange(0,len(MERRA_2_Monthly_Means['DATE']),1), MERRA_2_Monthly_Means[M2ColumnName],1)

    line_isd = [slope_ISD * i + intercept_ISD for i in np.arange(0,len(ISD_Monthly_means['DATE']),1)]
    line_M = [slope_M * i + intercept_M for i in  np.arange(0,len(MERRA_2_Monthly_Means['DATE']),1)]

    correlation_matrix_isd = np.corrcoef(ISD_Monthly_means[ISDColumnName], line_isd)
    corr_isd = correlation_matrix_isd[0,1]
    R2_isd = corr_isd**2
    correlation_matrix_M = np.corrcoef(MERRA_2_Monthly_Means[M2ColumnName], line_M)
    corr_M = correlation_matrix_M[0,1]
    R2_M = corr_M**2

    #Pearson Correlation Coefficient - returns value between -1 and 1. r = (sum of (x-mx)(y-my))/(sqrt(sum of (x-xm)^2 (y-my)^2))
    # pcc = scipy.stats.pearsonr(ISD_Monthly_means[ISDColumnName].dropna(), MERRA_2_Monthly_Means[M2ColumnName].dropna())

    #trendline plot
    # plt.plot(ISD_Monthly_means['DATE'], line_isd, label = 'ISD trend ' + str(np.round(slope_ISD,5)) + ' $R^2$: ' +str(round(R2_isd,3)))
    # plt.plot(MERRA_2_Monthly_Means['DATE'], line_M, label = 'MERRA-2 trend ' + str(np.round(slope_M,5))+ ' $R^2$: ' +str(round(R2_M,3)))

    #calculate the Mean Absolute Error (MAE)
    #sum of absolute errors divided by sample size
    absolute_errors = np.array(line_M) - np.array(line_isd)
    sum_abs_errors = sum(absolute_errors)
    MAE = sum_abs_errors/len(line_M)

    # plt.text(ISD_Monthly_means['DATE'][0], 7.5, np.round(pcc,3))
    plt.legend(fontsize = 16)
    plt.xlabel('Time [m]', fontsize = 16)

    plt.ylabel('Monthly Average [m/s]', fontsize = 16)
    # plt.ylabel('Monthly Average [$kg/m^2/month$]', fontsize = 16)

    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)

    # plt.ylim(0,8)
    ax.yaxis.offsetText.set_fontsize(18)
    # plt.ylim(0,1.15*max(max(MERRA_2_Monthly_Means[M2ColumnName]),max(ISD_Monthly_means[ISDColumnName])))
    plt.ylim(0, 0.015)
    # plt.title(str(station_ID) + ' All Monthly Averages, ' + str(format(MAE, '0.3f')) + ' MAE [m/s]', fontsize = 18)
    # plt.title(str(station_ID) + ' All Monthly Averages, ' + str(format(MAE, '0.3E')) + ' MAE [m/s], ' + str(np.round(pcc[0],3)) + ' PCC', fontsize = 18)
    plt.title(str(station_ID) + ' All Monthly Averages' , fontsize = 18)

    # plt.title(str(station_ID) + ' All Monthly Averages, \n Source Function at this Location: '+ str(np.round(Source_Value,3)), fontsize = 40)
    
    #plot number of measurements in each month 
    # ax2 = ax.twinx()
    # ax2.set_ylim(0,1000)
    # ax1.bar(x = np.arange(1,31,1) - .25, height = barValues, width = .5, label = 'Difference [ISD - MERRA-2 Recalculated] \n 20-Year Average $kg/m^2$', color = 'orange', alpha = .85)
    # for j in range(0, len(ISD_Monthly_means[ISDColumnName])):
    # ax2.bar(x = ISD_Monthly_means['DATE'], height = num_points[ISDColumnName], color="pink", width = 30, label = 'number of measurments in each month', alpha = .3)    
    # ax2.axhline(y=720, alpha=0.3, color = 'red')  
    SaveDirectoryAll = '/Users/emily/Documents/UMBC/Dr_LimaLab/MERRA2VsISD_YearsTimeSeries/'
    plt.savefig(SaveDirectoryAll  + str(station_ID) +'_timeCorrelated_TOPO_20year_SameScale'+str(date.today())+str(ISDColumnName)+'.png', dpi = 600)
    plt.close()
    # plt.show()

'''Depricated '''
def DepricatedScaleSourceFunction():
    S = nc.Dataset('/Users/emily/Documents/UMBC/Dr_LimaLab/Merra2_Wind/gocart.dust_source.v5a.x1152_y721.nc', low_memory = False)
    source_function_whole = np.array(S.variables['du_src'])
    #source function is 721, 1152 (lat,long)
    source_function = np.zeros((361,576))
    for i in range(0, 361):
        for j in range(0,576):
            # print(source_function_whole[0,2*i,2*j])
            source_function[i,j] = np.mean(source_function_whole[0,2*i:2*i+1,2*j:2*j+1])
    # print(source_function.shape)
    return source_function

'''Depricated '''
def DepricatedRetrieveSourceFunction(latitude, longitude):
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
def DepricatedFlux_MB(df,begin_date,end_date):
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

def DepricatedU_tvsR(df, begin_date,end_date):
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
def DepricatedFlux_Ginoux(df, begin_date, end_date):
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

'''Depricated '''
def DepricatedU_tVector(df,begin_date,end_date, mylat, mylong):
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
        # print(soil_wetness.head())
        # print(soil_wetness.dtypes)
        # test = np.log(soil_wetness['GWETTOP'])
        # print(test)
        # print(test.dtypes)
        U_t_vector = U_t * (1.2 + 0.2*np.log(soil_wetness['GWETTOP']))#same dimensions as soil_wetness
        print('length of U_t after ginoux correction: ', len(U_t_vector))
        # print(df.tail(5), U_t_vector.tail(5))
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
        # print(df.head(5))
        # print(len(df), 'length of df at end of Ut Vector')

    return U_t_df, df

'''Depricated '''
def DepricatedU_tVector_const_GWETTOP(df,begin_date,end_date):
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

'''Depricated'''
#makes a histogram of the wind speed values and data points. All points that have been time and space aligned
def DepricatedCombinationFlux(df,flux001,flux002,flux003,flux004,flux005, begin_date, end_date, mylong,mylat,map):
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

        '''Source Function retrieval

        1 -- topographic source (OG)
        2 -- Sediment supply map (FENGSHA)

        '''
        if map == '1':
            source_function = ScaleSourceFunction()
            #returns 360x575 - the entire map
            i = round(((float(mylong) + 180)/(5/8)) + 1)
            j = round(((float(mylat) + 90)/0.5) + 1)
            source_function = source_function[j,i]
            print('source function', source_function)
        if map == '2':
            source_function = source_function_vector_ssm(df, mylat, mylong, interpolation)
            # ds = xr.open_dataset('/Users/emily/Documents/UMBC/Dr_LimaLab/FENGSHA/FENGSHA_SOILGRIDS2017_GEFSv12_v1.2-2.nc')
            # source_function = ds.monet.nearest_latlon(lat = mylat, lon = mylong)
            # print('Sediment Supply Map value', float(source_function.ssm[6]))
            # source_function = float(source_function.ssm[6])
        # print(len(hourly_df['Speed_ISD']), ' ISD windspeed length after combining subhourly')
        for hour in df['Speed_ISD']:
            if df['Speed_ISD'][hour] > U_t[hour]:
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

'''Takes in aligned data from FAUCART and returns a dataframe that has combined
any sub-hourly measurements by averaging them '''
def combineSubHourly(df):
    '''https://towardsdatascience.com/pandas-dataframe-group-by-consecutive-same-values-128913875dba'''
    test_speed_group_ISD = df.groupby((df['hour'].shift() != df['hour']).cumsum())['Speed_ISD'].mean(numeric_only=True)

    # print(len(test_speed_group_ISD))
    # print(test_speed_group_ISD.head(20))
    test_speed_group_M2 = df.groupby((df['hour'].shift() != df['hour']).cumsum())['OG_winds_correct_box'].mean(numeric_only=True)
    # print(test_speed_group_M2.head(20))
    new_df = pd.DataFrame()
    new_df['Speed_ISD'] = np.array(test_speed_group_ISD)
    new_df['OG_winds_correct_box'] = np.array(test_speed_group_M2)
    new_df['LATITUDE'] = df.groupby((df['hour'].shift() != df['hour']).cumsum())['LATITUDE'].mean(numeric_only=True)
    new_df['LONGITUDE'] = df.groupby((df['hour'].shift() != df['hour']).cumsum())['LONGITUDE'].mean(numeric_only=True)
    df['DATE'] = pd.to_datetime(df['DATE'])
    new_df['DATE'] = df.groupby((df['hour'].shift() != df['hour']).cumsum())['DATE'].agg(pd.Series.mean)
    new_df['hour'] = df.groupby((df['hour'].shift() != df['hour']).cumsum())['hour'].mean(numeric_only=True)
    new_df['month'] = df.groupby((df['month'].shift() != df['month']).cumsum())['month'].mean(numeric_only=True)
    new_df['year'] = df.groupby((df['year'].shift() != df['year']).cumsum())['year'].mean(numeric_only=True)

    return new_df

'''Depricated'''
def Depricatedhistogram(df,station_ID,begin_date,end_date,mylong,mylat,diurnalText,map):

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
    filename = ParentDirectory +'PercentDifference_files/'+'ME('+ str(begin_date)[0:10] + '-' + str(end_date)[0:10] + ')_withSizeBins_omega_map_experiments' +str(map)+ '.xlsx'
    path = Path(ParentDirectory +'PercentDifference_files/' +'ME('+ str(begin_date)[0:10] + '-' + str(end_date)[0:10] + ')_withSizeBins_omega_map_experiments'+str(map) + '.xlsx')
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

def Unmodified_threshold():
    #U_t for each box, uncorrected by GWETTOP
    U_t = []
    Radius = [.73, 1.4, 2.4, 4.5, 8]
    s_p = [0.1, 0.25, 0.25, 0.25, 0.25]
    densities = [2650,2650,2650,2650,2650]
    density_air = 1.250 #kg/m^3
    g = 9.81 #m/s^2

    for i in range(0,len(Radius)):
        r = Radius[i]
        R = r * 10**-6 #m
        # sp = s_p[i]
        density_particle = densities[i]
        temp1 = .13*np.sqrt((density_particle*g*2*R)/density_air)
        temp2 = np.sqrt(1+6e-7/(density_particle*g*(2*R)**2.5))
        temp3 = np.sqrt(1.928*(((1331*(100*(2*R))**1.56)+.38)**.092)-1)
        U_t0 = (temp1*temp2)/temp3
        U_t.append(U_t0)

    return U_t
'''WIND SPEED histogram maker - takes a
df of the co-located data,
a string of the stationID,
diurnalText options: all time, day time, night time,
mylat, mylong is station location

saves histogram to Histogram outputs location '''
def Histogram_AllTime(df,station_ID, diurnalText):
    WhereToSave = '/Users/emily/Documents/UMBC/Dr_LimaLab/HistogramsISDvsMERRA2/'
    df = combineSubHourly(df)
    plt.figure(figsize = (17,7))
    df['Speed_ISD'].hist(alpha = 0.5, bins = 10, label = 'ISD Wind Speed', range = [0,12])
    df['OG_winds_correct_box'].hist(alpha = 0.5, bins = 10, label = 'MERRA-2 Wind Speed', range = [0,12])
    '''Threshold Velocity for size bins - TODO: add back vertical lines for the average threshold for each size bin'''
    # U_t = U_tVector_const_GWETTOP(df,begin_date,end_date)
    # U_t, df = U_tVector(df,begin_date,end_date, mylat, mylong)
    # print(U_t.keys())
    # print(U_t.head())
    colors = ['black','crimson','chartreuse','cyan','orchid']
    r = [0.73, 1.4, 2.4, 4.5, 8] #microns - size bins of MERRA-2
    # ut_keys = 'u_t at 0.73'
    U_t = Unmodified_threshold()
    plt.axvline(np.nanmean(U_t[0]), label = 'D = ' + str(np.round(2*r[0],3)) + 'um , u_t = ' + str(np.round(np.nanmean(U_t[0]),3)) + ' m/s', c = colors[0])
    plt.axvline(np.nanmean(U_t[1]), label = 'D = ' + str(np.round(2*r[1],3)) + 'um , u_t = ' + str(np.round(np.nanmean(U_t[1]),3)) + ' m/s', c = colors[1])
    plt.axvline(np.nanmean(U_t[2]), label = 'D = ' + str(np.round(2*r[2],3)) + 'um , u_t = ' + str(np.round(np.nanmean(U_t[2]),3)) + ' m/s', c = colors[2])
    plt.axvline(np.nanmean(U_t[3]), label = 'D = ' + str(np.round(2*r[3],3)) + 'um , u_t = ' + str(np.round(np.nanmean(U_t[3]),3)) + ' m/s', c = colors[3])
    plt.axvline(np.nanmean(U_t[4]), label = 'D = ' + str(np.round(2*r[4],3)) + 'um , u_t = ' + str(np.round(np.nanmean(U_t[4]),3)) + ' m/s', c = colors[4])
    plt.title(str(station_ID) + ' 2001-2020 10m Wind Speed ' + diurnalText, fontsize = 25)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.ylabel('Number of occurrences', fontsize = 20)
    plt.xlabel('Wind Speed [m/s]', fontsize = 20)
    plt.legend(fontsize = 18)
    plt.savefig(WhereToSave + str(station_ID) + diurnalText +str(datetime.date.today()) +'.png', dpi = 600)
    plt.close()
    return
'''Depricated '''
def DepricatedWrite_Flux(df,station_ID, diurnalText, begin_date, end_date, mylat, mylong):
    df = combineSubHourly(df)
    U_t, df = U_tVector(df,begin_date,end_date, mylat, mylong)


    return
'''makes scatter plot of the seasonal means as a 1:1 plot
takes top of directory where the colocated station data is
and the stationID as a string

Saves figure back to the same directory the colocated station data is'''
def AverageScatter(df, station_ID):
    df = Seasons(df) #provides seasonal labels for each measurement/value row
    df['Season'] = df['Season']
    plt.figure(figsize = (12,12))
    SeasonMeans_ISD = df.groupby( by = ['Season', 'year'])['Speed_ISD'].mean()
    SeasonSTD_ISD = df.groupby(by = ['Season', 'year'])['Speed_ISD'].std()
    SeasonMeans_MERRA2 = df.groupby( by = ['Season', 'year'])['OG_winds_correct_box'].mean()
    SeasonSTD_MERRA2 = df.groupby(by = ['Season', 'year'])['OG_winds_correct_box'].std()
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
    plt.scatter(centers[:,0], centers[:,1], label = 'Winter Centroid ' + str(np.round(distances[0],3)), marker = 's', s = 100, c = 'blue', edgecolors = 'black')
    plt.errorbar(centers[:,0], centers[:,1],  yerr = np.std(SeasonMeans_MERRA2['Winter']), xerr = np.std(SeasonMeans_ISD['Winter']), fmt="o")

    #summer
    kmeans_model = KMeans(n_clusters = K).fit(np.array(list(zip(SeasonMeans_ISD['Summer'],SeasonMeans_MERRA2['Summer']))))
    centers1 = np.array(kmeans_model.cluster_centers_)
    distances.append(np.abs(centers1[:,0]-centers1[:,1])/(np.sqrt(2)))
    plt.scatter(centers1[:,0], centers1[:,1],label = 'Summer Centroid ' + str(np.round(distances[1],3)), marker = 's', s = 100, c = 'orange', edgecolors = 'black')
    plt.errorbar(centers1[:,0], centers1[:,1],  yerr = np.std(SeasonMeans_MERRA2['Summer']), xerr = np.std(SeasonMeans_ISD['Summer']), fmt="o")

    #spring
    kmeans_model = KMeans(n_clusters = K).fit(np.array(list(zip(SeasonMeans_ISD['Spring'],SeasonMeans_MERRA2['Spring']))))
    centers2 = np.array(kmeans_model.cluster_centers_)
    distances.append(np.abs(centers2[:,0]-centers2[:,1])/(np.sqrt(2)))
    plt.scatter(centers2[:,0], centers2[:,1], label = 'Spring Centroid ' + str(np.round(distances[2],3)), marker = 's', s = 100, c = 'green', edgecolors = 'black')
    plt.errorbar(centers2[:,0], centers2[:,1],  yerr = np.std(SeasonMeans_MERRA2['Spring']), xerr = np.std(SeasonMeans_ISD['Spring']), fmt="o")

    #fall
    kmeans_model = KMeans(n_clusters = K).fit(np.array(list(zip(SeasonMeans_ISD['Fall'], SeasonMeans_MERRA2['Fall']))))
    centers3 = np.array(kmeans_model.cluster_centers_)
    distances.append(np.abs(centers3[:,0]-centers3[:,1])/(np.sqrt(2)))
    plt.scatter(centers3[:,0], centers3[:,1], label = 'Fall Centroid ' + str(np.round(distances[3],3)), marker = 's', s = 100,c = 'red', edgecolors = 'black')
    plt.errorbar(centers3[:,0], centers3[:,1],  yerr = np.std(SeasonMeans_MERRA2['Fall']), xerr = np.std(SeasonMeans_ISD['Fall']), fmt="o")
    
    xx = np.linspace(1,7,100)
    plt.plot(xx,xx, label = '1:1')
    plt.ylabel('MERRA-2 Average[m/s]', fontsize = 30)
    plt.xlabel('ISD Average[m/s]', fontsize = 30)
    plt.title('Modeled (MERRA-2) vs. Measured (ISD) Surface \n Wind Speed for station #' + str(station_ID), fontsize = 35)
    plt.legend(facecolor = '0.75', fontsize = 18)
    plt.xticks(fontsize = 22)
    plt.yticks(fontsize = 22)
    # plt.show()
    plt.savefig('/Users/emily/Documents/UMBC/Dr_LimaLab/SeasonalScatterPlots/' +str(station_ID)+ 'MERRA_2vsISD_seasonal_boxUpdate.png', dpi = 600)
    plt.close()
    return
'''Takes a data frame of colocated data and returns two data frames, one with only local
daylight values, and the second with only local night time valeus '''
def diurnal_df(df):
    daytime_on = 3 #local to the Middle East
    daytime_off = 15 #local to the Middle East

    daytime_df = df[(df['DATE'].dt.hour >= daytime_on) & (df['DATE'].dt.hour <= daytime_off)]
    nighttime_df = df[(df['DATE'].dt.hour <= daytime_on) | (df['DATE'].dt.hour >= daytime_off)]
    return daytime_df, nighttime_df

#makes the bubble plot
def SeasonMapPlot(winterAVG,summerAVG,springAVG,fallAVG,winterSTD,SummerSTD,SpringSTD,FallSTD,lat,long,titleString, color, min,max):
    crs = ccrs.PlateCarree()
    fig, ax = plt.subplots(2,2, subplot_kw = {'projection': crs}, figsize = (13,12))
    numbers = np.arange(1,31,1)
    '''middle east extent'''
    ax[0,0].set_extent([30, 60, 10, 40], ccrs.PlateCarree())
    '''southwest extent'''
    # ax[0,0].set_extent([-126, -93, 25, 50], ccrs.PlateCarree())
    # ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')
    # ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor='black')
    # ax.add_feature(cf.BOARDERS)
    # ax[0,0].states(resolution = '10m')
    MasterDf = pd.read_csv('/Users/emily/Documents/UMBC/Dr_LimaLab/ISDWind/MiddleEast/10m_confirmed_stations.csv')
    ax[0,0].coastlines(resolution='10m')
    winterScatter = ax[0,0].scatter(np.array(long), np.array(lat), s = winterSTD*100, c = winterAVG, cmap = color, vmin = min, vmax = max, alpha = .8, edgecolors = 'black')
    for i in range(len(MasterDf)):
        x = MasterDf['Long'][i]
        y = MasterDf['Lat'][i]
        ax[0,0].text(x * (1 + 0.01), y * (1 + 0.01) , i+1, fontsize=10, c = 'black')
    ax[0,0].title.set_text(titleString+' Winter 10m Wind Speed')
    # AVGlegend = ax[0,0].legend(*winterScatter.legend_elements(num = 6), loc = 'lower left', title = 'Average 10m \nWind Speed [m/s]', bbox_to_anchor = (-.35,-.2), frameon = False)
    # ax[0,0].add_artist(AVGlegend)
    # extraScatter = ax[0,0].scatter(np.array(np.append(long,long)), np.array(np.append(lat,lat)), s = np.append(winterSTD*100,SummerSTD*100), c = np.append(winterAVG, summerAVG), cmap = color, vmin = min, vmax = max, alpha = .7)
    handles, labels = winterScatter.legend_elements(prop = 'sizes', num = 6)
    labels = np.array(labels)
    labels = [int(''.join(char for char in string if char.isdigit())) for string in labels]
    labels = [labels/100 for labels in labels]
    # STDlegend = ax[0,0].legend(handles, labels, loc = 'upper right', title = 'STD [m/s]', bbox_to_anchor = (0,-.3))

    ax[0,1].set_extent([30, 60, 10, 40], ccrs.PlateCarree())
    # ax[0,1].states(resolution = '10m')
    ax[0,1].coastlines(resolution='10m')

    SpringScatter = ax[0,1].scatter(np.array(long), np.array(lat), s = SpringSTD*100, c = springAVG, cmap = color, vmin = min, vmax = max, alpha = .7, edgecolors = 'black')
    for i in range(len(MasterDf)):
        x = MasterDf['Long'][i]
        y = MasterDf['Lat'][i]
        ax[0,1].text(x * (1 + 0.01), y * (1 + 0.01) , i+1, fontsize=10, c = 'black')
    ax[0,1].title.set_text(titleString+' Spring 10m Wind Speed')
    # AVGlegend = ax[0,1].legend(*SpringScatter.legend_elements(num = 6), loc = 'lower left', title = 'Average 10m \nWind Speed [m/s]', bbox_to_anchor = (-.25,0))
    # ax[0,1].add_artist(AVGlegend)
    handles, labels = SpringScatter.legend_elements(prop = 'sizes', num = 6)
    labels = np.array(labels)
    labels = [int(''.join(char for char in string if char.isdigit())) for string in labels]
    labels = [labels/100 for labels in labels]
    # STDlegend = ax[0,1].legend(handles, labels, loc = 'upper right', title = 'STD [m/s]', bbox_to_anchor = (1.15,1))

    ax[1,0].set_extent([30, 60, 10, 40], ccrs.PlateCarree())
    ax[1,0].coastlines(resolution='10m')

    SummerScatter = ax[1,0].scatter(np.array(long), np.array(lat), s = SummerSTD*100, c = summerAVG, cmap = color, vmin = min, vmax = max, alpha = .7, edgecolors = 'black')
    for i in range(len(MasterDf)):
        x = MasterDf['Long'][i]
        y = MasterDf['Lat'][i]
        ax[1,0].text(x * (1 + 0.01), y * (1 + 0.01) , i+1, fontsize=10, c = 'black')
    ax[1,0].title.set_text(titleString+' Summer 10m Wind Speed')
    # AVGlegend = ax[1,0].legend(*SummerScatter.legend_elements(num = 6), loc = 'lower left', title = 'Average 10m \nWind Speed [m/s]', bbox_to_anchor = (-.25,0))
    # ax[1,0].add_artist(AVGlegend)
    handles, labels = SummerScatter.legend_elements(prop = 'sizes', num = 6)
    labels = np.array(labels)
    labels = [int(''.join(char for char in string if char.isdigit())) for string in labels]
    labels = [labels/100 for labels in labels]
    STDlegend = ax[1,0].legend(handles, labels, loc = 'upper right', title = 'STD [m/s]', bbox_to_anchor = (-.05,1), frameon = False)

    ax[1,1].set_extent([30, 60, 10, 40], ccrs.PlateCarree())
    ax[1,1].coastlines(resolution='10m')

    FallScatter = ax[1,1].scatter(np.array(long), np.array(lat), s = FallSTD*100, c = fallAVG, cmap = color, vmin = min, vmax = max, alpha = .7, edgecolors = 'black')
    for i in range(len(MasterDf)):
        x = MasterDf['Long'][i]
        y = MasterDf['Lat'][i]
        ax[1,1].text(x * (1 + 0.01), y * (1 + 0.01) , i+1, fontsize=10, c = 'black')
    ax[1,1].title.set_text(titleString+' Fall 10m Wind Speed')
    AVGlegend = ax[1,1].legend(*winterScatter.legend_elements(num = 8), loc = 'lower left', title = 'Seasonal Average 10m \nWind Speed difference \n[m/s] (ISD - MERRA-2)', bbox_to_anchor = (-1.75,1.2), frameon = False)
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
    # plt.show()
    plt.savefig('TimeCorrelatedBubblePlot_Outline_' + titleString +str(date.today()) +'.png', dpi = 300)
    return
'''takes in a
 df which should be filtered to what you want to plot - for example seasonally filtered,
 colorName is the name of the column that you want to use for the color of the bubbles
 sizeName is the name of the column that you want to be the size of the bubbles,
 titleString is what you are naming the plot

 Saves the bubble plot to a bubble plot output directory
 Only makes ONE bubble plot, must call multiple times for multiple plots
 returns nothing
 '''
def GeneralBubblePlot(df, titleString, color, size):
    #set up the map
    crs = ccrs.PlateCarree()
    fig, ax = plt.subplots(1,1, subplot_kw = {'projection': crs}, figsize = (9,9))
    ax.set_extent([30, 60, 10, 40], ccrs.PlateCarree()) #hardcoded middle east view
    ax.coastlines(resolution='10m')

    # df['MERRA-2_emissions_calculated']  = df['MERRA-2 emissions_calculated']*(10**9)
    # df['ISD_emissions'] = df['ISD total']*(10**9)

    #F-test:
    #https://www.cuemath.com/data/f-test/#:~:text=An%20F%20test%20is%20a,follows%20a%20Student%20t%2Ddistribution

    Scatter = ax.scatter(np.array(df['Long']), np.array(df['Lat']), c = (df['OG_winds_correct_box'] - df['Speed_ISD']), vmin= -8, vmax = 8, s = np.var(df['Speed_ISD'])/np.var(df['OG_winds_correct_box']), cmap = 'RdBu', alpha = .7, edgecolors = 'black')
    ax.title.set_text(titleString)

    color_legend = ax.legend(*Scatter.legend_elements(), title = 'Difference in Emissions' + "\n" +"[kg/m^2s] x10^-9", bbox_to_anchor = (1.05,.9), facecolor='white', framealpha=1)
    ax.add_artist(color_legend)

    handles, labels = Scatter.legend_elements(prop = 'sizes', alpha = 0.6)
    size_legend = ax.legend(handles,labels, title = '% Difference ISD vs ' + "\n" + "M2 Re-calculated", bbox_to_anchor = (1.05,.5),facecolor='white', framealpha=1)

    numbers = np.arange(1,31,1)
    for i in range(len(numbers)):
        x = df['LONGITUDE'][i]
        y = df['LATITUDE'][i]
        plt.text(x * (1 + 0.01), y * (1 + 0.01) , numbers[i], fontsize=12, c = 'black')
    # plt.savefig('ISDWind/MiddleEast/BubblePlot_outputs/2001-2020/' + titleString + '.png', dpi = 300)
    plt.show()
    plt.close()

    return

def MonthlyMeans12(df, stationID, ISDColumnName, M2ColumnName, titleString):
    MERRA_2_Means = df.groupby(by = ['month'])[M2ColumnName].mean()
    ISD_Means = df.groupby(by = ['month'])[ISDColumnName].mean()
    MERRA_2_Means = MERRA_2_Means.reset_index()
    ISD_Means = ISD_Means.reset_index()
    months = np.arange(1,13,1)
    monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']
    
    y_err1 = ISD_Means[ISDColumnName].std()
    y_err2 = MERRA_2_Means[M2ColumnName].std()

    # std_ISD = (df.groupby(by = ['month'])[ISDColumnName].std())/2
    # STD_M2 = (df.groupby(by = ['month'])[M2ColumnName].std())/2
    # print(ISD_Means[ISDColumnName][1:12] + std_ISD[0:11])
    fig,ax = plt.subplots(1,1, figsize = (12,6))
    plt.plot(months,ISD_Means[ISDColumnName] , label = 'ISD Measurements', linewidth = 8)
    plt.plot(months, MERRA_2_Means[M2ColumnName], label = 'MERRA-2 Data', linewidth = 8)
    plt.fill_between(months,ISD_Means[ISDColumnName] - y_err1, ISD_Means[ISDColumnName] + y_err1 , alpha = .2)
    plt.fill_between(months,MERRA_2_Means[M2ColumnName] - y_err2, MERRA_2_Means[M2ColumnName] + y_err2 , alpha = .2)
   
    plt.xticks(months, labels = monthNames, fontsize = 12)
    plt.xlabel('Time', fontsize = 16)
    plt.ylabel('Monthly Average ' + titleString, fontsize = 16)
    plt.yticks(fontsize = 12)
    plt.title(str(stationID) + ' Monthly Average ' + titleString, fontsize = 18)
    plt.legend(fontsize = 16)
    # plt.ylim(0,8)
    plt.ylim(0,1.15*max(max(MERRA_2_Means[M2ColumnName]),max(ISD_Means[ISDColumnName])))
    # plt.show()
    plt.savefig('/Users/emily/Documents/UMBC/Dr_LimaLab/MERRA2VsISD_MonthlyTimeSeries/' + str(date.today()) +  str(stationID) + titleString, dpi = 600)
    plt.close()
    return
'''Takes in a
dataframe
userText
folder
col1
col2

Saves a "Block Plot" to the output directory of block plots

DEPRICATED '''
def DepricatedBlockPlot(d, userText, folder, col1, col2):
    fig, ax = plt.subplots(figsize=(7,7))
    leapYearDays = [365,365,365,366,365,365,365,366,365,365,365,366,365,365,365,366,365,365,365,366]
    years = np.arange(2001,2021)
    station_number = np.arange(1,31,1)
    col = 0
    directory = '/Users/emily/Documents/UMBC/Dr_LimaLab/ISDWind/MiddleEast/BlockPlot/' + str(folder) + '/'
    for entry in os.scandir(directory):
        if entry.name.endswith('.xlsx') and entry.name.startswith('ME'):
            filelist.append(entry.name)
    filelist = sorted(filelist)
    print(filelist)
    for file in filelist:
            print(file)
            df = pd.read_excel(directory+file, na_values = ['Nan', 'nan'])
            thisYearData = pd.to_numeric(df[str(col1)]) - pd.to_numeric(df[col2])
            # print(thisYearData)
            d[:, col] = thisYearData*60*60*24*leapYearDays[col] #kg/m^2
            col += 1
    # d = np.zeros((30,20))
    '''Middle East Station IDs '''
    station_ids = np.array([40199099999,40155099999,41112099999,40420099999,41055099999,40361099999,40435099999,40357099999,41084099999,40405099999,40360099999,40394099999,40419499999,40416099999,41024099999,41140099999,41114099999,40437099999,40438099999,41128099999,40430099999,40373099999,40362099999,41136099999,40375099999,41036099999,40356099999,41061099999,40400099999,40439099999], dtype = "int")
    cmap = plt.cm.bwr
    cmap.set_bad(color = 'black')
    plt.imshow(d, cmap=cmap, vmin = -.15, vmax = .15)
    ax.set_yticks(np.arange(30))
    ax.set_yticklabels(station_number)
    ax.set_xticks(np.arange(20))
    ax.set_xticklabels(years, rotation = 45)
    plt.colorbar(label = 'kg/m\u00b2/yr') #https://www.geeksforgeeks.org/how-to-print-superscript-and-subscript-in-python/#
    plt.xlabel('Year')
    plt.ylabel('Station ID')
    titleString = 'Dust Emission ' + userText
    plt.title(titleString)
    plt.savefig(directory,titleString+str(datetime.date.today()))

    return
'''Takes no arguments and makes the dust emission difference Block plots and saves '''

def BlockPlot(directory, type):
    # directory = ''
    list_of_stations = '/Users/emily/Documents/UMBC/Dr_LimaLab/ISDWind/MiddleEast/10m_confirmed_stations.csv'
    MasterList = pd.read_csv(list_of_stations)
    stationIDs = MasterList['Station_ID']
    myyears = np.arange(2001,2021)
    station_number = np.arange(1,31,1)
    leapYearDays = [365,365,365,366,365,365,365,366,365,365,365,366,365,365,365,366,365,365,365,366]

    blocks = np.empty(shape = (30,20)) #numpy array of size number of years x number of stations 
    row = 0
    for stationID in stationIDs:
        # df = ApplesToApples.ReadData(,)
        df = pd.read_csv(directory + str(stationID) + '_FAUCART.csv', low_memory=False)
        # df = combineSubHourly(df)
        ISDMeans = df.groupby(by = ['year'])['ISD_flux'].mean()
        MERRA2Means = df.groupby(by = ['year'])['Recalculated_flux_correct_box'].mean()
        ISDMeans = ISDMeans.reset_index()
        MERRA2Means = MERRA2Means.reset_index()
        # print(ISDMeans)
        # print(MERRA2Means)
        if len(ISDMeans) != 20:
            missingYears = set(myyears).difference(ISDMeans['year'])
            print(missingYears)
            for missing in missingYears:
                ISDMeans.loc[len(ISDMeans)] = [missing, np.nan]
                MERRA2Means.loc[len(MERRA2Means)] = [missing, np.nan]
            ISDMeans = ISDMeans.sort_values(by=['year'])
            MERRA2Means = MERRA2Means.sort_values(by = ['year'])
            # print(ISDMeans)
            # print(MERRA2Means)
            # missingYearIndex = 
            # print(missingYearIndex)
        ISDMeans = ISDMeans['ISD_flux']
        MERRA2Means = MERRA2Means['Recalculated_flux_correct_box']
        for i in range(0,len(MERRA2Means)):
            ISDMeans[i] *= 60*60*24*leapYearDays[i]
            MERRA2Means[i] *= 60*60*24*leapYearDays[i]
        
        if type == 'mass':
            blocks[row,:] = (ISDMeans - MERRA2Means)
        elif type == 'percent':
           blocks[row,:] = ((ISDMeans - MERRA2Means)/MERRA2Means)*100
        print(blocks)
        row += 1
    fig, ax = plt.subplots(figsize=(7,7))
    cmap = plt.cm.bwr
    cmap.set_bad(color = 'black')
    plt.imshow(blocks, cmap=cmap, vmin = -.15, vmax = .15)
    # plt.imshow(blocks, cmap = cmap, vmin = -200, vmax = 200)
    ax.set_yticks(np.arange(30))
    ax.set_yticklabels(station_number)
    ax.set_xticks(np.arange(20))
    ax.set_xticklabels(myyears, rotation = 45)
    plt.colorbar(label = 'kg/m\u00b2/yr') #https://www.geeksforgeeks.org/how-to-print-superscript-and-subscript-in-python/#
    # plt.colorbar(label = 'Percent Difference [%]') #https://www.geeksforgeeks.org/how-to-print-superscript-and-subscript-in-python/#
    # plt.xlabel('Year')
    plt.ylabel('Station ID')
    # plt.title('Dust Emission Perrcent Difference \n (Using ISD winds - Using MERRA-2 winds)')
    plt.title('Dust mass Emission Difference \n (Dust Emission Using ISD winds - Dust Emission Using MERRA2 winds)')
    # plt.savefig(directory +str(datetime.date.today()), dpi = 300)
    plt.show()
    return

def DustImpactPlot(directory, filestring):
    list_of_stations = '/Users/emily/Documents/UMBC/Dr_LimaLab/ISDWind/MiddleEast/10m_confirmed_stations.csv'
    MasterList = pd.read_csv(list_of_stations)
    StationIDs = MasterList['Station_ID']
    # leapYearDays = [365,365,365,366,365,365,365,366,365,365,365,366,365,365,365,366,365,365,365,366]

    barValues = [] #raw average difference between ISD and M2 emission values in kg/m^2
    lineValues = [] #percent change to M2 emissions if ISD winds were used instead of M2
    for StationID in StationIDs:
        df = pd.read_csv(directory + str(StationID) + filestring, low_memory=False)
        print(StationID)
        ISDMean = np.nanmean(df['ISD_flux']) #kg/m^2/s
        MERRA2Mean = np.nanmean(df['Recalculated_flux_correct_box']) #kg/m^2/s
        ISDMean *= 60*60*24*365
        MERRA2Mean *= 60*60*24*365
        Difference = ISDMean - MERRA2Mean
        barValues.append(Difference)
        lineValues.append((np.nanmean(df['ISD_flux']) - np.nanmean(df['Recalculated_flux_correct_box']))/(np.nanmean(df['Recalculated_flux_correct_box']))*100)
        #if ISD AND M2 original flux is 0 (like station 2) then lineValue will be NaN
    
    #if ISD AND M2 original flux is 0 (like station 2) then lineValue will be NaN
    lineValues[1] = 0
    # print(lineValues)
    fig, ax1 = plt.subplots(figsize = (10,8))
    ax2 = ax1.twinx()
    ax1.set_ylim(-.15,.15)
    ax2.set_ylim(-250,250)
    ax1.bar(x = np.arange(1,31,1) - .25, height = barValues, width = .5, label = 'Difference [ISD - MERRA-2 Recalculated] \n 20-Year Average $kg/m^2/yr$', color = 'orange', alpha = .85)
    ax2.bar(x = np.arange(1,31,1) + .25, height = lineValues, width = .5, label = 'Percent Change \n to MERRA-2 Recalculated Emissions', color = 'blue', alpha = .75)
    
    # ax2.spines['bottom'].set_position(('data', 0))
    ax1.set_ylabel('Average Mass Difference [ISD - MERRA-2 Recalculated] $kg/m^2/yr$', color = 'orange', fontsize = 16)
    ax2.set_ylabel('Percent Change to MERRA-2 Recalculated Emissions (%)', color = 'blue', fontsize = 16)
    ax2.legend(loc='upper right')
    ax1.legend(loc='upper left')
    ax1.set_xlabel('Station Number')
    ax1.xaxis.set_ticks(np.arange(1, 31, 1))
    ax1.spines['bottom'].set_position(('data', 0))
    plt.title('Dust Emission Impacts - 20 Year Average', fontsize = 20)
    plt.show()
    # plt.savefig(str(datetime.date.today()) + '20YearDustEmissionImpact_orangeBlue.png', dpi = 300)
    return (barValues, lineValues)

def PercentChangeMap(percentChange):
    crs = ccrs.PlateCarree()
    fig, ax = plt.subplots(1,1, subplot_kw = {'projection': crs}, figsize = (12,12))
    S = nc.Dataset('/Users/emily/Documents/UMBC/Dr_LimaLab/Merra2_Wind/gocart.dust_source.v5a.x1152_y721.nc', low_memory = False)
    SSM = nc.Dataset('/Users/emily/Documents/UMBC/Dr_LimaLab/FAUXCART/FENGSHA_SOILGRIDS2017_GEFSv12_v1.2-2_M2gridsize.nc', low_memory = False)
    list_of_stations = '/Users/emily/Documents/UMBC/Dr_LimaLab/ISDWind/MiddleEast/10m_confirmed_stations.csv'
    MasterList = pd.read_csv(list_of_stations)
    source_function_whole = S.variables['du_src'][0,:,:]
    ssm = SSM.variables['ssm'][6,:,:]
    SSM_long = SSM.variables['longitude'][:,:]
    SSM_lat = SSM.variables['latitude'][:,:]

    long_source = S.variables['longitude'][:]
    lat_source = S.variables['latitude'][:]
    ax = plt.axes(projection = ccrs.PlateCarree())
    ax.set_extent([30, 60, 10, 40])
    plt.contourf(long_source, lat_source, source_function_whole, 30, transform = ccrs.PlateCarree())
    # plt.contourf(SSM_long, SSM_lat, ssm, 30, transform = ccrs.PlateCarree())

    ax.coastlines()
    #plot data points over this
    norm = mcolors.TwoSlopeNorm(vcenter=0)
    scatter = ax.scatter(MasterList['Long'],MasterList['Lat'], edgecolors = 'black', c = percentChange, s = 200, cmap = 'bwr', norm = norm)
    handles, labels = scatter.legend_elements(prop = 'colors', num = 9)
    numbers = np.arange(1,31,1)
    for i in range(len(numbers)):
        x = MasterList['Long'][i]
        y = MasterList['Lat'][i]
        plt.text(x * (1 + 0.01), y * (1 + 0.01) , numbers[i], fontsize=12, c = 'white')
    ax.legend(handles,labels, title = 'Yearly Average Percent Difference \n Between Measurement and Model', fontsize = 15)
    plt.title('Emission Percent Change From Measurment to Model (ISD - MERRA-2)', fontsize = 20)
    # plt.show()
    plt.savefig(str(datetime.date.today()) + 'PercentDifferenceMap_TOPO.png', dpi = 600)

    return 

def Zscore(df):
    list_of_stations = '/Users/emily/Documents/UMBC/Dr_LimaLab/ISDWind/MiddleEast/10m_confirmed_stations.csv'
    MasterList = pd.read_csv(list_of_stations)
    StationIDs = MasterList['Station_ID']


    zwinter = []
    zfall = []
    zspring = []
    zsummer = []

    for stationID in StationIDs:
        df = pd.read_csv('/Users/emily/Documents/UMBC/Dr_LimaLab/FAUXCART/FAUXCART_Run1/' + str(stationID) + '_FAUCART.csv', low_memory=False)
        df = Seasons(df)
        SeasonMeans_ISD = df.groupby( by = ['Season'])['Speed_ISD'].mean()
        # Fall Sp Su W
        SeasonSTD_ISD = df.groupby(by = ['Season'])['Speed_ISD'].std()
        SeasonMeans_MERRA2 = df.groupby( by = ['Season'])['OG_winds_correct_box'].mean()
        SeasonSTD_MERRA2 = df.groupby(by = ['Season'])['OG_winds_correct_box'].std()
        winterNumber = len(df[df["Season"]=="Winter"])
        fallNumber = len(df[df["Season"]=="Fall"])
        springNumber = len(df[df["Season"]=="Spring"])
        summerNumber = len(df[df["Season"]=="Summer"])
        # print(SeasonMeans_MERRA2)
        ztop = SeasonMeans_ISD[3] - SeasonMeans_MERRA2[3]
        zbottom = np.sqrt((SeasonSTD_ISD[3]**2 + SeasonSTD_MERRA2[3]**2)/winterNumber)
        print(ztop)
        print(zbottom)
        print(ztop/zbottom)
        print('----------')
        # zwinter.append(SeasonMeans_ISD[3] - SeasonMeans_MERRA2[3]/ (np.sqrt((SeasonSTD_ISD[3]**2 + SeasonSTD_MERRA2[3]**2)/winterNumber)))
        # print(zwinter)
    return

def taylorDiagram(df):
    #standard deviation of dataset
    STD = df['Speed_ISD'].std()
    #pearson's correlation coefficent 
    CC, pvalue = scipy.stats.pearsonr(df['Speed_ISD'], df['OG_winds_correct_box'])

    print(STD, CC)
    #create figure 
    fig = plt.figure(figsize=(12, 12))
    dia = gv.TaylorDiagram(fig=fig, label='REF')
    ax = plt.gca()

    plt.show()

    return

def ConfusingPlot(massDifferenceWinds,):
    list_of_stations = '/Users/emily/Documents/UMBC/Dr_LimaLab/ISDWind/MiddleEast/10m_confirmed_stations.csv'
    MasterList = pd.read_csv(list_of_stations)
    StationIDs = MasterList['Station_ID']
    topo_directory = '/Users/emily/Documents/UMBC/Dr_LimaLab/FAUXCART/FAUXCART_Run1/'
    ssm_directory = '/Users/emily/Documents/UMBC/Dr_LimaLab/FAUXCART/FAUXCART SSM Run#1/'

    difference_due_to_map = []
    for StationID in StationIDs:
        topo_df = pd.read_csv(topo_directory + str(StationID) + '_FAUCART.csv', low_memory=False)
        ssm_df = pd.read_csv(ssm_directory + str(StationID) + '_FAUXCART_SSM.csv', low_memory=False)

        topoM2Winds_emission = np.nanmean(topo_df['Recalculated_flux_correct_box'])
        ssmM2Winds_emission = np.nanmean(ssm_df['Recalculated_flux_correct_box'])

        topoM2Winds_emission_year = topoM2Winds_emission*60*60*24*365 #kg/m^2/year
        ssmM2Winds_emission_year = ssmM2Winds_emission*60*60*24*365

        #ssm - m2 (new - old)
        difference = ssmM2Winds_emission_year - topoM2Winds_emission_year
        difference_due_to_map.append(difference)

    print(difference_due_to_map)
    fig, ax1 = plt.subplots(figsize = (10,8), frameon = True)
    ax1.set_ylim(-.3,.3)
    ax2 = ax1.twinx()
    ax2.set_ylim(-.3, .3)
    ax1.bar(x = np.arange(1,31,1) - .25, height = massDifferenceWinds, width = .5, label = 'Difference [ISD - MERRA-2] due to Winds Only (with Topo Source) \n 20-Year Average $kg/m^2/yr$', color = 'orange')
    ax1.bar(x = np.arange(1,31,1) + .25, height = difference_due_to_map, width = .5, label = 'Difference [SSM - Topographic] due to Map \n 20-Year Average $kg/m^2/yr$', color = 'blue')
    ax1.set_xlabel('Station Number', fontsize = 14)
    ax1.xaxis.set_ticks(np.arange(1, 31, 1))
    ax1.spines['bottom'].set_position(('data', 0))
    ax1.set_ylabel('Dust Mass Emission Difference $kg/m^2/yr$', fontsize = 14)
    plt.title('Dust Emission Differences Quantification', fontsize = 18)
    ax1.legend(fontsize = 14)
    # plt.show()
    plt.savefig(str(datetime.date.today()) +'ConfusingPlot.png', dpi = 300)
    return

# '''Unfinished function'''
# def AERONET_PLOT():
#     # station_number = input('Which ISD station number would you like: ')
#     station_number = '40199099999'
#     # AERONET_station = input('Which AERONET station name?: ')
#     AERONET_station = 'Eilat'
#     # Begin_date = input('Begining date (dd-mm-yyyy): ')
#     Begin_date = '01/08/2015'
#     End_date = '15/09/2015'
#     # End_date = input('End date (dd-mm-yyyy): ')

#     ParentDirectory_ISD_MERRA2 = '/Users/emily/Documents/UMBC/Dr_LimaLab/ISDWind/MiddleEast/' + str(station_number) + '_DATA/'
#     ParentDirectory_AERONET = '/Users/Emily/Documents/UMBC/Dr_LimaLab/AERONET/' + str(AERONET_station) +'/'
#     ISD_MERRA2_df = ReadData(ParentDirectory_ISD_MERRA2, station_number)
#     # ISD_MERRA2_df = pd.read_csv(ParentDirectory_ISD_MERRA2 + 'MERRA2_intersection_' + station_number + '.csv', low_memory = False)
#     for entry in os.scandir(ParentDirectory_AERONET):
#         filename_string = entry.name
#     AERONET_df = pd.read_csv(ParentDirectory_AERONET + filename_string, skiprows = 6, low_memory = False)
#     #datetime object formatting
#     AERONET_df['Date(dd:mm:yyyy)'] = AERONET_df['Date(dd:mm:yyyy)'].str.replace(':', '/')
#     AERONET_df['Date(dd:mm:yyyy)'] = pd.to_datetime(AERONET_df['Date(dd:mm:yyyy)'], format ='%d/%m/%Y')
#     ISD_MERRA2_df['DATE'] = pd.to_datetime(ISD_MERRA2_df['DATE'], format = '%Y-%m-%d %H:%M:%S')
#     #https://sparkbyexamples.com/pandas/pandas-select-dataframe-rows-between-two-dates/
#     #choose dates from user input
#     AERONET_df_myDates = AERONET_df[AERONET_df['Date(dd:mm:yyyy)'].between(Begin_date, End_date)]
#     ISD_MERRA2_df_myDates = ISD_MERRA2_df[ISD_MERRA2_df['DATE'].between(Begin_date, End_date)]

#     # ISD_MERRA2_df = ISD_MERRA2_df['DATE'].between(Begin_date, End_date)
#     # ISD_MERRA2_df_myDates = ISD_MERRA2_df_myDates.to_frame()
#     print(ISD_MERRA2_df_myDates.head())
#     print(AERONET_df_myDates.head())
#     print(ISD_MERRA2_df_myDates.tail())
#     print(AERONET_df_myDates.tail())
#     # print(ISD_MERRA2_df_myDates.info())
#     # print(AERONET_df_myDates.info())
#     # print(AERONET_df_myDates.tail())
#     # print(ISD_MERRA2_df_myDates.head())
#     # print(ISD_MERRA2_df_myDates.tail())
#     # print(ISD_MERRA2_df_myDates.info())
#     # print(AERONET_df_myDates['AOD_675nm'])

#     #Plot AERONET AOD
#     plt.figure(figsize = (16,9))
#     plt.subplot(2,1,1)
#     plt.plot(AERONET_df_myDates['Date(dd:mm:yyyy)'],AERONET_df_myDates['AOD_675nm'], label = 'AERONET AOD @ 675nm')
#     plt.xlabel('Time')
#     plt.ylabel('AOD 675nm')
#     plt.title('AERONET - ' + str(AERONET_station) + ' AOD at 674nm')
#     plt.legend()

#     #plot ISD and MERRA-2 10 meter winds
#     plt.subplot(2,1,2)
#     plt.plot(ISD_MERRA2_df_myDates['DATE'], ISD_MERRA2_df_myDates['Speed_ISD'], label = 'ISD 10m wind speed')
#     plt.plot(ISD_MERRA2_df_myDates['DATE'], ISD_MERRA2_df_myDates['speed_MERRA2'], label = 'MERRA-2 10m wind speed')
#     plt.ylabel('10 meter wind speed [m/s]')
#     plt.xlabel('Time')
#     plt.title('ISD and MERRA-2 10m wind speed @' + str(station_number))
#     plt.legend()
#     # plt.show()
#     plt.savefig(ParentDirectory_AERONET + '/Comparison_with_ISD_' + str(station_number) + '_.png', dpi = 200)
#     return
