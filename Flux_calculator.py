#calculate flux for each hour just as M2 does it
import pandas as pd
import netCDF4 as nc
import os
import numpy as np

path_to_top_of_directory_of_intersection_files = '/Users/emily/Documents/UMBC/Dr_LimaLab/ISDWind/MiddleEast/'
path_to_master_list_of_all_stations = '/Users/emily/Documents/UMBC/Dr_LimaLab/ISDWind/MiddleEast/10m_confirmed_stations.csv'
path_GWETTOP = '/Users/emily/Documents/UMBC/Dr_LimaLab/MERRA-2_SurfaceWetness/'
OG_Flux = '/Users/emily/Documents/UMBC/Dr_LimaLab/MERRA-2_FluxBins/' #original dust emission from M2
OG_wind_files = '/media/arochali/backup_8TB/for_Emily_PhD/MERRA2/MERRA2_10mWind_Whole/'
df = pd.read_csv(path_to_master_list_of_all_stations)
station_IDs = df['Station_ID']

def getIndex(lat,long):
    i = int((long + 180)/(5/8)))+1
    j = int((lat + 90)/0.5))+1
    longIndex = i - 1
    latIndex = j - 1
    return longIndex, latIndex
#goes in the GWETTOP file directory and finds the value for GWETTOP at the requested
#time and location
def retrieveGWETTOP(lat,long,time):
    if time.month < 10:
        monthstring = '0' + str(time.month)
    else:
        monthstring = str(time.month)
    if time.day < 10:
        daystring = '0' + str(time.day)
    else:
        daystring = str(time.day)
    datestringtolookfor = str(time.year) + monthstring + daystring
    for file in os.scandir(path_GWETTOP):
        if file.name.endswith('.nc') and datestringtolookfor in file.name:
            gwettop_file = nc.Dataset(path_GWETTOP + file.name)# time = 24, lat = 361 , long = 576
            #now find the right hour and location
            long_select, lat_select = getIndex(lat,long, gwettop_file)
            gwettop = gwettop_file['GWETTOP'][int(time.hour),lat_select,long_select] #single value for that lat,long at that time
            gwettop_file.close()
    return  gwettop

#goes to the UNSCALED source function and returns the source value closest to the given lat and long
def retrieveSource(lat,long):
    source = np.load('/Users/emily/Downloads/du_src_new_2.npz')
    long_select, lat_select = getIndex(lat,long,source)
    S = source['du_src_new'][lat_select,long_select]
    source.close()

    return S

#goes to the original DUEM00# files and returns the total flux value for the given location
#and time
def retrieveOriginalEmissions(lat,long,time):
    datestringtolookfor = datestringtolookforBuild(time)
    OG_total = 0
    for file in os.scandir(OG_Flux):
        if file.name.endswith('.nc') and datestringtolookfor in file.name:
            OG_flux_file = nc.Dataset(OG_Flux + file.name)# time = 24, lat = 361 , long = 576
            #now find the right hour and location
            long_select, lat_select = getIndex(lat,long,OG_flux_file)
            DUEM001 = OG_flux_file['DUEM001'][int(time.hour),lat_select,long_select] #single value for that lat,long at that time
            DUEM002 = OG_flux_file['DUEM002'][int(time.hour),lat_select,long_select] #single value for that lat,long at that time
            DUEM003 = OG_flux_file['DUEM003'][int(time.hour),lat_select,long_select] #single value for that lat,long at that time
            DUEM004 = OG_flux_file['DUEM004'][int(time.hour),lat_select,long_select] #single value for that lat,long at that time
            DUEM005 = OG_flux_file['DUEM005'][int(time.hour),lat_select,long_select] #single value for that lat,long at that time
            OG_total = float(DUEM001) + float(DUEM002) + float(DUEM003) + float(DUEM004) + float(DUEM005)
            OG_flux_file.close()

    return OG_total

def datestringtolookforBuild(time):
    if time.month < 10:
        monthstring = '0' + str(time.month)
    else:
        monthstring = str(time.month)
    if time.day < 10:
        daystring = '0' + str(time.day)
    else:
        daystring = str(time.day)
    datestringtolookfor = str(time.year) + monthstring + daystring

    return datestringtolookfor

def retrieveM2Winds(lat,long,time):
    long_select,lat_select = getIndex(lat,long,OG_wind_file)
    datestringtolookfor = datestringtolookforBuild(time)
    for file in os.scandir(OG_Flux):
        if file.name.endswith('.nc') and datestringtolookfor in file.name:
            OG_wind_file = nc.Dataset(OG_wind_files + str(file.name))
            U10 = OG_wind_file['U10M'][int(time.hour), lat_select,long_select]
            V10 = OG_wind_file['V10M'][int(time.hour), lat_select,long_select]
            thisHourWindMagnitude = np.sqrt(U10**2 + V10**2)
    return thisHourWindMagnitude

def retrieveOceanFraction(lat,long):
    ocean_fraction = nc.Dataset('/Users/emily/Documents/UMBC/Dr_LimaLab/MERRA2_101.const_2d_asm_Nx.00000000.nc4.nc4')
    OCEAN_FRAK = np.array(ocean_fraction.variables['FROCEAN'])
    long_select,lat_select = getIndex(lat,long,ocean_fraction)
    OceanFraction = OCEAN_FRAK[0,lat_select,long_select]
    return OceanFraction

def retrieveLakeFraction(lat,long,time):
    lake_fraction = nc.Dataset('/Users/emily/Documents/UMBC/Dr_LimaLab/MERRA2_101.const_2d_ctm_Nx.00000000.nc4.nc4') #12 months, lat = 361, long = 576
    FRLAKE = np.array(lake_fraction.variables['FRLAKE'])
    long_select, lat_select = getIndex(lat,long, lake_fraction)
    FR_lake = FRLAKE[int(time.month-1),int(lat_select),int(long_select)]
    return FR_lake

effective_radii = [0.73, 1.4, 2.4, 4.5, 8] #microns - size bins of MERRA-2
density_air = 1.250 #kg/m^3
g = 9.81 #m/s^2
C = 1 #microgram *s^2/m^5
C  = C *(1*10**-9) #kg*s^2/m^5
C = C*0.08
s_p = [0.1, 0.25, 0.25, 0.25, 0.25]
densities = [2650,2650,2650,2650,2650]
U_t0 = [] #five basic values for Ut0 that are modified each hour by GWETTOP
for bin in range(0,len(effective_radii)):
    r = effective_radii[bin]
    R = r * 10**-6 #m
    sp = s_p[bin]
    density_particle = densities[bin]
    temp1 = .13*np.sqrt((density_particle*g*2*R)/density_air)
    temp2 = np.sqrt(1+6e-7/(density_particle*g*(2*R)**2.5))
    temp3 = np.sqrt(1.928*(((1331*(100*(2*R))**1.56)+.38)**.092)-1)
    U_t = (temp1*temp2)/temp3
    U_t0.append(U_t)

for station_ID in station_IDs:
    #go and find the intersection file
    correlated_data_file = pd.read_csv(path_to_top_of_directory_of_intersection_files + str(station_ID) +'_DATA/MERRA2_intersection_' + str(station_ID) + '.csv')
    #for each hour in correlated data file go and find the associated GWETTOP, V10m, U10M
    correlated_data_file['DATE'] = pd.to_datetime(correlated_data_file['DATE'])
    station_lat = correlated_data_file['LATITUDE'][0]
    station_long = correlated_data_file['LONGITUDE'][0]
    ISD_10m_windSpeed = correlated_data_file['Speed_ISD'][:]
    M2_10m_windSpeed = correlated_data_file['speed_MERRA2'][:]
    #retrieve source function at this location
    S = retrieveSource(station_lat, station_long)
    print('station number ', station_ID, 'at lat, long', station_lat, station_long, 'source', S)

    ocean_fraction = retrieveOceanFraction(station_lat,station_long)

    correlated_data_file_ISD_flux = []
    correlated_data_file_M2_recalculated_flux = []
    correlated_data_file_M2_original_flux = []
    correlated_data_file_gwettop = []
    correlated_data_file_source_used = []
    correlated_data_file_OG_winds = []
    correlated_data_file_m2_recalculated_flux_correct_box = []
    for hour in range(0, len(correlated_data_file['DATE'])):
        thishour = correlated_data_file['DATE'][hour]
        GWETTOP_this_hour = retrieveGWETTOP(station_lat, station_long, thishour)
        OG_flux_this_hour = retrieveOriginalEmissions(station_lat, station_long, thishour)
        lake_fraction = retrieveLakeFraction(station_lat, station_long, thishour)
        OG_winds_this_hour = retrieveM2Winds(lat,long,time)
        # calcualte emission in 5 bins
        f_isd = 0 #this hour's flux gets reset
        f_M2_recalculated = 0
        f_M2_recalculated_correct_box = 0
        for bin in range(0,len(effective_radii)):
            U_t = max(0, U_t0[bin] * (1.2+0.2*np.log10(max(.001,GWETTOP_this_hour)))) #threshold for this hour for this bin
            sp = s_p[bin]
            if ISD_10m_windSpeed[hour] > U_t:
                f_isd += (1-ocean_fraction)*(1-lake_fraction)*C*S*sp*(ISD_10m_windSpeed[hour]**2)*(ISD_10m_windSpeed[hour] - U_t)
            if M2_10m_windSpeed[hour] > U_t:
                f_M2_recalculated += (1-ocean_fraction)*(1-lake_fraction)*C*S*sp*(M2_10m_windSpeed[hour]**2)*(M2_10m_windSpeed[hour] - U_t)
            if OG_winds_this_hour > U_t:
                f_M2_recalculated_correct_box += (1-ocean_fraction)*(1-lake_fraction)*C*S*sp*(OG_winds_this_hour**2)*(OG_winds_this_hour - U_t)
        #write back to file
        correlated_data_file_ISD_flux.append(f_isd)
        correlated_data_file_M2_recalculated_flux.append(f_M2_recalculated)
        correlated_data_file_M2_original_flux.append(OG_flux_this_hour)
        correlated_data_file_gwettop.append(GWETTOP_this_hour)
        correlated_data_file_source_used.append(S)
        correlated_data_file_OG_winds.append(OG_winds_this_hour)
        correlated_data_file_m2_recalculated_flux_correct_box.append(f_M2_recalculated_correct_box)

    correlated_data_file['ISD_flux'] = correlated_data_file_ISD_flux
    correlated_data_file['M2_recalculated_flux'] = correlated_data_file_M2_recalculated_flux
    correlated_data_file['M2_original_flux'] = correlated_data_file_M2_original_flux
    correlated_data_file['gwettop'] = correlated_data_file_gwettop
    correlated_data_file['source_used'] = correlated_data_file_source_used
    correlated_data_file['OG_winds_correct_box'] = correlated_data_file_OG_winds
    correlated_data_file['Recalculated_flux_correct_box'] = correlated_data_file_m2_recalculated_flux_correct_box

    correlated_data_file.to_csv(path_to_top_of_directory_of_intersection_files + str(station_ID) + '_FAUCART.csv')







                #
