READ ME - FAUXCART

FA - Faber
UX - User eXperimental
C - Chemistry 
A - Aerosol
R - Radiation &
T - Transport model

This package requires a list of stations from the ISD and runs a wind speed and other user defined experiments on the dust emission scheme developed for GOCART.

The first time this program runs, it will download all necessary files from ISD and MERRA-2

File and function descriptions

Wind_and_Dust.py

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

This file contains functions names commented out that should be called in a loop over the list of station numbers as well as functions that should only be called once to analyze the whole study region.
=====================================================================
Histogram_AllTime(df, station_ID, diurnalText)
 - df == a data frame that has all the data in it you want to plot. It will look specifically for ’Speed_ISD’ and ‘OG_winds_correct_box’, so this function only works on wind speeds currently.
- station_ID == this is a string of the station ID number. The function is intended to be called within a loop over each station number in the list of stations for the analysis.
- diurnalText == ‘string’ to appear in the title of the figure for if you want to modify df BEFORE sending it to this function to only include daytime/nighttime values

This function also will plot the mean threshold friction velocity for each bin (unmodified for soil wetness) as 5 vertical lines.

This function has a path for saving figures hard coded as WhereToSave

=====================================================================
DustImpactPlot(‘/path/to/data/‘, filestring)

‘/path/to/data/‘ == string path to where FAUXCART run data is stored

Filestring == string with filenames of runs. Ie ‘_FAUXCART.csv’ for TOPO, ‘_FAUXCART_SSM.csv’ for SSM

This function returns mass difference between ISD and M2 

Works with PercentChagneMap to generate a percent change

Makes the two-bar plot with mass difference and percent change to M2 reanalysis original DUEM values if the M2 winds were replaced with ISD winds
