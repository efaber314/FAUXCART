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

STATISTICAL ANALYSIS
9.ztest scores

''' 
import ApplesToApples
import pandas as pd

#1.histograms (after combine subhourly)
df = pd.read_csv('/Users/emily/Downloads/TestDataOutput40199099999_FAUCART.csv')
df_hist = ApplesToApples.combineSubHourly(df)
ApplesToApples.Histogram_AllTime(df_hist,'40199099999 TEST DATA','All Time')

#2.monthly means (240)
ISDmeans,m2means,numpoints = ApplesToApples.allYearsPlot(df,'40199099999 TEST DATA')
print(ISDmeans,m2means,numpoints)