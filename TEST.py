# import ApplesToApples
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_excel('/Users/emily/Documents/UMBC/Dr_LimaLab/ISDWind/MiddleEast/PercentDifference_files/ME(2001-01-01-2020-12-31)_withSizeBins_omega_map_experiments.xlsx')
barValues = (df['ISD total']*60*60*24*1000*7304) - (df['MERRA-2 emissions_calculated']*60*60*24*1000*7304)
barValues = barValues/1000/20
# lineValues = ((df['M2 emissions given g/m^2']-df['M2 emissions g/m^2'])/1000)/20
lineValues = df['Percent Change to M2 emissions calculated']*100
fig, ax1 = plt.subplots(figsize = (8,8))
ax2 = ax1.twinx()
ax1.set_ylim(-.1,.1)
ax2.set_ylim(-160,160)
ax1.bar(x = np.arange(1,31,1), height = barValues, label = 'Difference [ISD - MERRA-2] \n 20-Year Average kg/m^2', color = 'orange')
ax2.plot(lineValues, label = 'Potential Percent Change \n to MERRA-2 Emissions', color = 'blue')
ax1.set_xlabel('Station Number')
ax1.set_ylabel('Difference [ISD - MERRA-2] kg/m^2', color = 'orange', fontsize = 16)
ax2.set_ylabel('Percent Change to MERRA-2 Emissions (%)', color = 'blue', fontsize = 16)
# plt.legend()
ax2.legend(loc='upper right')
ax1.legend(loc='upper left')
plt.title('Dust Emissions - 20 Year Average', fontsize = 20)
plt.show()
