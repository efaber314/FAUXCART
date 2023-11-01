#MERRA2 wind nc file downloader

import sys
import os
import requests
from bs4 import BeautifulSoup
import numpy as np
import urllib.request

#get list of urls
AreaName = input('What station are you gettig data for? ')
FolderName = 'Merra2Wind_' + AreaName

ParentDirectory = "/Users/emily/Documents/UMBC/Dr_LimaLab/Merra2_Wind"
path = os.path.join(ParentDirectory, FolderName)
os.mkdir(path)
print('Directory made')
print('Downloading data... this may take some time')
with open('/Users/emily/Documents/UMBC/Dr_LimaLab/Merra2_Wind/subset_M2TMNXFLX_5.12.4_20210706_151852.txt') as f:
    urls = f.readlines()
urls = ([s.strip('\n') for s in urls ])
for url in urls:
    urllib.request.urlretrieve(url,url[86:])
    print(url[86:] + " saved")
