#FAUCART Driver
#Emily Faber
#August 2023
import ISD_Downloader
import Parallelizer

#asks for user input in function, returns user-input path/name of csv
my_stations = ISD_Downloader.ISD_Downloader()
print('Downloader DONE')
Parallelizer.parallelize(my_stations)
