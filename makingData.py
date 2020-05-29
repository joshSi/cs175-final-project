import numpy as np 
import pandas as pd

# Load the data
my_data = pd.read_csv('us-counties.csv', sep=',')

# Get the unique combos of County and State
all_county_state = my_data["county"] + ", " +my_data["state"] + ", No"
county_state = all_county_state.unique()

# Get the list of dates
dates = my_data["date"].unique()
#dates = np.concatenate((["County, State"], dates))


# Table consisting of [weather conditions][restriction lvl][case growth rate]
final_result = np.zeros((1,1))

np.savetxt("counties.csv", county_state, fmt='%s')
np.savetxt("dataset.csv", final_result, fmt='%s')

'''
Want to make table of:
 [lat][long][elevation][temp][sea level pressure][visibility][wind speed][prcp][humidity][etc.][level of restrictions][case growth]

To better understand the weather report:

"STATION"               = location ID
"DATE"                  = date
"LATITUDE"              = location's lat
"LONGITUDE"             = location's long
"ELEVATION"             = location's elevation
"NAME"                  = location's name
"TEMP"                  = temperature
"TEMP_ATTRIBUTES"
"DEWP"                  = dewpoint temperature
"DEWP_ATTRIBUTES
"SLP"                   = sea level pressure
"SLP_ATTRIBUTES"
"STP"                   = station pressure
"STP_ATTRIBUTES"
"VISIB"                 = visibility
"VISIB_ATTRIBUTES"
"WDSP"                  = wind speed
"WDSP_ATTRIBUTES"
"MXSPD","GUST"
"MAX"
"MAX_ATTRIBUTES"
"MIN"
"MIN_ATTRIBUTES"    
"PRCP"                  = precipitation
"PRCP_ATTRIBUTES"
"SNDP"                  = snow depth
"FRSHTT"                = prediction for fog, rain, snow, hail, thunder, tornado
'''
