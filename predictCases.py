'''
Python script for our Machine Learning models to predict a growth in COVID19 cases

User will be able to run the script and be given instructions on choosing the 
state, county, weather attributes and initial case number, which they will 
follow through to correctly use the predictor 

'''
import NNReg
import LinearReg
import numpy as np 

state_option = {
    1: "California", 2: "Maryland", 3: "New Jersey", 4: "Illinois",
    5: "Ohio", 6: "Texas", 7: "Colorado", 8: "New York", 9: "Virginia",
    10: "Connecticut", 11: "Georgia", 12: "Washington", 13: "Florida",
    14: "Pennsylvania", 15: "Michigan", 16: "Indiana", 17: "Wisconsin",
    18: "South Dakota", 19: "Louisiana", 20: "Massachusetts", 21: "Rhode Island",
    22: "Utah", 23: "Delaware"
    }

statesMultCounty = [1,2,3,4,6,8,9,10,13,15]

NY_counties = {
    1: "Dutchess", 2:"Erie", 3:"Monroe", 4:"Nassau", 5:"New York City", 
    6:"Orange", 7:"Suffolk", 8:"Westchester"
}
CA_counties = {
    1: "Alameda", 2:"Los Angeles", 3: "San Diego", 4: "San Bernardino", 
    5: "Riverside"
}
NJ_counties = {
    1:"Burlington", 2:"Hudson", 3:"Mercer", 4:"Monmouth", 5:"Ocean",
    6:"Somerset"
}
MD_counties = {
    1:"Anne Arundel", 2:"Baltimore"
}
IL_counties = {
    1: "Cook", 2:"Du Page"
}
TX_counties = {
    1: "Dallas", 2:"Harris", 3:"Travis"
}
FL_counties = {
    1: "MiamiDade", 2:"Palm Beach"
}
CT_counties = {
    1: "Fairfield", 2:"Hartford"
}
VA_counties = {
    1: "Fairfax", 2:"Lehigh", 3:"Prince William"
}
MI_counties = {
    1:"Macomb", 2:"Oakland"
}
locations = {
    "alameda":(37.811, -122.33, 10.0),
    "annearundel":(38.98, -76.48, 0.0),
    "baltimore":(39.27, -76.57, 10.0),
    "burlington":(40.08, -74.88, 9.0),
    "cook":(42.0, -87.5, 202.0),
    "cuyahoga":(41.53,-81.63,210.0),
    "dallas":(32.68,-96.87, 200.6),
    "denver":(39.83,-104.65, 1650.2),
    "dupage":(42.0,-87.5,202.0),
    "dutchess":(41.79,-73.74, 125.9),
    "erie":(42.88,-78.89,178.3),
    "fairfax":(36.97,-76.42,2.0,),
    "fairfield":(41.17,-73.18,10.0),
    "fultown":(33.78,-84.52,256.0,),
    "harris":(29.68,-94.98,10.0),
    "hartford":(41.73611,-72.65056,5.8),
    "hudson":(40.65,-74.0666666,72.0),
    "king":(47.53028,-122.30083,6.1),
    "lehigh":(40.64985,-75.44771,118.9),
    "losangeles":(34.008,-118.5,2.0),
    "macomb":(42.466666, -82.8666666, 180.0),
    "marion":(40.48333,-85.68333,263.0),
    "mercer":(40.27679, -74.8159, 57.9),
    "miamidade":(25.73, -80.15, 10.0),
    "milwaukee":(43.05,-87.88, 188.0),
    "minehana":(43.7346,-96.6222,485.9),
    "monmouth":(40.47,-74.0,10.0),
    "monroe":(43.25,-77.5833333,83.0),
    "nassaue":(40.8,-73.77,10.0),
    "newhaven":(41.28,-72.88,10.0),
    "newyork":(40.701,-74.014,10.0),
    "oakland":(42.665,-83.41806,297.5),
    "ocean":(39.5333333,-74.45,12.0),
    "orange":(41.50917,-74.265,111.3),
    "orleans":(30.0166666,-90.1166666,4.0),
    "palmbeach":(26.61,-80.03,0.0),
    "philadelphia":(40.8,-74.75,18.0),
    "plymouth":(41.90972,-70.72944,45.4),
    "princewilliam":(37.25,-76.33,10.0),
    "providence":(41.8,-71.4,10.0),
    "riverside":(33.95194,-117.43861,245.2),
    "saltlake":(40.7781,-111.9694,1287.8),
    "sanbernardino":(34.09528,-117.23472,353.3),
    "sandiego":(32.867,-117.258,2.0),
    "somerset":(40.62405,-74.66905,32.0),
    "suffolk":(41.05,-71.97,10.0),
    "sussex":(38.68974,-75.36253,14.0),
    "travis":(30.6222,-98.0846,414.8),
    "westchester":(42.0166666,-73.9166666,12.0),
    "worcester":(42.2706,-71.8731,304.8)
}

def getInputs():
    print("Welcome to COVID19 Case Growth predictor.\nTo get started please choose a state of the simulation.")
    for k,v in state_option.items():
        print(k,v)
    state = int(input("Enter the name of the state: "))
    while state not in range(1,24):
        state = int(input("Enter the name of the state: "))
    if state in statesMultCounty:
        if state==1:
            for num,county in CA_counties.items():
                print(num, county)
            county = int(input("Enter the number of the county: "))
            latitude, longitude, elevation = locations[CA_counties[county].lower().replace(" ", "")]
        elif state==2:
            for num,county in MD_counties.items():
                print(num, county)
            county = int(input("Enter the number of the county: "))
            latitude, longitude, elevation = locations[MD_counties[county].lower().replace(" ", "")]
        elif state==3:
            for num,county in NJ_counties.items():
                print(num, county)
            county = int(input("Enter the number of the county: "))
            latitude, longitude, elevation = locations[NJ_counties[county].lower().replace(" ", "")]
        elif state==4: 
            for num,county in IL_counties.items():
                print(num, county)
            county = int(input("Enter the number of the county: "))
            latitude, longitude, elevation = locations[IL_counties[county].lower().replace(" ", "")]
        elif state==6:
            for num,county in TX_counties.items():
                print(num, county)
            county = int(input("Enter the number of the county: "))
            latitude, longitude, elevation = locations[TX_counties[county].lower().replace(" ", "")]
        elif state==8:
            for num,county in NY_counties.items():
                print(num, county)
            county = int(input("Enter the number of the county: "))
            latitude, longitude, elevation = locations[NY_counties[county].lower().replace(" ", "")]
        elif state==9:
            for num,county in VA_counties.items():
                print(num, county)
            county = int(input("Enter the number of the county: "))
            latitude, longitude, elevation = locations[VA_counties[county].lower().replace(" ", "")]
        elif state==10:
            for num,county in CT_counties.items():
                print(num, county)
            county = int(input("Enter the number of the county: "))
            latitude, longitude, elevation = locations[CT_counties[county].lower().replace(" ", "")]
        elif state==13:
            for num,county in FL_counties.items():
                print(num, county)
            county = int(input("Enter the number of the county: "))
            latitude, longitude, elevation = locations[FL_counties[county].lower().replace(" ", "")]
        elif state==15:
            for num,county in MI_counties.items():
                print(num, county)
            county = int(input("Enter the number of the county: "))
            latitude, longitude, elevation = locations[MI_counties[county].lower().replace(" ", "")]
    else:
        if state == 5:
            latitude,longitude,elevation = locations['cuyahoga']
        elif state == 7:
            latitude,longitude,elevation = locations['denver']
        elif state == 11:
            latitude,longitude,elevation = locations['fultown']
        elif state == 12:
            latitude,longitude,elevation = locations['king']
        elif state == 14:
            latitude,longitude,elevation = locations['philadelphia']
        elif state == 16:
            latitude,longitude,elevation = locations['marion']
        elif state == 17:
            latitude,longitude,elevation = locations['milwaukee']
        elif state == 18:
            latitude,longitude,elevation = locations['minnehana']
        elif state == 19:
            latitude,longitude,elevation = locations['orleans']
        elif state == 20:
            latitude,longitude,elevation = locations['worcester']
        elif state == 21:
            latitude,longitude,elevation = locations['providence']
        elif state == 22:
            latitude,longitude,elevation = locations['saltlake']
        elif state == 23:
            latitude,longitude,elevation = locations['sussex']
            
    temperature = float(input("Enter the temperature in Fahrenheits: "))
    wind = float(input("Enter the wind speed: "))
    rain = float(input("Enter the rain amount: "))
    initCases = int(input("Enter the starting number of cases: "))
    return latitude, longitude, elevation, temperature, wind, rain, initCases

def main():
    latitude    = None
    longitude   = None
    elevation   = None
    temperature = None
    wind        = None
    rain        = None
    initCases   = None
    latitude,longitude, elevation, temperature, wind, rain, initCases =getInputs()
    print("--------------\nPredcting for:\nLatitude:", latitude,"\nLongitude: ", longitude, "\nElevation: ", elevation, "\nTemperature: ", temperature, 
            "\nWind: ", wind, "\nRain: ", rain, "\nInitial Cases: ", initCases, "\n--------------\n")

    pred = np.array([[latitude, longitude, elevation, temperature, wind, rain]])
    nnetPrediction = NNReg.traingAndPredictCases(pred)
    linPrediction = LinearReg.trainAndPredictCases(pred)
    nnetPrediction = round(nnetPrediction[0], 4)*100
    linPrediction = round(linPrediction[0][0], 4)*100

    msg = "In a week the cases would have changed by\n"+ str(nnetPrediction)+"% (Neural Network)\n"+str(linPrediction)+"% (Linear Regression)"
    print(msg)

if __name__=="__main__":
    main()

