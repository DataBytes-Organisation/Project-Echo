## app.routers.weather_data.py
import datetime
import time
from ftplib import FTP
import pandas as pd
import io
import os
import json 
import tempfile
import math

username = 'anonymous'
password = 'deakinuni@projectecho.com'

lat = -38.80266562762370
lon = 143.563
timestamp = 1709462927 #Random chosen date March 3 2024

#Function to download weather station list with coordinates from FTP server
def download_weather_stations_from_ftp(ftp_server, ftp_directory, local_directory):

    with FTP(ftp_server) as ftp:

        ftp.login(username,password)
        ftp.cwd(ftp_directory)
        filename = f"{ftp_directory}/stations_db.txt"

        remote_filepath = f"{filename}"
        local_filepath = f"{local_directory}/stations_db.txt"
        try:
            with open(local_filepath, "wb") as local_file:
                ftp.retrbinary(f"RETR {remote_filepath}", local_file.write)   
                print(f"File '{remote_filepath}' downloaded to '{local_filepath}'")

        except Exception as e:
            print(f"Error downloading file '{remote_filepath}': {e}")

# Function to calculate distance between two coordinates using the Haversine formula
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = math.sin(d_lat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c  # Distance in kilometers

# Function to process the coordinate file and find the closest weather station
def find_closest_station(lat, lon, filepath):
    # Read the file and create a DataFrame
    stations = []
    with open(filepath, 'r') as file:
        #print("helloo")
        for line in file:
            parts = line.split()
            #print(parts)
            station_code = parts[0]
            state = parts[1]
            lat_station = float(parts[-2])
            lon_station = float(parts[-1])
            station_name = ' '.join(parts[3:-3]).lower()
            station_name_formatted = station_name.lower().replace(' ', '_')
            stations.append({
                'Station Code': station_code,
                'State': state,
                'Station Name': station_name_formatted,
                'Latitude': lat_station,
                'Longitude': lon_station
            })
            

    # Create a DataFrame from the parsed data
    df_stations = pd.DataFrame(stations)

    # Calculate the distance from the provided location to each weather station
    df_stations['Distance (km)'] = df_stations.apply(
        lambda row: haversine(lat, lon, row['Latitude'], row['Longitude']), axis=1)

    # Find the station with the smallest distance
    closest_station = df_stations.loc[df_stations['Distance (km)'].idxmin()]
    print("closest is",closest_station)
    return closest_station['Station Name']


# Function to download files from FTP server
def download_file_from_ftp(ftp_server, ftp_directory, local_directory,year_month, station_name):
      with FTP(ftp_server) as ftp:

        ftp.login(username, password)
        #ftp_directory_location = ftp_directory +  observe_location + '/'
        ftp.cwd(ftp_directory)
        filename = f"{ftp_directory}/{station_name}/{station_name}-{year_month}"

        remote_filepath = f"{filename}.csv"            
        local_filepath = f"{local_directory}/{station_name}-{year_month}.csv"
        try:
            with open(local_filepath, "wb") as local_file:
                ftp.retrbinary(f"RETR {remote_filepath}", local_file.write)   
                print(f"File '{remote_filepath}' downloaded to '{local_filepath}'")

        except Exception as e:
            print(f"Error downloading file '{remote_filepath}': {e}")
                      
# Define FTP server details
ftp_server = "ftp.bom.gov.au"
ftp_directory = "/anon/gen/clim_data/IDCKWCDEA0/tables/vic"

# Read file and get weather data for the day
def read_file(filepath, date):
    try:
        print(f"Reading file: {filepath}")

        # Check if file exists right before reading
        if not os.path.exists(filepath):
            print(f"File not found right before reading: {filepath}")
            raise FileNotFoundError(f"File not found: {filepath}")

        df = pd.read_csv(filepath, skiprows=11, encoding='ISO-8859-1')  # Skip the first 11 rows to get to the actual data
        print(f"File read successfully, dataframe shape: {df.shape}")

        df.rename(columns={
            'Station Name': 'Station',
            'Date': 'Date',
            '0000-2400': 'Total Transpiration (mm)',
            '0900-0900': 'Rainfall (mm)',
            '0900-0900.1': 'Evaporation (mm)',
            'Temperature': 'Max Temperature (°C)',
            'Temperature.1': 'Min Temperature (°C)',
            'Humidity': 'Max Humidity (%)',
            'Humidity.1': 'Min Humidity (%)',
            'Speed': 'Wind Speed (m/sec)',
            'Radiation': 'Solar Radiation (MJ/sq m)'
        }, inplace=True)

        print(f"Dataframe columns: {df.columns}")

        # Filter the DataFrame for the specific date
        filtered_df = df[df['Date'] == date]

        # Check if data for the specific date is available
        if not filtered_df.empty:
            print(f"Weather data for {date}:")
            print(filtered_df)
            return filtered_df
        else:
            print(f"No data available for {date}.")

    except FileNotFoundError as e:
        print(f"FileNotFoundError: {e}")
        raise
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        raise

#if __name__ == "__main__":


    #dt_object = datetime.datetime.fromtimestamp(timestamp)

    #year_month = dt_object.strftime('%Y%m') #For filename in remote repo for every month
    #day_month_year = dt_object.strftime('%d%m%Y') #For row in csv file for exact date
    #download_file_from_ftp(ftp_server, ftp_directory, local_directory,year_month)
    #local_filepath = f"{local_directory}/cape_otway_lighthouse-{year_month}.csv"  
    #read_file(local_filepath,day_month_year)


