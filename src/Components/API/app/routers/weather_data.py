import datetime
import time
from ftplib import FTP
import pandas as pd
import io
import os
import json 
import tempfile

username = 'anonymous'
password = 'deakinuni@projectecho.com'

def parse_args():
    parser = argparse.ArgumentParser(description='Download weather data based on location and date.')
    parser.add_argument('--lat', type=float, required=True, help='Latitude of the location')
    parser.add_argument('--lon', type=float, required=True, help='Longitude of the location')
    parser.add_argument('--timestamp', type=str, required=True, help='Timestamp in YYYY-MM-DD-HH format')  # Modified to include hour
    
    return parser.parse_args()

lat = -38.80266562762370
lon = 143.563
timestamp = 1709462927 #Random chosen date March 3 2024


# Function to download files from FTP server
def download_file_from_ftp(ftp_server, ftp_directory, local_directory,year_month):
      with FTP(ftp_server) as ftp:

        ftp.login(username, password)
        #ftp_directory_location = ftp_directory +  observe_location + '/'
        ftp.cwd(ftp_directory)
        filename = f"{ftp_directory}/cape_otway_lighthouse-{year_month}"

        remote_filepath = f"{filename}.csv"            
        local_filepath = f"{local_directory}/cape_otway_lighthouse-{year_month}.csv"
        try:
            with open(local_filepath, "wb") as local_file:
                ftp.retrbinary(f"RETR {remote_filepath}", local_file.write)   
                print(f"File '{remote_filepath}' downloaded to '{local_filepath}'")

        except Exception as e:
            print(f"Error downloading file '{remote_filepath}': {e}")
                      
# Define FTP server details
ftp_server = "ftp.bom.gov.au"
ftp_directory = "/anon/gen/clim_data/IDCKWCDEA0/tables/vic/cape_otway_lighthouse"

# Define local directory to save files
local_directory = "C:/Users/vishn/Documents/Project-Echo/Weather"

# Download files from FTP server

def read_file(filepath, date):
    try:
        print(f"Reading file: {filepath}")

        # Ensure the file exists right before reading
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

if __name__ == "__main__":
    print("yo")

    dt_object = datetime.datetime.fromtimestamp(timestamp)

    year_month = dt_object.strftime('%Y%m') #For filename in remote repo for every month
    day_month_year = dt_object.strftime('%d%m%Y') #For row in csv file for exact date
    download_file_from_ftp(ftp_server, ftp_directory, local_directory,year_month)
    local_filepath = f"{local_directory}/cape_otway_lighthouse-{year_month}.csv"  
    read_file(local_filepath,day_month_year)


