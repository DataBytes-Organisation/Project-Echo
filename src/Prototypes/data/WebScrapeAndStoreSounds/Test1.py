from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
import csv

# Define the URL to scrape
base_url = "http://freesound.org"

# Set up the Selenium WebDriver
service = Service(executable_path='./chromedriver-mac-arm64/chromedriver')
options = webdriver.ChromeOptions()
driver = webdriver.Chrome(service=service, options=options)
#driver = webdriver.Chrome(executable_path="/chromedriver_mac_arm64/chromedriver")

# Open the webpage
driver.get(base_url)

# Find the search bar and enter the search query (e.g., "weather")
search_bar = driver.find_element(By.NAME, "q")
search_bar.send_keys("weather")
search_bar.send_keys(Keys.RETURN)

# Initialize BeautifulSoup
soup = BeautifulSoup(driver.page_source, "html.parser")

# Find the audio clips on the page
audio_clips = soup.find_all("div", class_="sample_player_small")

# Create a list to store audio clip information
audio_info = []

# Loop through each audio clip and collect information
for clip in audio_clips:
    clip_name = clip.find("div", class_="sample_player").find("div", class_="sound_title").find("div", class_="sound_filename").text.strip()
    clip_info = clip.find("div", class_="sample_information")
    clip_uploaded_by = clip_info.find("a", class_="user").text.strip()
    #clip_length = clip.find("div", class_="sample_player").find("div", class_="small_player").find("div", class_="player small").find("div", class_="time-indicator-container").find("div", class_="time-indicator").text.strip()
    clip_date = clip_info.find("span", class_="date").text.strip()
    clip_download = clip_info.find("span", class_="download_count").text.strip()
    
    audio_info.append({
        "Name": clip_name,
        "Uploaded By": clip_uploaded_by,
        #"Length": clip_length,
        "Date": clip_date,
        "Download": clip_download
    })

# Write collected data to a CSV file
with open("weather_audio_info.csv", "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["Name", "Uploaded By", "Date", "Download"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for clip in audio_info:
        writer.writerow(clip)

# Close the browser
driver.quit()
