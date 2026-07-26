import csv
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service

# Set up the web driver (make sure to provide the path to your WebDriver executable)
service = Service(executable_path='./chromedriver-mac-arm64/chromedriver')
options = webdriver.ChromeOptions()
driver = webdriver.Chrome(service=service, options=options)

# URL to scrape
url = "http://freesound.org"

# Function to scrape and save weather audio clips
def scrape_weather_audio():
    driver.get(url)
    time.sleep(2)  # Wait for the page to load

    # Find the search bar and enter the search query (e.g., "weather")
    search_bar = driver.find_element(By.NAME, "q")
    search_bar.send_keys("weather")
    search_bar.send_keys(Keys.RETURN)

    # Wait for search results to load
    time.sleep(5)

    # Get page source and parse with BeautifulSoup
    page_source = driver.page_source
    soup = BeautifulSoup(page_source, "html.parser")

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

    # Extract audio clip information
    # audio_clips = []
    # for item in soup.find_all("div", class_="audio-item"):
    #     name = item.find("div", class_="title").get_text()
    #     uploaded_by = item.find("div", class_="username").get_text()
    #     length = item.find("div", class_="duration").get_text()
    #     audio_clips.append({"name": name, "uploaded_by": uploaded_by, "length": length})


    return audio_info

# Scrape audio clips and save them in a CSV file
def save_to_csv(data):
    with open("weather_audio_clips.csv", "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["Name", "Uploaded By", "Date", "Download"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for clip in data:
            writer.writerow(clip)

if __name__ == "__main__":
    try:
        audio_data = scrape_weather_audio()
        save_to_csv(audio_data)
        print("Scraping and CSV creation successful.")
    except Exception as e:
        print("An error occurred:", str(e))
    finally:
        driver.quit()
