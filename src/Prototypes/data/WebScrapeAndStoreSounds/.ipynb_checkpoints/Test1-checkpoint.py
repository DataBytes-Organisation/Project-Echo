from selenium import webdriver
import time

# Set up Selenium WebDriver
driver = webdriver.Chrome()  # Make sure chromedriver is in the same directory
base_url = "https://freesound.org/search/?q=weather"

# Open the URL
driver.get(base_url)

# Function to scroll and load more content
def scroll_to_bottom():
    SCROLL_PAUSE_TIME = 3
    last_height = driver.execute_script("return document.body.scrollHeight")

    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCROLL_PAUSE_TIME)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

# Scroll to the bottom of the page to load more content
scroll_to_bottom()

# Find audio elements
audio_elements = driver.find_elements_by_tag_name("audio")

# Extract audio URLs
audio_urls = [audio.get_attribute("src") for audio in audio_elements]

# Close the browser
driver.quit()

# Print the collected audio URLs
for url in audio_urls:
    print(url)
