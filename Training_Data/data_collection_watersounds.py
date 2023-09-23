from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
import os
import requests as r
import csv
from lxml import etree


driver = webdriver.Chrome()
with open('G:/water_sounds/water_sounds.csv', 'w', newline='\n') as file:
        writer = csv.writer(file)
        writer.writerow(['name', 'uploaded_by', 'length', 'file_size', 'sample_rate', 'bit_depth', 'description', 'link'])

url = 'https://freesound.org/search/?q=forest+water'
i = 0

# Increase the range value to get more audio files loaded into specified directory
for x in range(1000):
        try:
                i += 1
                if i == 15:
                        i = 0
                        url = driver.find_element(By.XPATH, '/html/body/div[2]/div/div/div[30]/ul/li[2]/a').get_attribute('href')
                driver.get(url)
                driver.implicitly_wait(3)
                titles = driver.find_elements(By.CLASS_NAME, 'sound_filename')
                titles[i].click()
                soup = BeautifulSoup(driver.page_source, 'html.parser')
                video_element = soup.find(class_="mp3_file")
                video_url = video_element["href"]
                file_name = (video_element.text).replace(" ", "_").replace(".", "_")
                doc = r.get(video_url)
                sound_author = soup.find(id = 'sound_author').text
                sound_description = soup.find(id='sound_description').text
                body = soup.find("body")
                dom = etree.HTML(str(body))
                sound_information_length = dom.xpath('/html/body/div[2]/div/div/div[4]/dl/dd[2]')[0].text
                file_size = dom.xpath('/html/body/div[2]/div/div/div[4]/dl/dd[3]')[0].text
                sample_rate = dom.xpath('/html/body/div[2]/div/div/div[4]/dl/dd[4]')[0].text
                bit_depth = dom.xpath('/html/body/div[2]/div/div/div[4]/dl/dd[5]')[0].text
                driver.implicitly_wait(3)
                with open(f'G:/water_sounds/{file_name}.mp3', 'wb') as file:
                    file.write(doc.content)
                    driver.implicitly_wait(5)
                    with open('G:/water_sounds/water_sounds.csv', 'a', newline='\n') as file:
                            writer = csv.writer(file)
                            writer.writerow([file_name, sound_author, sound_information_length, file_size, sample_rate,
                                             bit_depth, sound_description, video_url])
                driver.back()
                driver.find_element(By.XPATH, '//*[@id="cookie-accept"]').click()
                driver.implicitly_wait(2)

        except:
                print("Exception")

driver.quit()
