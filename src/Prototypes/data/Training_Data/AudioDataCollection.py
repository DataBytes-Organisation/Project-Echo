from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from bs4 import BeautifulSoup
import pandas as pd
import time
import requests

df=pd.DataFrame({'links':[''], 'Title':[''], 'Uploaded By':[''], 'Length':[''],'Date Uploaded':[''],'Description':['']})

#file location of chrome driver
driver=webdriver.Chrome()

#Site URL
driver.get('https://freesound.org/home/login/?next=/')


###Code to login (need to log in to download audio)### 
username=input('Please enter your Free sounds username: ')
password=input('Please enter your Free sounds Password: ')
login_field = driver.find_element(By.XPATH,'//*[@id="id_username"]')
password_field =driver.find_element(By.XPATH,'//*[@id="id_password"]') 
login_field.send_keys(username)
password_field.send_keys(password)

button=driver.find_element(By.XPATH,'//*[@id="content_full"]/form/input[2]').click()


###CODE TO SEARCH###
#Please change to type of audio required
SearchFor = input('Please enter the type of audio you would like to scrape: ')

#Seach for chosen subject
audioSearch=driver.find_element(By.XPATH,'//*[@id="search"]/form/fieldset/input[1]')
audioSearch.send_keys(SearchFor)
button=driver.find_element(By.XPATH,'//*[@id="search_submit"]').click()


##SCRAPING THE RESULTS##
anoutherPage=True
lastPage=False
#Continue scraping until the last page is reached
while anoutherPage==True:
    soup=BeautifulSoup(driver.page_source, 'lxml')
    results = soup.find_all('div',class_="sample_player_small")
    try:
        next_page = soup.find('a', {'title':'Next Page'}).get('href')
        baseURL='https://freesound.org'
        next_page= baseURL+next_page
    except:
        lastPage=True
    for result in results:
        try:
            link = result.find('a', class_='title').get('href')
            full_link = 'https://freesound.org'+ link
            print(full_link)
        except:
            full_link = ''
            print('Unable to find full link')
        try:
            title = result.find('div', class_='sound_filename').text.strip()
            print(title)
        except:
            title=''
            print('Unable to find title')
        try:
            uploaded_by = result.find('a', class_='user').text.strip()
            print(uploaded_by)
        except:
            uploaded_by=''
            print('Unable to find uploaded by')
        try:
            length= result.find('div', class_='time-indicator').text.strip()
            print(length)
        except:
            length=''
            print('Unable to find length')
        try:
            date = result.find('span', class_='date').text.strip()
            print(date)
        except:
            date=''
            print('Unable to find date')
        try:
            desc= soup.find('p', class_='description').text.strip()
            print(desc)
        except:
            desc=''
            print('Unable to find Description')
        try:
            driver.get(full_link)
            soup=BeautifulSoup(driver.page_source, 'lxml')
        
            button=driver.find_element(By.XPATH,'//*[@id="download_button"]').click()
        except:
            pass
        df=df.append({'links':[full_link], 'Title':[title], 'Uploaded By':[uploaded_by], 'length':[length],'Date Uploaded':[date],'Description':[desc]}, ignore_index=True)
        if lastPage==True:
            anoutherPage=False
        else:
            driver.get(next_page)

#Export data to CSV program        
df.to_csv('weather.csv')