import requests
import os

 

API_KEY = "S5zntBxGdMTaBLM0RxQQcf2TJp2pbuVohIrs9WRq"  # Replace with your API key
BASE_URL = "https://freesound.org/apiv2"
DOWNLOAD_DIR = "Downloaded_sounds"

 

def search_freesound(query, limit=10):
    """
    Search for sounds in Freesound.
    """
    endpoint = f"{BASE_URL}/search/text/"
    params = {
        "query": query,
        "token": API_KEY,
        "fields": "id,name,previews",
        "page_size": limit
    }
    response = requests.get(endpoint, params=params)
    response.raise_for_status()  # Raise an error for bad responses
    return response.json()['results']

 

def download_sound(sound_id, preview_url, filename):
    """
    Download a sound from Freesound.
    """
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)

    response = requests.get(preview_url)
    response.raise_for_status()

    with open(os.path.join(DOWNLOAD_DIR, filename), 'wb') as f:
        f.write(response.content)
    print(f"Downloaded {filename}")

 

def main():
    query = input("Enter the search term: ")
    results = search_freesound(query)
    for result in results:
        sound_id = result['id']
        preview_url = result['previews']['preview-hq-mp3']
        filename = f"{result['name']}.mp3"
        download_sound(sound_id, preview_url, filename)

 

if __name__ == "__main__":
    main()