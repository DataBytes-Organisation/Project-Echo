import requests
from bs4 import BeautifulSoup
import csv

class BirdNameScraper:
    def __init__(self, url):
        self.url = url
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    def fetch_page_content(self):
        try:
            response = requests.get(self.url, headers=self.headers)
            response.raise_for_status()
            return response.content
        except requests.RequestException as e:
            print(f"Error fetching page content: {e}")
            return None

    def extract_scientific_names(self, content):
        soup = BeautifulSoup(content, 'html.parser')
        scientific_names = []

        for row in soup.find_all('tr'):
            cells = row.find_all('td')
            if cells and cells[0].a:
                scientific_names.append(cells[0].a.text)

        return scientific_names

    def save_to_csv(self, names):
        with open('Australian_bird_names.csv', 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["Scientific Name"])
            for name in names:
                writer.writerow([name])

    def run(self):
        content = self.fetch_page_content()
        if content:
            names = self.extract_scientific_names(content)
            self.save_to_csv(names)
            print("Saved bird names to Australian_bird_names.csv")

if __name__ == "__main__":
    URL = "https://www.ozanimals.com/australian-bird-index/genus.html"
    scraper = BirdNameScraper(URL)
    scraper.run()

