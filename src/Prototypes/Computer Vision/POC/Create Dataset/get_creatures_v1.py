import os
import io
import zipfile
import time
import requests
from PIL import Image
import re

API_URL = "https://commons.wikimedia.org/w/api.php"
SEARCH_TERM = "Aythya australi"
LIMIT = 200

SEARCH_TERM2 = re.sub(r"\s+", "_", SEARCH_TERM.strip()).lower()
out_dir = f"{SEARCH_TERM}_300x300a"
os.makedirs(out_dir, exist_ok=True)

# Wikimedia requires a descriptive User-Agent
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "BirdImageCollector/1.1 (Deakin University Capstone Project; s225149998@deakin.edu.au)"
})


def fetch_image_pages(limit):
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": SEARCH_TERM,
        "gsrlimit": 50,            # max per request
        "gsrnamespace": 6,         # File: namespace
        "prop": "imageinfo",
        "iiprop": "url",
        "iiurlwidth": 300,
        "iiurlheight": 300,
    }

    fetched = 0

    while fetched < limit:
        resp = SESSION.get(API_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        pages = data.get("query", {}).get("pages", {})
        if not pages:
            break

        for page in pages.values():
            if fetched >= limit:
                break

            infos = page.get("imageinfo", [])
            if not infos:
                continue

            info = infos[0]
            url = info.get("thumburl") or info.get("url")
            if not url:
                continue

            yield url
            fetched += 1

        cont = data.get("continue")
        if not cont:
            break

        params.update(cont)


def download_and_save(url, idx):
    resp = SESSION.get(url, timeout=30)
    resp.raise_for_status()
    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    img = img.resize((300, 300), Image.LANCZOS)
    filename = f"{SEARCH_TERM2}_{idx:03d}.png"
    path = os.path.join(out_dir, filename)
    img.save(path, format="PNG")
    return path


paths = []

for i, url in enumerate(fetch_image_pages(LIMIT), start=1):
    print(f"Downloading {i}: {url}")
    try:
        p = download_and_save(url, i)
        paths.append(p)
        time.sleep(0.2)  # be polite
    except Exception as e:
        print(f"Failed on {url}: {e}")

zip_path = f"{SEARCH_TERM2}_300x300.zip"

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in paths:
        zf.write(p, arcname=os.path.basename(p))

print(f"Saved {len(paths)} images and zipped to {zip_path}")
