# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 18:25:08 2025

@author: jblair
"""

# -*- coding: utf-8 -*-

import os
import io
import zipfile
import time
import requests
from PIL import Image
import re
import json

API_URL = "https://commons.wikimedia.org/w/api.php"
BIRDS_JSON = "creatures.json"   # <--*.json file with scientific bird names
LIMIT_PER_BIRD = 100          # images per bird

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "BirdImageCollector/1.1 (Deakin University Capstone Project; s225149998@deakin.edu.au)"
})


def fetch_image_pages(search_term, limit):
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": search_term,
        "gsrlimit": 50,          # Wikimedia max per request
        "gsrnamespace": 6,       # File: namespace (images)
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


def download_and_save(url, search_term2, out_dir, idx):
    resp = SESSION.get(url, timeout=30)
    resp.raise_for_status()
    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    img = img.resize((300, 300), Image.LANCZOS)
    filename = f"{search_term2}_{idx:03d}.png"
    path = os.path.join(out_dir, filename)
    img.save(path, format="PNG")
    return path


def main():
    with open(BIRDS_JSON, "r", encoding="utf-8") as f:
        birds_data = json.load(f)

    birds = birds_data.get("birds", [])
    print(f"Found {len(birds)} birds in {BIRDS_JSON}")

    for bird_name in birds:
        search_term = bird_name
        search_term2 = re.sub(r"\s+", "_", search_term.strip()).lower()
        out_dir = f"{search_term2}_300x300a"
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n=== Processing bird: {bird_name} ===")
        print(f"Output dir: {out_dir}")

        paths = []
        for i, url in enumerate(fetch_image_pages(search_term, LIMIT_PER_BIRD), start=1):
            print(f"  Downloading {i}: {url}")
            try:
                p = download_and_save(url, search_term2, out_dir, i)
                paths.append(p)
                time.sleep(0.2)  # be polite to the API
            except Exception as e:
                print(f"  Failed on {url}: {e}")

        if not paths:
            print(f"  No images downloaded for {bird_name}, skipping zip.")
            continue

        zip_path = f"{search_term2}_300x300.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in paths:
                zf.write(p, arcname=os.path.basename(p))

        print(f"  Saved {len(paths)} images and zipped to {zip_path}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
