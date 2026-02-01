-- coding: utf-8 -- 

""" Created on Thu Nov 27 18:16:50 2025 

@author: jblair """ 

import os import random import numpy as np from PIL import Image, ImageEnhance, ImageFilter 

--- CONFIG --- 

input_dir = "Aythya australi_300x300a" # original 200-ish images output_dir = "Aythya_australi_augmented" # where augmented images go target_count = 1000 # total images  

os.makedirs(output_dir, exist_ok=True) random.seed(42) np.random.seed(42) 

def random_augment(img): """Apply a random sequence of augmentations to a PIL Image (RGB, 300x300).""" w, h = img.size 

# Horizontal flip 
if random.random() < 0.5: 
    img = img.transpose(Image.FLIP_LEFT_RIGHT) 
 
# Small rotation 
if random.random() < 0.5: 
    angle = random.uniform(-20, 20)  # degrees 
    img = img.rotate(angle, resample=Image.BICUBIC) 
 
# Random crop & resize back (acts like zoom/translation) 
if random.random() < 0.7: 
    scale = random.uniform(0.7, 1.0)  # 0.7 = tighter crop, 1.0 = no crop 
    new_w, new_h = int(w * scale), int(h * scale) 
    if new_w < w and new_h < h: 
        left = random.randint(0, w - new_w) 
        top = random.randint(0, h - new_h) 
        img = img.crop((left, top, left + new_w, top + new_h)) 
        img = img.resize((w, h), Image.LANCZOS) 
 
# Brightness 
if random.random() < 0.7: 
    enhancer = ImageEnhance.Brightness(img) 
    img = enhancer.enhance(random.uniform(0.7, 1.3)) 
 
# Contrast 
if random.random() < 0.7: 
    enhancer = ImageEnhance.Contrast(img) 
    img = enhancer.enhance(random.uniform(0.7, 1.3)) 
 
# Colour / saturation 
if random.random() < 0.7: 
    enhancer = ImageEnhance.Color(img) 
    img = enhancer.enhance(random.uniform(0.7, 1.3)) 
 
# Slight blur 
if random.random() < 0.3: 
    radius = random.uniform(0.3, 1.0) 
    img = img.filter(ImageFilter.GaussianBlur(radius=radius)) 
 
# Add Gaussian noise 
if random.random() < 0.5: 
    arr = np.asarray(img).astype("float32") 
    noise = np.random.normal(0, 8, arr.shape)  # mean 0, std 8 
    arr = arr + noise 
    arr = np.clip(arr, 0, 255).astype("uint8") 
    img = Image.fromarray(arr) 
 
return img 
  

--- MAIN LOGIC --- 

Collect source images 

image_paths = [ os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")) ] 

if not image_paths: raise RuntimeError(f"No images found in '{input_dir}'") 

print(f"Found {len(image_paths)} base images in '{input_dir}'.") 

idx = 0 

1) Copy originals into the augmented folder first (keeps raw images in the set) 

for path in image_paths: img = Image.open(path).convert("RGB") idx += 1 out_path = os.path.join(output_dir, f"image_{idx:04d}.png") img.save(out_path) print(f"Copied {idx} original images to '{output_dir}'.") 

2) Keep generating augmentations until we hit target_count 

while idx < target_count: src_path = random.choice(image_paths) img = Image.open(src_path).convert("RGB") aug = random_augment(img) 

idx += 1 
out_path = os.path.join(output_dir, f"eagle_{idx:04d}.png") 
aug.save(out_path) 
 
if idx % 100 == 0 or idx == target_count: 
    print(f"Created {idx} images so far...") 
  

print(f"Done. Saved {idx} images total in '{output_dir}'.") 

 