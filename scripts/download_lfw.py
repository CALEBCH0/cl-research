#!/usr/bin/env python3
"""Script to download LFW dataset with better error handling."""
import os
import sys
import urllib.request
import tarfile
import requests
from tqdm import tqdm

def download_with_progress(url, filename):
    """Download file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc='Downloading') as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
                pbar.update(len(chunk))

def main():
    root = "datasets/face_datasets/lfw"
    os.makedirs(root, exist_ok=True)
    
    # Check if already exists
    if os.path.exists(os.path.join(root, 'lfw')):
        print("LFW dataset already exists!")
        return
    
    # Try multiple sources
    urls = [
        "http://vis-www.cs.umass.edu/lfw/lfw.tgz",  # Original (often down)
        "https://ndownloader.figshare.com/files/5976018",  # Alternative source
        "http://conradsanderson.id.au/lfwcrop/lfwcrop_grey.zip"  # Cropped version
    ]
    
    filename = os.path.join(root, "lfw.tgz")
    
    print("Downloading LFW dataset...")
    
    downloaded = False
    for i, url in enumerate(urls):
        print(f"\nTrying source {i+1}/{len(urls)}: {url}")
        
        # Adjust filename based on URL
        if url.endswith('.zip'):
            filename = os.path.join(root, "lfw.zip")
        else:
            filename = os.path.join(root, "lfw.tgz")
            
        try:
            # Try with requests first
            download_with_progress(url, filename)
            downloaded = True
            break
        except Exception as e:
            print(f"Failed with requests: {e}")
            try:
                print("Trying with urllib...")
                urllib.request.urlretrieve(url, filename)
                downloaded = True
                break
            except Exception as e2:
                print(f"Failed with urllib: {e2}")
                continue
    
    if not downloaded:
        print("\nAll download sources failed!")
        print("\nAlternative: Use Kaggle dataset")
        print("1. Go to: https://www.kaggle.com/datasets/jessicali9530/lfw-dataset")
        print("2. Download and extract to: datasets/face_datasets/")
        print("\nOr try wget with different sources:")
        for url in urls:
            print(f"wget {url} -O {filename}")
        sys.exit(1)
    
    print("Extracting...")
    if filename.endswith('.zip'):
        import zipfile
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(root)
    else:
        with tarfile.open(filename, 'r:gz') as tar:
            tar.extractall(root)
    
    os.remove(filename)
    print("✓ LFW dataset downloaded successfully!")

if __name__ == "__main__":
    main()