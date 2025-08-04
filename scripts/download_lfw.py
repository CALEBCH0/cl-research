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
    
    url = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"
    filename = os.path.join(root, "lfw.tgz")
    
    print("Downloading LFW dataset...")
    print(f"URL: {url}")
    print(f"Destination: {filename}")
    
    try:
        # Try with requests first (better for WSL)
        download_with_progress(url, filename)
    except Exception as e:
        print(f"Failed with requests: {e}")
        print("Trying with urllib...")
        try:
            urllib.request.urlretrieve(url, filename)
        except Exception as e2:
            print(f"Failed with urllib: {e2}")
            print("\nPlease download manually:")
            print(f"wget {url} -O {filename}")
            print(f"tar -xzf {filename} -C {root}")
            sys.exit(1)
    
    print("Extracting...")
    with tarfile.open(filename, 'r:gz') as tar:
        tar.extractall(root)
    
    os.remove(filename)
    print("✓ LFW dataset downloaded successfully!")

if __name__ == "__main__":
    main()