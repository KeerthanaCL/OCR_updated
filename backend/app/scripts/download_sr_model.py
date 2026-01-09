"""Script to download ESPCN super-resolution model"""

import os
import urllib.request

def download_sr_model():
    """Download ESPCN super-resolution model"""
    model_dir = "./models"
    os.makedirs(model_dir, exist_ok=True)
    
    # Fixed URL: Use raw.githubusercontent.com
    model_url = "https://github.com/fannymonori/TF-ESPCN/raw/master/export/ESPCN_x4.pb"
    model_path = os.path.join(model_dir, "ESPCN_x4.pb")
    
    if os.path.exists(model_path):
        print(f"Model already exists: {model_path}")
        return
    
    print(f"Downloading super-resolution model from:\n{model_url}")
    
    try:
        urllib.request.urlretrieve(model_url, model_path)
        print(f"Downloaded: {model_path}")
    except Exception as e:
        print(f"Download failed: {e}")
        print("You can manually download from:")
        print("https://github.com/fannymonori/TF-ESPCN/tree/master/export")
        raise

if __name__ == "__main__":
    download_sr_model()