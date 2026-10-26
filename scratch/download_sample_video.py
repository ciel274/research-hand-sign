import urllib.request
import os

def download_file(url, save_path):
    print(f"Downloading {url} to {save_path}...")
    try:
        # User-agent header to avoid bot blocking
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        )
        with urllib.request.urlopen(req) as response:
            with open(save_path, 'wb') as out_file:
                out_file.write(response.read())
        print("Download successful!")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False

if __name__ == "__main__":
    target_dir = "/Users/reo/Desktop/5semester/研究/research-hand-sign/src/ui/static/temp"
    os.makedirs(target_dir, exist_ok=True)
    
    # Try Bhuribhat/ASL-Finger-Spelling-To-Text first
    url1 = "https://raw.githubusercontent.com/Bhuribhat/ASL-Finger-Spelling-To-Text/master/Practice_ASL_1.mp4"
    save_path = os.path.join(target_dir, "sample_fingerspelling.mp4")
    
    success = download_file(url1, save_path)
    if not success:
        # Fallback to jingjie00
        url2 = "https://raw.githubusercontent.com/jingjie00/Fingerspelling-A-Sign-Language-Recognition/master/RecognitionSimulationVideo.mp4"
        success = download_file(url2, save_path)
        
    if success:
        print(f"Sample video saved to: {save_path}")
    else:
        print("Failed to download any sample video.")
