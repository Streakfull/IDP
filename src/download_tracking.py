import SoccerNet
from SoccerNet.Downloader import SoccerNetDownloader
import os
import shutil
import time

path = "./raw_dataset/soccernet-reid/raw"
mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory=path)
mySoccerNetDownloader.password = "s0cc3rn3t"
# Maximum retry attempts
max_retries = 20


def download_tracking():
    for attempt in range(1, max_retries + 1):
        print(f"Attempt {attempt}/{max_retries} to download SoccerNet data...")

        try:
            mySoccerNetDownloader.downloadDataTask(
                task="tracking", split=["test"])
            print("✅ Download completed successfully!")
            break  # Exit loop if download is successful

        except Exception as e:
            print(f"❌ Download failed (Attempt {attempt}): {e}")

            # Remove the dataset folder if download fails
            if os.path.exists(path):
                print(f"⚠️ Deleting folder: {path}")
                # Remove directory and ignore errors
                shutil.rmtree(path, ignore_errors=True)

            if attempt < max_retries:
                print("⏳ Waiting 1 minute before retrying...\n")
                time.sleep(180)  # Wait for 60 seconds before retrying
            else:
                print("❌ Max retries reached. Download failed.")


def download_jersey():
    mySoccerNetDownloader.downloadDataTask(
        task="jersey-2023", split=["train", "test"])


def download_reid():
    mySoccerNetDownloader.downloadDataTask(
        task="reid", split=["train", "valid", "test"])


download_reid()
