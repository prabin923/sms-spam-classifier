"""Download the UCI SMS Spam Collection dataset."""

import urllib.request
import zipfile
from pathlib import Path

from config import DATA_DIR, RAW_DATA_FILE

# Current UCI hosting (legacy /ml/... path returns 404)
DATASET_ZIP_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"


def download() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_DATA_FILE.exists():
        print(f"Dataset already exists: {RAW_DATA_FILE}")
        return RAW_DATA_FILE

    zip_path = DATA_DIR / "sms_spam.zip"
    print(f"Downloading from {DATASET_ZIP_URL} ...")
    urllib.request.urlretrieve(DATASET_ZIP_URL, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if name.endswith("SMSSpamCollection"):
                zf.extract(name, DATA_DIR)
                extracted = DATA_DIR / name
                extracted.rename(RAW_DATA_FILE)
                break
        else:
            raise FileNotFoundError("SMSSpamCollection not found inside zip")

    zip_path.unlink(missing_ok=True)
    print(f"Saved to {RAW_DATA_FILE}")
    return RAW_DATA_FILE


if __name__ == "__main__":
    download()
