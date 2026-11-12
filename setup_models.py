"""Extract (or download, then extract) the Vosk models the notebook expects.

    python setup_models.py

For each model: if models/<name>/ already exists it is left alone; otherwise
models/<name>.zip is extracted, downloading it from alphacephei.com first if missing.
"""
import sys
import urllib.request
import zipfile
from pathlib import Path

MODELS = [
    "vosk-model-small-en-us-0.15",
    "vosk-model-small-es-0.42",
    "vosk-model-small-ru-0.22",
]
BASE_URL = "https://alphacephei.com/vosk/models/"
HERE = Path(__file__).resolve().parent / "models"


def main():
    HERE.mkdir(exist_ok=True)
    for name in MODELS:
        folder, archive = HERE / name, HERE / f"{name}.zip"
        if folder.is_dir():
            print(f"ok        {name}")
            continue
        if not archive.exists():
            print(f"download  {name}.zip ...", flush=True)
            urllib.request.urlretrieve(BASE_URL + archive.name, archive)
        print(f"extract   {name}.zip")
        with zipfile.ZipFile(archive) as z:
            z.extractall(HERE)
        if not folder.is_dir():
            sys.exit(f"Expected {folder} after extracting {archive.name}")
    print("Models ready.")


if __name__ == "__main__":
    main()
