"""
nuScenes dataset downloader for HiP-AD training.

Downloads all required data:
  - v1.0-trainval metadata + 10 blob files (camera images)
  - nuScenes map expansion
  - CAN bus data
  - (optional) v1.0-mini for smoke testing

Usage:
  # Set credentials via environment variables or edit below
  export NUSCENES_EMAIL="your@email.com"
  export NUSCENES_PASSWORD="yourpassword"

  python download.py --output-dir /path/to/nuscenes
  python download.py --output-dir /path/to/nuscenes --mini-only   # mini set only
  python download.py --output-dir /path/to/nuscenes --skip-blobs  # metadata/maps/canbus only
"""

import requests
import os
import hashlib
import argparse
import json
import tarfile
import gzip
import zipfile
from tqdm import tqdm


# ==================== credentials ====================
# Override via environment variables or edit directly
USEREMAIL = os.environ.get("NUSCENES_EMAIL", "your@email.com")
PASSWORD = os.environ.get("NUSCENES_PASSWORD", "yourpassword")

# ==================== download manifest ====================
# filename -> md5 (None = skip md5 check)

TRAINVAL_META = {
    "v1.0-trainval_meta.tgz": "b428ce833dfc8e0568a8d448300e09e5",
}

TRAINVAL_BLOBS = {
    "v1.0-trainval01_blobs.tgz": "75540bfa03b6443b56edbdffe913c5ee",
    "v1.0-trainval02_blobs.tgz": "1e013ac7e5e67f58ae23388b0e5f3a78",
    "v1.0-trainval03_blobs.tgz": "0ad55e5711b404e61aadc98e1e7ad498",
    "v1.0-trainval04_blobs.tgz": "5d62c7032e8f24e22e6fde3e1df8c142",
    "v1.0-trainval05_blobs.tgz": "20e78490deb0e4318dcdf7485b9c0e33",
    "v1.0-trainval06_blobs.tgz": "96d48b6e2b0c4e87ef6abb839ae2f4c8",
    "v1.0-trainval07_blobs.tgz": "bc75a2cf00cdbde3c22e519c3b10e429",
    "v1.0-trainval08_blobs.tgz": "e3e2c6ebba2b9990bc9e5e1d2a4a3f4a",
    "v1.0-trainval09_blobs.tgz": "b1e5fd47c0953ef9a3e20e26de2a5121",
    "v1.0-trainval10_blobs.tgz": "a2e80d4e5e20179aa14c5db9e5e27aaa",
}

MAPS = {
    "nuScenes-map-expansion-v1.3.zip": "c08e5a81e31b4db2aae0ca7a01a4a7e4",
}

CANBUS = {
    "can_bus.zip": "3d2089b91a0a40be6c904e3cfa210a67",
}

MINI = {
    "v1.0-mini.tgz": "d7503a57b83ac3bead58a5dae4603bd1",
}


def login(username, password):
    headers = {
        "Content-Type": "application/x-amz-json-1.1",
        "X-Amz-Target": "AWSCognitoIdentityProviderService.InitiateAuth",
    }
    data = json.dumps({
        "AuthFlow": "USER_PASSWORD_AUTH",
        "ClientId": "7fq5jvs5ffs1c50hd3toobb3b9",
        "AuthParameters": {
            "USERNAME": username,
            "PASSWORD": password,
        },
        "ClientMetadata": {},
    })
    response = requests.post(
        "https://cognito-idp.us-east-1.amazonaws.com/",
        headers=headers,
        data=data,
    )
    if response.status_code == 200:
        try:
            return json.loads(response.content)["AuthenticationResult"]["IdToken"]
        except KeyError:
            print("Authentication failed: 'AuthenticationResult' not in response")
    else:
        print(f"Login failed (status {response.status_code})")
    return None


def download_file(url, save_file, md5):
    response = requests.get(url, stream=True)
    if save_file.endswith(".tgz"):
        content_type = response.headers.get("Content-Type", "")
        if content_type == "application/x-tar":
            save_file = save_file.replace(".tgz", ".tar")
        elif content_type not in ("application/octet-stream", "application/zip", "binary/octet-stream"):
            print(f"  unknown content type: {content_type}")
            return save_file

    if os.path.exists(save_file) and md5:
        md5obj = hashlib.md5()
        with open(save_file, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                md5obj.update(chunk)
        if md5obj.hexdigest() == md5:
            print(f"  {os.path.basename(save_file)} already downloaded (md5 ok)")
            return save_file
        else:
            print(f"  {os.path.basename(save_file)} md5 mismatch, re-downloading")

    file_size = int(response.headers.get("Content-Length", 0))
    progress_bar = tqdm(
        total=file_size, unit="B", unit_scale=True, unit_divisor=1024,
        desc=f"  {os.path.basename(save_file)}", ascii=True,
    )
    md5obj = hashlib.md5()
    with open(save_file, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                md5obj.update(chunk)
                f.write(chunk)
                progress_bar.update(len(chunk))
    progress_bar.close()

    if md5:
        if md5obj.hexdigest() != md5:
            print(f"  WARNING: {os.path.basename(save_file)} md5 mismatch!")
        else:
            print(f"  {os.path.basename(save_file)} md5 ok")
    return save_file


def extract(filepath):
    dirname = os.path.dirname(filepath)
    basename = os.path.basename(filepath)
    print(f"  extracting {basename} ...")
    if filepath.endswith(".zip"):
        with zipfile.ZipFile(filepath, "r") as z:
            z.extractall(dirname)
    elif filepath.endswith(".tgz"):
        with gzip.open(filepath, "rb") as f_in:
            with tarfile.open(fileobj=f_in, mode="r") as tar:
                tar.extractall(dirname)
    elif filepath.endswith(".tar"):
        with tarfile.open(filepath, "r") as tar:
            tar.extractall(dirname)
    else:
        print(f"  unknown archive type: {basename}")


def get_download_url(filename, headers, region="us"):
    api_url = (
        f"https://o9k5xn5546.execute-api.us-east-1.amazonaws.com/v1/archives"
        f"/v1.0/{filename}?region={region}&project=nuScenes"
    )
    response = requests.get(api_url, headers=headers)
    if response.status_code == 200:
        return response.json()["url"]
    else:
        print(f"  failed to get URL for {filename} (status {response.status_code})")
        print(f"  {response.text}")
        return None


def download_and_extract(file_dict, headers, output_dir, region):
    os.makedirs(output_dir, exist_ok=True)
    for filename, md5 in file_dict.items():
        print(f"\n[{filename}]")
        url = get_download_url(filename, headers, region)
        if url is None:
            continue
        save_path = os.path.join(output_dir, filename)
        save_path = download_file(url, save_path, md5)
        extract(save_path)


def main():
    parser = argparse.ArgumentParser(description="Download nuScenes dataset for HiP-AD")
    parser.add_argument("--output-dir", type=str, default="./data/nuscenes",
                        help="Output directory (default: ./data/nuscenes)")
    parser.add_argument("--region", type=str, default="us", choices=["us", "asia"],
                        help="Download region (default: us)")
    parser.add_argument("--mini-only", action="store_true",
                        help="Download mini set only (for smoke testing)")
    parser.add_argument("--skip-blobs", action="store_true",
                        help="Skip blob files (download metadata/maps/canbus only)")
    parser.add_argument("--email", type=str, default=None,
                        help="nuscenes.org email (or set NUSCENES_EMAIL env var)")
    parser.add_argument("--password", type=str, default=None,
                        help="nuscenes.org password (or set NUSCENES_PASSWORD env var)")
    args = parser.parse_args()

    email = args.email or USEREMAIL
    pw = args.password or PASSWORD
    if email == "your@email.com" or pw == "yourpassword":
        print("Set credentials via --email/--password or NUSCENES_EMAIL/NUSCENES_PASSWORD env vars")
        return

    print(f"Output: {args.output_dir}")
    print(f"Region: {args.region}")
    print(f"Login: {email}")

    token = login(email, pw)
    if token is None:
        return
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    print("Login success\n")

    if args.mini_only:
        # Mini set only
        print("=" * 50)
        print("Downloading v1.0-mini")
        print("=" * 50)
        download_and_extract(MINI, headers, args.output_dir, args.region)
    else:
        # Full trainval
        print("=" * 50)
        print("Downloading v1.0-trainval metadata")
        print("=" * 50)
        download_and_extract(TRAINVAL_META, headers, args.output_dir, args.region)

        if not args.skip_blobs:
            print("\n" + "=" * 50)
            print("Downloading v1.0-trainval blobs (10 files, ~300GB total)")
            print("=" * 50)
            download_and_extract(TRAINVAL_BLOBS, headers, args.output_dir, args.region)

    # Maps and CAN bus (always needed)
    print("\n" + "=" * 50)
    print("Downloading maps")
    print("=" * 50)
    download_and_extract(MAPS, headers, args.output_dir, args.region)

    print("\n" + "=" * 50)
    print("Downloading CAN bus")
    print("=" * 50)
    download_and_extract(CANBUS, headers, args.output_dir, args.region)

    print("\n" + "=" * 50)
    print("Done!")
    print(f"\nNext steps:")
    print(f"  1. Symlink: ln -s {os.path.abspath(args.output_dir)} data/nuscenes")
    print(f"  2. Generate infos: see nusc.md section 2")
    print(f"  3. Generate kmeans anchors: see nusc.md section 2")
    print("=" * 50)


if __name__ == "__main__":
    main()
