"""Download the generated outputs (kept out of git) from the Zenodo record named in zenodo_record.json,
verify their SHA-256 checksums and unpack them into outputs/ and samples_table10/.
Usage: python download_outputs.py [--only outputs|samples_table10] [--files NAME.zip ...] [--keep-zips]
  e.g. python download_outputs.py --files outputs_LaMa.zip outputs_SCAT.zip"""
import argparse, hashlib, json, os, sys, urllib.request, zipfile
HERE = os.path.dirname(os.path.abspath(__file__))
cfg = json.load(open(os.path.join(HERE, "zenodo_record.json"), encoding="utf-8"))
rid = str(cfg["record_id"])
if not rid.isdigit():
    sys.exit("zenodo_record.json has no record ID yet (placeholder %s); see PUBLISHING.md" % rid)
ap = argparse.ArgumentParser(); ap.add_argument("--only", choices=["outputs", "samples_table10"]); ap.add_argument("--keep-zips", action="store_true")
ap.add_argument("--files", nargs="+", help="download only these archives (names as in zenodo_record.json)")
args = ap.parse_args()
known = {a["file"] for a in cfg["archives"]}
if args.files and set(args.files) - known:
    sys.exit("unknown archive(s): " + ", ".join(sorted(set(args.files) - known)))
dl = os.path.join(HERE, "_downloads"); os.makedirs(dl, exist_ok=True)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()


for a in cfg["archives"]:
    if args.only and not a["file"].startswith(args.only + "_"): continue
    if args.files and a["file"] not in args.files: continue
    dst = os.path.join(dl, a["file"])
    if not (os.path.exists(dst) and sha256(dst) == a["sha256"]):
        url = f"https://zenodo.org/records/{rid}/files/{a['file']}?download=1"
        print("downloading", a["file"], f"({a['bytes'] / 2**20:.0f} MB)")
        urllib.request.urlretrieve(url, dst)
        if sha256(dst) != a["sha256"]:
            sys.exit("checksum mismatch for " + a["file"])
    with zipfile.ZipFile(dst) as z: z.extractall(HERE)
    if not args.keep_zips: os.remove(dst)
    print("ok", a["extracts_to"], a["n_files"], "files")
