"""C1: copy each method repository's own dependency declaration (at the pinned commit) into
release/environments/<method>/repo_declared_<file>, and record it in release/configs/<method>.yaml. Idempotent."""
import os, shutil, subprocess
W = os.path.dirname(os.path.abspath(__file__)); V = os.path.join(W, ".."); R = os.path.join(V, "..")
FI = os.path.join(R, "FI", "000_Related work"); NII = os.path.join(R, "NII")
SRC = {"BAT-Fill": (NII, "BAT-Fill", "requirements.txt"), "CTSDG": (NII, "CTSDG", "requirements.txt"),
       "EdgeConnect": (NII, "edge-connect", "requirements.txt"), "MISF": (NII, "misf", "requirements.txt"),
       "StrDiff": (NII, "StrDiffusion", "requirements.txt"), "ZITS": (NII, "ZITS_inpainting", "requirement.txt"),
       "MADIN": (FI, "MADIN", "requirements.txt"), "spa-former": (FI, "spa-former", "requirements.txt"),
       "WaveFill": (FI, "WaveFill", "requirements.txt"), "SCAT": (FI, "Generative-Image-Inpainting", "environment.yml"),
       "VHII": (FI, "variable-hyperparameter-image-impainting", "environment.yml")}
for m, (base, d, f) in SRC.items():
    repo = os.path.join(base, d); cfg = os.path.join(V, "release", "configs", f"{m}.yaml")
    head = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    pinned = [l.split(":", 1)[1].strip() for l in open(cfg, encoding="utf-8") if l.startswith("commit:")][0]
    # read the file as committed at the pinned commit (not the working tree)
    blob = subprocess.run(["git", "-C", repo, "show", f"{pinned}:{f}"], capture_output=True)
    assert blob.returncode == 0, (m, blob.stderr[:200])
    dst = os.path.join(V, "release", "environments", m, "repo_declared_" + f)
    open(dst, "wb").write(blob.stdout)
    c = open(cfg, encoding="utf-8").read()
    if "repo_dependencies:" not in c:
        c = c.rstrip("\n") + f"\nrepo_dependencies: \"environments/{m}/repo_declared_{f} (as declared by the repository at the pinned commit; the versions actually installed for the original inference run were not recorded)\"\n"
        open(cfg, "w", encoding="utf-8").write(c)
    print(m, f, "HEAD==pinned" if head == pinned else f"HEAD {head[:8]} != pinned {pinned[:8]}", len(blob.stdout), "bytes")
print(len(SRC), "methods")
