"""Publish a verified HYB1 improvement to main using a private temporary checkout."""
import argparse
from itertools import product
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from verify import CAP, TENSORS, evaluate

REMOTE = "https://github.com/fkesheh/hy4-flash-compressed.git"


def git(*args, cwd=None, check=True):
    return subprocess.run(["git", "-c", "credential.helper=", "-c",
                           "credential.helper=!gh auth git-credential", *args],
                          cwd=cwd, check=check, text=True, capture_output=False)


def promote(repo, candidates, method, refs):
    folder = repo / "champion"
    previous = json.loads((folder / "result.json").read_text())
    if previous.get("format") != "HYB1":
        raise ValueError("Champion uses another codec. Compare the candidate in a dedicated PR.")
    current = [folder / f"{name}.hyb" for name in TENSORS]
    best = evaluate(current, refs)
    chosen = None
    for paths in product(*zip(current, candidates)):
        if paths == tuple(current) or sum(p.stat().st_size for p in paths) > CAP:
            continue
        result = evaluate(paths, refs)
        if result["score"] < best["score"] - 1e-10:
            best, chosen = result, paths
    if chosen is None:
        print(f"Main already scores {best['score']:.10f} or better than this candidate")
        return False
    best["format"] = "HYB1"
    for name, old, new in zip(TENSORS, current, chosen):
        if old != new:
            shutil.copyfile(new, old)
            provenance = method
        else:
            provenance = previous["projections"][name]["method"]
        best["projections"][name]["method"] = provenance
    (folder / "result.json").write_text(json.dumps(best, indent=2) + "\n")
    git("add", "champion", cwd=repo)
    git("-c", "user.name=fkesheh", "-c", "user.email=fkesheh@users.noreply.github.com",
        "commit", "-m", f"Improve expert 0 to {best['score']:.10f} at {best['bytes']} bytes", cwd=repo)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gate_up", type=Path)
    parser.add_argument("down", type=Path)
    parser.add_argument("--method", required=True, help="Public algorithm and reproduction settings")
    parser.add_argument("--references", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        # Snapshot both candidates before comparison, even if another run overwrites them.
        snapshots = []
        for name, source in zip(TENSORS, [args.gate_up, args.down]):
            if source.stat().st_size > CAP:
                raise ValueError("Projection exceeds the expert budget")
            dest = root / f"{name}.hyb"
            shutil.copyfile(source, dest)
            snapshots.append(dest)
        for i in range(5):
            repo = root / str(i)
            git("clone", "--quiet", "--depth=1", "--branch", "main", REMOTE, str(repo))
            if not promote(repo, snapshots, args.method, args.references):
                return
            result = git("push", "origin", "HEAD:main", cwd=repo, check=False)
            if result.returncode == 0:
                return
            remote = subprocess.check_output(["git", "ls-remote", REMOTE, "refs/heads/main"], text=True).split()[0]
            parent = subprocess.check_output(["git", "rev-parse", "HEAD^"], cwd=repo, text=True).strip()
            if remote == parent:
                raise RuntimeError("Push failed. Check GitHub credentials and permissions.")
        raise RuntimeError("Main changed repeatedly. Retry with the same candidates.")


if __name__ == "__main__":
    main()
