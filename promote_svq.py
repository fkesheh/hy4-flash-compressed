"""Validate and locally commit an SVQ1 candidate against current main."""
import argparse
from itertools import product
import json
from pathlib import Path
import shutil
import subprocess

from verify import CAP, TENSORS, evaluate


def git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True)


def promote(repo, candidates, method, refs):
    folder = repo / "champion"
    manifest_path = folder / "result.json"
    previous = json.loads(manifest_path.read_text())
    current = [folder / previous.get("projections", {}).get(n, {}).get("file", f"{n}.hyb")
               for n in TENSORS]
    best = evaluate(current, refs)
    chosen = None
    for paths in product(*zip(current, candidates)):
        if paths == tuple(current) or sum(p.stat().st_size for p in paths) > CAP:
            continue
        result = evaluate(paths, refs)
        if result["score"] < best["score"] - 1e-10:
            best, chosen = result, paths
    if chosen is None:
        print(f"Current champion scores {best['score']:.10f}; candidate pair does not improve it")
        return False

    used_files = set()
    for name, source in zip(TENSORS, chosen, strict=True):
        fmt = source.read_bytes()[:4].decode("ascii")
        suffix = {"HYB1": "hyb", "SVQ1": "svq"}.get(fmt)
        if suffix is None:
            raise ValueError(f"Unsupported format {fmt}")
        dest_name = f"{name}.{suffix}"
        dest = folder / dest_name
        if source.resolve() != dest.resolve():
            shutil.copyfile(source, dest)
        used_files.add(dest_name)
        best["projections"][name]["file"] = dest_name
        best["projections"][name]["format"] = fmt
        if source in candidates:
            best["projections"][name]["method"] = method
        else:
            best["projections"][name]["method"] = previous["projections"][name].get("method", "previous champion")

    for name in TENSORS:
        for suffix in ("hyb", "svq"):
            old = folder / f"{name}.{suffix}"
            if old.name not in used_files:
                old.unlink(missing_ok=True)
    formats = {info["format"] for info in best["projections"].values()}
    best["format"] = next(iter(formats)) if len(formats) == 1 else "MIXED"
    manifest_path.write_text(json.dumps(best, indent=2) + "\n")
    git("add", "champion", "README.md", "verify.py", "scale_vq.py", "scale_itq.py",
        "promote_svq.py", cwd=repo)
    git("-c", "user.name=fkesheh", "-c", "user.email=fkesheh@users.noreply.github.com",
        "commit", "-m", f"Add SVQ scale-codebook codec; score {best['score']:.10f}", cwd=repo)
    print(json.dumps(best, indent=2))
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gate_up", type=Path)
    parser.add_argument("down", type=Path)
    parser.add_argument("--method", required=True)
    parser.add_argument("--references", type=Path)
    args = parser.parse_args()
    promote(Path(__file__).resolve().parent,
            [args.gate_up.resolve(), args.down.resolve()], args.method, args.references)


if __name__ == "__main__":
    main()
