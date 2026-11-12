"""Generate the gallery/probe partition files used for the CloGait evaluation.

The files shipped in ``partitions/`` were produced with this script. Run it
only if you want to regenerate them or define new protocols.

    python create_partition_gallery_probe.py --root /path/to/CloGait/silhouettes

Protocols written to ``--out`` (default: ./partitions):
  AllViews/G_Casual__P_<cloth>.json   gallery: Casual, probe: <cloth>, all 5 views
  View<N>/G_Casual__P_<cloth>.json    same, restricted to view N
  CrossView/G_View<i>__P_View<j>.json gallery view i, probe view j (all clothing)
"""
import argparse
import json
from itertools import product
from pathlib import Path

CLOTHES = ["Casual", "AoDai", "Bag", "Coat", "Longdress"]
PROBE_CLOTHES = [c for c in CLOTHES if c != "Casual"]
VIEWS = [f"View{i}" for i in range(1, 6)]


def list_sequences(root: Path):
    seqs = []
    for pid in sorted(p for p in root.iterdir() if p.is_dir()):
        for cloth in sorted(p for p in pid.iterdir() if p.is_dir()):
            for view in sorted(p for p in cloth.iterdir() if p.is_dir()):
                seqs.append((pid.name, cloth.name, view.name))
    return seqs


def select(seqs, clothes, views):
    return sorted(f"{p}-{c}-{v}" for p, c, v in seqs if c in clothes and v in views)


def dump(path: Path, gallery, probe):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"GALLERY": gallery, "PROBE": probe}, f, indent=4)
    print(f"{path}: gallery={len(gallery)} probe={len(probe)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="CloGait/silhouettes directory (person/clothing/view)")
    ap.add_argument("--out", default="./partitions")
    args = ap.parse_args()
    seqs, out = list_sequences(Path(args.root)), Path(args.out)

    for cloth in PROBE_CLOTHES:
        dump(out / "AllViews" / f"G_Casual__P_{cloth}.json",
             select(seqs, {"Casual"}, VIEWS), select(seqs, {cloth}, VIEWS))
        for v in VIEWS:
            dump(out / v / f"G_Casual__P_{cloth}.json",
                 select(seqs, {"Casual"}, {v}), select(seqs, {cloth}, {v}))
    for g, p in product(VIEWS, VIEWS):
        if g != p:
            dump(out / "CrossView" / f"G_{g}__P_{p}.json",
                 select(seqs, set(CLOTHES), {g}), select(seqs, set(CLOTHES), {p}))


if __name__ == "__main__":
    main()
