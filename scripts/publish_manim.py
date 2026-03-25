"""Render repository teaching GIFs and copy them to stable asset names."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "utils" / "manim_animations_full.py"
ASSETS_DIR = PROJECT_ROOT / "assets" / "gifs"
MEDIA_ROOT = PROJECT_ROOT / "media"

SCENE_ORDER = [
    "GraphBasicsScene",
    "GCNMessagePassingScene",
    "FeatureVectorUpdateScene",
    "PyGEdgeIndexScene",
    "ProteinBackboneScene",
    "ProteinDistanceGraphScene",
    "GraphPoolingScene",
    "SaliencyVsSubgraphScene",
    "LRPNumericConservationScene",
    "GNNLRPRelevantWalkScene",
    "WildtypeMutantDeltaScene",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quality",
        choices=["l", "m", "h", "p", "k"],
        default="l",
        help="Manim quality flag. Uses -q{quality}. Default: l",
    )
    parser.add_argument(
        "--format",
        default="gif",
        choices=["gif", "mp4"],
        help="Output format to collect from the media directory.",
    )
    parser.add_argument(
        "--scene",
        dest="scenes",
        action="append",
        choices=SCENE_ORDER,
        help="Render only a specific scene. Repeat the flag to render multiple scenes.",
    )
    parser.add_argument(
        "--skip-render",
        action="store_true",
        help="Only recopy previously rendered outputs into assets/gifs.",
    )
    return parser.parse_args()


def render_scenes(scene_names: list[str], quality: str, output_format: str) -> None:
    command = [
        "manim",
        f"-q{quality}",
        str(SCRIPT_PATH),
        *scene_names,
        f"--format={output_format}",
    ]
    print("Rendering scenes:")
    for scene in scene_names:
        print(f"  - {scene}")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def find_rendered_asset(scene_name: str, extension: str) -> Path:
    patterns = [
        f"{scene_name}_ManimCE_v*.{extension}",
        f"{scene_name}.{extension}",
    ]
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(MEDIA_ROOT.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"Could not find a rendered {extension} for {scene_name}.")
    return max(matches, key=lambda path: path.stat().st_mtime)


def copy_assets(scene_names: list[str], output_format: str) -> list[tuple[str, Path]]:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    copied: list[tuple[str, Path]] = []
    for scene_name in scene_names:
        source = find_rendered_asset(scene_name, output_format)
        destination = ASSETS_DIR / f"{scene_name}.{output_format}"
        shutil.copy2(source, destination)
        copied.append((scene_name, destination))
    return copied


def print_summary(copied_assets: list[tuple[str, Path]]) -> None:
    print("\nStable asset paths:")
    for scene_name, path in copied_assets:
        print(f"  - {scene_name}: {path.relative_to(PROJECT_ROOT)}")


def main() -> None:
    args = parse_args()
    scene_names = args.scenes or SCENE_ORDER
    if not args.skip_render:
        render_scenes(scene_names, args.quality, args.format)
    copied_assets = copy_assets(scene_names, args.format)
    print_summary(copied_assets)


if __name__ == "__main__":
    main()

