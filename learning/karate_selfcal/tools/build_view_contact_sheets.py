"""Build contact sheets for all camera views in Harmony4D karate zip files."""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


IMAGE_RE = re.compile(r"^([^/]+/[^/]+)/exo/(cam\d+)/images/([^/]+\.(?:jpg|jpeg|png))$", re.I)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract one frame per camera and build contact sheets.")
    parser.add_argument("--zips", nargs="+", required=True, help="Harmony4D zip files.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--thumb-width", type=int, default=320)
    parser.add_argument("--thumb-height", type=int, default=180)
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument(
        "--frame-position",
        type=float,
        default=0.5,
        help="Frame position in each camera image list, from 0.0 to 1.0.",
    )
    return parser.parse_args()


def camera_sort_key(camera_id: str) -> int:
    match = re.search(r"\d+", camera_id)
    return int(match.group()) if match else 9999


def select_frame(names: list[str], frame_position: float) -> str:
    if not names:
        raise ValueError("Cannot select a frame from an empty list.")
    pos = min(max(float(frame_position), 0.0), 1.0)
    index = int(round((len(names) - 1) * pos))
    return sorted(names)[index]


def build_tile(image: Image.Image, label: str, thumb_width: int, thumb_height: int, font: ImageFont.ImageFont) -> Image.Image:
    thumb = image.copy()
    thumb.thumbnail((thumb_width, thumb_height), Image.LANCZOS)
    canvas = Image.new("RGB", (thumb_width, thumb_height + 24), "white")
    x = (thumb_width - thumb.width) // 2
    y = (thumb_height - thumb.height) // 2
    canvas.paste(thumb, (x, y))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, thumb_height, thumb_width, thumb_height + 24), fill=(20, 20, 20))
    draw.text((8, thumb_height + 6), label, fill=(255, 255, 255), font=font)
    return canvas


def write_contact_sheet(
    tiles: list[Image.Image],
    output_path: Path,
    cols: int,
) -> None:
    if not tiles:
        return
    tile_w, tile_h = tiles[0].size
    rows = (len(tiles) + int(cols) - 1) // int(cols)
    sheet = Image.new("RGB", (int(cols) * tile_w, rows * tile_h), (245, 245, 245))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % int(cols)) * tile_w, (index // int(cols)) * tile_h))
    sheet.save(output_path, quality=92)


def process_zip(
    zip_path: Path,
    output_root: Path,
    thumb_width: int,
    thumb_height: int,
    cols: int,
    frame_position: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    font = ImageFont.load_default()
    with zipfile.ZipFile(zip_path) as archive:
        by_sequence: dict[str, dict[str, list[str]]] = {}
        for name in archive.namelist():
            match = IMAGE_RE.match(name)
            if not match:
                continue
            sequence_id, camera_id, _ = match.groups()
            by_sequence.setdefault(sequence_id, {}).setdefault(camera_id, []).append(name)

        for sequence_id in sorted(by_sequence):
            cameras = by_sequence[sequence_id]
            sequence_slug = sequence_id.replace("/", "__")
            sequence_dir = output_root / sequence_slug
            frames_dir = sequence_dir / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)

            tiles: list[Image.Image] = []
            camera_rows: list[dict[str, object]] = []
            for camera_id in sorted(cameras, key=camera_sort_key):
                image_names = sorted(cameras[camera_id])
                selected = select_frame(image_names, frame_position)
                image = Image.open(io.BytesIO(archive.read(selected))).convert("RGB")
                frame_path = frames_dir / f"{camera_id}.jpg"
                image.save(frame_path, quality=92)
                label = f"{camera_id} | {Path(selected).name}"
                tiles.append(build_tile(image, label, thumb_width, thumb_height, font))
                camera_rows.append(
                    {
                        "camera": camera_id,
                        "frame": Path(selected).name,
                        "source": selected,
                        "output": str(frame_path),
                    }
                )

            sheet_path = sequence_dir / "contact_sheet.jpg"
            write_contact_sheet(tiles, sheet_path, cols)

            md_lines = [
                f"# {sequence_id}",
                "",
                f"- source zip: `{zip_path}`",
                f"- cameras: {len(camera_rows)}",
                f"- selected frame position: {frame_position:.2f}",
                "",
                f"![contact_sheet]({sheet_path})",
                "",
                "| camera | frame | file |",
                "|---|---|---|",
            ]
            for row in camera_rows:
                md_lines.append(f"| {row['camera']} | {row['frame']} | `{row['output']}` |")
            (sequence_dir / "view_index.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

            rows.append(
                {
                    "zip": str(zip_path),
                    "sequence": sequence_id,
                    "num_cameras": len(camera_rows),
                    "contact_sheet": str(sheet_path),
                    "frames_dir": str(frames_dir),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, object]] = []
    for zip_name in args.zips:
        zip_path = Path(zip_name)
        if not zip_path.exists():
            summary.append({"zip": str(zip_path), "status": "missing"})
            continue
        summary.extend(
            process_zip(
                zip_path=zip_path,
                output_root=output_root,
                thumb_width=int(args.thumb_width),
                thumb_height=int(args.thumb_height),
                cols=int(args.cols),
                frame_position=float(args.frame_position),
            )
        )

    index_lines = [
        "# Karate View Contact Sheets",
        "",
        "| sequence | cameras | contact sheet |",
        "|---|---:|---|",
    ]
    for item in summary:
        if item.get("status") == "missing":
            index_lines.append(f"| {item['zip']} | 0 | missing |")
            continue
        sheet = str(item["contact_sheet"])
        rel_sheet = os.path.relpath(sheet, output_root)
        index_lines.append(f"| {item['sequence']} | {item['num_cameras']} | `{rel_sheet}` |")

    (output_root / "index.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "num_sequences": sum(1 for item in summary if item.get("sequence")),
                "index": str(output_root / "index.md"),
                "summary": str(output_root / "summary.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
