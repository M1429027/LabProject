"""Build selected-camera contact sheets from extracted view contact frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create selected-view contact sheets for Stage A.")
    parser.add_argument("--contact-root", required=True, help="Root from build_view_contact_sheets.py")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--views", nargs="+", required=True, help="Camera ids, e.g. cam02 cam03 cam07 cam16")
    parser.add_argument("--thumb-width", type=int, default=360)
    parser.add_argument("--thumb-height", type=int, default=210)
    return parser.parse_args()


def tile_image(path: Path, label: str, width: int, height: int, font: ImageFont.ImageFont) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((width, height), Image.LANCZOS)
    canvas = Image.new("RGB", (width, height + 28), "white")
    canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, height, width, height + 28), fill=(20, 20, 20))
    draw.text((10, height + 8), label, fill=(255, 255, 255), font=font)
    return canvas


def main() -> None:
    args = parse_args()
    contact_root = Path(args.contact_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    views = [str(view) for view in args.views]
    font = ImageFont.load_default()

    summary_path = contact_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    selected_rows = []

    for item in summary:
        if "sequence" not in item:
            continue
        sequence = str(item["sequence"])
        sequence_slug = sequence.replace("/", "__")
        frames_dir = contact_root / sequence_slug / "frames"
        out_sequence_dir = output_dir / sequence_slug
        out_sequence_dir.mkdir(parents=True, exist_ok=True)

        tiles = []
        missing = []
        selected_frames = {}
        for view in views:
            frame_path = frames_dir / f"{view}.jpg"
            if not frame_path.exists():
                missing.append(view)
                continue
            selected_frames[view] = str(frame_path)
            tiles.append(tile_image(frame_path, view, int(args.thumb_width), int(args.thumb_height), font))

        if tiles:
            tile_w, tile_h = tiles[0].size
            sheet = Image.new("RGB", (2 * tile_w, 2 * tile_h), (245, 245, 245))
            for index, tile in enumerate(tiles[:4]):
                sheet.paste(tile, ((index % 2) * tile_w, (index // 2) * tile_h))
            sheet_path = out_sequence_dir / "selected_views.jpg"
            sheet.save(sheet_path, quality=92)
        else:
            sheet_path = None

        row = {
            "sequence": sequence,
            "selected_views": views,
            "missing_views": missing,
            "selected_frames": selected_frames,
            "selected_sheet": str(sheet_path) if sheet_path else None,
        }
        selected_rows.append(row)

        md_lines = [
            f"# {sequence} selected views",
            "",
            f"- views: `{', '.join(views)}`",
            f"- missing: `{', '.join(missing) if missing else 'none'}`",
            "",
        ]
        if sheet_path:
            md_lines.extend([f"![selected_views]({sheet_path})", ""])
        md_lines.extend(["| view | frame |", "|---|---|"])
        for view in views:
            md_lines.append(f"| {view} | `{selected_frames.get(view, '')}` |")
        (out_sequence_dir / "selected_views.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    index_lines = [
        "# Selected Karate Views",
        "",
        f"Selected views: `{', '.join(views)}`",
        "",
        "| sequence | missing | sheet |",
        "|---|---|---|",
    ]
    for row in selected_rows:
        sheet = row.get("selected_sheet")
        rel_sheet = Path(sheet).relative_to(output_dir) if sheet else ""
        missing = ", ".join(row["missing_views"]) if row["missing_views"] else "none"
        index_lines.append(f"| {row['sequence']} | {missing} | `{rel_sheet}` |")

    (output_dir / "selected_views_index.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    (output_dir / "selected_views_manifest.json").write_text(
        json.dumps(
            {
                "selected_views": views,
                "sequences": selected_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "num_sequences": len(selected_rows),
                "views": views,
                "index": str(output_dir / "selected_views_index.md"),
                "manifest": str(output_dir / "selected_views_manifest.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
