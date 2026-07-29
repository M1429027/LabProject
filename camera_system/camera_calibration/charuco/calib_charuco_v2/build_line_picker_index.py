import argparse
import base64
import html
from pathlib import Path

import cv2

from pick_line_points import extract_frame, make_html, parse_point_ids, load_world_points


def safe_prefix(path):
    return path.stem.replace(" ", "_").replace("(", "").replace(")", "")


def write_picker(video_path, out_dir, frame_index, time_sec, point_ids, world_points):
    prefix = safe_prefix(video_path)
    frame, metadata = extract_frame(video_path, frame_index=frame_index, time_sec=time_sec)
    frame_path = out_dir / f"{prefix}_frame.jpg"
    if not cv2.imwrite(str(frame_path), frame):
        raise RuntimeError(f"Cannot write frame: {frame_path}")

    image_b64 = base64.b64encode(frame_path.read_bytes()).decode("ascii")
    html_path = out_dir / f"{prefix}_line_picker.html"
    html_path.write_text(
        make_html(
            image_b64=image_b64,
            metadata=metadata,
            point_ids=point_ids,
            world_points=world_points,
            output_name=f"{prefix}_line_points.json",
        ),
        encoding="utf-8",
    )
    return {
        "label": prefix,
        "video": str(video_path),
        "href": html_path.name,
        "frame_index": metadata["frame_index"],
        "frame_count": metadata["frame_count"],
        "image_size": metadata["image_size"],
    }


def write_index(entries, index_path):
    cards = []
    for item in entries:
        cards.append(
            '<a class="card" href="{}"><strong>{}</strong><span>{}</span>'
            "<small>{}x{}, frame {}/{}</small></a>".format(
                html.escape(item["href"]),
                html.escape(item["label"]),
                html.escape(item["video"]),
                int(item["image_size"][0]),
                int(item["image_size"][1]),
                int(item["frame_index"]),
                int(item["frame_count"]),
            )
        )

    index_path.write_text(
        """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>line point picker index</title>
<style>
body { margin:0; background:#101214; color:#eef2f5; font-family:system-ui,-apple-system,Segoe UI,sans-serif; }
header { padding:16px; background:#181c20; border-bottom:1px solid #333b44; }
h1 { margin:0; font-size:20px; }
main { padding:16px; display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:12px; }
.card { display:flex; flex-direction:column; gap:6px; padding:14px; border:1px solid #333b44; border-radius:8px; background:#181c20; color:#eef2f5; text-decoration:none; }
.card:hover { border-color:#ffd23f; }
.card span, .card small { color:#aab3bd; }
</style>
</head>
<body>
<header><h1>Line point picker</h1></header>
<main>
{}
</main>
</body>
</html>
""".replace("{}", "\n".join(cards)),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description="Build an index page for multiple floor-line point pickers.")
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--pattern", default="cam*out.mkv")
    parser.add_argument("--out-dir", default="line_picker_batch")
    parser.add_argument("--frame", type=int, default=None)
    parser.add_argument("--time-sec", type=float, default=None)
    parser.add_argument("--point-ids", default="")
    parser.add_argument("--world-points-json", default="")
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    videos = sorted(video_dir.glob(args.pattern))
    if not videos:
        raise FileNotFoundError(f"No videos matched: {video_dir / args.pattern}")

    point_ids = parse_point_ids(args.point_ids)
    world_points = load_world_points(args.world_points_json)
    entries = [write_picker(video, out_dir, args.frame, args.time_sec, point_ids, world_points) for video in videos]
    index_path = out_dir / "index.html"
    write_index(entries, index_path)

    print("[OK] Picker index created")
    print(f"  Open: {index_path}")
    for entry in entries:
        print(f"  {entry['label']}: {out_dir / entry['href']}")


if __name__ == "__main__":
    main()
