import argparse
import base64
import json
from pathlib import Path

import cv2


def parse_point_ids(value):
    if not value:
        return []
    return [item.strip() for item in value.split(',') if item.strip()]


def load_world_points(path):
    if not path:
        return []
    with open(path, 'r', encoding='utf-8') as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and 'points' in payload:
        payload = payload['points']
    if not isinstance(payload, list):
        raise ValueError('world points JSON must be a list or an object with a points list')

    points = []
    for index, item in enumerate(payload):
        if isinstance(item, dict):
            point_id = str(item.get('id', f'P{index:02d}'))
            world = item.get('world', [item['x'], item['y'], item.get('z', 0.0)])
        else:
            point_id = f'P{index:02d}'
            world = item
        if len(world) != 3:
            raise ValueError(f'world point {point_id} must have 3 values')
        points.append({'id': point_id, 'world': [float(v) for v in world]})
    return points


def extract_frame(video_path, frame_index=None, time_sec=None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f'Cannot open video: {video_path}')

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if time_sec is not None:
        if fps <= 0:
            raise RuntimeError('Video FPS is unavailable; use --frame instead of --time-sec')
        frame_index = int(round(float(time_sec) * fps))
    if frame_index is None:
        frame_index = frame_count // 2 if frame_count > 0 else 0
    frame_index = max(0, int(frame_index))
    if frame_count > 0:
        frame_index = min(frame_index, frame_count - 1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f'Cannot read frame {frame_index} from {video_path}')

    metadata = {
        'video_path': str(video_path),
        'frame_index': int(frame_index),
        'fps': fps,
        'frame_count': frame_count,
        'image_size': [width, height],
    }
    return frame, metadata


def make_html(image_b64, metadata, point_ids, world_points, output_name):
    metadata_json = json.dumps(metadata, ensure_ascii=True)
    point_ids_json = json.dumps(point_ids, ensure_ascii=True)
    world_points_json = json.dumps(world_points, ensure_ascii=True)
    output_name_json = json.dumps(output_name, ensure_ascii=True)
    return f'''
<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>line point picker</title>
<style>
body {{ margin:0; background:#101214; color:#eef2f5; font-family:system-ui,-apple-system,Segoe UI,sans-serif; }}
header {{ position:sticky; top:0; z-index:2; display:flex; gap:10px; align-items:center; flex-wrap:wrap; padding:12px; background:#181c20; border-bottom:1px solid #333b44; }}
button {{ border:1px solid #333b44; border-radius:6px; background:#222831; color:#eef2f5; padding:7px 10px; font:inherit; cursor:pointer; }}
button:hover {{ border-color:#ffd23f; }}
.primary {{ background:#ffd23f; color:#111; border-color:#ffd23f; }}
.danger {{ color:#ff6b6b; }}
.status,.hint {{ color:#aab3bd; }}
main {{ padding:12px; }}
.work {{ display:flex; gap:12px; align-items:flex-start; flex-wrap:wrap; }}
.stage {{ position:relative; display:inline-block; max-width:100%; cursor:crosshair; }}
.stage img {{ max-width:100%; height:auto; display:block; }}
canvas {{ position:absolute; inset:0; width:100%; height:100%; display:block; pointer-events:none; }}
.side {{ min-width:220px; max-width:360px; }}
.pointList {{ margin:0; padding-left:22px; color:#eef2f5; }}
.pointList li {{ margin:4px 0; }}
textarea {{ width:min(920px,100%); height:190px; margin-top:12px; padding:10px; border:1px solid #333b44; border-radius:6px; background:#0b0d0f; color:#eef2f5; }}
.hint {{ margin-left:auto; }}
</style>
</head>
<body>
<header>
<button id='undoBtn'>Undo</button>
<button id='resetBtn' class='danger'>Reset</button>
<button id='exportBtn' class='primary'>Export JSON</button>
<button id='copyBtn'>Copy JSON</button>
<span id='status' class='status'></span>
<span class='hint'>Left click: add. Keys: u undo, r reset, e export.</span>
</header>
<main>
<div class='work'>
<div id='stage' class='stage'>
<img id='frameImage' src='data:image/jpeg;base64,{image_b64}' alt='extracted frame'>
<canvas id='canvas'></canvas>
</div>
<aside class='side'>
<p class='status'>Clicked points</p>
<ol id='pointList' class='pointList'></ol>
</aside>
</div>
<textarea id='jsonBox' spellcheck='false'></textarea>
</main>
<script>
const metadata = {metadata_json};
const pointIds = {point_ids_json};
const worldPoints = {world_points_json};
const outputName = {output_name_json};
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const stage = document.getElementById('stage');
const frameImage = document.getElementById('frameImage');
const statusEl = document.getElementById('status');
const jsonBox = document.getElementById('jsonBox');
const pointList = document.getElementById('pointList');
const points = [];
function nextId(index) {{
  if (index < worldPoints.length && worldPoints[index].id) return String(worldPoints[index].id);
  if (index < pointIds.length) return String(pointIds[index]);
  return 'P' + String(index).padStart(2, '0');
}}
function payload() {{
  return {{ metadata: metadata, points: points.map(function(p) {{
    const out = {{ id: p.id, image: [Number(p.image[0].toFixed(3)), Number(p.image[1].toFixed(3))] }};
    if (p.world) out.world = p.world;
    return out;
  }}), notes: {{ image_coordinate: 'pixels, origin at top-left', world_coordinate: 'meters if supplied; z=0 for floor tape points' }} }};
}}
function updateJson() {{
  jsonBox.value = JSON.stringify(payload(), null, 2);
  statusEl.textContent = 'points: ' + points.length + ' | next: ' + nextId(points.length);
  pointList.innerHTML = '';
  points.forEach(function(p) {{
    const li = document.createElement('li');
    li.textContent = p.id + '  x=' + p.image[0].toFixed(1) + ', y=' + p.image[1].toFixed(1);
    pointList.appendChild(li);
  }});
}}
function draw() {{
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.lineWidth = Math.max(2, canvas.width / 800);
  ctx.lineWidth = Math.max(2, canvas.width / 800);
  ctx.font = Math.max(16, canvas.width / 70) + 'px system-ui, sans-serif';
  points.forEach(function(p) {{
    const x = p.image[0];
    const y = p.image[1];
    ctx.fillStyle = '#ffd23f';
    ctx.strokeStyle = '#111';
    ctx.lineWidth = Math.max(3, canvas.width / 500);
    ctx.beginPath();
    ctx.arc(x, y, Math.max(8, canvas.width / 140), 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.beginPath();
    ctx.strokeStyle = '#ffd23f';
    ctx.lineWidth = Math.max(3, canvas.width / 450);
    ctx.arc(x, y, Math.max(18, canvas.width / 80), 0, Math.PI * 2);
    ctx.stroke();
    ctx.lineWidth = 6;
    ctx.strokeStyle = '#111';
    ctx.strokeText(p.id, x + 14, y - 12);
    ctx.lineWidth = 2;
    ctx.fillText(p.id, x + 14, y - 12);
  }});
  updateJson();
}}
function canvasPoint(event) {{
  const rect = frameImage.getBoundingClientRect();
  return [(event.clientX - rect.left) * canvas.width / rect.width, (event.clientY - rect.top) * canvas.height / rect.height];
}}
stage.addEventListener('click', function(event) {{
  const index = points.length;
  const point = {{ id: nextId(index), image: canvasPoint(event) }};
  if (index < worldPoints.length && worldPoints[index].world) point.world = worldPoints[index].world;
  points.push(point);
  draw();
}});
document.getElementById('undoBtn').onclick = function() {{ points.pop(); draw(); }};
document.getElementById('resetBtn').onclick = function() {{ points.length = 0; draw(); }};
document.getElementById('copyBtn').onclick = async function() {{ updateJson(); await navigator.clipboard.writeText(jsonBox.value); }};
document.getElementById('exportBtn').onclick = function() {{
  updateJson();
  const blob = new Blob([jsonBox.value + String.fromCharCode(10)], {{ type: 'application/json' }});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = outputName;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}};
document.addEventListener('keydown', function(event) {{
  if (event.key === 'u' || event.key === 'U') {{ points.pop(); draw(); }}
  if (event.key === 'r' || event.key === 'R') {{ points.length = 0; draw(); }}
  if (event.key === 'e' || event.key === 'E') document.getElementById('exportBtn').click();
}});
function fitCanvas() {{
  canvas.width = frameImage.naturalWidth || frameImage.clientWidth;
  canvas.height = frameImage.naturalHeight || frameImage.clientHeight;
  draw();
}}
if (frameImage.complete) fitCanvas();
frameImage.onload = fitCanvas;
window.addEventListener('resize', draw);
</script>
</body>
</html>
'''


def main():
    parser = argparse.ArgumentParser(description='Extract one frame from an extrinsic video and create a browser point picker.')
    parser.add_argument('--video', required=True, help='Extrinsic video path.')
    parser.add_argument('--frame', type=int, default=None, help='Frame index to extract. Defaults to the middle frame.')
    parser.add_argument('--time-sec', type=float, default=None, help='Timestamp in seconds to extract.')
    parser.add_argument('--out-dir', default='line_pick_out', help='Output directory.')
    parser.add_argument('--prefix', default='cam', help='Output filename prefix.')
    parser.add_argument('--point-ids', default='', help='Optional comma-separated labels in click order.')
    parser.add_argument('--world-points-json', default='', help='Optional JSON list of world points in click order.')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame, metadata = extract_frame(Path(args.video), frame_index=args.frame, time_sec=args.time_sec)
    frame_path = out_dir / f'{args.prefix}_frame.jpg'
    if not cv2.imwrite(str(frame_path), frame):
        raise RuntimeError(f'Cannot write frame: {frame_path}')

    image_b64 = base64.b64encode(frame_path.read_bytes()).decode('ascii')
    html_path = out_dir / f'{args.prefix}_line_picker.html'
    html_path.write_text(
        make_html(
            image_b64=image_b64,
            metadata=metadata,
            point_ids=parse_point_ids(args.point_ids),
            world_points=load_world_points(args.world_points_json),
            output_name=f'{args.prefix}_line_points.json',
        ),
        encoding='utf-8',
    )

    print('[OK] HTML point picker created')
    print(f'  Frame: {frame_path}')
    print(f'  Open:  {html_path}')
    print('  Click points in the browser, then press Export JSON or Copy JSON.')


if __name__ == '__main__':
    main()
