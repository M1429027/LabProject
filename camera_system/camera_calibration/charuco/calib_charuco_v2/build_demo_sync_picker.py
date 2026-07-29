import argparse
import base64
import json
from pathlib import Path

import cv2


def video_info(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f'Cannot open video: {video_path}')
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return {
        'video_path': str(video_path),
        'fps': fps,
        'frame_count': frame_count,
        'duration_s': frame_count / fps if fps > 0 else None,
        'image_size': [width, height],
    }


def extract_thumbs(video_path, thumbs_dir, step):
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f'Cannot open video: {video_path}')

    info = video_info(video_path)
    frame_count = info['frame_count']
    width, height = info['image_size']
    thumb_w = 240
    thumb_h = max(1, int(round(height * thumb_w / max(width, 1))))
    frames = []

    for frame_index in range(0, frame_count, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        thumb = cv2.resize(frame, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        name = f'f{frame_index:06d}.jpg'
        cv2.imwrite(str(thumbs_dir / name), thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        frames.append({
            'frame': int(frame_index),
            'time_s': frame_index / info['fps'] if info['fps'] > 0 else None,
            'src': str((thumbs_dir / name).relative_to(thumbs_dir.parent.parent)).replace('\\', '/'),
        })
    cap.release()
    return info, frames


def make_html(payload, output_json_name):
    payload_json = json.dumps(payload, ensure_ascii=True)
    output_json = json.dumps(output_json_name, ensure_ascii=True)
    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>demo sync picker</title>
<style>
body {{ margin:0; background:#101214; color:#eef2f5; font-family:system-ui,-apple-system,Segoe UI,sans-serif; }}
header {{ position:sticky; top:0; z-index:5; padding:12px; background:#181c20; border-bottom:1px solid #333b44; display:flex; flex-wrap:wrap; gap:8px; align-items:center; }}
button {{ border:1px solid #333b44; border-radius:6px; background:#222831; color:#eef2f5; padding:7px 10px; font:inherit; cursor:pointer; }}
button:hover {{ border-color:#ffd23f; }}
.primary {{ background:#ffd23f; border-color:#ffd23f; color:#111; }}
.status {{ color:#aab3bd; }}
main {{ padding:12px; }}
.cam {{ border:1px solid #333b44; border-radius:10px; margin-bottom:14px; overflow:hidden; background:#15191d; }}
.camHead {{ padding:10px 12px; display:flex; flex-wrap:wrap; gap:10px; align-items:center; border-bottom:1px solid #333b44; }}
.camTitle {{ font-weight:600; color:#ffd23f; }}
.strip {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(180px,1fr)); gap:8px; padding:10px; }}
.thumb {{ position:relative; border:2px solid transparent; border-radius:8px; overflow:hidden; background:#0b0d0f; cursor:pointer; }}
.thumb img {{ width:100%; display:block; }}
.thumb .label {{ position:absolute; left:6px; top:6px; background:rgba(0,0,0,.7); padding:2px 5px; border-radius:4px; font-size:12px; }}
.thumb.start {{ border-color:#45f587; }}
.thumb.end {{ border-color:#ff6b6b; }}
.thumb.start::after {{ content:'START'; position:absolute; right:6px; top:6px; background:#45f587; color:#111; padding:2px 6px; border-radius:4px; font-weight:700; font-size:12px; }}
.thumb.end::before {{ content:'END'; position:absolute; right:6px; bottom:6px; background:#ff6b6b; color:#111; padding:2px 6px; border-radius:4px; font-weight:700; font-size:12px; }}
textarea {{ box-sizing:border-box; width:100%; height:220px; margin-top:12px; background:#0b0d0f; color:#eef2f5; border:1px solid #333b44; border-radius:8px; padding:10px; }}
.hint {{ max-width:900px; color:#aab3bd; line-height:1.45; }}
</style>
</head>
<body>
<header>
  <button id="modeStart" class="primary">選 START</button>
  <button id="modeEnd">選 END</button>
  <button id="exportBtn" class="primary">Export JSON</button>
  <button id="copyBtn">Copy JSON</button>
  <span id="status" class="status"></span>
</header>
<main>
  <p class="hint">操作：先按「選 START」，在每台相機點開頭舉手拍手那一格；再按「選 END」，點結尾舉手拍手那一格。縮圖每隔固定幀數抽一張，先求同步用，通常夠估 offset；如果要更精細，可以之後把 step 調小重產。</p>
  <div id="cams"></div>
  <textarea id="jsonBox" spellcheck="false"></textarea>
</main>
<script>
const payload = {payload_json};
const outputJson = {output_json};
let mode = 'start';
const picks = {{}};
for (const cam of payload.cameras) picks[cam.id] = {{ start_frame: null, end_frame: null }};
const camsEl = document.getElementById('cams');
const statusEl = document.getElementById('status');
const jsonBox = document.getElementById('jsonBox');
function setMode(next) {{
  mode = next;
  document.getElementById('modeStart').className = next === 'start' ? 'primary' : '';
  document.getElementById('modeEnd').className = next === 'end' ? 'primary' : '';
  update();
}}
function resultPayload() {{
  return {{
    metadata: payload.metadata,
    cameras: payload.cameras.map(cam => {{
      const p = picks[cam.id];
      return {{
        id: cam.id,
        video_path: cam.info.video_path,
        fps: cam.info.fps,
        frame_count: cam.info.frame_count,
        duration_s: cam.info.duration_s,
        start_frame: p.start_frame,
        start_time_s: p.start_frame == null ? null : p.start_frame / cam.info.fps,
        end_frame: p.end_frame,
        end_time_s: p.end_frame == null ? null : p.end_frame / cam.info.fps
      }};
    }})
  }};
}}
function update() {{
  for (const cam of payload.cameras) {{
    for (const fr of cam.frames) {{
      const el = document.getElementById(`${{cam.id}}-${{fr.frame}}`);
      if (!el) continue;
      el.classList.toggle('start', picks[cam.id].start_frame === fr.frame);
      el.classList.toggle('end', picks[cam.id].end_frame === fr.frame);
    }}
  }}
  const done = Object.values(picks).filter(p => p.start_frame != null && p.end_frame != null).length;
  statusEl.textContent = `mode: ${{mode.toUpperCase()}} | completed cameras: ${{done}}/${{payload.cameras.length}}`;
  jsonBox.value = JSON.stringify(resultPayload(), null, 2);
}}
function build() {{
  for (const cam of payload.cameras) {{
    const section = document.createElement('section');
    section.className = 'cam';
    const head = document.createElement('div');
    head.className = 'camHead';
    head.innerHTML = `<span class="camTitle">${{cam.id}}</span><span class="status">${{cam.info.image_size[0]}}x${{cam.info.image_size[1]}}, ${{cam.info.fps.toFixed(3)}} fps, ${{cam.info.frame_count}} frames, ${{cam.info.duration_s.toFixed(2)}}s</span>`;
    const strip = document.createElement('div');
    strip.className = 'strip';
    for (const fr of cam.frames) {{
      const div = document.createElement('div');
      div.className = 'thumb';
      div.id = `${{cam.id}}-${{fr.frame}}`;
      div.innerHTML = `<img src="${{fr.src}}" loading="lazy"><span class="label">f${{fr.frame}} / ${{fr.time_s.toFixed(2)}}s</span>`;
      div.onclick = () => {{
        if (mode === 'start') picks[cam.id].start_frame = fr.frame;
        else picks[cam.id].end_frame = fr.frame;
        update();
      }};
      strip.appendChild(div);
    }}
    section.appendChild(head);
    section.appendChild(strip);
    camsEl.appendChild(section);
  }}
  update();
}}
document.getElementById('modeStart').onclick = () => setMode('start');
document.getElementById('modeEnd').onclick = () => setMode('end');
document.getElementById('copyBtn').onclick = async () => {{ update(); await navigator.clipboard.writeText(jsonBox.value); }};
document.getElementById('exportBtn').onclick = () => {{
  update();
  const blob = new Blob([jsonBox.value + '\\n'], {{type:'application/json'}});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = outputJson;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}};
build();
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description='Build a browser filmstrip picker for multi-camera demo sync clap events.')
    parser.add_argument('--video-dir', default='camtest/line')
    parser.add_argument('--glob', default='cam*demo.mkv')
    parser.add_argument('--out-dir', default='camtest/line/demo_sync_picker')
    parser.add_argument('--step', type=int, default=5, help='Extract one thumbnail every N frames.')
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    out_dir = Path(args.out_dir)
    thumbs_root = out_dir / 'thumbs'
    out_dir.mkdir(parents=True, exist_ok=True)
    videos = sorted(video_dir.glob(args.glob))
    if not videos:
        raise FileNotFoundError(f'No videos matched {video_dir / args.glob}')

    cameras = []
    for video in videos:
        cam_id = video.stem
        info, frames = extract_thumbs(video, thumbs_root / cam_id, max(1, args.step))
        cameras.append({'id': cam_id, 'info': info, 'frames': frames})
        print(f'[INFO] {cam_id}: {len(frames)} thumbnails')

    payload = {
        'metadata': {
            'thumbnail_step_frames': max(1, args.step),
            'instruction': 'Pick start/end overhead clap frames for each camera.',
        },
        'cameras': cameras,
    }
    html = make_html(payload, 'demo_sync_events.json')
    html_path = out_dir / 'index.html'
    html_path.write_text(html, encoding='utf-8')
    print(f'[OK] Open: {html_path.resolve()}')


if __name__ == '__main__':
    main()
