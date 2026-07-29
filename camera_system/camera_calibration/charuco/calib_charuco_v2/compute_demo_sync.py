import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Compute multi-camera demo sync from picked start/end clap events.')
    parser.add_argument('--events', required=True, help='JSON exported from build_demo_sync_picker.py page.')
    parser.add_argument('--reference', default='cam1demo', help='Reference camera id.')
    parser.add_argument('--out', default=None, help='Output sync JSON. Defaults next to events.')
    args = parser.parse_args()

    events_path = Path(args.events)
    with open(events_path, 'r', encoding='utf-8') as handle:
        payload = json.load(handle)

    cameras = {cam['id']: cam for cam in payload.get('cameras', [])}
    if args.reference not in cameras:
        raise KeyError(f'Reference camera {args.reference!r} not found. Available: {sorted(cameras)}')

    ref = cameras[args.reference]
    required = ['start_time_s', 'end_time_s', 'fps', 'frame_count']
    for key in required:
        if ref.get(key) is None:
            raise ValueError(f'Reference camera missing {key}')

    ref_start = float(ref['start_time_s'])
    ref_end = float(ref['end_time_s'])
    if ref_end <= ref_start:
        raise ValueError('Reference end event must be after start event')

    result = {
        'reference_camera': args.reference,
        'meaning': {
            'scale': 'reference_time = scale * local_time + offset_s',
            'offset_s': 'reference_time = scale * local_time + offset_s',
            'trim_local_start_s': 'local timestamp to start reading this video for common synced overlap',
            'trim_local_end_s': 'local timestamp to stop reading this video for common synced overlap',
        },
        'cameras': {},
    }

    ref_overlap_start = 0.0
    ref_overlap_end = float(ref['frame_count']) / float(ref['fps'])

    tmp = {}
    for cam_id, cam in sorted(cameras.items()):
        for key in required:
            if cam.get(key) is None:
                raise ValueError(f'{cam_id} missing {key}')
        start = float(cam['start_time_s'])
        end = float(cam['end_time_s'])
        if end <= start:
            raise ValueError(f'{cam_id} end event must be after start event')

        scale = (ref_end - ref_start) / (end - start)
        offset = ref_start - scale * start
        fps = float(cam['fps'])
        duration = float(cam['frame_count']) / fps

        # Convert this camera's local video interval into reference-time interval.
        local_ref_start = offset
        local_ref_end = scale * duration + offset
        ref_overlap_start = max(ref_overlap_start, local_ref_start)
        ref_overlap_end = min(ref_overlap_end, local_ref_end)

        tmp[cam_id] = {
            'video_path': cam['video_path'],
            'fps': fps,
            'frame_count': int(cam['frame_count']),
            'duration_s': duration,
            'start_frame': int(cam['start_frame']),
            'start_time_s': start,
            'end_frame': int(cam['end_frame']),
            'end_time_s': end,
            'scale': scale,
            'offset_s': offset,
            'local_start_minus_ref_start_s': start - ref_start,
            'local_end_minus_ref_end_s': end - ref_end,
        }

    if ref_overlap_end <= ref_overlap_start:
        raise RuntimeError('No common synchronized overlap; check picked events.')

    for cam_id, item in tmp.items():
        scale = item['scale']
        offset = item['offset_s']
        trim_local_start = (ref_overlap_start - offset) / scale
        trim_local_end = (ref_overlap_end - offset) / scale
        fps = item['fps']
        item['trim_local_start_s'] = trim_local_start
        item['trim_local_end_s'] = trim_local_end
        item['trim_local_start_frame'] = int(round(trim_local_start * fps))
        item['trim_local_end_frame'] = int(round(trim_local_end * fps))
        item['trim_num_frames_est'] = max(0, item['trim_local_end_frame'] - item['trim_local_start_frame'])
        result['cameras'][cam_id] = item

    result['common_reference_time_s'] = {
        'start': ref_overlap_start,
        'end': ref_overlap_end,
        'duration': ref_overlap_end - ref_overlap_start,
    }

    out_path = Path(args.out) if args.out else events_path.with_name('demo_sync_result.json')
    with open(out_path, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)

    print(f'[OK] wrote {out_path}')
    print(f"common synced duration: {result['common_reference_time_s']['duration']:.3f}s")
    for cam_id, item in result['cameras'].items():
        print(
            f"{cam_id}: scale={item['scale']:.8f} offset={item['offset_s']:+.4f}s "
            f"trim=[{item['trim_local_start_frame']}, {item['trim_local_end_frame']}) "
            f"frames={item['trim_num_frames_est']}"
        )


if __name__ == '__main__':
    main()
