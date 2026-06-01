import argparse
import json
import random
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np
from tqdm import tqdm


TASK_NAME = "cover_blocks"

GLOBAL_TASK = (
    "On the table, red, green, and blue blocks are arranged randomly along with three lids. "
    "From the current viewpoint, cover the blocks from right to left using the lids, "
    "and then uncover them again in the sequence red, green, and blue."
)


def natural_episode_id(path: Path) -> int:
    nums = re.findall(r"\d+", path.stem)
    if not nums:
        raise ValueError(f"Cannot parse episode id from {path.name}")
    return int(nums[-1])


def find_hdf5_files(task_root: Path) -> List[Path]:
    data_dir = task_root / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Cannot find RMBench hdf5 dir: {data_dir}")

    h5_files = list(data_dir.glob("episode*.hdf5")) + list(data_dir.glob("episode*.h5"))
    h5_files = sorted(set(h5_files), key=natural_episode_id)

    if not h5_files:
        raise FileNotFoundError(f"No episode*.hdf5 found under {data_dir}")

    return h5_files


def find_source_video(task_root: Path, episode_id: int) -> Optional[Path]:
    """
    优先找 RMBench 采集时已经生成的 head-camera 原视频。
    这样可以避免从 HDF5 重新写视频时出现 RGB/BGR 通道不一致。
    """
    candidate_names = [
        f"episode{episode_id}.mp4",
        f"episode_{episode_id}.mp4",
        f"episode{episode_id:04d}.mp4",
        f"episode_{episode_id:04d}.mp4",
        f"{episode_id}.mp4",
    ]

    candidate_dirs = [
        task_root / "video",
        task_root / "videos",
        task_root / "video" / "head_camera",
        task_root / "videos" / "head_camera",
    ]

    for d in candidate_dirs:
        for name in candidate_names:
            p = d / name
            if p.exists():
                return p

    # fallback: 在 video/videos 目录递归找所有 mp4，并按文件名里的数字匹配 episode id
    for d in [task_root / "video", task_root / "videos"]:
        if not d.exists():
            continue
        for p in sorted(d.rglob("*.mp4")):
            try:
                if natural_episode_id(p) == episode_id:
                    return p
            except ValueError:
                continue

    return None


def get_video_frame_count(video_path: Path) -> Optional[int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count if count > 0 else None


def map_hdf5_range_to_source_video(
    start_frame: int,
    end_frame: int,
    hdf5_episode_len: int,
    source_video_frame_count: Optional[int],
) -> Tuple[int, int]:
    """
    如果源视频帧数和 HDF5 帧数不同，按比例映射。
    例如原视频只有 120 帧，HDF5 有 1000 帧时，子任务边界需要同步缩放。
    """
    if source_video_frame_count is None or source_video_frame_count <= 0:
        return start_frame, end_frame

    if hdf5_episode_len <= 0:
        return start_frame, end_frame

    if abs(source_video_frame_count - hdf5_episode_len) <= 2:
        return start_frame, end_frame

    mapped_start = round(start_frame / hdf5_episode_len * source_video_frame_count)
    mapped_end = round(end_frame / hdf5_episode_len * source_video_frame_count)

    mapped_start = max(0, min(mapped_start, source_video_frame_count - 1))
    mapped_end = max(mapped_start + 1, min(mapped_end, source_video_frame_count))

    return mapped_start, mapped_end


def run_ffmpeg(cmd: List[str]) -> None:
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Cannot find ffmpeg. Please install ffmpeg first, or use HDF5 fallback with --video_codec mp4v."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "ffmpeg command failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stderr:\n{exc.stderr}"
        ) from exc


def ffmpeg_encode_args(video_codec: str, h264_crf: int, h264_preset: str) -> List[str]:
    if video_codec == "h264":
        return [
            "-c:v", "libx264",
            "-preset", str(h264_preset),
            "-crf", str(h264_crf),
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
        ]

    if video_codec == "mp4v":
        return [
            "-c:v", "mpeg4",
            "-q:v", "5",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
        ]

    raise ValueError(f"Unsupported video_codec={video_codec}. Use h264 or mp4v.")


def write_full_from_source_video(
    src_video: Path,
    out_path: Path,
    overwrite: bool,
    video_codec: str,
    ffmpeg_path: str,
    h264_crf: int,
    h264_preset: str,
) -> int:
    """
    从 RMBench 原视频转码到输出目录。
    这里不经过 OpenCV，不会发生 RGB/BGR 通道交换。
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not overwrite:
        return -1

    tmp_path = out_path.with_name(out_path.stem + "__tmp.mp4")
    if tmp_path.exists():
        tmp_path.unlink()

    vf = "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p"
    cmd = [
        ffmpeg_path,
        "-y",
        "-i", str(src_video),
        "-an",
        "-vf", vf,
        *ffmpeg_encode_args(video_codec, h264_crf, h264_preset),
        str(tmp_path),
    ]

    run_ffmpeg(cmd)
    tmp_path.replace(out_path)

    return get_video_frame_count(out_path) or -1


def write_clip_from_source_video(
    src_video: Path,
    out_path: Path,
    start_frame: int,
    end_frame: int,
    overwrite: bool,
    video_codec: str,
    ffmpeg_path: str,
    h264_crf: int,
    h264_preset: str,
) -> int:
    """
    用 ffmpeg 直接从 RMBench 原视频按帧切 clip。
    关键点：不从 HDF5 重新解码/编码，避免颜色通道问题。
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not overwrite:
        return -1

    start_frame = int(start_frame)
    end_frame = int(end_frame)
    if start_frame < 0 or end_frame <= start_frame:
        raise ValueError(f"Invalid source video clip range: {start_frame=} {end_frame=}")

    tmp_path = out_path.with_name(out_path.stem + "__tmp.mp4")
    if tmp_path.exists():
        tmp_path.unlink()

    # trim=start_frame/end_frame 按解码帧号裁剪；setpts 重置时间戳。
    vf = (
        f"trim=start_frame={start_frame}:end_frame={end_frame},"
        "setpts=PTS-STARTPTS,"
        "scale=trunc(iw/2)*2:trunc(ih/2)*2,"
        "format=yuv420p"
    )

    cmd = [
        ffmpeg_path,
        "-y",
        "-i", str(src_video),
        "-an",
        "-vf", vf,
        *ffmpeg_encode_args(video_codec, h264_crf, h264_preset),
        str(tmp_path),
    ]

    run_ffmpeg(cmd)
    tmp_path.replace(out_path)

    return get_video_frame_count(out_path) or (end_frame - start_frame)


def decode_hdf5_frame_to_bgr(frame_obj, hdf5_array_color_order: str = "rgb") -> np.ndarray:
    """
    HDF5 fallback only.

    返回 OpenCV VideoWriter 需要的 BGR 图像。
    - 如果 frame 是压缩 bytes，cv2.imdecode 默认返回 BGR。
    - 如果 frame 是 HWC ndarray，则用 hdf5_array_color_order 指定它原本是 RGB 还是 BGR。
    """
    if isinstance(frame_obj, np.ndarray):
        arr = frame_obj

        if arr.ndim == 1:
            img_bgr = cv2.imdecode(np.frombuffer(arr.tobytes(), np.uint8), cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise RuntimeError("cv2.imdecode failed for ndarray bytes frame")
            return img_bgr

        if arr.ndim == 3 and arr.shape[-1] == 3:
            arr = arr.astype(np.uint8)

            if hdf5_array_color_order == "rgb":
                return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            if hdf5_array_color_order == "bgr":
                return arr

            raise ValueError(f"Unsupported hdf5_array_color_order={hdf5_array_color_order}")

        raise RuntimeError(f"Unsupported ndarray frame shape: {arr.shape}")

    raw = bytes(frame_obj)
    img_bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError("cv2.imdecode failed for bytes frame")
    return img_bgr


def get_episode_length(h5: h5py.File) -> int:
    if "joint_action" in h5 and "left_arm" in h5["joint_action"]:
        return len(h5["joint_action"]["left_arm"])
    return len(h5["observation"]["head_camera"]["rgb"])


def transcode_to_h264(
    src_path: Path,
    dst_path: Path,
    ffmpeg_path: str,
    crf: int,
    preset: str,
) -> None:
    tmp_h264_path = dst_path.with_name(dst_path.stem + "__h264_tmp.mp4")
    if tmp_h264_path.exists():
        tmp_h264_path.unlink()

    cmd = [
        ffmpeg_path,
        "-y",
        "-i", str(src_path),
        "-an",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
        "-c:v", "libx264",
        "-preset", str(preset),
        "-crf", str(crf),
        "-movflags", "+faststart",
        str(tmp_h264_path),
    ]

    run_ffmpeg(cmd)
    tmp_h264_path.replace(dst_path)


def write_video_from_hdf5(
    hdf5_path: Path,
    out_path: Path,
    start_frame: int,
    end_frame: int,
    fps: int,
    overwrite: bool,
    video_codec: str,
    ffmpeg_path: str,
    h264_crf: int,
    h264_preset: str,
    hdf5_array_color_order: str,
) -> int:
    """
    fallback：从 HDF5 重建视频。
    优先推荐使用 source video；只有找不到 RMBench 原视频时才走这里。
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not overwrite:
        return -1

    write_path = out_path
    tmp_mp4v_path = None

    if video_codec == "h264":
        tmp_mp4v_path = out_path.with_name(out_path.stem + "__opencv_tmp.mp4")
        if tmp_mp4v_path.exists():
            tmp_mp4v_path.unlink()
        write_path = tmp_mp4v_path

    with h5py.File(hdf5_path, "r") as h5:
        rgb_ds = h5["observation"]["head_camera"]["rgb"]
        episode_len = min(get_episode_length(h5), len(rgb_ds))

        start_frame = max(0, int(start_frame))
        end_frame = min(int(end_frame), episode_len)

        if start_frame >= end_frame:
            raise ValueError(
                f"Invalid clip range for {hdf5_path.name}: "
                f"start={start_frame}, end={end_frame}, episode_len={episode_len}"
            )

        first = decode_hdf5_frame_to_bgr(
            rgb_ds[start_frame],
            hdf5_array_color_order=hdf5_array_color_order,
        )
        height, width = first.shape[:2]

        writer = cv2.VideoWriter(
            str(write_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        if not writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter for {write_path}")

        writer.write(first)
        written = 1

        for idx in range(start_frame + 1, end_frame):
            frame = decode_hdf5_frame_to_bgr(
                rgb_ds[idx],
                hdf5_array_color_order=hdf5_array_color_order,
            )
            writer.write(frame)
            written += 1

        writer.release()

    if video_codec == "h264":
        transcode_to_h264(
            src_path=write_path,
            dst_path=out_path,
            ffmpeg_path=ffmpeg_path,
            crf=h264_crf,
            preset=h264_preset,
        )
        if tmp_mp4v_path is not None and tmp_mp4v_path.exists():
            tmp_mp4v_path.unlink()

    return written


def load_language_annotation(task_root: Path) -> Dict:
    anno_path = task_root / "language_annotation.json"
    if not anno_path.exists():
        print(f"[WARN] language_annotation.json not found: {anno_path}")
        return {}

    with open(anno_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_episode_annotations(language_annotations: Dict, episode_id: int) -> List[Tuple[str, int]]:
    keys = [
        f"episode_{episode_id}",
        f"episode{episode_id}",
        str(episode_id),
    ]

    value = None
    for key in keys:
        if key in language_annotations:
            value = language_annotations[key]
            break

    if value is None:
        return []

    result = []
    for item in value:
        if isinstance(item, list) and len(item) >= 2:
            subtask_text = str(item[0]).strip()
            duration = int(item[1])
            if subtask_text and duration > 0:
                result.append((subtask_text, duration))
        elif isinstance(item, dict):
            subtask_text = str(
                item.get("subtask")
                or item.get("instruction")
                or item.get("text")
                or ""
            ).strip()
            duration = int(item.get("duration", 0))
            if subtask_text and duration > 0:
                result.append((subtask_text, duration))

    return result


def build_boundaries(
    annotations: List[Tuple[str, int]],
    episode_len: int,
) -> List[Tuple[int, int, str]]:
    boundaries = []
    cursor = 0

    for subtask_text, duration in annotations:
        start = cursor
        end = min(cursor + duration, episode_len)

        if start < end:
            boundaries.append((start, end, subtask_text))

        cursor = end

        if cursor >= episode_len:
            break

    return boundaries


def make_conversation(question: str, answer: str) -> List[Dict[str, str]]:
    return [
        {"from": "human", "value": "<video>\n" + question.strip()},
        {"from": "gpt", "value": answer.strip()},
    ]


def add_full_episode_sample(records: List[Dict], episode_id: int, rel_video_path: str):
    records.append(
        {
            "_episode_id": episode_id,
            "id": f"cover_blocks_ep{episode_id:04d}_full_task",
            "video": [rel_video_path],
            "conversations": make_conversation(
                question=(
                    "What is the global task being performed by the robot in this video? "
                    "Answer with a concise task description."
                ),
                answer=GLOBAL_TASK,
            ),
        }
    )

    records.append(
        {
            "_episode_id": episode_id,
            "id": f"cover_blocks_ep{episode_id:04d}_summary",
            "video": [rel_video_path],
            "conversations": make_conversation(
                question=(
                    "Describe the robot's behavior in this episode. "
                    "Focus on the order of covering and uncovering the blocks."
                ),
                answer=(
                    "The robot covers the visible blocks from right to left using the lids, "
                    "then uncovers the blocks again in the order red, green, and blue."
                ),
            ),
        }
    )


def add_subtask_clip_sample(
    records: List[Dict],
    episode_id: int,
    subtask_idx: int,
    rel_video_path: str,
    subtask_text: str,
):
    records.append(
        {
            "_episode_id": episode_id,
            "id": f"cover_blocks_ep{episode_id:04d}_subtask{subtask_idx:02d}",
            "video": [rel_video_path],
            "conversations": make_conversation(
                question=(
                    "What subtask is the robot executing in this short clip? "
                    "Answer with the current subtask only."
                ),
                answer=subtask_text,
            ),
        }
    )


def add_next_subtask_sample(
    records: List[Dict],
    episode_id: int,
    subtask_idx: int,
    rel_video_path: str,
    next_subtask_text: str,
):
    records.append(
        {
            "_episode_id": episode_id,
            "id": f"cover_blocks_ep{episode_id:04d}_next{subtask_idx:02d}",
            "video": [rel_video_path],
            "conversations": make_conversation(
                question=(
                    "Global task: "
                    + GLOBAL_TASK
                    + "\nBased only on the video so far, what should the robot do next? "
                    "Answer with the next subtask."
                ),
                answer=next_subtask_text,
            ),
        }
    )


def split_and_write_jsonl(
    records: List[Dict],
    out_root: Path,
    train_name: str,
    val_name: str,
    val_ratio: float,
    seed: int,
):
    episode_ids = sorted({r["_episode_id"] for r in records})
    rng = random.Random(seed)
    rng.shuffle(episode_ids)

    val_count = int(round(len(episode_ids) * val_ratio))
    if val_ratio > 0 and val_count == 0 and len(episode_ids) > 1:
        val_count = 1

    val_episode_ids = set(episode_ids[:val_count])

    train_records = []
    val_records = []

    for r in records:
        clean_r = {k: v for k, v in r.items() if not k.startswith("_")}
        if r["_episode_id"] in val_episode_ids:
            val_records.append(clean_r)
        else:
            train_records.append(clean_r)

    train_path = out_root / train_name
    val_path = out_root / val_name

    with open(train_path, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for r in val_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nSaved train jsonl: {train_path}  ({len(train_records)} samples)")
    print(f"Saved val jsonl:   {val_path}  ({len(val_records)} samples)")
    print(f"\nUse for VideoLLaMA3:")
    print(f"  --data_folder {out_root}")
    print(f"  --data_path {train_path}")


def write_output_video(
    hdf5_path: Path,
    source_video_path: Optional[Path],
    out_path: Path,
    start_frame: int,
    end_frame: int,
    hdf5_episode_len: int,
    args,
) -> int:
    """
    优先从 RMBench 原视频裁剪/转码；没有原视频时再 fallback 到 HDF5。
    """
    if source_video_path is not None and args.prefer_source_video:
        source_frame_count = get_video_frame_count(source_video_path)
        src_start, src_end = map_hdf5_range_to_source_video(
            start_frame=start_frame,
            end_frame=end_frame,
            hdf5_episode_len=hdf5_episode_len,
            source_video_frame_count=source_frame_count,
        )

        if start_frame == 0 and end_frame >= hdf5_episode_len:
            return write_full_from_source_video(
                src_video=source_video_path,
                out_path=out_path,
                overwrite=args.overwrite,
                video_codec=args.video_codec,
                ffmpeg_path=args.ffmpeg_path,
                h264_crf=args.h264_crf,
                h264_preset=args.h264_preset,
            )

        return write_clip_from_source_video(
            src_video=source_video_path,
            out_path=out_path,
            start_frame=src_start,
            end_frame=src_end,
            overwrite=args.overwrite,
            video_codec=args.video_codec,
            ffmpeg_path=args.ffmpeg_path,
            h264_crf=args.h264_crf,
            h264_preset=args.h264_preset,
        )

    return write_video_from_hdf5(
        hdf5_path=hdf5_path,
        out_path=out_path,
        start_frame=start_frame,
        end_frame=end_frame,
        fps=args.fps,
        overwrite=args.overwrite,
        video_codec=args.video_codec,
        ffmpeg_path=args.ffmpeg_path,
        h264_crf=args.h264_crf,
        h264_preset=args.h264_preset,
        hdf5_array_color_order=args.hdf5_array_color_order,
    )


def convert_cover_blocks(args):
    rmbench_root = Path(args.rmbench_root).expanduser().resolve()
    task_root = rmbench_root / "data" / TASK_NAME / args.task_config

    if not task_root.exists():
        raise FileNotFoundError(
            f"Cannot find task root: {task_root}\n"
            f"Expected structure: RMBench/data/cover_blocks/{args.task_config}"
        )

    out_root = Path(args.out_root).expanduser().resolve()
    video_root = out_root / "videos" / TASK_NAME
    out_root.mkdir(parents=True, exist_ok=True)

    h5_files = find_hdf5_files(task_root)
    if args.max_episodes is not None:
        h5_files = h5_files[: args.max_episodes]

    language_annotations = load_language_annotation(task_root)

    print(f"RMBench root:        {rmbench_root}")
    print(f"Task root:           {task_root}")
    print(f"Output root:         {out_root}")
    print(f"Episodes:            {len(h5_files)}")
    print(f"Modes:               {args.modes}")
    print(f"Video codec:         {args.video_codec}")
    print(f"Prefer source video: {args.prefer_source_video}")

    records: List[Dict] = []

    missing_source_count = 0

    for hdf5_path in tqdm(h5_files, desc="Converting cover_blocks"):
        episode_id = natural_episode_id(hdf5_path)

        with h5py.File(hdf5_path, "r") as h5:
            episode_len = get_episode_length(h5)

        source_video_path = find_source_video(task_root, episode_id)
        if source_video_path is None:
            missing_source_count += 1

        annotations = get_episode_annotations(language_annotations, episode_id)
        boundaries = build_boundaries(annotations, episode_len)

        if "full" in args.modes:
            full_video_path = video_root / "full" / f"episode{episode_id}.mp4"
            write_output_video(
                hdf5_path=hdf5_path,
                source_video_path=source_video_path,
                out_path=full_video_path,
                start_frame=0,
                end_frame=episode_len,
                hdf5_episode_len=episode_len,
                args=args,
            )

            rel_full = str(full_video_path.relative_to(out_root))
            add_full_episode_sample(records, episode_id, rel_full)

        if not boundaries:
            print(f"[WARN] No subtask annotation for episode {episode_id}; skip subtask/next samples.")
            continue

        if "subtask" in args.modes:
            for subtask_idx, (start, end, subtask_text) in enumerate(boundaries):
                clip_path = (
                    video_root
                    / "subtask"
                    / f"episode{episode_id}_subtask{subtask_idx:02d}_{start}_{end}.mp4"
                )

                write_output_video(
                    hdf5_path=hdf5_path,
                    source_video_path=source_video_path,
                    out_path=clip_path,
                    start_frame=start,
                    end_frame=end,
                    hdf5_episode_len=episode_len,
                    args=args,
                )

                rel_clip = str(clip_path.relative_to(out_root))
                add_subtask_clip_sample(
                    records=records,
                    episode_id=episode_id,
                    subtask_idx=subtask_idx,
                    rel_video_path=rel_clip,
                    subtask_text=subtask_text,
                )

        if "next" in args.modes:
            for subtask_idx, (start, end, subtask_text) in enumerate(boundaries):
                if start <= 0:
                    continue

                prefix_path = (
                    video_root
                    / "prefix"
                    / f"episode{episode_id}_before_subtask{subtask_idx:02d}_0_{start}.mp4"
                )

                write_output_video(
                    hdf5_path=hdf5_path,
                    source_video_path=source_video_path,
                    out_path=prefix_path,
                    start_frame=0,
                    end_frame=start,
                    hdf5_episode_len=episode_len,
                    args=args,
                )

                rel_prefix = str(prefix_path.relative_to(out_root))
                add_next_subtask_sample(
                    records=records,
                    episode_id=episode_id,
                    subtask_idx=subtask_idx,
                    rel_video_path=rel_prefix,
                    next_subtask_text=subtask_text,
                )

    if missing_source_count > 0:
        print(
            f"\n[WARN] {missing_source_count}/{len(h5_files)} episodes did not have source videos; "
            f"those episodes used HDF5 fallback."
        )

    split_and_write_jsonl(
        records=records,
        out_root=out_root,
        train_name="cover_blocks_videollama3_train.jsonl",
        val_name="cover_blocks_videollama3_val.jsonl",
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert RMBench cover_blocks M(n) data to VideoLLaMA3 JSONL format. "
            "This version prefers RMBench source videos to avoid RGB/BGR color swaps."
        )
    )

    parser.add_argument(
        "--rmbench_root",
        type=str,
        required=True,
        help="Path to RMBench repo root, e.g. /path/to/RMBench",
    )
    parser.add_argument(
        "--task_config",
        type=str,
        default="demo_clean",
        help="RMBench task config. For RMBench usually demo_clean.",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default="./datasets/rmbench_cover_blocks_vl3",
        help="Output dataset root for VideoLLaMA3.",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Limit number of episodes for debugging.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Fallback output fps when rebuilding from HDF5.",
    )
    parser.add_argument(
        "--video_codec",
        type=str,
        default="h264",
        choices=["h264", "mp4v"],
        help="Output video codec. h264 is VSCode/browser friendly.",
    )
    parser.add_argument(
        "--ffmpeg_path",
        type=str,
        default="ffmpeg",
        help="Path to ffmpeg executable.",
    )
    parser.add_argument(
        "--h264_crf",
        type=int,
        default=23,
        help="H.264 CRF quality. Lower is better/larger. Typical range: 18-28.",
    )
    parser.add_argument(
        "--h264_preset",
        type=str,
        default="medium",
        help="x264 preset, e.g. ultrafast, veryfast, medium, slow.",
    )
    parser.add_argument(
        "--prefer_source_video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Prefer RMBench original videos under task_root/video when available. "
            "Disable with --no-prefer_source_video to rebuild videos from HDF5."
        ),
    )
    parser.add_argument(
        "--hdf5_array_color_order",
        type=str,
        default="rgb",
        choices=["rgb", "bgr"],
        help=(
            "Only used by HDF5 fallback for uncompressed HWC arrays. "
            "Use rgb if HDF5 arrays are RGB; use bgr if they are already BGR."
        ),
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["full", "subtask", "next"],
        choices=["full", "subtask", "next"],
        help=(
            "full: whole episode QA; "
            "subtask: subtask clip QA; "
            "next: prefix-video next-subtask planning QA."
        ),
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Episode-level validation split ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for episode-level train/val split.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite generated videos if they already exist.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    convert_cover_blocks(parse_args())
