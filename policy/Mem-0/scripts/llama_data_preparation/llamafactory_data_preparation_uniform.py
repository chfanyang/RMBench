import argparse
import cv2
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

# python scripts/llama_data_preparation/llamafactory_data_preparation_uniform.py --lerobot_dataset_path /mnt/hwdata/cfy/RMBench/policy/Mem-0/lerobot_datasets/cover_blocks_eval --episode_start_id 0 --episode_end_id 100

workspace = os.path.dirname(os.path.abspath(__file__))
Mem0_workspace = os.path.join(workspace, "..", "..")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare LLaMA-Factory format data from LeRobot dataset (uniform sampling of historical frames).")
    parser.add_argument(
        "--lerobot_dataset_path",
        type=str,
        default=os.path.join(Mem0_workspace, "lerobot_datasets", "battery_try"),
        help="Path to the LeRobot dataset.",
    )
    parser.add_argument("--episode_start_id", type=int, default=0, help="Start episode id (inclusive).")
    parser.add_argument("--episode_end_id", type=int, default=50, help="End episode id (exclusive).")
    parser.add_argument("--sampling_interval", type=int, default=36, help="Sample one frame every N frames for historical observations.")
    return parser.parse_args()


# parameter need to be set (overridden by CLI args when run from automation script)
_args = parse_args()
lerobot_dataset_path = _args.lerobot_dataset_path
episode_start_id = _args.episode_start_id
episode_end_id = _args.episode_end_id
sampling_interval = _args.sampling_interval


def save_image(img, path):
    # save image
    img = img.numpy()
    # Convert image format (C, H, W) -> (H, W, C)
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))

    # Convert to uint8 and create PIL Image
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

    image = Image.fromarray(img, mode='RGB')

    image.save(path)


system_prompt = (
    "You are a robotic assistant specialized in subtask planning. "
    "I will provide you with:\n"
    "1. global_task: A global task instruction.\n"
    "2. historical_observations: A sequence of images uniformly sampled every "
    f"{sampling_interval} frames from the episode history, showing the visual "
    "observations at different time steps. The images are presented in temporal "
    "order, where Step 0 corresponds to the initial observation (frame 0), "
    "Step 1 corresponds to the observation at frame {sampling_interval}, "
    "Step 2 corresponds to the observation at frame {sampling_interval}*2, and so on. "
    "Each step captures the robot's visual state at that moment.\n\n"
    "Format:\n"
    "<global_task>: {global task instruction}.\n"
    "<historical_observations>:\n"
    "Step 0: <image>.\n"
    "Step 1: <image>.\n"
    "...\n\n"
    "IMPORTANT: The steps indicate the temporal order of observations. "
    "Higher step indices represent more recent observations. "
    "At the very beginning of the task (frame 0), the historical_observations "
    "only contain Step 0 (the initial observation).\n\n"
    "Based on all the provided information, output the next subtask to execute "
    "in the format: 'next_subtask: {subtask name}.'.\n"
)

llamafactory_dataset_name = lerobot_dataset_path.split("/")[-1]

# set initial dataset list
high_level_finetune_data = []

# create folder
finetune_dataset_path = f"{Mem0_workspace}/llamafactory_data/{llamafactory_dataset_name}_uniform"
os.makedirs(finetune_dataset_path, exist_ok=True)

total_episodes = episode_end_id - episode_start_id
print(f"\n[LlamaFactory Data Preparation (Uniform)] Total episodes to process: {total_episodes} (episode_id {episode_start_id} ~ {episode_end_id - 1})", flush=True)
print(f"[LlamaFactory Data Preparation (Uniform)] Sampling interval: every {sampling_interval} frames", flush=True)

for idx, episode_id in enumerate(range(episode_start_id, episode_end_id), start=1):
    print(f"[ {idx}/{total_episodes} ] Processing episode {episode_id} ...", flush=True)
    # create episode folder
    episode_folder_path = f"{finetune_dataset_path}/{llamafactory_dataset_name}_uniform_images/episode_{episode_id}"
    os.makedirs(episode_folder_path, exist_ok=True)
    # images list (accumulated historical frames for this episode)
    images_list = []

    dataset = LeRobotDataset(lerobot_dataset_path, video_backend="pyav", episodes=[episode_id])

    episode_length = len(dataset)

    for frame_id in tqdm(range(episode_length), desc=f"Episode {idx}/{total_episodes} (id={episode_id}) frames"):

        # Sample frame every `sampling_interval` frames
        if frame_id % sampling_interval == 0:
            # save current frame image
            image = dataset[frame_id]['observation.image.head_camera']
            image_path = f"{episode_folder_path}/frame_{frame_id:06d}.png"
            save_image(image, image_path)
            images_list.append(
                f"{llamafactory_dataset_name}_uniform_images/episode_{episode_id}/frame_{frame_id:06d}.png"
            )

            # build training sample
            frame_information = {"messages": [], "images": []}

            # system prompt message
            system_prompt_msg = {"role": "system", "content": system_prompt}
            frame_information["messages"].append(system_prompt_msg)

            # user prompt message
            global_task_txt = "<global_task>: " + dataset[frame_id]['global_task'] + "\n"
            historical_obs = "<historical_observations>:\n"
            for i, img_path in enumerate(images_list):
                historical_obs += f"Step {i}: <image>.\n"
            combined_message = global_task_txt + historical_obs

            user_prompt_msg = {"role": "user", "content": combined_message}
            frame_information["messages"].append(user_prompt_msg)

            # assistant prompt message
            subtask_name = dataset[frame_id]['subtask']
            assistant_prompt_msg = {"role": "assistant", "content": f"next_subtask: {subtask_name}"}
            frame_information["messages"].append(assistant_prompt_msg)

            # add images
            frame_information["images"] = images_list.copy()

            high_level_finetune_data.append(frame_information.copy())

# save high_level_finetune_data to json file
output_path = f"{finetune_dataset_path}/{llamafactory_dataset_name}_uniform_high_level_finetune_data.json"
with open(output_path, "w") as f:
    json.dump(high_level_finetune_data, f, indent=2, ensure_ascii=False)

print(f"\n[LlamaFactory Data Preparation (Uniform)] Done. Processed {total_episodes} episodes (episode_id {episode_start_id} ~ {episode_end_id - 1}).", flush=True)
print(f"[LlamaFactory Data Preparation (Uniform)] Total training samples: {len(high_level_finetune_data)}", flush=True)
print(f"[LlamaFactory Data Preparation (Uniform)] Output saved to: {output_path}", flush=True)
