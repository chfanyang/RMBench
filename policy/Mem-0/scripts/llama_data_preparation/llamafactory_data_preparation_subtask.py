import argparse
import os
import json
import numpy as np
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

# Usage:
# python scripts/llama_data_preparation/llamafactory_data_preparation_subtask.py --lerobot_dataset_path /mnt/hwdata/cfy/RMBench/policy/Mem-0/lerobot_datasets/cover_blocks --episode_start_id 0 --episode_end_id 1

workspace = os.path.dirname(os.path.abspath(__file__))
Mem0_workspace = os.path.join(workspace, "..", "..")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare LLaMA-Factory format data: given historical frames "
                    "within the current subtask plus already-finished subtasks, "
                    "predict current subtask name and whether it has finished."
    )
    parser.add_argument(
        "--lerobot_dataset_path",
        type=str,
        default=os.path.join(Mem0_workspace, "lerobot_datasets", "battery_try"),
        help="Path to the LeRobot dataset.",
    )
    parser.add_argument("--episode_start_id", type=int, default=0, help="Start episode id (inclusive).")
    parser.add_argument("--episode_end_id", type=int, default=50, help="End episode id (exclusive).")
    parser.add_argument(
        "--sampling_interval", type=int, default=36,
        help="Sample one frame every N frames within the current subtask for "
             "historical observations.",
    )
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
    "You are a robotic assistant specialized in subtask recognition. "
    "I will provide you with:\n"
    "1. global_task: The overall goal instruction.\n"
    "2. finished_subtasks: Subtasks already completed, in temporal order "
    "(index 0 = first completed, higher = more recent). Each comes with an "
    "image showing the visual state at the end of that subtask. "
    "If none yet, this is null.\n"
    "3. historical_observations: Images uniformly sampled every "
    f"{sampling_interval} frames from the current subtask, in temporal order "
    "(Step 0 = earliest observation, higher steps = more recent).\n\n"
    "Input format:\n"
    "<global_task>: {global task instruction}.\n"
    "<finished_subtasks>:\n"
    "0: {subtask instruction}, the corresponding image is: <image>.\n"
    "...\n"
    "<historical_observations>:\n"
    "Step 0: <image>.\n"
    "...\n\n"
    "IMPORTANT:\n"
    "- At the beginning of a subtask, historical_observations only contains Step 0.\n"
    "- At the beginning of the task, finished_subtasks is null.\n\n"
    "Output: 'current_subtask: {subtask name}, finished: {yes/no}'.\n"
)

llamafactory_dataset_name = lerobot_dataset_path.split("/")[-1]

# set initial dataset list
high_level_finetune_data = []

# create folder
finetune_dataset_path = f"{Mem0_workspace}/llamafactory_data/{llamafactory_dataset_name}_subtask"
os.makedirs(finetune_dataset_path, exist_ok=True)

total_episodes = episode_end_id - episode_start_id
print(f"\n[LlamaFactory Data Preparation (Subtask Status)] Total episodes to process: "
      f"{total_episodes} (episode_id {episode_start_id} ~ {episode_end_id - 1})", flush=True)
print(f"[LlamaFactory Data Preparation (Subtask Status)] Sampling interval: every "
      f"{sampling_interval} frames", flush=True)

for idx, episode_id in enumerate(range(episode_start_id, episode_end_id), start=1):
    print(f"[ {idx}/{total_episodes} ] Processing episode {episode_id} ...", flush=True)
    # create episode folder
    episode_folder_path = (
        f"{finetune_dataset_path}/{llamafactory_dataset_name}_subtask_images/episode_{episode_id}"
    )
    os.makedirs(episode_folder_path, exist_ok=True)

    # List of relative image paths for frames within the current subtask
    current_subtask_frames = []

    # List of {name, image_path} for already-finished subtasks
    finished_subtasks_list = []

    # Counter for subtask-end frame images (unique filenames)
    finished_frame_counter = 0

    dataset = LeRobotDataset(lerobot_dataset_path, video_backend="pyav", episodes=[episode_id])
    episode_length = len(dataset)

    for frame_id in tqdm(range(episode_length),
                         desc=f"Episode {idx}/{total_episodes} (id={episode_id}) frames"):

        # --- Detect subtask change: finalize previous subtask ------------------
        if frame_id > 0 and dataset[frame_id]['subtask'] != dataset[frame_id - 1]['subtask']:
            # Previous subtask just ended — save its last frame as completion image
            prev_frame_id = frame_id - 1
            prev_image = dataset[prev_frame_id]['observation.image.head_camera']
            finished_image_path = (
                f"{episode_folder_path}/finished_{finished_frame_counter:06d}.png"
            )
            save_image(prev_image, finished_image_path)
            finished_subtasks_list.append({
                "name": dataset[prev_frame_id]['subtask'],
                "image": (
                    f"{llamafactory_dataset_name}_subtask_images/episode_{episode_id}"
                    f"/finished_{finished_frame_counter:06d}.png"
                ),
            })
            finished_frame_counter += 1

            # Reset current subtask frame history
            current_subtask_frames = []

        # --- Only generate samples at sampling_interval boundaries -------------
        if frame_id % sampling_interval == 0:
            # save current frame image
            image = dataset[frame_id]['observation.image.head_camera']
            image_path = f"{episode_folder_path}/frame_{frame_id:06d}.png"
            save_image(image, image_path)
            current_subtask_frames.append(
                f"{llamafactory_dataset_name}_subtask_images/episode_{episode_id}/frame_{frame_id:06d}.png"
            )

            # build training sample
            frame_information = {"messages": [], "images": []}

            # system prompt message
            system_prompt_msg = {"role": "system", "content": system_prompt}
            frame_information["messages"].append(system_prompt_msg)

            # --- user prompt message ------------------------------------------
            global_task_txt = "<global_task>: " + dataset[frame_id]['global_task'] + "\n"

            # finished_subtasks section
            if len(finished_subtasks_list) == 0:
                finished_subtasks_txt = "<finished_subtasks>: null.\n"
            else:
                finished_subtasks_txt = "<finished_subtasks>:\n"
                for i, item in enumerate(finished_subtasks_list):
                    finished_subtasks_txt += (
                        f"{i}: {item['name']}, "
                        f"the corresponding image is: <image>.\n"
                    )

            # historical_observations section
            historical_obs = "<historical_observations>:\n"
            for i, img_path in enumerate(current_subtask_frames):
                historical_obs += f"Step {i}: <image>.\n"

            combined_message = global_task_txt + finished_subtasks_txt + historical_obs

            user_prompt_msg = {"role": "user", "content": combined_message}
            frame_information["messages"].append(user_prompt_msg)

            # --- assistant prompt message -------------------------------------
            subtask_name = dataset[frame_id]['subtask']
            is_finished = "yes" if dataset[frame_id]['subtask_end'] else "no"
            assistant_prompt_msg = {
                "role": "assistant",
                "content": f"current_subtask: {subtask_name}, finished: {is_finished}",
            }
            frame_information["messages"].append(assistant_prompt_msg)

            # --- collect images (finished subtask images + current frames) ----
            all_images = (
                [item["image"] for item in finished_subtasks_list]
                + current_subtask_frames.copy()
            )
            frame_information["images"] = all_images

            high_level_finetune_data.append(frame_information.copy())

# save high_level_finetune_data to json file
output_path = f"{finetune_dataset_path}/{llamafactory_dataset_name}_subtask_high_level_finetune_data.json"
with open(output_path, "w") as f:
    json.dump(high_level_finetune_data, f, indent=2, ensure_ascii=False)

print(f"\n[LlamaFactory Data Preparation (Subtask Status)] Done. "
      f"Processed {total_episodes} episodes (episode_id {episode_start_id} ~ {episode_end_id - 1}).",
      flush=True)
print(f"[LlamaFactory Data Preparation (Subtask Status)] Total training samples: "
      f"{len(high_level_finetune_data)}", flush=True)
print(f"[LlamaFactory Data Preparation (Subtask Status)] Output saved to: {output_path}", flush=True)
