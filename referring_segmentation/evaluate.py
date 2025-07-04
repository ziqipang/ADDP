import os
import json
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, default="refcoco")
    parser.add_argument("--split", type=str, default="val_flatten")
    parser.add_argument("--result-dir", type=str, required=True)
    parser.add_argument("--threshold", type=int, default=50)
    parser.add_argument("--save-path", type=str, default=None)
    args = parser.parse_args()

    json_path = os.path.join(args.dataset_root, "anns", args.dataset_name, f"{args.split}.json")
    mask_dir = os.path.join(args.dataset_root, "masks", args.dataset_name)
    with open(json_path, "r") as f:
        anns = json.load(f)

    IoU_list, I_list, U_list = [], [], []

    pbar = tqdm(total=len(anns))
    for index, sample in enumerate(anns):
        mask = Image.open(os.path.join(mask_dir, f"{sample['segment_id']}.png")).convert("L")
        mask = mask.resize((512, 512), Image.Resampling.NEAREST)
        mask = np.array(mask)

        predicted_img = Image.open(os.path.join(args.result_dir, f"{index}.png")).convert("RGB")
        predicted_img = predicted_img.resize((512, 512), Image.Resampling.LANCZOS)
        predicted_img = np.array(predicted_img)

        tgt_color = np.array([255, 0, 0])[np.newaxis, np.newaxis, :]
        predicted_mask = np.linalg.norm(predicted_img - tgt_color, axis=-1) < args.threshold

        I = np.logical_and(mask, predicted_mask).sum()
        U = np.logical_or(mask, predicted_mask).sum()
        IoU = I / (U + 1e-10)
        I_list.append(float(I))
        U_list.append(float(U))
        IoU_list.append(IoU)

        pbar.update(1)
    pbar.close()

    if args.save_path:
        result = {"IoU": IoU_list, "I": I_list, "U": U_list}
        json.dump(result, open(args.save_path, "w"))

    print("mIoU: ", np.average(IoU_list))
    print("oIoU: ", np.average(I_list) / np.average(U_list))


if __name__ == "__main__":
    main()
