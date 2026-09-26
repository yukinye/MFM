"""Download AFHQ and build the cached tensors that mfm.dataloaders.image_data expects.

ImageDataModule only builds its caches itself when they are missing, and that path is
broken in the repo (`_get_ambient_space` calls `image_base_dataset("val",)` without a
transform, and never sets `ambient_x0/ambient_x1`). Building all four files up front
makes the datamodule take its working `_load_*` path instead.

Files written to <working_dir>/data/afhq/:
  afhq_{train,val}_ambient_dataset_64.pt   {"mean": [N,3,64,64] in [0,1], "label": [N]}
                                           (ambient space, used for RBF k-means, App. D.2)
  afhq_{train,val}_latent_dataset_128.pt   {"mean": [N,4,16,16], "label": [N]}
                                           (SD v1 VAE latent mean of 128x128 images)
  afhq_val_pixels_128.pt                   {"mean": [N,3,128,128] in [0,1], "label": [N]}
                                           (used by evaluate_afhq.py for FID / LPIPS)
Labels follow ImageFolder order: cat=0, dog=1, wild=2.

Usage:
  python scripts/afhq/prepare_afhq.py --working_dir runs/afhq --download
"""

import argparse
import subprocess
from pathlib import Path

import torch
import torchvision.transforms as T
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

AFHQ_URL = "https://www.dropbox.com/s/t9l9o3vsx2jai3z/afhq.zip?dl=1"  # from clovaai/stargan-v2


def make_transform(size, preprocess):
    if preprocess == "paper":
        # App. D.2: "upsizing to 313x256, center cropping to 256x256, resizing to 128x128"
        return T.Compose([T.Resize((313, 256)), T.CenterCrop(256), T.Resize((size, size)), T.ToTensor()])
    # What mfm/dataloaders/image_data.py does: a plain resize (AFHQ images are 512x512).
    return T.Compose([T.Resize((size, size)), T.ToTensor()])


def load_split(root, split, size, preprocess):
    dataset = ImageFolder(root / split, make_transform(size, preprocess))
    assert dataset.class_to_idx == {"cat": 0, "dog": 1, "wild": 2}, dataset.class_to_idx
    images, labels = [], []
    for x, y in tqdm(DataLoader(dataset, batch_size=64, num_workers=8), desc=f"{split} {size}px"):
        images.append(x)
        labels.append(y)
    return torch.cat(images), torch.cat(labels)


@torch.no_grad()
def encode(vae, process, images, device):
    means = []
    for batch in tqdm(images.split(64), desc="VAE encode"):
        batch = process.preprocess(batch).to(device)  # [0,1] -> [-1,1]
        means.append(vae.encode(batch).latent_dist.mean.cpu())
    return torch.cat(means)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--working_dir", required=True)
    p.add_argument("--download", action="store_true", help="download and unzip AFHQ (~730 MB)")
    p.add_argument("--preprocess", choices=["repo", "paper"], default="repo",
                   help="repo: plain resize as in image_data.py; paper: resize 313x256 + center crop (App. D.2)")
    p.add_argument("--image_size", type=int, default=128)
    args = p.parse_args()

    data_root = Path(args.working_dir).resolve() / "data"
    root = data_root / "afhq"
    if args.download and not (root / "train").exists():
        data_root.mkdir(parents=True, exist_ok=True)
        zip_path = data_root / "afhq.zip"
        subprocess.run(["curl", "-L", "-o", str(zip_path), AFHQ_URL], check=True)
        subprocess.run(["unzip", "-q", str(zip_path), "-d", str(data_root)], check=True)
    if not (root / "train").exists():
        raise SystemExit(f"AFHQ not found at {root} (expected train/ and val/ with cat/dog/wild)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", use_safetensors=True).to(device)
    process = VaeImageProcessor(do_convert_rgb=True)

    for split in ["train", "val"]:
        images, labels = load_split(root, split, 64, args.preprocess)
        torch.save({"mean": images, "label": labels}, root / f"afhq_{split}_ambient_dataset_64.pt")

        images, labels = load_split(root, split, args.image_size, args.preprocess)
        latents = encode(vae, process, images, device)
        torch.save({"mean": latents, "label": labels}, root / f"afhq_{split}_latent_dataset_{args.image_size}.pt")
        if split == "val":
            torch.save({"mean": images, "label": labels}, root / f"afhq_val_pixels_{args.image_size}.pt")
        print(f"{split}: {len(labels)} images, latent shape {tuple(latents.shape[1:])}")


if __name__ == "__main__":
    main()
