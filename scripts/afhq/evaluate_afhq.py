"""Evaluate a trained dog->cat flow model on AFHQ as described in App. D.2 / Table 2.

- source: AFHQ validation dogs (VAE latents)
- generation: integrate v_theta from t=0 to t=1 with Tsit5 (atol=rtol=1e-5, 100 output
  steps), decode the final latent with the SD v1 VAE
- FID: generated cats vs. the *validation* cat set (Inception features, fld)
- LPIPS (VGG): each source dog vs. the cat generated from it

The repo's built-in FlowNetTrainImage test is not used, because it does not match the
paper: it compares against *training* images at 64px, feeds [0,1] images through a
[-1,1] postprocess for FID, and pairs 4.7k train dogs with 500 generated cats for LPIPS.

Usage:
  python scripts/afhq/evaluate_afhq.py --config_path configs/images/mfm.yaml \
      --working_dir runs/afhq/mfm --ckpt runs/afhq/mfm/checkpoints/image/<id>/flow_model/<x>.ckpt
"""

import argparse
import json
import sys
from pathlib import Path

import lpips
import torch
from fld.features.InceptionFeatureExtractor import InceptionFeatureExtractor
from fld.metrics.FID import FID
from torchdyn.core import NeuralODE
from torchvision.utils import save_image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from mfm.dataloaders.image_data import ImageDataModule, LABELS_MAP  # noqa: E402
from mfm.flow_matchers.ema import EMA  # noqa: E402
from mfm.networks.unet_base import UNetModelWrapper as UNetModel  # noqa: E402
from mfm.networks.utils import flow_model_torch_wrapper  # noqa: E402
from mfm.train.parsers import parse_args  # noqa: E402
from mfm.train.train_utils import dataset_name2datapath, load_config, merge_config  # noqa: E402


def build_args(config_path, working_dir):
    sys.argv = [sys.argv[0], "--working_dir", working_dir]
    args = merge_config(parse_args(), load_config(config_path))
    args.data_path = dataset_name2datapath(args.data_name, args.working_dir)
    return args


def load_flow_net(args, dim, ckpt_path, device):
    # Same construction as mfm/train/main.py
    flow_net = UNetModel(
        geopath_model=False,
        dim=dim,
        num_channels=args.unet_num_channels,
        num_res_blocks=args.unet_num_res_blocks,
        channel_mult=args.unet_channel_mult,
        dropout=args.unet_dropout,
        resblock_updown=args.unet_resblock_updown,
        use_new_attention_order=args.unet_use_new_attention_order,
        attention_resolutions=args.unet_attention_resolutions,
        num_heads=args.unet_num_heads,
    )
    if args.ema_decay is not None:
        flow_net = EMA(model=flow_net, decay=args.ema_decay)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    state = {k[len("flow_net."):]: v for k, v in state.items() if k.startswith("flow_net.")}
    flow_net.load_state_dict(state)
    flow_net.to(device)
    flow_net.eval()  # EMA.train(False) swaps in the EMA weights; it returns None, so don't chain
    return flow_net


@torch.no_grad()
def generate(flow_net, vae, x0, batch_size, device):
    node = NeuralODE(flow_model_torch_wrapper(flow_net), solver="tsit5", sensitivity="adjoint", atol=1e-5, rtol=1e-5)
    t_span = torch.linspace(0, 1, 100, device=device)
    outputs = []
    for i, batch in enumerate(x0.split(batch_size)):
        x1 = node.trajectory(batch.to(device), t_span=t_span)[-1]
        outputs.append(vae.decode(x1).sample.clamp(-1, 1).cpu())
        print(f"generated {min((i + 1) * batch_size, len(x0))}/{len(x0)}", flush=True)
    return torch.cat(outputs)  # [-1, 1]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config_path", required=True)
    p.add_argument("--working_dir", required=True)
    p.add_argument("--ckpt", required=True, help="flow model checkpoint (flow_model/*.ckpt)")
    p.add_argument("--out", help="output dir (default: <working_dir>/eval)")
    p.add_argument("--batch_size", type=int, default=16)
    cli = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = build_args(cli.config_path, cli.working_dir)
    datamodule = ImageDataModule(args=args)  # loads the cached tensors from prepare_afhq.py
    flow_net = load_flow_net(args, datamodule.dim, cli.ckpt, device)
    vae = datamodule.vae.to(device)

    labels = LABELS_MAP[args.data_name]
    pixels = torch.load(Path(args.data_path) / f"afhq_val_pixels_{args.image_size}.pt")
    val_dogs = pixels["mean"][pixels["label"] == labels[args.x0_label]]  # same order as datamodule.val_x0
    val_cats = pixels["mean"][pixels["label"] == labels[args.x1_label]]
    assert len(val_dogs) == len(datamodule.val_x0)

    generated = generate(flow_net, vae, datamodule.val_x0, cli.batch_size, device)
    generated_01 = (generated + 1) / 2

    extractor = InceptionFeatureExtractor()
    fid = FID().compute_metric(extractor.get_tensor_features(val_cats), None, extractor.get_tensor_features(generated_01))

    loss_fn = lpips.LPIPS(net="vgg").to(device)
    lpips_vals = [
        loss_fn(d.to(device), g.to(device), normalize=True).flatten().cpu()
        for d, g in zip(val_dogs.split(50), generated_01.split(50))
    ]
    lpips_score = torch.cat(lpips_vals).mean().item()

    out = Path(cli.out or Path(cli.working_dir) / "eval")
    out.mkdir(parents=True, exist_ok=True)
    save_image(torch.cat([val_dogs[:16], generated_01[:16]]), out / "dogs_to_cats.png", nrow=16)
    torch.save(generated, out / "generated_cats.pt")
    result = {"FID": float(fid), "LPIPS": lpips_score, "n": len(generated), "ckpt": str(cli.ckpt)}
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
