import torch
import torch.nn as nn

from mfm.networks.unet_base import UNetModelWrapper


class GeoPathUNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        kwargs.pop("geopath_model", None)
        self.time_geopath = True
        self.mainnet = UNetModelWrapper(geopath_model=True, *args, **kwargs)

    def forward(
        self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat([x0, x1], dim=1)
        return self.mainnet(t, x)
