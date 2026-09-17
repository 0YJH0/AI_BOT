"""Lightweight RGB-D object-position estimator used by Phase 12."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as functional


@dataclass(frozen=True)
class RGBDPoseModelCfg:
    """Serializable model and preprocessing configuration."""

    image_height: int = 120
    image_width: int = 160
    depth_clip_m: float = 5.0
    base_channels: int = 32


def preprocess_rgbd(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    *,
    image_height: int = 120,
    image_width: int = 160,
    depth_clip_m: float = 5.0,
) -> torch.Tensor:
    """Convert uint8 NHWC RGB and NHW/NHWC depth into normalized NCHW input."""

    if rgb.ndim == 3:
        rgb = rgb.unsqueeze(0)
    if depth.ndim == 2:
        depth = depth.unsqueeze(0)
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if rgb.ndim != 4 or rgb.shape[-1] not in (3, 4):
        raise ValueError(f"Expected RGB shape [N,H,W,3/4], got {tuple(rgb.shape)}")
    if depth.ndim != 3 or depth.shape[:3] != rgb.shape[:3]:
        raise ValueError(
            f"Expected depth shape [N,H,W] matching RGB, got {tuple(depth.shape)}"
        )
    if min(image_height, image_width) < 8 or depth_clip_m <= 0.0:
        raise ValueError("Invalid RGB-D preprocessing configuration")

    rgb_float = rgb[..., :3].float().permute(0, 3, 1, 2) / 255.0
    depth_float = torch.nan_to_num(
        depth.float(), nan=depth_clip_m, posinf=depth_clip_m, neginf=0.0
    ).clamp_(0.0, depth_clip_m)
    depth_float = depth_float.unsqueeze(1) / depth_clip_m
    inputs = torch.cat((rgb_float, depth_float), dim=1)
    inputs = functional.interpolate(
        inputs,
        size=(image_height, image_width),
        mode="bilinear",
        align_corners=False,
    )
    return (inputs - 0.5) / 0.5


class _ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int) -> None:
        super().__init__()
        groups = min(8, output_channels)
        self.block = nn.Sequential(
            nn.Conv2d(
                input_channels, output_channels, kernel_size=3, stride=stride, padding=1
            ),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class RGBDObjectPositionNet(nn.Module):
    """Regress object xyz in the ROS optical camera frame from one RGB-D frame."""

    def __init__(
        self,
        config: RGBDPoseModelCfg | None = None,
        target_mean: torch.Tensor | None = None,
        target_std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.config = config or RGBDPoseModelCfg()
        channels = self.config.base_channels
        self.backbone = nn.Sequential(
            _ConvBlock(4, channels, 2),
            _ConvBlock(channels, channels * 2, 2),
            _ConvBlock(channels * 2, channels * 4, 2),
            _ConvBlock(channels * 4, channels * 4, 2),
            # Preserve a coarse 2-D feature grid.  Global average pooling
            # would make direct xyz regression almost translation invariant
            # and discard the image-location cue needed for x/y prediction.
            nn.AdaptiveAvgPool2d((4, 5)),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(channels * 4 * 4 * 5, channels * 4),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.05),
            nn.Linear(channels * 4, 3),
        )
        mean = torch.zeros(3) if target_mean is None else target_mean.float()
        std = torch.ones(3) if target_std is None else target_std.float()
        if mean.shape != (3,) or std.shape != (3,) or bool((std <= 0.0).any()):
            raise ValueError("target_mean and target_std must be positive three-vectors")
        self.register_buffer("target_mean", mean.clone())
        self.register_buffer("target_std", std.clone())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return normalized camera-frame xyz for supervised optimization."""

        return self.head(self.backbone(inputs))

    def predict_position(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return metric camera-frame xyz in metres."""

        return self(inputs) * self.target_std + self.target_mean

    def checkpoint_payload(self, metrics: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "format_version": "1.0.0",
            "model": "RGBDObjectPositionNet",
            "model_config": asdict(self.config),
            "target_mean": self.target_mean.detach().cpu(),
            "target_std": self.target_std.detach().cpu(),
            "model_state_dict": self.state_dict(),
            "metrics": metrics or {},
        }


class RGBDKeypointPositionNet(nn.Module):
    """Localize the object in 2-D and reconstruct xyz with camera geometry."""

    def __init__(
        self,
        config: RGBDPoseModelCfg | None = None,
        depth_mean: torch.Tensor | float = 0.0,
        depth_std: torch.Tensor | float = 1.0,
    ) -> None:
        super().__init__()
        self.config = config or RGBDPoseModelCfg(image_height=60, image_width=80, base_channels=24)
        channels = self.config.base_channels
        self.features = nn.Sequential(
            _ConvBlock(4, channels, 2),
            _ConvBlock(channels, channels * 2, 2),
            _ConvBlock(channels * 2, channels * 4, 1),
        )
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(channels * 4, channels * 2, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels * 2, 1, kernel_size=1),
        )
        self.depth_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((3, 4)),
            nn.Flatten(),
            nn.Linear(channels * 4 * 3 * 4, channels * 2),
            nn.SiLU(inplace=True),
            nn.Linear(channels * 2, 1),
        )
        mean = torch.as_tensor(depth_mean, dtype=torch.float32).reshape(1)
        std = torch.as_tensor(depth_std, dtype=torch.float32).reshape(1)
        if bool((std <= 0.0).any()):
            raise ValueError("depth_std must be positive")
        self.register_buffer("depth_mean", mean.clone())
        self.register_buffer("depth_std", std.clone())

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(inputs)
        heatmap_logits = functional.interpolate(
            self.heatmap_head(features),
            size=(self.config.image_height, self.config.image_width),
            mode="bilinear",
            align_corners=False,
        )
        normalized_depth = self.depth_head(features).squeeze(-1)
        return heatmap_logits, normalized_depth

    def keypoint_pixels(
        self,
        heatmap_logits: torch.Tensor,
        valid_pixel_mask: torch.Tensor | None = None,
        *,
        hard: bool = False,
    ) -> torch.Tensor:
        """Differentiable soft-argmax in resized-input pixel coordinates."""

        batch, _, height, width = heatmap_logits.shape
        if valid_pixel_mask is not None:
            if valid_pixel_mask.shape != (batch, height, width):
                raise ValueError("valid_pixel_mask must match the heatmap spatial shape")
            valid_pixel_mask = valid_pixel_mask.bool()
            if bool((~valid_pixel_mask.flatten(1).any(dim=-1)).any()):
                raise ValueError("Every valid-pixel mask must contain at least one pixel")
            heatmap_logits = heatmap_logits.masked_fill(
                ~valid_pixel_mask.unsqueeze(1), torch.finfo(heatmap_logits.dtype).min
            )
        flattened_logits = heatmap_logits.reshape(batch, -1)
        if hard:
            peak = torch.argmax(flattened_logits, dim=-1)
            x_peak = (peak.remainder(width).to(heatmap_logits.dtype) + 0.5) * (
                self.config.image_width / width
            )
            y_peak = (torch.div(peak, width, rounding_mode="floor").to(heatmap_logits.dtype) + 0.5) * (
                self.config.image_height / height
            )
            return torch.stack((x_peak, y_peak), dim=-1)
        probabilities = torch.softmax(flattened_logits, dim=-1)
        y_grid, x_grid = torch.meshgrid(
            torch.arange(height, device=heatmap_logits.device, dtype=heatmap_logits.dtype),
            torch.arange(width, device=heatmap_logits.device, dtype=heatmap_logits.dtype),
            indexing="ij",
        )
        # Cell centres map back into the preprocessed input image.
        x_grid = (x_grid.reshape(-1) + 0.5) * (self.config.image_width / width)
        y_grid = (y_grid.reshape(-1) + 0.5) * (self.config.image_height / height)
        return torch.stack(
            (
                torch.sum(probabilities * x_grid, dim=-1),
                torch.sum(probabilities * y_grid, dim=-1),
            ),
            dim=-1,
        )

    def predict_position(
        self,
        inputs: torch.Tensor,
        resized_intrinsics: torch.Tensor,
        valid_pixel_mask: torch.Tensor | None = None,
        hard_keypoint: bool = False,
    ) -> torch.Tensor:
        if resized_intrinsics.shape != (inputs.shape[0], 3, 3):
            raise ValueError("resized_intrinsics must have shape [N,3,3]")
        heatmap_logits, normalized_depth = self(inputs)
        return self.position_from_outputs(
            inputs,
            heatmap_logits,
            normalized_depth,
            resized_intrinsics,
            valid_pixel_mask=valid_pixel_mask,
            hard_keypoint=hard_keypoint,
        )

    def position_from_outputs(
        self,
        inputs: torch.Tensor,
        heatmap_logits: torch.Tensor,
        normalized_depth: torch.Tensor,
        resized_intrinsics: torch.Tensor,
        valid_pixel_mask: torch.Tensor | None = None,
        hard_keypoint: bool = False,
    ) -> torch.Tensor:
        """Geometrically reconstruct xyz without a second network forward pass."""

        pixels = self.keypoint_pixels(
            heatmap_logits, valid_pixel_mask, hard=hard_keypoint
        )
        grid = torch.stack(
            (
                pixels[:, 0] / max(self.config.image_width - 1, 1) * 2.0 - 1.0,
                pixels[:, 1] / max(self.config.image_height - 1, 1) * 2.0 - 1.0,
            ),
            dim=-1,
        ).clamp(-1.0, 1.0).reshape(-1, 1, 1, 2)
        normalized_surface = functional.grid_sample(
            inputs[:, 3:4], grid, mode="bilinear", align_corners=True
        ).reshape(-1)
        surface_depth = (normalized_surface + 1.0) * 0.5 * self.config.depth_clip_m
        # Predict only the small visible-surface to cube-centre offset.  RTX
        # depth supplies metric scale, which generalizes better than asking a
        # small network to relearn absolute camera depth.
        depth = surface_depth + normalized_depth * self.depth_std + self.depth_mean
        x = (pixels[:, 0] - resized_intrinsics[:, 0, 2]) / resized_intrinsics[:, 0, 0] * depth
        y = (pixels[:, 1] - resized_intrinsics[:, 1, 2]) / resized_intrinsics[:, 1, 1] * depth
        return torch.stack((x, y, depth), dim=-1)

    def checkpoint_payload(self, metrics: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "format_version": "1.0.0",
            "model": "RGBDKeypointPositionNet",
            "model_config": asdict(self.config),
            "depth_mean": self.depth_mean.detach().cpu(),
            "depth_std": self.depth_std.detach().cpu(),
            "model_state_dict": self.state_dict(),
            "metrics": metrics or {},
        }


class RGBDPoseEstimator:
    """Checkpoint-backed batched inference interface for simulator control loops."""

    def __init__(
        self, model: RGBDObjectPositionNet | RGBDKeypointPositionNet, device: torch.device | str
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()

    @classmethod
    def load(
        cls, checkpoint_path: str | Path, device: torch.device | str = "cpu"
    ) -> "RGBDPoseEstimator":
        checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
        config = RGBDPoseModelCfg(**checkpoint["model_config"])
        if checkpoint.get("model") == "RGBDObjectPositionNet":
            model = RGBDObjectPositionNet(
                config,
                target_mean=checkpoint["target_mean"],
                target_std=checkpoint["target_std"],
            )
        elif checkpoint.get("model") == "RGBDKeypointPositionNet":
            model = RGBDKeypointPositionNet(
                config,
                depth_mean=checkpoint["depth_mean"],
                depth_std=checkpoint["depth_std"],
            )
        else:
            raise ValueError("Unsupported visual checkpoint")
        model.load_state_dict(checkpoint["model_state_dict"])
        return cls(model, device)

    @torch.inference_mode()
    def predict(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        camera_intrinsics: torch.Tensor | None = None,
        valid_pixel_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        config = self.model.config
        inputs = preprocess_rgbd(
            rgb.to(self.device),
            depth.to(self.device),
            image_height=config.image_height,
            image_width=config.image_width,
            depth_clip_m=config.depth_clip_m,
        )
        if isinstance(self.model, RGBDKeypointPositionNet):
            if camera_intrinsics is None:
                raise ValueError("The keypoint model requires camera intrinsics")
            if rgb.ndim == 3:
                raw_height, raw_width = rgb.shape[:2]
            else:
                raw_height, raw_width = rgb.shape[1:3]
            resized_intrinsics = camera_intrinsics.to(self.device).float().clone()
            if resized_intrinsics.ndim == 2:
                resized_intrinsics = resized_intrinsics.unsqueeze(0)
            resized_intrinsics[:, 0, :] *= config.image_width / raw_width
            resized_intrinsics[:, 1, :] *= config.image_height / raw_height
            resized_mask = None
            if valid_pixel_mask is not None:
                if valid_pixel_mask.ndim == 2:
                    valid_pixel_mask = valid_pixel_mask.unsqueeze(0)
                resized_mask = functional.interpolate(
                    valid_pixel_mask.to(self.device).float().unsqueeze(1),
                    size=(config.image_height, config.image_width),
                    mode="nearest",
                ).squeeze(1).bool()
            return self.model.predict_position(
                inputs,
                resized_intrinsics,
                resized_mask,
                hard_keypoint=resized_mask is not None,
            )
        return self.model.predict_position(inputs)


def quaternion_conjugate(quaternion_wxyz: torch.Tensor) -> torch.Tensor:
    """Return the conjugate of a wxyz quaternion batch."""

    if quaternion_wxyz.shape[-1] != 4:
        raise ValueError("Quaternion final dimension must be four")
    result = quaternion_wxyz.clone()
    result[..., 1:] *= -1.0
    return result


def quaternion_apply(quaternion_wxyz: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by normalized wxyz quaternions."""

    if quaternion_wxyz.shape[-1] != 4 or vector.shape[-1] != 3:
        raise ValueError("Expected quaternion [...,4] and vector [...,3]")
    quaternion_wxyz = quaternion_wxyz / torch.linalg.vector_norm(
        quaternion_wxyz, dim=-1, keepdim=True
    ).clamp_min(1.0e-12)
    scalar = quaternion_wxyz[..., :1]
    imaginary = quaternion_wxyz[..., 1:]
    twice_cross = 2.0 * torch.linalg.cross(imaginary, vector, dim=-1)
    return vector + scalar * twice_cross + torch.linalg.cross(
        imaginary, twice_cross, dim=-1
    )


def camera_to_world_position(
    position_camera: torch.Tensor,
    camera_position_w: torch.Tensor,
    camera_quaternion_wxyz: torch.Tensor,
) -> torch.Tensor:
    """Transform ROS-optical camera positions into world coordinates."""

    return camera_position_w + quaternion_apply(camera_quaternion_wxyz, position_camera)


def world_to_root_position(
    position_w: torch.Tensor,
    root_position_w: torch.Tensor,
    root_quaternion_wxyz: torch.Tensor,
) -> torch.Tensor:
    """Transform world positions into a robot-root frame."""

    return quaternion_apply(
        quaternion_conjugate(root_quaternion_wxyz), position_w - root_position_w
    )
