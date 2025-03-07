# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import shutil
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import matplotlib
import matplotlib.cm as cm
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from superglue.models.matching import Matching as SuperGlueMatching

# matplotlib.use("Agg")


class SPGMatching(nn.Module):
    """Implement matching between images using SuperGlue
    github.com/magicleap/SuperGluePretrainedNetwork."""

    def __init__(
        self,
        # config,
        default_vis_dir: str,
        print_images: bool,
        only_print_match: bool = True,
    ) -> None:
        super().__init__()
        self._device = torch.device("cuda:0")
        self.matcher = SuperGlueMatching().eval().to(self._device)
        self.print_images = print_images
        self.only_print_match = only_print_match
        self.default_vis_dir = default_vis_dir
        self.vis_dir = default_vis_dir
        self.match_conf_threshold = 6.0

    def set_vis_dir(self, episode_id: str) -> None:
        if self.print_images:
            self.vis_dir = os.path.join(self.default_vis_dir, str(episode_id))
            shutil.rmtree(self.vis_dir, ignore_errors=True)
            os.makedirs(self.vis_dir, exist_ok=True)

    @staticmethod
    def _make_matching_plot(
        img0: np.ndarray,
        img1: np.ndarray,
        kpts0: np.ndarray,
        kpts1: np.ndarray,
        mkpts0: np.ndarray,
        mkpts1: np.ndarray,
        color: np.ndarray,
        text: List[str],
        path: str,
        show_keypoints=False,
        small_text: List[str] = [],
    ) -> None:
        """Visualizes matching inference using matplotlib and saves the result
        to disk."""

        def plot_image_pair(imgs: List[np.ndarray], dpi: int = 200) -> None:
            fig, ax = plt.subplots(1, 2, figsize=(7.5, 4), dpi=dpi)
            fig.set_facecolor("black")
            for i in range(2):
                ax[i].imshow(imgs[i], cmap=plt.get_cmap("gray"), vmin=0, vmax=255)
                ax[i].get_yaxis().set_ticks([])
                ax[i].get_xaxis().set_ticks([])
                for spine in ax[i].spines.values():  # remove frame
                    spine.set_visible(False)
            plt.tight_layout()

        def plot_keypoints(
            kpts0: np.ndarray, kpts1: np.ndarray, color: str = "w", ps: int = 2
        ) -> None:
            ax = plt.gcf().axes
            ax[0].scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
            ax[1].scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)

        def plot_matches(
            kpts0: np.ndarray,
            kpts1: np.ndarray,
            color: str,
            lw: float = 1.5,
            ps: int = 4,
        ) -> None:
            fig = plt.gcf()
            ax = fig.axes
            fig.canvas.draw()

            transFigure = fig.transFigure.inverted()
            fkpts0 = transFigure.transform(ax[0].transData.transform(kpts0))
            fkpts1 = transFigure.transform(ax[1].transData.transform(kpts1))

            fig.lines = [
                matplotlib.lines.Line2D(
                    (fkpts0[i, 0], fkpts1[i, 0]),
                    (fkpts0[i, 1], fkpts1[i, 1]),
                    zorder=1,
                    transform=fig.transFigure,
                    c=color[i],
                    linewidth=lw,
                )
                for i in range(len(kpts0))
            ]
            ax[0].scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
            ax[1].scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)

        plot_image_pair([img0, img1])
        if show_keypoints:
            plot_keypoints(kpts0, kpts1, color="k", ps=4)
            plot_keypoints(kpts0, kpts1, color="w", ps=2)
        plot_matches(mkpts0, mkpts1, color)

        fig = plt.gcf()
        txt_color = "black"
        txt = fig.text(
            0.01,
            0.99,
            "\n".join(text),
            transform=fig.axes[0].transAxes,
            fontsize=15,
            va="top",
            ha="left",
            color=txt_color,
        )
        txt.set_path_effects([patheffects.withStroke(linewidth=2, foreground="w")])

        txt = fig.text(
            0.01,
            0.01,
            "\n".join(small_text),
            transform=fig.axes[0].transAxes,
            fontsize=5,
            va="bottom",
            ha="left",
            color=txt_color,
        )
        txt.set_path_effects([patheffects.withStroke(linewidth=2, foreground="w")])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(str(path), bbox_inches="tight", pad_inches=0)
        plt.close()

    def get_confidence_score(self, matches, confidences):
        if matches.size != 0:
            # valid = matches > -1
            # return confidences[valid].sum()
            return confidences[matches > -1].sum()
        else:
            return 0

    def visualize(
        self,
        matcher_inputs: Dict[str, Any],
        matcher_outputs: Dict[str, Any],
        step: int,
        filename: str = None,
    ) -> None:
        """Visualize the input/output of running SuperPoint and SuperGlue
        inference."""
        if not self.print_images:
            return

        data = {**matcher_inputs, **matcher_outputs}
        data = {k: v[0].cpu().numpy() for k, v in data.items()}

        kpts0 = data["keypoints0"]
        kpts1 = data["keypoints1"]
        matches = data["matches0"]
        conf = data["matching_scores0"]

        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid]
        conf_score = self.get_confidence_score(matches, conf)

        color = cm.jet(mconf)

        text = [
            "SuperGlue",
            "Keypoints: {}:{}".format(len(kpts0), len(kpts1)),
            f"Matches: {len(mkpts0)}",
            f"Match Conf: {conf_score:.2f}",
        ]

        # Display extra parameter info.
        k_thresh = self.matcher.superpoint.config["keypoint_threshold"]
        m_thresh = self.matcher.superglue.config["match_threshold"]
        small_text = [
            f"Keypoint Threshold: {k_thresh:.4f}",
            f"Match Threshold: {m_thresh:.2f}",
        ]

        if filename is None:
            filename = f"superglue_{str(step+1).zfill(3)}.png"
        if not self.only_print_match or (
            self.only_print_match and conf_score >= self.match_conf_threshold
        ):
            self._make_matching_plot(
                data["image0"][0] * 255,
                data["image1"][0] * 255,
                kpts0,
                kpts1,
                mkpts0,
                mkpts1,
                color,
                text,
                os.path.join(self.vis_dir, filename),
                small_text=small_text,
            )

    def _preprocess_image(self, img: np.ndarray) -> Tensor:
        """Prepare an image for SuperPoint inference."""
        img_in = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_in = img_in.astype("float32") / 255.0
        img_in = torch.from_numpy(img_in)[None, None]
        return img_in.to(self._device)

    @torch.no_grad()
    def get_goal_image_keypoints(
        self, goal_image: np.ndarray, idx: int = 0
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Run SuperPoint inference on a single image."""
        goal_img = self._preprocess_image(goal_image)
        pred = self.matcher.superpoint({"image": goal_img})
        return goal_img, {f"{k}{idx}": v for k, v in pred.items()}

    @torch.no_grad()
    def forward(
        self,
        rgb_image_0: np.ndarray,
        rgb_image_1: Union[np.ndarray, torch.Tensor],
        rgb_image_0_keypoints: Optional[Dict[str, Any]] = None,
        rgb_image_1_keypoints: Optional[Dict[str, Any]] = None,
        rgb_image_0_descriptors: Optional[Dict[str, Any]] = None,
        rgb_image_1_descriptors: Optional[Dict[str, Any]] = None,
        step: Optional[int] = None,
        filename: Optional[str] = None,
    ):
        """Computes and describes keypoints using SuperPoint and matches
        keypoints between 2 RGB images using SuperGlue.

        Either goal_image or goal_image_keypoints must be provided.
        Returns:
            tensor of goal image keypoints
            tensor of rgb image keypoints
            tensor of keypoint matches
            tensor of match confidences
        """
        if rgb_image_0_keypoints is None:
            rgb_image_0_keypoints = {}
        if rgb_image_1_keypoints is None:
            rgb_image_1_keypoints = {}
        if isinstance(rgb_image_0, np.ndarray):
            rgb_image_0 = self._preprocess_image(rgb_image_0)
        if isinstance(rgb_image_1, np.ndarray):
            rgb_image_1 = self._preprocess_image(rgb_image_1)

        matcher_inputs = {
            "image0": rgb_image_0,
            "image1": rgb_image_1,
            **rgb_image_0_keypoints,
            **rgb_image_1_keypoints,
        }
        pred = self.matcher(matcher_inputs)
        matches = pred["matches0"].cpu().numpy()
        confidence = pred["matching_scores0"].cpu().numpy()

        if self.print_images:
            self.visualize(matcher_inputs, pred, step, filename)

        if "keypoints0" in matcher_inputs:
            keypoints0 = matcher_inputs["keypoints0"]
        else:
            keypoints0 = pred["keypoints0"]

        if "keypoints1" in matcher_inputs:
            keypoints1 = matcher_inputs["keypoints1"]
        else:
            keypoints1 = pred["keypoints1"]

        return keypoints0, keypoints1, matches, confidence
