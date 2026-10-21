# EgoForesight modifications Copyright 2026 Chubin Zhang.

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
import re
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import tree
from einops import rearrange
from PIL import Image
from pydantic import Field, PrivateAttr
from transformers import AutoProcessor, ProcessorMixin
from transformers.data.data_collator import DataCollatorMixin
from transformers.feature_extraction_utils import BatchFeature

from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EMBODIMENT_TAG_MAPPING, EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.schema import DatasetMetadata
from starVLA.dataloader.gr00t_lerobot.transform.base import InvertibleModalityTransform

# from starVLA.dataloader.gr00t_lerobot.backbone.eagle_backbone import DEFAULT_EAGLE_PATH
# from starVLA.dataloader.gr00t_lerobot.backbone.qwen_vl_utils import process_vision_info

# from starVLA.dataloader.gr00t_lerobot.fusevla.models.dinov3_vit.image_processing_dinov3_vit_fast import DINOv3ViTImageProcessorFast

# def formalize_language(language: str) -> str:
#     """
#     1. Force lowercase
#     2. Remove all punctuations
#     """
#     language = language.lower()
#     language = re.sub(r"[^\w\s]", "", language)
#     return language


# def build_eagle_processor(eagle_path: str) -> ProcessorMixin:
#     eagle_processor = AutoProcessor.from_pretrained(
#         eagle_path, trust_remote_code=True, use_fast=True
#     )
#     eagle_processor.tokenizer.padding_side = "left"
#     return eagle_processor

# def build_dino_processor(pretrained_path: str):
#     processor = DINOv3ViTImageProcessorFast.from_pretrained(
#         pretrained_path, trust_remote_code=True, use_fast=True
#     )
#     return processor


def collate(features: List[dict]) -> dict:
    batch = {}
    keys = features[0].keys()

    for key in keys:
        values = [elem[key] for elem in features]

        # if key == "eagle_content":
        #     text_list = []
        #     image_inputs = []
        #     for v in values:
        #         curr_text_list = v["text_list"]
        #         curr_image_inputs = v["image_inputs"]
        #         text_list += curr_text_list
        #         image_inputs += curr_image_inputs
        #     eagle_inputs = eagle_processor(
        #         text=text_list, images=image_inputs, return_tensors="pt", padding=True
        #     )

        #     if dino_processor is not None:
        #         vit_inputs = dino_processor(images=image_inputs, return_tensors="pt")
        #         #import pdb; pdb.set_trace()
        #         batch['vit_inputs'] = vit_inputs['pixel_values']

        #     for k, v in eagle_inputs.items():
        #         k = "eagle_" + k
        #         batch[k] = v
        #     if 'future_image_inputs' in values[0]:
        #         # enable_latent_embedding
        #         future_image_inputs = []
        #         for v in values:
        #             future_image = v["future_image_inputs"]
        #             future_image_inputs += future_image
        #         future_inputs = eagle_processor(
        #             text=text_list, images=future_image_inputs, return_tensors="pt", padding=True
        #         )
        #         batch['future_pixel_values'] = future_inputs['pixel_values']
        if key in ("pixel_values", "image_grid_thw", "attention_mask", "input_ids"):
            # Concat in existing batch dimension.
            batch[key] = torch.cat(values)
        else:
            # state, state_mask, action and action_mask.
            # Stack to form the batch dimension.
            batch[key] = torch.from_numpy(np.stack(values))
    return batch


class DefaultDataCollator(DataCollatorMixin):
    def __init__(self):
        super().__init__()
        # self.eagle_processor = build_eagle_processor(eagle_path)
        # if dino_path is not None:
        #     self.dino_processor = build_dino_processor(dino_path)
        # else:
        #     self.dino_processor = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        return collate(features)


class GR00TTransform(InvertibleModalityTransform):

    # -- We inherit from ModalityTransform, so we keep apply_to as well --
    apply_to: list[str] = Field(default_factory=list, description="Not used in this transform, kept for compatibility.")
    training: bool = Field(default=True, description="Whether to apply the transform in training mode.")
    # formalize_language: bool = Field(default=False, description="Formalize language if True.")
    embodiment_tag_mapping: dict[str, int] = Field(
        description="The projector index of each embodiment tag.",
        default=EMBODIMENT_TAG_MAPPING,
    )
    language_dropout_prob: float = Field(
        default=0.0,
        description="Dropout probability for language.",
    )

    # # Private attributes to keep track of shapes/dimensions across apply/unapply
    # _language_key: Optional[list[str]] = PrivateAttr(default=None)

    # eagle_processor: ProcessorMixin = Field(default=build_eagle_processor(DEFAULT_EAGLE_PATH))
    # dino_processor: Optional[Any] = Field(default=None, description="DINO processor for LWA inference")

    # XEmbDiT arguments
    default_instruction: str = Field(default="Perform the default behavior.")
    max_state_dim: int
    max_action_dim: int
    state_horizon: int
    action_horizon: int

    max_length: int = 512
    embodiment_tag: EmbodimentTag | None = None

    def set_metadata(self, dataset_metadata: DatasetMetadata):
        """Set the metadata for the transform."""
        super().set_metadata(dataset_metadata)
        self.embodiment_tag = dataset_metadata.embodiment_tag

    def get_embodiment_tag(self) -> int:
        """Get the embodiment tag from the data."""
        assert self.embodiment_tag is not None, "Embodiment tag not set. Please call set_metadata first."
        return self.embodiment_tag_mapping[self.embodiment_tag.value]

    def check_keys_and_batch_size(self, data):
        grouped_keys = {}
        for key in data.keys():
            if "annotation" in key:
                modality = "language"
            else:
                try:
                    modality, _ = key.split(".")
                except:  # noqa: E722
                    modality = "others"  # will contain the video, state, and action
            if modality not in grouped_keys:
                grouped_keys[modality] = []
            grouped_keys[modality].append(key)
        # Use video key to determine batch size.
        video_ndim = data["video"].ndim
        if video_ndim == 5:  # Interpret as [T, V, H, W, C]
            is_batched = False
            batch_size = 1
        elif video_ndim == 6:  # Interpret as [B, T, V, H, W, C]
            is_batched = True
            batch_size = data["video"].shape[0]
        else:
            raise ValueError(f"Unsupported video number of dimensions: {video_ndim}")

        # Handle language
        # if "language" in grouped_keys:
        #     language_keys = grouped_keys["language"]
        #     assert len(language_keys) == 1, f"{language_keys=}"
        #     self._language_key = language_keys[0]
        return is_batched, batch_size

    # def _apply_vlm_processing(self, batch: dict) -> BatchFeature:
    #     """
    #     Args:
    #         batch:
    #             video: [V, T, C, H, W]
    #     Returns: required input with the format `BatchFeature`
    #     """
    #     # TODO(YL, FH): check if this is correct
    #     images = batch["images"]  # [V, T, C, H, W]
    #     images.shape[0]

    #     np_images = rearrange(images, "v t c h w -> (t v) c h w")
    #     text_content = []

    #     # handle language
    #     lang = batch["language"]
    #     if isinstance(lang, list):
    #         lang = lang[0]
    #     text_content.append({"type": "text", "text": lang})

    #     eagle_images = [Image.fromarray(np.transpose(v, (1, 2, 0))) for v in np_images]
    #     eagle_image = [{"type": "image", "image": img} for img in eagle_images]
    #     eagle_conversation = [
    #         {
    #             "role": "user",
    #             "content": eagle_image + text_content,
    #         }
    #     ]

    #     text_list = [
    #         self.eagle_processor.apply_chat_template(
    #             eagle_conversation, tokenize=False, add_generation_prompt=True
    #         )
    #     ]
    #     image_inputs, video_inputs = self._process_vision_info(eagle_conversation)
    #     eagle_content = {
    #         "image_inputs": image_inputs,
    #         "video_inputs": video_inputs,
    #         "text_list": text_list,
    #     }
    #     inputs = {}
    #     inputs["eagle_content"] = eagle_content
    #     return inputs

    # def _process_vision_info(self, eagle_conversation):
    #     # TODO
    #     if "eagle" in DEFAULT_EAGLE_PATH.lower():
    #         # Use the eagle processor to process the vision info
    #         return self.eagle_processor.process_vision_info(eagle_conversation)
    #     return process_vision_info(eagle_conversation)

    # def _prepare_video(self, data: dict):
    #     """Process, stack, and pad images from data['video']."""
    #     ## TODO(YL, FH): check if this is correct
    #     images = rearrange(
    #         data["video"],
    #         "t v h w c -> v t c h w",
    #     )
    #     return images

    # def _prepare_language(self, data: dict):
    #     """Tokenize data['language'] (or default_instruction if missing)."""
    #     if self._language_key is not None:
    #         raw_language = data[self._language_key]
    #         if isinstance(raw_language, list):
    #             raw_language = raw_language[0]

    #         # Language dropout
    #         if self.training and self.language_dropout_prob > 1e-9:
    #             if random.random() < self.language_dropout_prob:
    #                 raw_language = self.default_instruction
    #     else:
    #         raw_language = self.default_instruction
    #     return raw_language

    def _prepare_state(self, data: dict):
        """
        Gathers final state from data['state'], then pads to max_state_dim.
        Return (state, state_mask, n_state_tokens).
        """
        if "state" not in data:
            state = np.zeros((self.state_horizon, self.max_state_dim))
            state_mask = np.zeros((self.state_horizon, self.max_state_dim), dtype=bool)
            n_state_tokens = self.state_horizon
            return state, state_mask, n_state_tokens

        state = data["state"]
        assert state.shape[0] == self.state_horizon, f"{state.shape=}, {self.state_horizon=}"

        # Canonical-aligned upstream already produced [T, max_state_dim] with a
        # correct mask. Trust it instead of re-padding.
        if state.shape[-1] == self.max_state_dim and "state_mask" in data:
            n_state_tokens = state.shape[0]
            return state, data["state_mask"], n_state_tokens

        n_state_dims = state.shape[-1]

        # Instead of asserting, just take the first max_state_dim dimensions if needed
        if n_state_dims > self.max_state_dim:
            state = state[:, : self.max_state_dim]
            n_state_dims = self.max_state_dim
        else:
            # Pad up to max_state_dim if smaller
            state = np.pad(state, ((0, 0), (0, self.max_state_dim - n_state_dims)), "constant")

        # Create mask for real state dims
        state_mask = np.zeros_like(state).astype(bool)
        state_mask[:, :n_state_dims] = True

        # We only have 1 "proprio" token to represent the entire state
        n_state_tokens = state.shape[0]
        return state, state_mask, n_state_tokens

    # action horizon padding
    # def _prepare_action(self, data: dict):
    #     """
    #     Pad/truncate to action_horizon and max_action_dim, return masks.
    #     Supports mixed datasets with different action horizons.
    #     """
    #     if "action" not in data:
    #         actions = np.zeros((self.action_horizon, self.max_action_dim))
    #         actions_mask = np.zeros((self.action_horizon, self.max_action_dim), dtype=bool)
    #         n_action_tokens = self.action_horizon
    #         return actions, actions_mask, n_action_tokens

    #     actions = data["action"]
    #     actual_horizon = actions.shape[0]  # T
    #     print('actual_horizon',actual_horizon)
    #     n_action_dims = actions.shape[1]  # D

    #     assert (
    #         n_action_dims <= self.max_action_dim
    #     ), f"Action dim {n_action_dims} exceeds max allowed {self.max_action_dim}."

    #     # Handle action horizon mismatch: pad or truncate
    #     if actual_horizon < self.action_horizon:
    #         # Pad temporal dimension with zeros
    #         pad_length = self.action_horizon - actual_horizon
    #         actions = np.pad(actions, ((0, pad_length), (0, 0)), mode='constant', constant_values=0)
    #         # Create temporal mask (True for real data, False for padding)
    #         temporal_mask = np.concatenate([
    #             np.ones(actual_horizon, dtype=bool),
    #             np.zeros(pad_length, dtype=bool)
    #         ])
    #     elif actual_horizon > self.action_horizon:
    #         # Truncate to target horizon (take first action_horizon steps)
    #         actions = actions[:self.action_horizon]
    #         temporal_mask = np.ones(self.action_horizon, dtype=bool)
    #     else:
    #         temporal_mask = np.ones(self.action_horizon, dtype=bool)

    #     # Pad the action dimension
    #     actions = np.pad(actions, ((0, 0), (0, self.max_action_dim - n_action_dims)), mode='constant', constant_values=0)

    #     # Create mask: [action_horizon, max_action_dim]
    #     # True only for valid (temporal, dimension) pairs
    #     actions_mask = np.zeros((self.action_horizon, self.max_action_dim), dtype=bool)
    #     actions_mask[:, :n_action_dims] = True
    #     # Apply temporal mask: set padded timesteps to False
    #     actions_mask = actions_mask & temporal_mask[:, np.newaxis]
    #     # print('actions masks',actions_mask.shape)
    #     n_action_tokens = self.action_horizon
    #     return actions, actions_mask, n_action_tokens

    def _prepare_action(self, data: dict):
        """
        Pad to max_action_dim, return masks.
        """
        if "action" not in data:
            actions = np.zeros((self.action_horizon, self.max_action_dim))
            actions_mask = np.zeros((self.action_horizon, self.max_action_dim), dtype=bool)
            n_action_tokens = self.action_horizon
            return actions, actions_mask, n_action_tokens

        actions = data["action"]

        assert actions.shape[0] == self.action_horizon, f"{actions.shape=}, {self.action_horizon=}"

        # Canonical-aligned upstream already produced [T, max_action_dim] with a
        # correct binary mask. Trust it instead of re-padding / re-masking.
        if actions.shape[-1] == self.max_action_dim and "action_mask" in data:
            return actions, data["action_mask"], actions.shape[0]

        n_action_tokens = actions.shape[0]  # T
        n_action_dims = actions.shape[1]

        assert (
            n_action_dims <= self.max_action_dim
        ), f"Action dim {n_action_dims} exceeds max allowed {self.max_action_dim}."

        # Pad the channel dimension
        actions = np.pad(actions, ((0, 0), (0, self.max_action_dim - n_action_dims)), "constant")

        # Create mask: [T, max_action_dim]
        actions_mask = np.zeros((n_action_tokens, self.max_action_dim), dtype=bool)
        actions_mask[:, :n_action_dims] = True

        return actions, actions_mask, n_action_tokens

    def apply_single(self, data: dict) -> dict:
        # Keep all original data and only update/add new fields
        transformed_data = data.copy()

        # # 1) Prepare video and language with vlm processing.
        # images = self._prepare_video(data)
        # images = images.astype(np.uint8)
        # language = self._prepare_language(data)
        # batch_data = {"images": images, "language": language}
        # vlm_outputs = self._apply_vlm_processing(batch_data)

        # Initialize vlm_outputs as empty dict since VLM processing is commented out
        vlm_outputs = {}

        # 2) Prepare state and update/add to transformed_data
        state, state_mask, _ = self._prepare_state(data)
        transformed_data["state"] = state
        transformed_data["state_mask"] = state_mask

        if self.training:
            # 3) Prepare actions and add to transformed_data
            transformed_data["segmentation_target"] = np.zeros((2,))
            transformed_data["segmentation_target_mask"] = np.zeros((1,))
            transformed_data["has_real_action"] = np.ones((), dtype=bool)
            actions, actions_mask, _ = self._prepare_action(data)
            transformed_data["action"] = actions
            transformed_data["action_mask"] = actions_mask

        # Add VLM outputs to transformed data (currently empty)
        for k, v in vlm_outputs.items():
            assert k not in transformed_data, f"Key {k} already exists in transformed_data."
            transformed_data[k] = v

        transformed_data["embodiment_id"] = self.get_embodiment_tag()

        if self.training:
            action_and_mask_keys = ["action", "action_mask"]
            assert all(
                transformed_data[key].shape == transformed_data["action"].shape for key in action_and_mask_keys
            ), f"Shape mismatch: {[(key, transformed_data[key].shape) for key in action_and_mask_keys]}"

        return transformed_data

    #####训练的时候不会调用这个batch方法，只有在推理的时候会调用#######

    def apply_batch(self, data: dict, batch_size: int) -> dict:
        # Split on batch dimension.
        data_split = [tree.map_structure(lambda x: x[i], data) for i in range(batch_size)]
        # Process each element.
        data_split_processed = [self.apply_single(elem) for elem in data_split]

        # import pdb; pdb.set_trace()
        return collate(data_split_processed)

    def apply(self, data: dict) -> dict:
        is_batched, batch_size = self.check_keys_and_batch_size(data)
        if is_batched:
            return self.apply_batch(data, batch_size)
        else:
            return self.apply_single(data)

    def unapply(self, data: dict) -> dict:
        # Leave as is so that ConcatTransform can split the values
        return data

    def __call__(self, data: dict) -> dict:
        return self.apply(data)
