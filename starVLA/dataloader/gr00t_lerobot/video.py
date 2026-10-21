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


import warnings

import av
import cv2
import numpy as np

import torch  # noqa: F401 # isort: skip
import torchvision  # noqa: F401 # isort: skip

# Suppress torchvision video deprecation warning
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.io._video_deprecation_warning")

# Import decord with graceful fallback
try:
    import decord  # noqa: F401

    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False

try:
    import torchcodec

    TORCHCODEC_AVAILABLE = True
except (ImportError, RuntimeError):
    TORCHCODEC_AVAILABLE = False


def get_frames_by_indices(
    video_path: str,
    indices: list[int] | np.ndarray,
    video_backend: str = "decord",
    video_backend_kwargs: dict = {},
) -> np.ndarray:
    if video_backend == "decord":
        if not DECORD_AVAILABLE:
            raise ImportError("decord is not available.")
        vr = decord.VideoReader(video_path, **video_backend_kwargs)
        frames = vr.get_batch(indices)
        return frames.asnumpy()
    elif video_backend == "torchcodec":
        if not TORCHCODEC_AVAILABLE:
            raise ImportError("torchcodec is not available.")
        decoder = torchcodec.decoders.VideoDecoder(
            video_path, device="cpu", dimension_order="NHWC", num_ffmpeg_threads=0
        )
        return decoder.get_frames_at(indices=indices).data.numpy()
    elif video_backend == "opencv":
        frames = []
        cap = cv2.VideoCapture(video_path, **video_backend_kwargs)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"Unable to read frame at index {idx}")
            frames.append(frame)
        cap.release()
        frames = np.array(frames)
        return frames
    elif video_backend == "torchvision_av":
        # Optimized version using PyAV directly for better performance
        container = av.open(video_path)
        video_stream = container.streams.video[0]
        fps = float(video_stream.average_rate)

        # Convert frame indices to timestamps
        timestamps = np.array(indices) / fps

        # Sort indices to optimize seeking
        sorted_order = np.argsort(timestamps)
        sorted_timestamps = timestamps[sorted_order]

        loaded_frames = [None] * len(indices)

        try:
            last_seek_pos = -1
            for i, target_ts in zip(sorted_order, sorted_timestamps):
                # Only seek if we're going backwards or far ahead
                if target_ts < last_seek_pos or target_ts - last_seek_pos > 2.0:
                    # Seek to slightly before target to ensure we don't miss it
                    seek_ts = max(0, target_ts - 0.5)
                    container.seek(int(seek_ts * av.time_base), stream=video_stream)

                closest_frame = None
                closest_ts_diff = float("inf")

                # Read frames until we pass the target timestamp
                for frame in container.decode(video=0):
                    current_ts = frame.pts * video_stream.time_base
                    current_diff = abs(current_ts - target_ts)

                    if current_diff < closest_ts_diff:
                        closest_ts_diff = current_diff
                        closest_frame = frame
                    else:
                        # Timestamps are increasing, we've passed the target
                        break

                    # If we're very close, stop early
                    if current_diff < 0.001:
                        break

                if closest_frame is not None:
                    frame_array = closest_frame.to_ndarray(format="rgb24")
                    loaded_frames[i] = frame_array
                    last_seek_pos = target_ts

        finally:
            container.close()

        # Filter out any None values (shouldn't happen, but safety check)
        loaded_frames = [f for f in loaded_frames if f is not None]

        if len(loaded_frames) == 0:
            raise ValueError(f"No frames could be loaded from {video_path}")

        return np.array(loaded_frames)
    else:
        raise NotImplementedError


def get_frames_by_timestamps(
    video_path: str,
    timestamps: list[float] | np.ndarray,
    video_backend: str = "decord",
    video_backend_kwargs: dict = {},
) -> np.ndarray:
    """Get frames from a video at specified timestamps.
    Args:
        video_path (str): Path to the video file.
        timestamps (list[int] | np.ndarray): Timestamps to retrieve frames for, in seconds.
        video_backend (str, optional): Video backend to use. Defaults to "decord".
    Returns:
        np.ndarray: Frames at the specified timestamps.
    """
    if video_backend == "decord":
        # For some GPUs, AV format data cannot be read
        if not DECORD_AVAILABLE:
            raise ImportError("decord is not available.")
        vr = decord.VideoReader(video_path, **video_backend_kwargs)
        num_frames = len(vr)
        # Retrieve the timestamps for each frame in the video
        frame_ts: np.ndarray = vr.get_frame_timestamp(range(num_frames))
        # Map each requested timestamp to the closest frame index
        # Only take the first element of the frame_ts array which corresponds to start_seconds
        indices = np.abs(frame_ts[:, :1] - timestamps).argmin(axis=0)
        frames = vr.get_batch(indices)
        return frames.asnumpy()
    elif video_backend == "torchcodec":
        if not TORCHCODEC_AVAILABLE:
            raise ImportError("torchcodec is not available.")
        decoder = torchcodec.decoders.VideoDecoder(
            video_path, device="cpu", dimension_order="NHWC", num_ffmpeg_threads=0
        )
        return decoder.get_frames_played_at(seconds=timestamps).data.numpy()
    elif video_backend == "opencv":
        # Open the video file
        cap = cv2.VideoCapture(video_path, **video_backend_kwargs)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")
        # Retrieve the total number of frames
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Calculate timestamps for each frame
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_ts = np.arange(num_frames) / fps
        frame_ts = frame_ts[:, np.newaxis]  # Reshape to (num_frames, 1) for broadcasting
        # Map each requested timestamp to the closest frame index
        indices = np.abs(frame_ts - timestamps).argmin(axis=0)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"Unable to read frame at index {idx}")
            frames.append(frame)
        cap.release()
        frames = np.array(frames)
        return frames
    elif video_backend == "torchvision_av":
        # PyAV 单次 decode 版本：避免反复 decode 导致 dav1d-worker 线程暴涨
        ts = np.asarray(timestamps, dtype=np.float64)
        if ts.ndim != 1:
            ts = ts.reshape(-1)

        order = np.argsort(ts)
        ts_sorted = ts[order]

        container = av.open(video_path, options={"threads": "1"})
        stream = container.streams.video[0]

        # 更强的“钉死线程”措施（不同版本可能有差异，try/except 保守）
        try:
            stream.thread_type = "NONE"
        except Exception:
            pass
        try:
            stream.codec_context.thread_count = 1
        except Exception:
            pass

        out_sorted = [None] * len(ts_sorted)

        try:
            # seek 到最早 timestamp 之前一点
            start_ts = max(0.0, float(ts_sorted[0]) - 0.5)
            try:
                container.seek(
                    int(start_ts / float(stream.time_base)),
                    stream=stream,
                    backward=True,
                    any_frame=False,
                )
            except Exception:
                # seek 不可靠就从头开始
                container.seek(0)

            j = 0
            last_frame_arr = None

            # 一次 decode 顺序扫过去
            for frame in container.decode(video=0):
                if frame.pts is None:
                    continue
                t = float(frame.pts * stream.time_base)

                # 如果还没到下一个目标，继续 decode
                if t + 1e-6 < ts_sorted[j]:
                    last_frame_arr = frame.to_ndarray(format="rgb24")
                    continue

                # 到达/超过目标：把当前帧作为该目标的匹配帧
                frame_arr = frame.to_ndarray(format="rgb24")
                last_frame_arr = frame_arr

                while j < len(ts_sorted) and t + 1e-6 >= ts_sorted[j]:
                    out_sorted[j] = frame_arr
                    j += 1
                    if j >= len(ts_sorted):
                        break

                if j >= len(ts_sorted):
                    break

            # 如果尾部还有没填的，用最后一帧补（或你也可以 raise）
            for k in range(len(out_sorted)):
                if out_sorted[k] is None:
                    out_sorted[k] = last_frame_arr

        finally:
            container.close()

        if out_sorted[0] is None:
            raise ValueError(f"No frames could be loaded from {video_path}")

        frames_sorted = np.stack(out_sorted, axis=0)

        # 还原到原 timestamps 顺序
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        return frames_sorted[inv]

    else:
        raise NotImplementedError


def get_all_frames(
    video_path: str,
    video_backend: str = "decord",
    video_backend_kwargs: dict = {},
    resize_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Get all frames from a video.
    Args:
        video_path (str): Path to the video file.
        video_backend (str, optional): Video backend to use. Defaults to "decord".
        video_backend_kwargs (dict, optional): Keyword arguments for the video backend.
        resize_size (tuple[int, int], optional): Resize size for the frames. Defaults to None.
    """
    if video_backend == "decord":
        if not DECORD_AVAILABLE:
            raise ImportError("decord is not available.")
        vr = decord.VideoReader(video_path, **video_backend_kwargs)
        frames = vr.get_batch(range(len(vr))).asnumpy()
    elif video_backend == "torchcodec":
        if not TORCHCODEC_AVAILABLE:
            raise ImportError("torchcodec is not available.")
        decoder = torchcodec.decoders.VideoDecoder(
            video_path, device="cpu", dimension_order="NHWC", num_ffmpeg_threads=0
        )
        frames = decoder.get_frames_at(indices=range(len(decoder)))
        return frames.data.numpy(), frames.pts_seconds.numpy()
    elif video_backend == "pyav":
        container = av.open(video_path)
        frames = []
        for frame in container.decode(video=0):
            frame = frame.to_ndarray(format="rgb24")
            frames.append(frame)
        frames = np.array(frames)
    elif video_backend == "torchvision_av":
        # 全局 backend 只设一次（避免反复改全局状态）
        if getattr(get_all_frames, "_tv_backend_set", False) is False:
            torchvision.set_video_backend("pyav")
            setattr(get_all_frames, "_tv_backend_set", True)

        reader = torchvision.io.VideoReader(video_path, "video")
        try:
            frames = [f["data"].numpy() for f in reader]
        finally:
            # 不同 torchvision 版本行为不同，尽量显式释放
            if hasattr(reader, "close"):
                try:
                    reader.close()
                except Exception:
                    pass
            del reader

        frames = np.array(frames)
        frames = frames.transpose(0, 2, 3, 1)
    else:
        raise NotImplementedError(f"Video backend {video_backend} not implemented")
    # resize frames if specified
    if resize_size is not None:
        frames = [cv2.resize(frame, resize_size) for frame in frames]
        frames = np.array(frames)
    return frames
