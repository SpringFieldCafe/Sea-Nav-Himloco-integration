#!/usr/bin/env python3
"""Record the Go2 head camera through Unitree's official VideoClient."""

import argparse
import subprocess
import sys
import time

import cv2
import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient


class FrontCameraRecorder:
    def __init__(self, net, output, fps, size, timeout):
        self.output = output
        self.fps = fps
        self.width, self.height = size
        self.timeout = timeout
        self.ffmpeg = None
        self.frames = 0
        ChannelFactoryInitialize(0, net)
        self.client = VideoClient()
        self.client.SetTimeout(3.0)
        self.client.Init()

    def start_encoder(self):
        self.ffmpeg = subprocess.Popen(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{self.width}x{self.height}", "-r", str(self.fps),
                "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "ultrafast",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", self.output,
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        print("first frame received; H264 recording started", flush=True)

    def record(self):
        startup_deadline = time.monotonic() + self.timeout
        recovery_deadline = None
        next_frame_at = time.monotonic()
        frame_period = 1.0 / self.fps
        while True:
            code, data = self.client.GetImageSample()
            if code != 0:
                now = time.monotonic()
                if self.frames == 0:
                    if now >= startup_deadline:
                        raise RuntimeError(f"VideoClient.GetImageSample failed: code={code}")
                else:
                    if recovery_deadline is None:
                        recovery_deadline = now + self.timeout
                    if now >= recovery_deadline:
                        raise RuntimeError(
                            f"VideoClient.GetImageSample failed after recovery timeout: code={code}"
                        )
                time.sleep(0.05)
                continue
            recovery_deadline = None
            image = cv2.imdecode(
                np.frombuffer(bytes(data), dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if image is None:
                print("warning: VideoClient returned a non-decodable image", flush=True)
                continue
            if (image.shape[1], image.shape[0]) != (self.width, self.height):
                image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)
            if self.ffmpeg is None:
                self.start_encoder()
            if self.ffmpeg.poll() is not None:
                raise RuntimeError("FFmpeg exited while writing the MP4")
            self.ffmpeg.stdin.write(image.tobytes())
            self.ffmpeg.stdin.flush()
            self.frames += 1
            # VideoClient may return the latest image faster than the requested
            # rate. Pace writes so MP4 duration follows wall-clock time instead
            # of becoming longer because duplicate frames were over-sampled.
            next_frame_at += frame_period
            delay = next_frame_at - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_frame_at = time.monotonic()

    def close(self):
        if self.ffmpeg is not None:
            self.ffmpeg.stdin.close()
            try:
                self.ffmpeg.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.ffmpeg.kill()
                self.ffmpeg.wait()
        print(f"saved {self.frames} frames to {self.output}", flush=True)


def parse_size(value):
    try:
        width, height = (int(part) for part in value.lower().split("x", 1))
    except (ValueError, AttributeError):
        raise argparse.ArgumentTypeError("size must be WIDTHxHEIGHT")
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("size must be positive")
    return width, height


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--net", required=True, help="network interface connected to Go2")
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--size", type=parse_size, default=(1280, 720))
    parser.add_argument("--startup-timeout", type=float, default=12.0)
    args = parser.parse_args(argv)
    recorder = None
    try:
        recorder = FrontCameraRecorder(
            args.net, args.output, args.fps, args.size, args.startup_timeout
        )
        recorder.record()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"front camera recorder error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    finally:
        if recorder is not None:
            recorder.close()
    return 0 if recorder is not None and recorder.frames else 2


if __name__ == "__main__":
    raise SystemExit(main())
