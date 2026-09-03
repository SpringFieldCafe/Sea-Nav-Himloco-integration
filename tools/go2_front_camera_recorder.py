#!/usr/bin/env python3
"""Save Go2's head front-camera ROS stream as an MP4 using OpenCV."""

import argparse
import re
import sys
import subprocess
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from unitree_go.msg import Go2FrontVideoData


def parse_size(value):
    match = re.fullmatch(r"(\d+)x(\d+)", value)
    if not match:
        raise argparse.ArgumentTypeError("size must be WIDTHxHEIGHT")
    return int(match.group(1)), int(match.group(2))


class FrontCameraRecorder(Node):
    def __init__(self, topic, output, fps, size, startup_timeout):
        super().__init__("go2_front_camera_recorder")
        self.output = output
        self.fps = fps
        self.width, self.height = size
        self.writer = None
        self.ffmpeg = None
        self.mode = None
        self.frames = 0
        self.started_at = time.monotonic()
        self.startup_timeout = startup_timeout
        self.subscription = self.create_subscription(
            Go2FrontVideoData, topic, self.on_frame, 10
        )
        self.timeout_timer = self.create_timer(0.5, self.check_startup)
        self.get_logger().info(f"recording {topic} to {output}")

    def check_startup(self):
        if self.frames == 0 and time.monotonic() - self.started_at > self.startup_timeout:
            self.get_logger().error("no frames received before startup timeout")
            rclpy.shutdown()

    def on_frame(self, message):
        encoded = bytes(message.video720p or message.video360p or message.video180p)
        if not encoded:
            return
        if self.mode is None:
            self.mode = "opencv" if encoded.startswith(b"\xff\xd8") else "h264"
        if self.mode == "h264":
            self.write_h264(encoded)
            return
        frame = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            self.get_logger().warning("OpenCV could not decode a JPEG camera frame")
            return
        if (frame.shape[1], frame.shape[0]) != (self.width, self.height):
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if self.writer is None:
            self.writer = cv2.VideoWriter(
                self.output, cv2.VideoWriter_fourcc(*"mp4v"), self.fps,
                (self.width, self.height),
            )
            if not self.writer.isOpened():
                self.get_logger().error("OpenCV could not open output video")
                rclpy.shutdown()
                return
            self.get_logger().info("first frame received; MP4 recording started")
        self.writer.write(frame)
        self.frames += 1

    def write_h264(self, encoded):
        if self.ffmpeg is None:
            self.ffmpeg = subprocess.Popen(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error",
                    "-f", "h264", "-framerate", str(self.fps), "-i", "pipe:0",
                    "-an", "-c:v", "libx264", "-preset", "ultrafast",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", self.output,
                ],
                stdin=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            self.get_logger().info("first encoded frame received; H264 recording started")
        if self.ffmpeg.poll() is not None:
            self.get_logger().error("FFmpeg exited while recording H264 stream")
            rclpy.shutdown()
            return
        try:
            self.ffmpeg.stdin.write(encoded)
            self.ffmpeg.stdin.flush()
        except (BrokenPipeError, OSError):
            self.get_logger().error("FFmpeg input pipe closed while recording")
            rclpy.shutdown()
            return
        self.frames += 1

    def close(self):
        if self.writer is not None:
            self.writer.release()
        if self.ffmpeg is not None:
            if self.ffmpeg.stdin is not None:
                self.ffmpeg.stdin.close()
            try:
                self.ffmpeg.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.ffmpeg.kill()
                self.ffmpeg.wait()
        self.get_logger().info(f"saved {self.frames} encoded frames to {self.output}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="/frontvideostream")
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--size", type=parse_size, default=(1280, 720))
    parser.add_argument("--startup-timeout", type=float, default=10.0)
    args = parser.parse_args(argv)
    if args.fps <= 0 or args.startup_timeout <= 0:
        parser.error("fps and startup-timeout must be positive")
    rclpy.init()
    recorder = FrontCameraRecorder(
        args.topic, args.output, args.fps, args.size, args.startup_timeout
    )
    try:
        rclpy.spin(recorder)
    except KeyboardInterrupt:
        pass
    finally:
        recorder.close()
        recorder.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0 if recorder.frames else 2


if __name__ == "__main__":
    sys.exit(main())
