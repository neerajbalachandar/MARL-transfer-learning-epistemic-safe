#!/usr/bin/env python3
"""Run BenchMARL with a small video writer adapter for installed torchvision."""

import sys
from pathlib import Path

import av
import numpy as np
import torchvision


def write_video(path, frames, fps=30, **_kwargs):
    """Write TorchRL's [time, height, width, channels] video array with PyAV."""
    frames = np.asarray(frames)
    with av.open(str(path), mode="w") as output:
        stream = output.add_stream("mpeg4", rate=fps)
        for frame in frames:
            video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
            for packet in stream.encode(video_frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


def main():
    default_root = Path(__file__).resolve().parents[2] / "BenchMARL"
    benchmarl_root = Path(sys.argv.pop(1) if len(sys.argv) > 1 else default_root).resolve()
    sys.path.insert(0, str(benchmarl_root))

    # TorchRL 0.11's CSV logger expects torchvision.io.write_video, which is
    # absent in this torchvision build. PyAV is already a project dependency.
    if not hasattr(torchvision.io, "write_video"):
        torchvision.io.write_video = write_video

    from benchmarl.run import hydra_experiment

    hydra_experiment()


if __name__ == "__main__":
    main()
