"""Bounded history shared by the playback loop and the viewer's HTTP threads."""

from collections import OrderedDict, deque
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensordict import TensorDict


type DecisionCheckpoint = tuple["TensorDict", "TensorDict", "TensorDict"]


@dataclass(frozen=True)
class TimelineFrame:
    branch: int
    tick: int
    poses: bytes
    checkpoint: DecisionCheckpoint | None
    size: int
    episode_split: bool


class Timeline:
    def __init__(self, max_frames: int, max_bytes: int, control_dt: float) -> None:
        self.max_frames = max_frames
        self.max_bytes = max_bytes
        self.control_dt = control_dt
        self.live_branch = 0
        self._frames: OrderedDict[int, TimelineFrame] = OrderedDict()
        self._branches: dict[int, deque[int]] = {}
        self._bytes = 0
        self._lock = Lock()
        self._pending: TimelineFrame | None = None

    def append(
        self, frame_id: int, tick: int, poses: bytes, checkpoint: DecisionCheckpoint | None,
        *, episode_split: bool = False,
    ) -> None:
        size = len(poses)
        if checkpoint is not None:
            size += sum(
                value.numel() * value.element_size()
                for state in checkpoint
                for value in state.values(include_nested=True, leaves_only=True)
            )
        if size > self.max_bytes:
            raise ValueError("A timeline frame exceeds history_mib. Increase the memory budget.")
        with self._lock:
            branch = self.live_branch
            self._frames[frame_id] = TimelineFrame(
                branch, tick, poses, checkpoint, size, episode_split
            )
            self._branches.setdefault(branch, deque()).append(frame_id)
            self._bytes += size
            while len(self._frames) > self.max_frames or self._bytes > self.max_bytes:
                _, evicted = self._frames.popitem(last=False)
                self._bytes -= evicted.size
                ids = self._branches[evicted.branch]
                ids.popleft()
                if not ids:
                    del self._branches[evicted.branch]

    def index(self):
        with self._lock:
            return {
                "liveBranch": self.live_branch,
                "controlDt": self.control_dt,
                "branches": [
                    {
                        "id": branch,
                        "first": ids[0],
                        "last": ids[-1],
                        "firstTick": self._frames[ids[0]].tick,
                        "lastTick": self._frames[ids[-1]].tick,
                        "splits": [
                            {"id": frame_id, "tick": frame.tick}
                            for frame_id in ids
                            if (frame := self._frames[frame_id]).episode_split
                        ],
                    }
                    for branch, ids in self._branches.items()
                ],
            }

    def frame(self, frame_id: int) -> TimelineFrame:
        with self._lock:
            return self._frames[frame_id]

    def request_branch(self, frame_id: int) -> None:
        with self._lock:
            frame = self._frames[frame_id]
            if frame.checkpoint is None:
                raise ValueError("Choose a decision frame to branch")
            if self._pending is not None:
                raise ValueError("A branch is already queued")
            self._pending = frame

    def take_branch(self) -> TimelineFrame | None:
        with self._lock:
            frame = self._pending
            self._pending = None
            if frame is not None:
                self.live_branch += 1
            return frame
