from __future__ import annotations
from typing import NamedTuple

import numpy as np


class Size(NamedTuple):
    width: int
    height: int


class Flock:
    def __init__(self, count: int, size: Size):
        self.count = count
        self.size = size
        self.max_vel = 2.0
        self.max_acc = 0.03
        self.rad = 15.0

        self.vel = np.zeros((count, 2), dtype=np.float32)
        self.pos = np.zeros((count, 2), dtype=np.float32)

        angle = np.random.uniform(0, 2 * np.pi, count)
        self.vel[:, 0], self.vel[:, 1] = np.cos(angle), np.sin(angle)
        self.pos[:, 0], self.pos[:, 1] = self.size.width / 2, self.size.height / 2

    def run(self):
        vel, pos = self.vel, self.pos
        max_vel, max_acc = self.max_vel, self.max_acc
        n = self.count

        dx = np.subtract.outer(pos[:, 0], pos[:, 0])
        dy = np.subtract.outer(pos[:, 1], pos[:, 1])
        dist = np.hypot(dx, dy)
        
        mask = (dist > 0) & (dist < self.rad)
        mask_count = np.maximum(mask.sum(axis=1), 1)

        def compute_norm(x: np.ndarray) -> np.ndarray:
            return np.sqrt((x * x).sum(axis=1)).reshape(n, 1)

        def limit_acc(steer: np.ndarray) -> None:
            norm = compute_norm(steer * steer)
            return np.multiply(steer, max_acc / norm, out=steer, where=norm > max_acc)

        def separate() -> np.ndarray:
            target = np.dstack((dx, dy))
            target = np.divide(target, dist.reshape(n, n, 1) ** 2, out=target, where=dist.reshape(n, n, 1) != 0)
            steer = (target * mask.reshape(n, n, 1)).sum(axis=1) / mask_count.reshape(n, 1)
            norm = compute_norm(steer)
            steer = max_vel * np.divide(steer, norm, out=steer, where=norm != 0) - vel
            return limit_acc(steer)
        
        def align() -> np.ndarray:
            target = np.dot(mask, vel) / mask_count.reshape(n, 1)
            norm = compute_norm(target)
            target = max_vel * np.divide(target, norm, out=target, where=norm != 0)
            steer = target - vel
            return limit_acc(steer)
        
        def cohesion() -> np.ndarray:
            target = np.dot(mask, pos) / mask_count.reshape(n, 1)
            desired = target - pos
            norm = compute_norm(desired)
            desired *= max_vel / norm
            steer = desired - vel
            return limit_acc(steer)
        
        def wrap(pos: np.ndarray) -> np.ndarray:
            pos += (self.size.width, self.size.height)
            pos %= (self.size.width, self.size.height)
            return pos
        
        acc = 1.5 * separate() + 1.0 * align() + 1.0 * cohesion()
        vel += acc
        norm = compute_norm(vel)
        vel = np.multiply(vel, max_vel / norm, out=vel, where=norm > max_vel)
        pos += vel
        pos = wrap(pos)


    

if __name__ == "__main__":
    from matplotlib.animation import FuncAnimation

    import numpy as np
    import matplotlib.pyplot as plt


    n = 1_000
    flock = Flock(n, Size(width=640, height=360))
    P = np.zeros((n, 2))

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], frameon=True)
    scatter = ax.scatter(P[:, 0], P[:, 1], s=30, facecolor="red", edgecolor="None", alpha=0.5)

    def update(*args):
        flock.run()
        scatter.set_offsets(flock.pos)

    animation = FuncAnimation(fig, update, interval=10)
    ax.set_xlim(0, 640)
    ax.set_ylim(0, 360)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()