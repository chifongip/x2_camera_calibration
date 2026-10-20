"""Select grid order by confidence, independently of mounting accuracy.

The operator confirms the initial correspondence. Its approximate pose only
selects grid permutations; translation disagreement is diagnostic rather than
an accuracy prerequisite for collecting calibration data.
"""

from collections import deque

import numpy as np

from .geometry import board_pose, difference, inverse

PERMUTATIONS = ('identity', 'rotate_180', 'flip_rows', 'flip_columns')
_BITS = (0, 3, 2, 1)


def variants(points, rows=6, columns=8):
    grid = np.asarray(points, dtype=float).reshape(rows, columns, 2)
    return [
        grid.reshape(-1, 2),
        grid[::-1, ::-1].reshape(-1, 2),
        grid[::-1].reshape(-1, 2),
        grid[:, ::-1].reshape(-1, 2),
    ]


def compose_permutation(first, second):
    return _BITS.index(_BITS[first] ^ _BITS[second])


class CornerTracker:
    """Track all four rectangular-grid orderings across detector changes."""

    def __init__(self, rows=6, columns=8):
        self.rows, self.columns = rows, columns
        self.previous = None
        self.stamp = None
        self.permutation = 0

    def reset(self):
        self.previous = None
        self.stamp = None

    def update(self, corners, stamp):
        points = np.asarray(corners, dtype=float).reshape(-1, 2)
        if not np.isfinite(points).all() or not np.isfinite(stamp):
            raise ValueError('Non-finite checkerboard corners or timestamp')
        span = float(np.linalg.norm(np.ptp(points, axis=0)))
        if span < 10:
            raise ValueError('Checkerboard is too small to establish ordering')
        choices = variants(points, self.rows, self.columns)
        reseed = (
            self.previous is None
            or stamp < self.stamp
            or stamp - self.stamp > 1.0
        )
        if reseed:
            choice = min(range(4), key=lambda i: tuple(choices[i][0][::-1]))
        else:
            scores = [
                float(np.sqrt(np.mean((p - self.previous) ** 2)))
                for p in choices
            ]
            ranked = np.argsort(scores)
            if scores[ranked[1]] - scores[ranked[0]] < 0.1 * span:
                self.reset()
                raise ValueError(
                    'Corner tracking uncertain; hold the board still'
                )
            choice = int(ranked[0])
        ordered = choices[choice].copy()
        self.previous, self.stamp, self.permutation = ordered, stamp, choice
        return ordered, choice == 1, reseed


class BoardSettling:
    """Require three distinct pairs spanning 0.3 seconds of quiet corners."""

    def __init__(self):
        self.history = deque()
        self.ready = False
        self.reason = 'Board settling: hold still for 0.3 seconds'

    def reset(self):
        self.history.clear()
        self.ready = False
        self.reason = 'Board settling: hold still for 0.3 seconds'

    def update(self, corners, pair):
        if self.history and pair == self.history[-1][0]:
            return
        stamp = min(pair)
        if self.history and (
            stamp - min(self.history[-1][0]) > 1.0
            or stamp < min(self.history[-1][0])
        ):
            self.reset()
        moving = None
        for _, previous in self.history:
            for name, current in corners.items():
                span = np.linalg.norm(np.ptp(current, axis=0))
                motion = float(
                    np.sqrt(
                        np.mean(
                            np.sum((current - previous[name]) ** 2, axis=1)
                        )
                    )
                )
                if motion > max(2.0, 0.01 * span):
                    moving = f'Board moving in {name} image ({motion:.1f}px); hold still'
                    break
            if moving:
                break
        if moving:
            self.reset()
        self.history.append(
            (pair, {name: p.copy() for name, p in corners.items()})
        )
        while len(self.history) > 2 and min(self.history[1][0]) < stamp - 0.3:
            self.history.popleft()
        elapsed = stamp - min(self.history[0][0])
        self.ready = len(self.history) >= 3 and elapsed >= 0.3 - 1e-6
        self.reason = (
            moving
            or f'Board settling: {elapsed:.1f}/0.3s, {len(self.history)} pairs'
        )


class PairOrdering:
    """Select corner identity by orientation separation, then check settling."""

    def __init__(self, objects):
        self.objects = objects
        self.columns = len(np.unique(objects[:, 0]))
        self.rows = len(np.unique(objects[:, 1]))
        self.trackers = {
            name: CornerTracker(self.rows, self.columns)
            for name in ('head', 'odin')
        }
        self.anchor = None
        self.pending = None
        self.last_pair = None
        self.streak = 0
        self.settling = BoardSettling()

    def lost(self):
        for tracker in self.trackers.values():
            tracker.reset()
        self.pending = None
        self.last_pair = None
        self.streak = 0
        self.settling.reset()

    def clear_confirmation(self):
        self.anchor = None
        self.pending = None
        self.streak = 0
        self.last_pair = None
        self.settling.reset()

    def confirm(self):
        if self.pending is None:
            raise ValueError('Wait for both complete boards before confirming')
        self.anchor = self.pending.copy()
        self.streak = 0
        self.last_pair = None
        self.settling.reset()

    def stabilize(self, corners, stamps):
        self.pending = None
        # A brief missed detection must not erase the confirmation or history.
        for name in ('head', 'odin'):
            if corners.get(name) is None:
                raise ValueError(f'{name}: complete checkerboard not detected')
        ordered, reversed_flags = {}, {}
        for name in ('head', 'odin'):
            try:
                ordered[name], reversed_flags[name], reseed = self.trackers[
                    name
                ].update(corners[name], stamps[name])
            except ValueError:
                self.streak = 0
                self.settling.reset()
                raise
            if reseed:
                self.streak = 0
                self.settling.reset()
        return ordered, reversed_flags

    def match(
        self, corners, models, torso_from_head, pair, reverse_odin=False
    ):
        self.pending = None
        head_board = board_pose(models['head'], corners['head'], self.objects)
        candidates = []
        choices = variants(corners['odin'], self.rows, self.columns)
        for points in choices:
            odin_board = board_pose(models['odin'], points, self.objects)
            candidate = torso_from_head @ head_board @ inverse(odin_board)
            translation, angle = (
                (0.0, 0.0)
                if self.anchor is None
                else difference(self.anchor, candidate)
            )
            candidates.append((candidate, translation, angle))
        margin = None
        if self.anchor is None:
            choice = int(reverse_odin)
        else:
            ranked = sorted(range(4), key=lambda i: candidates[i][2])
            choice = ranked[0]
            margin = candidates[ranked[1]][2] - candidates[choice][2]
            if margin < 45.0:
                self.streak = 0
                self.settling.reset()
                raise ValueError(
                    f'Corner-order confidence low (separation {margin:.1f}deg < 45deg); '
                    'hold still or use R/C to recheck correspondence'
                )
        self.pending = candidates[choice][0]
        corners['odin'] = choices[choice].copy()
        if self.last_pair != pair:
            if self.last_pair and min(pair) - min(self.last_pair) > 1.0:
                self.streak = 0
            self.streak += 1
            self.last_pair = pair
        self.settling.update(corners, pair)
        head_perm = self.trackers['head'].permutation
        odin_perm = compose_permutation(
            self.trackers['odin'].permutation, choice
        )
        return choice, {
            'stable_pairs': self.streak,
            'head_corner_permutation': PERMUTATIONS[head_perm],
            'odin_corner_permutation': PERMUTATIONS[odin_perm],
            'order_separation_deg': margin,
            'reference_translation_delta_m': candidates[choice][1],
            'reference_rotation_delta_deg': candidates[choice][2],
            'board_settled': self.settling.ready,
        }

    @property
    def ready(self):
        return (
            self.pending is not None
            and self.streak >= 3
            and self.settling.ready
        )

    @property
    def wait_reason(self):
        return self.settling.reason
