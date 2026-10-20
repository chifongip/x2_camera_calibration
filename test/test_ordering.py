from copy import deepcopy

import numpy as np
import pytest
import cv2

from x2_camera_calibration.geometry import (
    HeadKinematics,
    board_points,
    difference,
    inverse,
    pose,
)
from x2_camera_calibration.ordering import (
    BoardSettling,
    CornerTracker,
    PairOrdering,
    variants,
)
from test_solver import dataset as _dataset, URDF

dataset = _dataset


def test_alternating_detector_reversal_keeps_same_visual_origin():
    tracker = CornerTracker()
    points = board_points()[:, :2] * 500 + [150.0, 100.0]
    first, _, _ = tracker.update(points, 1.0)
    for index in range(1, 20):
        raw = points[::-1] if index % 2 else points
        ordered, reverse, _ = tracker.update(raw, 1.0 + index * 0.1)
        np.testing.assert_allclose(ordered, first)
        assert reverse == bool(index % 2)


def test_ambiguous_quarter_turn_blocks_ordering():
    tracker = CornerTracker()
    points = board_points()[:, :2] * 500 + [150.0, 100.0]
    tracker.update(points, 1.0)
    center = points.mean(axis=0)
    turned = (points - center) @ np.array([[0.0, 1.0], [-1.0, 0.0]]) + center
    with pytest.raises(ValueError, match='uncertain'):
        tracker.update(turned, 1.1)


def _step(guard, meta, sample, index):
    raw = {
        name: np.array(sample[f'{name}_corners']) for name in ('head', 'odin')
    }
    if index % 2:
        raw['head'] = raw['head'][::-1]
    if index % 3:
        raw['odin'] = raw['odin'][::-1]
    pair = (100.0 + index * 0.1, 100.0 + index * 0.1)
    ordered, flags = guard.stabilize(raw, dict(zip(('head', 'odin'), pair)))
    extra, details = guard.match(
        ordered,
        meta['calibration'],
        HeadKinematics(URDF).transform(sample['head_positions']),
        pair,
    )
    flags['odin'] ^= extra
    return ordered, flags, details


def test_cross_camera_reversals_and_head_motion_preserve_reference(dataset):
    meta, _, _ = dataset
    guard = PairOrdering(board_points())
    initial = deepcopy(meta['samples'][0])
    # Align using operator knowledge of the initial numbered board.
    ordered, _, _ = _step(guard, meta, initial, 0)
    head_reference = ordered['head'].copy()
    odin_reference = ordered['odin'].copy()
    guard.confirm()
    for index in range(1, 10):
        ordered, _, _ = _step(guard, meta, initial, index)
        np.testing.assert_allclose(ordered['head'], head_reference)
        np.testing.assert_allclose(ordered['odin'], odin_reference)
    assert guard.ready
    reference = guard.anchor.copy()
    guard.lost()
    assert not guard.ready
    # A held-out board and head posture must not require a new camera anchor.
    for index in range(10, 14):
        _step(guard, meta, meta['samples'][32], index)
    assert guard.ready
    assert difference(reference, guard.pending)[0] < 0.01
    assert difference(reference, guard.pending)[1] < 1.0


def test_approximate_initial_pose_does_not_block_corner_collection(dataset):
    meta, _, _ = dataset
    guard = PairOrdering(board_points())
    sample = deepcopy(meta['samples'][0])
    _step(guard, meta, sample, 0)
    guard.confirm()
    # Planar PnP uncertainty must not impose a 50 mm / 5 degree capture cutoff.
    guard.anchor = guard.anchor @ pose([0.0, 0.14, 0.0], [0.1, 0.0, 0.0])
    for index in range(1, 6):
        _, _, details = _step(guard, meta, sample, index)
    assert guard.ready
    assert details['reference_translation_delta_m'] > 0.05
    assert details['reference_rotation_delta_deg'] > 5.0
    assert details['order_separation_deg'] > 45.0


def test_repeated_frame_does_not_satisfy_three_pair_gate(dataset):
    meta, _, _ = dataset
    guard = PairOrdering(board_points())
    _step(guard, meta, meta['samples'][0], 0)
    guard.confirm()
    for _ in range(5):
        _step(guard, meta, meta['samples'][0], 1)
    assert not guard.ready
    assert guard.streak == 1


def test_missing_board_blocks_new_confirmation(dataset):
    meta, _, _ = dataset
    guard = PairOrdering(board_points())
    _step(guard, meta, meta['samples'][0], 0)
    with pytest.raises(ValueError, match='not detected'):
        guard.stabilize(
            {'head': None, 'odin': None}, {'head': 1.0, 'odin': 1.0}
        )
    assert guard.pending is None
    with pytest.raises(ValueError, match='complete boards'):
        guard.confirm()


def test_all_grid_permutations_keep_the_same_corner_identity():
    tracker = CornerTracker()
    points = board_points()[:, :2] * 500 + [150.0, 100.0]
    reference, _, _ = tracker.update(points, 1.0)
    for index in range(1, 13):
        raw = variants(points)[index % 4]
        ordered, _, _ = tracker.update(raw, 1.0 + 0.1 * index)
        np.testing.assert_allclose(reference, ordered)


def test_low_separation_between_order_hypotheses_blocks_capture(dataset):
    meta, _, _ = dataset
    guard = PairOrdering(board_points())
    sample = meta['samples'][0]
    ordered, _, _ = _step(guard, meta, sample, 0)
    direct = guard.pending.copy()
    guard.match(
        deepcopy(ordered),
        meta['calibration'],
        HeadKinematics(URDF).transform(sample['head_positions']),
        (100.1, 100.1),
        reverse_odin=1,
    )
    other = guard.pending.copy()
    axis = cv2.Rodrigues((inverse(direct) @ other)[:3, :3])[0].reshape(3)
    guard.anchor = direct @ pose(axis / 2, [0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match='confidence low'):
        _step(guard, meta, sample, 2)
    assert not guard.ready


def test_board_movement_blocks_capture_then_recovers_when_held():
    gate = BoardSettling()
    points = board_points()[:, :2] * 500 + [150.0, 100.0]
    for index in range(5):
        gate.update({'head': points, 'odin': points}, (1.0 + 0.1 * index,) * 2)
    assert gate.ready
    moved = points + [20.0, 5.0]
    gate.update({'head': moved, 'odin': moved}, (1.5, 1.5))
    assert not gate.ready
    assert 'Board moving' in gate.reason
    for index in range(6, 10):
        gate.update({'head': moved, 'odin': moved}, (1.0 + 0.1 * index,) * 2)
    assert gate.ready


def test_brief_detection_dropout_does_not_restart_confirmation(dataset):
    meta, _, _ = dataset
    guard = PairOrdering(board_points())
    sample = meta['samples'][0]
    _step(guard, meta, sample, 0)
    guard.confirm()
    for index in range(1, 6):
        _step(guard, meta, sample, index)
    assert guard.ready
    anchor = guard.anchor.copy()
    with pytest.raises(ValueError, match='not detected'):
        guard.stabilize(
            {'head': None, 'odin': None}, {'head': 100.6, 'odin': 100.6}
        )
    assert not guard.ready
    _step(guard, meta, sample, 7)
    assert guard.ready
    np.testing.assert_array_equal(anchor, guard.anchor)
