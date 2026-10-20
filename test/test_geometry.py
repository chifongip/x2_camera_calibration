from pathlib import Path

import numpy as np
import pytest

from x2_camera_calibration.geometry import (
    HeadKinematics,
    board_points,
    difference,
    inverse,
    pose,
    project,
    rigid,
    transform_record,
    unproject,
)
from x2_camera_calibration.session import (
    camera_models_match,
    consistent,
    fingerprint,
    load_yaml,
    stable_head,
    write_yaml,
)


@pytest.fixture
def fish():
    return {
        'cam_model': 'FishPoly',
        'image_width': 1600,
        'image_height': 1296,
        'A11': 733.666,
        'A12': -0.01947,
        'A22': 733.38,
        'u0': 816.4,
        'v0': 671.1,
        'k2': -0.006866,
        'k3': 0.018203,
        'k4': -0.08031,
        'k5': 0.08882,
        'k6': -0.05007,
        'k7': 0.009814,
        'p1': 0.0,
        'p2': 0.0,
    }


def test_fishpoly_matches_sdk_equations_and_axis(fish):
    points = np.array([[0.0, 0.0, 1.0], [0.3, -0.4, 1.2], [-0.8, 0.4, 1.0]])
    actual = project(fish, points)
    np.testing.assert_allclose(actual[0], [fish['u0'], fish['v0']])
    for point, pixel in zip(points[1:], actual[1:]):
        radius = np.linalg.norm(point[:2])
        theta = np.arccos(point[2] / np.linalg.norm(point))
        distorted = theta + sum(fish[f'k{i}'] * theta**i for i in range(2, 8))
        x, y = point[:2] * distorted / radius
        np.testing.assert_allclose(
            pixel,
            [
                x * fish['A11'] + y * fish['A12'] + fish['u0'],
                y * fish['A22'] + fish['v0'],
            ],
            atol=1e-10,
        )
    np.testing.assert_allclose(
        unproject(fish, actual), points[:, :2] / points[:, 2:], atol=1e-10
    )


def test_eight_coefficient_head_model():
    head = {
        'K': [1454.0, 0, 1356.0, 0, 1453.0, 1018.0, 0, 0, 1],
        'D': [8.9, 4.1, -0.0000057, 0.0000096, 0.157, 9.32, 7.66, 1.04],
        'model': 'plumb_bob',
    }
    points = np.array([[0.3, 0.4, 1.0], [-0.5, -0.3, 1.0]])
    np.testing.assert_allclose(
        unproject(head, project(head, points)), points[:, :2], atol=1e-8
    )


def test_invalid_calibration_and_behind_camera(fish):
    with pytest.raises(ValueError, match='behind'):
        project(fish, [[1, 0, -1]])
    with pytest.raises(ValueError, match='FishPoly'):
        project({**fish, 'cam_model': 'equidistant'}, [[0, 0, 1]])
    with pytest.raises(ValueError):
        rigid(np.zeros((4, 4)))
    assert not consistent({'K': [1.0, 2.0]}, {'K': [1.0, 3.0]})


def test_transform_composition_and_xyzw():
    transform = pose(np.array([0.2, -0.3, 0.4]), [1, 2, 3])
    np.testing.assert_allclose(
        transform @ inverse(transform), np.eye(4), atol=1e-12
    )
    record = transform_record(transform, 'parent', 'child')
    x, y, z, w = record['quaternion_xyzw']
    recovered = np.array(
        [
            [
                1 - 2 * (y * y + z * z),
                2 * (x * y - z * w),
                2 * (x * z + y * w),
            ],
            [
                2 * (x * y + z * w),
                1 - 2 * (x * x + z * z),
                2 * (y * z - x * w),
            ],
            [
                2 * (x * z - y * w),
                2 * (y * z + x * w),
                1 - 2 * (x * x + y * y),
            ],
        ]
    )
    np.testing.assert_allclose(recovered, transform[:3, :3], atol=1e-12)
    assert difference(transform, transform) == (0.0, 0.0)
    assert board_points().shape == (48, 3)
    np.testing.assert_allclose(board_points()[-1], [0.245, 0.175, 0])


def test_actual_urdf_head_kinematics():
    path = (
        Path(__file__).resolve().parents[2]
        / 'x2_description/urdf/x2_ultra.urdf'
    )
    if not path.exists():
        pytest.skip('Source-workspace URDF regression check')
    kinematics = HeadKinematics(path.read_text())
    zero = kinematics.transform(
        {'head_yaw_joint': 0.0, 'head_pitch_joint': 0.0}
    )
    np.testing.assert_allclose(
        zero[:3, 3],
        [
            0.00834772789983268 + 0.0684,
            -0.00021713,
            0.309397077276121 + 0.0889 + 0.05,
        ],
        atol=1e-12,
    )
    changed = kinematics.transform(
        {'head_yaw_joint': 0.2, 'head_pitch_joint': -0.1}
    )
    assert difference(zero, changed)[1] > 10
    with pytest.raises(KeyError):
        kinematics.transform({})


def test_stationarity_and_timestamp_bracketing():
    samples = [
        {
            'stamp': t / 100,
            'positions': {'head_yaw_joint': 0.1, 'head_pitch_joint': -0.1},
        }
        for t in range(100)
    ]
    assert stable_head(samples, 0.755)['head_yaw_joint'] == pytest.approx(0.1)
    with pytest.raises(ValueError, match='bracketing'):
        stable_head(samples, 1.2)
    with pytest.raises(ValueError, match='history'):
        stable_head(samples, 0.1)
    samples[65]['positions']['head_yaw_joint'] = 0.2
    with pytest.raises(ValueError, match='moved'):
        stable_head(samples, 0.755)


def test_ros_numpy_arrays_serialize_without_changing_fingerprint(tmp_path):
    value = {'K': list(np.array([1454.0, 0.0, 1356.0])), 'array': np.eye(2)}
    write_yaml(tmp_path / 'snapshot.yaml', value)
    restored = load_yaml(tmp_path / 'snapshot.yaml')
    assert fingerprint(value) == fingerprint(restored)


def test_online_extrinsic_changes_do_not_hide_intrinsic_changes():
    reference = {
        'head': {'K': [100.0]},
        'odin': {'A11': 200.0},
        'internal': {'T_imu_lidar': np.eye(4).tolist()},
    }
    current = {
        'head': {'K': [100.0]},
        'odin': {'A11': 200.0},
        'internal': {'T_imu_lidar': pose([0, 0, 0], [0.0004, 0, 0]).tolist()},
    }
    assert camera_models_match(reference, current)
    current['head']['K'][0] += 0.01
    assert not camera_models_match(reference, current)
