"""Capture admission and reproducible, versioned dataset files."""

from pathlib import Path
import hashlib
import json

import numpy as np
import yaml

from .geometry import rigid

HEAD_JOINTS = ('head_yaw_joint', 'head_pitch_joint')
FACTORY_GEOMETRY_SOURCE = (
    'https://manifoldtechltd.github.io/wiki/en/odin_series/odin1/'
    '5.%20Data%20output_.html#52-extrinsic-description'
)



def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(plain(value), sort_keys=True).encode()
    ).hexdigest()


def load_yaml(path):
    with Path(path).open(encoding='utf-8') as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f'{path}: expected a YAML mapping')
    return value


def write_yaml(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        yaml.safe_dump(plain(value), sort_keys=False), encoding='utf-8'
    )
    temporary.replace(path)


def stable_head(samples, stamp, max_gap=0.1, window=0.5, spread_deg=0.2):
    """Interpolate state only with close bracketing data and a stable history."""
    ordered = sorted(samples, key=lambda sample: sample['stamp'])
    before = [s for s in ordered if s['stamp'] <= stamp]
    after = [s for s in ordered if s['stamp'] >= stamp]
    if (
        not before
        or not after
        or stamp - before[-1]['stamp'] > max_gap
        or after[0]['stamp'] - stamp > max_gap
    ):
        raise ValueError(
            'Waiting for head state bracketing the image within 100 ms'
        )
    history = [
        s for s in ordered if stamp - window - max_gap <= s['stamp'] <= stamp
    ]
    if len(history) < 2 or history[0]['stamp'] > stamp - window:
        raise ValueError('Waiting for 0.5 seconds of head stability history')
    angles = np.array(
        [[s['positions'][name] for name in HEAD_JOINTS] for s in history]
    )
    if not np.isfinite(angles).all() or np.max(
        np.ptp(angles, axis=0)
    ) > np.deg2rad(spread_deg):
        raise ValueError(
            'Head moved more than 0.2 degrees; hold it stationary'
        )
    left, right = before[-1], after[0]
    duration = right['stamp'] - left['stamp']
    fraction = (stamp - left['stamp']) / duration if duration > 0 else 0
    return {
        name: float(
            left['positions'][name] * (1 - fraction)
            + right['positions'][name] * fraction
        )
        for name in HEAD_JOINTS
    }


def factory_extrinsics(calibration):
    """Use device Tcl_0 and documented fixed IMU-LiDAR hardware geometry."""
    if 'Tcl_0' not in calibration:
        raise ValueError('Odin factory calibration must contain Tcl_0')
    values = np.asarray(calibration['Tcl_0'], dtype=float)
    if values.size != 16:
        raise ValueError('Tcl_0 must contain 16 values')
    camera_from_lidar = rigid(values.reshape(4, 4)).copy()
    # Factory YAML rounds rotation entries; URDF requires an SO(3) rotation.
    u, _, vt = np.linalg.svd(camera_from_lidar[:3, :3])
    camera_from_lidar[:3, :3] = u @ vt
    imu_from_lidar = np.eye(4)
    imu_from_lidar[:3, 3] = [-0.02663, 0.03447, 0.02174]
    return {
        'T_camera_lidar': camera_from_lidar.tolist(),
        'T_imu_lidar': imu_from_lidar.tolist(),
        'source': FACTORY_GEOMETRY_SOURCE,
        'camera_source': 'odin_calib.yaml:Tcl_0',
        'rotation_normalization': 'SVD projection onto SO(3)',
    }


def consistent(reference, current, tolerance=1e-7):
    if isinstance(reference, dict):
        return (
            isinstance(current, dict)
            and reference.keys() == current.keys()
            and all(
                consistent(reference[k], current[k], tolerance)
                for k in reference
            )
        )
    if isinstance(reference, (list, tuple, int, float)):
        try:
            return np.allclose(
                reference, current, atol=tolerance, rtol=tolerance
            )
        except (TypeError, ValueError):
            return reference == current
    return reference == current


def camera_models_match(reference, current):
    """Compare fixed camera intrinsics independently of legacy diagnostics."""
    return all(
        consistent(reference[name], current[name]) for name in ('head', 'odin')
    )


def load_session(directory):
    directory = Path(directory)
    metadata = load_yaml(directory / 'session.yaml')
    if metadata.get('version') != 1:
        raise ValueError('Unsupported session version')
    if not metadata.get('samples'):
        raise ValueError('No observations were captured in this session')
    urdf = (directory / 'robot.urdf').read_text(encoding='utf-8')
    if fingerprint(urdf) != metadata['urdf_sha256']:
        raise ValueError('URDF snapshot does not match session fingerprint')
    identifiers = [sample['id'] for sample in metadata['samples']]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError('Duplicate observation IDs')
    for sample in metadata['samples']:
        if sample['calibration_sha256'] != metadata['calibration_sha256']:
            raise ValueError('Observation calibration fingerprint mismatch')
        if sample['role'] not in ('train', 'validation'):
            raise ValueError('Invalid observation role')
        stamps = sample['stamps']
        if (
            not np.isfinite([stamps['head'], stamps['odin']]).all()
            or min(stamps['head'], stamps['odin']) <= 0
            or abs(stamps['head'] - stamps['odin'])
            > metadata['max_pair_delta_sec'] + 1e-9
        ):
            raise ValueError('Observation exceeds image pairing tolerance')
        for name in HEAD_JOINTS:
            if not np.isfinite(sample['head_positions'][name]):
                raise ValueError('Non-finite head position')
        if 'internal' in sample:
            for name in ('T_camera_lidar', 'T_imu_lidar'):
                rigid(sample['internal'][name])
    if fingerprint(metadata['calibration']) != metadata['calibration_sha256']:
        raise ValueError('Calibration snapshot fingerprint mismatch')
    # Legacy sessions retain their original fingerprints and diagnostic matrices.
    # Always derive physical geometry from the saved device calibration file.
    factory_file = load_yaml(directory / 'odin_calib.yaml')
    factory = factory_extrinsics(factory_file)
    if 'factory_internal' in metadata and not consistent(
        metadata['factory_internal'], factory
    ):
        raise ValueError('Factory geometry differs from odin_calib.yaml snapshot')
    if not consistent(
        metadata['calibration']['odin'],
        factory_file['cam_0'],
    ):
        raise ValueError('Factory camera intrinsics differ from session snapshot')
    metadata['factory_internal'] = factory
    return metadata, urdf
