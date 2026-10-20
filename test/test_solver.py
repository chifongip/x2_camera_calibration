import numpy as np
import pytest
import cv2
from copy import deepcopy

from x2_camera_calibration.geometry import (
    HeadKinematics,
    board_points,
    difference,
    inverse,
    pose,
    project,
    transform_points,
)
from x2_camera_calibration.offline import (
    evaluate,
    solve_main,
    solve_session,
    validate_main,
)
from x2_camera_calibration.session import (
    factory_extrinsics, fingerprint, load_session, write_yaml,
)

URDF = '''<robot name="test">
<joint name="head_yaw_joint" type="revolute"><parent link="torso_link"/>
<child link="head_yaw_link"/><origin xyz="0 0 0.3"/><axis xyz="0 0 1"/></joint>
<joint name="head_pitch_joint" type="revolute"><parent link="head_yaw_link"/>
<child link="head_pitch_link"/><origin xyz="0 0 0.08"/><axis xyz="0 1 0"/></joint>
<joint name="camera" type="fixed"><parent link="head_pitch_link"/>
<child link="rgb_head_center"/><origin xyz="0.07 0 0.05" rpy="-1.5708 0 -1.574"/></joint>
</robot>'''


@pytest.fixture
def dataset():
    rng = np.random.default_rng(71)
    head = {
        'K': [1454.0, 0, 1356.0, 0, 1453.0, 1018.0, 0, 0, 1],
        'D': [8.9, 4.1, -0.0000057, 0.0000096, 0.157, 9.32, 7.66, 1.04],
        'model': 'plumb_bob',
        'width': 2688,
        'height': 1944,
        'frame_id': 'rgb_head_front_center',
    }
    odin = {
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
    }
    camera_lidar = pose([-0.2, 0.05, 0.1], [0.04, -0.005, -0.008])
    internal = factory_extrinsics({'Tcl_0': camera_lidar.ravel().tolist()})
    imu_camera = (
        np.array(internal['T_imu_lidar']) @ inverse(camera_lidar)
    )
    meta = {
        'version': 1,
        'factory_internal': internal,
        'board': {'columns': 8, 'rows': 6, 'square_size_m': 0.035},
        'max_pair_delta_sec': 0.03,
        'urdf_sha256': fingerprint(URDF),
        'calibration': {'head': head, 'odin': odin, 'internal': internal},
        'samples': [],
    }
    meta['calibration_sha256'] = fingerprint(meta['calibration'])
    kin = HeadKinematics(URDF)
    objects = board_points()
    camera_transform = pose([0.04, -0.07, 0.03], [0.16, -0.04, 0.08])
    zero_positions = {'head_yaw_joint': 0.0, 'head_pitch_joint': 0.0}
    mounting = (
        kin.transform(zero_positions) @ camera_transform @ inverse(imu_camera)
    )
    for index in range(42):
        role = 'train' if index < 30 else 'validation'
        positions = zero_positions.copy()
        if role == 'validation':
            positions = {
                'head_yaw_joint': [-0.12, 0.0, 0.13][index % 3],
                'head_pitch_joint': [-0.08, 0.06, 0.12][index % 3],
            }
        head_odin = inverse(kin.transform(positions)) @ mounting @ imu_camera
        head_board = pose(
            rng.uniform(-0.45, 0.45, 3),
            [
                rng.uniform(-0.25, 0.1),
                rng.uniform(-0.2, 0.1),
                rng.uniform(0.8, 1.8),
            ],
        )
        odin_board = inverse(head_odin) @ head_board
        head_pixels = project(head, transform_points(head_board, objects))
        odin_pixels = project(odin, transform_points(odin_board, objects))
        # Small corner noise exercises estimation without making thresholds vacuous.
        head_pixels += rng.normal(0, 0.03, head_pixels.shape)
        odin_pixels += rng.normal(0, 0.03, odin_pixels.shape)
        meta['samples'].append(
            {
                'id': f'synthetic_{index}',
                'role': role,
                'head_positions': positions,
                'stamps': {'head': index + 100.0, 'odin': index + 100.01},
                'calibration_sha256': meta['calibration_sha256'],
                'head_corners': head_pixels.tolist(),
                'odin_corners': odin_pixels.tolist(),
                'head_image': f'images/{index}_head.jpg',
                'odin_image': f'images/{index}_odin.jpg',
            }
        )
    return meta, mounting, camera_transform


def test_noisy_metric_recovery_and_independent_validation(dataset):
    meta, expected_mount, expected_camera = dataset
    calibration, report = solve_session(meta, URDF)
    assert report['status'] == 'accepted', report
    assert report['training_pairs'] == 30
    assert report['validation_pairs'] == 12
    translation, angle = difference(
        expected_mount, calibration['T_torso_odin_imu']['matrix']
    )
    assert translation < 0.001
    assert angle < 0.1
    translation, angle = difference(
        expected_camera, calibration['T_head_camera_odin_camera']['matrix']
    )
    assert translation < 0.001
    assert angle < 0.1


def test_wiwc_snapshots_do_not_change_factory_mounting(dataset):
    meta, expected_mount, _ = dataset
    # Neither the session-wide nor per-observation legacy WIWC matrices may
    # affect the physical mounting calculation or independent validation.
    meta['calibration']['internal'] = deepcopy(meta['factory_internal'])
    meta['calibration']['internal']['T_imu_lidar'][0][3] += 0.25
    for index, sample in enumerate(meta['samples']):
        sample['internal'] = deepcopy(meta['calibration']['internal'])
        sample['internal']['T_camera_lidar'][1][3] += index * 0.01
    calibration, report = solve_session(meta, URDF)
    assert report['status'] == 'accepted', report
    translation, angle = difference(
        expected_mount, calibration['T_torso_odin_imu']['matrix']
    )
    assert translation < 0.001
    assert angle < 0.1
    assert calibration['version'] == 2
    assert calibration['factory_internal'] == meta['factory_internal']


def write_factory_snapshot(directory, meta):
    write_yaml(directory / 'odin_calib.yaml', {
        'cam_0': meta['calibration']['odin'],
        'Tcl_0': np.array(
            meta['factory_internal']['T_camera_lidar']
        ).ravel().tolist(),
    })


def test_reversed_order_and_corrupt_pair_rejected(dataset):
    meta, _, _ = dataset
    meta['samples'][0]['odin_corners'].reverse()
    meta['samples'][1]['odin_corners'] = (
        np.array(meta['samples'][1]['odin_corners']) + 40
    ).tolist()
    _, report = solve_session(meta, URDF)
    assert report['status'] == 'accepted'
    assert {s['id'] for s in report['rejected']} == {
        'synthetic_0',
        'synthetic_1',
    }


def test_insufficient_data_or_changed_head_posture(dataset):
    meta, _, _ = dataset
    meta['samples'][0]['head_positions']['head_yaw_joint'] = 0.01
    with pytest.raises(ValueError, match='posture'):
        solve_session(meta, URDF)
    meta['samples'] = meta['samples'][:10]
    with pytest.raises(ValueError, match='20'):
        solve_session(meta, URDF)


def test_validation_does_not_hide_bad_mounting(dataset):
    meta, _, _ = dataset
    calibration, _ = solve_session(meta, URDF)
    calibration['T_torso_odin_imu']['matrix'][0][3] += 0.04
    report = evaluate(
        meta,
        URDF,
        calibration,
        [s for s in meta['samples'] if s['role'] == 'validation'],
    )
    assert report['status'] == 'requires_review'
    assert any('10 mm' in failure for failure in report['failures'])


def test_no_validation_is_not_accepted(dataset):
    meta, _, _ = dataset
    meta['samples'] = meta['samples'][:30]
    calibration, report = solve_session(meta, URDF)
    assert calibration['status'] == 'requires_review'
    assert any('10 independent' in failure for failure in report['failures'])


def test_session_tampering_and_pairing(tmp_path, dataset):
    meta, _, _ = dataset
    (tmp_path / 'robot.urdf').write_text(URDF)
    write_factory_snapshot(tmp_path, meta)
    write_yaml(tmp_path / 'session.yaml', meta)
    loaded, _ = load_session(tmp_path)
    assert len(loaded['samples']) == 42
    meta['samples'][0]['stamps']['odin'] += 1
    write_yaml(tmp_path / 'session.yaml', meta)
    with pytest.raises(ValueError, match='pairing'):
        load_session(tmp_path)
    (tmp_path / 'robot.urdf').write_text(URDF + ' ')
    with pytest.raises(ValueError, match='URDF snapshot'):
        load_session(tmp_path)


def test_offline_commands_export_reports_and_validation_overlays(
    tmp_path, dataset
):
    meta, _, _ = dataset
    session = tmp_path / 'session'
    session.mkdir()
    (session / 'images').mkdir()
    (session / 'robot.urdf').write_text(URDF)
    write_factory_snapshot(session, meta)
    write_yaml(session / 'session.yaml', meta)
    for sample in meta['samples']:
        if sample['role'] == 'validation':
            for name in ('head', 'odin'):
                assert cv2.imwrite(
                    str(session / sample[f'{name}_image']),
                    np.zeros((480, 640, 3), dtype=np.uint8),
                )
    result = tmp_path / 'result'
    assert (
        solve_main(['--session', str(session), '--output', str(result)]) == 0
    )
    assert (result / 'calibration.yaml').exists()
    assert (result / 'report.yaml').exists()
    assert len(list((result / 'overlays').glob('*.jpg'))) == 24
    validation = tmp_path / 'validation'
    assert (
        validate_main(
            [
                '--session',
                str(session),
                '--output',
                str(validation),
                '--calibration',
                str(result / 'calibration.yaml'),
            ]
        )
        == 0
    )
    assert len(list((validation / 'overlays').glob('*.jpg'))) == 24


def test_legacy_session_uses_saved_factory_file(tmp_path, dataset):
    meta, _, _ = dataset
    write_factory_snapshot(tmp_path, meta)
    del meta['factory_internal']
    (tmp_path / 'robot.urdf').write_text(URDF)
    write_yaml(tmp_path / 'session.yaml', meta)
    loaded, _ = load_session(tmp_path)
    assert loaded['factory_internal']['T_imu_lidar'][0][3] == -0.02663
    assert loaded['calibration_sha256'] == meta['calibration_sha256']
    (tmp_path / 'odin_calib.yaml').unlink()
    with pytest.raises(OSError):
        load_session(tmp_path)


def test_validation_rejects_old_result_and_changed_factory(dataset):
    meta, _, _ = dataset
    calibration, _ = solve_session(meta, URDF)
    calibration['version'] = 1
    with pytest.raises(ValueError, match='Legacy result'):
        evaluate(meta, URDF, calibration, [])
    calibration['version'] = 2
    calibration['factory_internal'] = deepcopy(calibration['factory_internal'])
    calibration['factory_internal']['T_camera_lidar'][0][3] += 0.01
    with pytest.raises(ValueError, match='factory geometry differs'):
        evaluate(meta, URDF, calibration, [])


def test_factory_rotation_rounding_and_invalid_input():
    cl = np.array([
        [-0.01267, -0.99991, -0.00533, 0.04060],
        [0.00056, 0.00533, -0.99999, -0.00472],
        [0.99992, -0.01267, 0.00049, -0.00794],
        [0, 0, 0, 1],
    ])
    internal = factory_extrinsics({'Tcl_0': cl.ravel().tolist()})
    normalized = np.array(internal['T_camera_lidar'])
    np.testing.assert_allclose(normalized[:3, 3], cl[:3, 3], atol=1e-12)
    np.testing.assert_allclose(
        normalized[:3, :3].T @ normalized[:3, :3], np.eye(3), atol=1e-12
    )
    assert np.linalg.det(normalized[:3, :3]) == pytest.approx(1)
    with pytest.raises(ValueError, match='contain Tcl_0'):
        factory_extrinsics({})
    with pytest.raises(ValueError, match='16 values'):
        factory_extrinsics({'Tcl_0': [1, 2]})
    cl[0, 0] = 2
    with pytest.raises(ValueError, match='rotation'):
        factory_extrinsics({'Tcl_0': cl.tolist()})


@pytest.mark.parametrize('change', ['extrinsics', 'intrinsics'])
def test_saved_factory_mismatch_rejected(tmp_path, dataset, change):
    meta, _, _ = dataset
    write_factory_snapshot(tmp_path, meta)
    (tmp_path / 'robot.urdf').write_text(URDF)
    write_yaml(tmp_path / 'session.yaml', meta)
    from x2_camera_calibration.session import load_yaml
    snapshot = load_yaml(tmp_path / 'odin_calib.yaml')
    if change == 'extrinsics':
        snapshot['Tcl_0'][3] += 0.01
    else:
        snapshot['cam_0']['A11'] += 1
    write_yaml(tmp_path / 'odin_calib.yaml', snapshot)
    with pytest.raises(ValueError, match='differs?'):
        load_session(tmp_path)
