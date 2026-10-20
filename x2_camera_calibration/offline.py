"""Fixed-intrinsic stereo solve and independent mounting validation."""

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

from .geometry import (
    HeadKinematics,
    board_points,
    board_pose,
    difference,
    inverse,
    mean_transform,
    project,
    rigid,
    transform_points,
    transform_record,
    unproject,
)
from .session import (
    HEAD_JOINTS,
    camera_models_match,
    fingerprint,
    load_session,
    load_yaml,
    write_yaml,
)

VIRTUAL_K = np.array(
    [[1000.0, 0.0, 1000.0], [0.0, 1000.0, 1000.0], [0.0, 0.0, 1.0]]
)


def objects_for(metadata):
    board = metadata['board']
    return board_points(
        board['columns'], board['rows'], board['square_size_m']
    )


def models_for(metadata):
    calibration = metadata['calibration']
    return calibration['head'], calibration['odin']


def imu_from_camera(metadata):
    if 'factory_internal' not in metadata:
        raise ValueError(
            'Missing factory geometry; load the session with its odin_calib.yaml'
        )
    internal = metadata['factory_internal']
    return rigid(internal['T_imu_lidar']) @ inverse(internal['T_camera_lidar'])


def corners_for(sample, camera, count):
    points = np.asarray(sample[f'{camera}_corners'], dtype=float)
    if points.shape != (count, 2) or not np.isfinite(points).all():
        raise ValueError(
            f"{sample['id']}: expected {count} finite {camera} corners"
        )
    return points


def virtual_points(model, points):
    return (unproject(model, points) * 1000 + 1000).astype(np.float32)


def stereo_fit(metadata, samples):
    objects = objects_for(metadata)
    head, odin = models_for(metadata)
    head_points = [
        virtual_points(head, corners_for(s, 'head', len(objects)))
        for s in samples
    ]
    odin_points = [
        virtual_points(odin, corners_for(s, 'odin', len(objects)))
        for s in samples
    ]
    fit = cv2.stereoCalibrate(
        [objects.astype(np.float32)] * len(samples),
        head_points,
        odin_points,
        VIRTUAL_K.copy(),
        np.zeros(5),
        VIRTUAL_K.copy(),
        np.zeros(5),
        (2000, 2000),
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 150, 1e-9),
    )
    # OpenCV returns camera 2 <- camera 1; the public output is head <- Odin.
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = fit[5], fit[6].reshape(3)
    return inverse(result)


def observation_errors(metadata, sample, head_from_odin):
    objects = objects_for(metadata)
    head, odin = models_for(metadata)
    head_points = corners_for(sample, 'head', len(objects))
    odin_points = corners_for(sample, 'odin', len(objects))
    head_from_board = board_pose(head, head_points, objects)
    odin_from_board = inverse(head_from_odin) @ head_from_board
    predictions = {
        'head': project(head, transform_points(head_from_board, objects)),
        'odin': project(odin, transform_points(odin_from_board, objects)),
    }
    errors = {
        'head': np.linalg.norm(predictions['head'] - head_points, axis=1),
        'odin': np.linalg.norm(predictions['odin'] - odin_points, axis=1),
    }
    return errors, predictions


def torso_from_imu(metadata, kinematics, sample, head_from_odin):
    return (
        kinematics.transform(sample['head_positions'])
        @ rigid(head_from_odin)
        @ inverse(imu_from_camera(metadata))
    )


def error_summary(values):
    values = np.asarray(values, dtype=float)
    if not values.size:
        return {'count': 0, 'rms_px': None, 'p95_px': None}
    return {
        'count': int(values.size),
        'rms_px': float(np.sqrt(np.mean(values**2))),
        'p95_px': float(np.percentile(values, 95)),
    }


def evaluate(
    metadata, urdf, calibration, samples, session_dir=None, output_dir=None
):
    if calibration.get('version') != 2:
        raise ValueError(
            'Legacy result uses unverified internal geometry; solve again '
            'into a new output directory with the updated tool'
        )
    if fingerprint(calibration['factory_internal']) != fingerprint(
        metadata['factory_internal']
    ):
        raise ValueError('Validation factory geometry differs from training')
    kinematics = HeadKinematics(urdf)
    mounting = rigid(calibration['T_torso_odin_imu']['matrix'])
    head, odin = models_for(metadata)
    objects = objects_for(metadata)
    residuals = {'head': [], 'odin': []}
    observations = []
    postures = []
    for sample in samples:
        head_from_odin = (
            inverse(kinematics.transform(sample['head_positions']))
            @ mounting
            @ imu_from_camera(metadata)
        )
        errors, predictions = observation_errors(
            metadata, sample, head_from_odin
        )
        independent = board_pose(
            head, corners_for(sample, 'head', len(objects)), objects
        ) @ inverse(
            board_pose(
                odin, corners_for(sample, 'odin', len(objects)), objects
            )
        )
        independent_mount = torso_from_imu(
            metadata, kinematics, sample, independent
        )
        translation, rotation = difference(mounting, independent_mount)
        observations.append(
            {
                'id': sample['id'],
                'translation_error_m': translation,
                'rotation_error_deg': rotation,
                **{name: error_summary(e) for name, e in errors.items()},
            }
        )
        angles = np.array(
            [sample['head_positions'][name] for name in HEAD_JOINTS]
        )
        if all(
            np.max(np.abs(angles - other)) > np.deg2rad(1)
            for other in postures
        ):
            postures.append(angles)
        for name in residuals:
            residuals[name].extend(errors[name].tolist())
            if session_dir is not None and output_dir is not None:
                relative = sample[f'{name}_image']
                path = Path(session_dir) / relative
                if not path.resolve().is_relative_to(
                    Path(session_dir).resolve()
                ):
                    raise ValueError(
                        'Image path escapes the session directory'
                    )
                image = cv2.imread(str(path))
                if image is None:
                    raise ValueError(f'Cannot read observation image: {path}')
                for observed, predicted in zip(
                    sample[f'{name}_corners'], predictions[name]
                ):
                    cv2.circle(
                        image,
                        tuple(np.rint(observed).astype(int)),
                        5,
                        (0, 255, 0),
                        1,
                    )
                    cv2.drawMarker(
                        image,
                        tuple(np.rint(predicted).astype(int)),
                        (0, 0, 255),
                        cv2.MARKER_CROSS,
                        9,
                        1,
                    )
                target = Path(output_dir) / f"{sample['id']}_{name}.jpg"
                if not cv2.imwrite(str(target), image):
                    raise ValueError(
                        f'Cannot write validation overlay: {target}'
                    )
    summaries = {
        name: error_summary(values) for name, values in residuals.items()
    }
    failures = []
    if len(samples) < 10:
        failures.append('Need at least 10 independent validation observations')
    if len(postures) < 3:
        failures.append(
            'Need at least three head postures separated by more than 1 degree'
        )
    for name, summary in summaries.items():
        if (
            summary['rms_px'] is None
            or summary['rms_px'] > 1.5
            or summary['p95_px'] > 3
        ):
            failures.append(
                f'{name}: exceeds 1.5 px RMS or 3 px p95 reprojection target'
            )
    if any(
        s['translation_error_m'] > 0.010 or s['rotation_error_deg'] > 1
        for s in observations
    ):
        failures.append(
            'Independent mounting estimates exceed 10 mm or 1 degree'
        )
    return {
        'status': 'accepted' if not failures else 'requires_review',
        'failures': failures,
        'validation_pairs': len(samples),
        'distinct_head_postures': len(postures),
        'reprojection': summaries,
        'observations': observations,
    }


def solve_session(metadata, urdf):
    samples = [s for s in metadata['samples'] if s['role'] == 'train']
    if len(samples) < 20:
        raise ValueError('At least 20 training observations are required')
    positions = np.array(
        [[s['head_positions'][name] for name in HEAD_JOINTS] for s in samples]
    )
    if np.max(np.ptp(positions, axis=0)) > np.deg2rad(0.2):
        raise ValueError(
            'Training head posture changed by more than 0.2 degrees; use validation role'
        )
    kinematics = HeadKinematics(urdf)
    rejected = []
    # Seed the robust fit with the median-consistent independent PnP estimate.
    head, odin = models_for(metadata)
    objects = objects_for(metadata)
    individual = [
        board_pose(head, corners_for(s, 'head', len(objects)), objects)
        @ inverse(
            board_pose(odin, corners_for(s, 'odin', len(objects)), objects)
        )
        for s in samples
    ]
    distances = np.array(
        [
            [
                difference(a, b)[0] + np.deg2rad(difference(a, b)[1])
                for b in individual
            ]
            for a in individual
        ]
    )
    head_from_odin = individual[int(np.argmin(np.median(distances, axis=1)))]
    for iteration in range(4):
        scores = []
        for sample in samples:
            try:
                errors, _ = observation_errors(
                    metadata, sample, head_from_odin
                )
                scores.append(float(np.sqrt(np.mean(errors['odin'] ** 2))))
            except (ValueError, cv2.error):
                scores.append(float('inf'))
        finite = np.asarray([score for score in scores if np.isfinite(score)])
        if not finite.size:
            raise ValueError(
                'No geometrically consistent training observations'
            )
        median = np.median(finite)
        limit = max(
            3.0,
            float(median + 3 * 1.4826 * np.median(np.abs(finite - median))),
        )
        retained = []
        for sample, score in zip(samples, scores):
            if score <= limit:
                retained.append(sample)
            else:
                rejected.append(
                    {
                        'id': sample['id'],
                        'reason': 'inconsistent reprojection',
                        'odin_rms_px': score if np.isfinite(score) else None,
                    }
                )
        if len(retained) < 20:
            raise ValueError(
                'Fewer than 20 training pairs remain after outlier rejection'
            )
        unchanged = len(retained) == len(samples)
        samples = retained
        head_from_odin = stereo_fit(metadata, samples)
        if unchanged and iteration > 0:
            break
    mounting = mean_transform(
        [
            torso_from_imu(metadata, kinematics, s, head_from_odin)
            for s in samples
        ]
    )
    calibration = {
        'version': 2,
        'factory_internal': metadata['factory_internal'],
        'transform_convention': 'T_A_B maps coordinates in B into A',
        'units': {'translation': 'm', 'quaternion': 'xyzw'},
        'board': metadata['board'],
        'calibration_sha256': metadata['calibration_sha256'],
        'urdf_sha256': fingerprint(urdf),
        'source_calibration': metadata['calibration'],
        'assumptions': [
            'rgb_head_front_center is an identity alias of rgb_head_center',
            'URDF camera axes are optical: x right, y down, z forward',
            'factory intrinsics and head joint offsets are fixed',
        ],
        'training_ids': [s['id'] for s in samples],
        'rejected': rejected,
        'T_head_camera_odin_camera': transform_record(
            head_from_odin, 'rgb_head_front_center', 'camera_0'
        ),
        'T_torso_odin_imu': transform_record(mounting, 'torso_link', 'imu'),
    }
    report = evaluate(
        metadata,
        urdf,
        calibration,
        [s for s in metadata['samples'] if s['role'] == 'validation'],
    )
    calibration['status'] = report['status']
    training_residuals = {'head': [], 'odin': []}
    for sample in samples:
        errors, _ = observation_errors(metadata, sample, head_from_odin)
        for name in errors:
            training_residuals[name].extend(errors[name])
    report['training_pairs'] = len(samples)
    report['training_reprojection'] = {
        name: error_summary(e) for name, e in training_residuals.items()
    }
    for name, summary in report['training_reprojection'].items():
        if summary['rms_px'] > 1.5 or summary['p95_px'] > 3:
            report['failures'].append(
                f'{name}: training reprojection exceeds commissioning targets'
            )
    if report['failures']:
        report['status'] = calibration['status'] = 'requires_review'
    report['rejected'] = rejected
    return calibration, report


def _parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='New output directory; never overwritten',
    )
    return parser


def solve_main(argv=None):
    parser = _parser(
        'Solve X2/Odin calibration from captured checkerboard observations'
    )
    args = parser.parse_args(argv)
    try:
        metadata, urdf = load_session(args.session)
        calibration, report = solve_session(metadata, urdf)
        args.output.mkdir(parents=True, exist_ok=False)
        validation = [
            s for s in metadata['samples'] if s['role'] == 'validation'
        ]
        if validation:
            overlays = args.output / 'overlays'
            overlays.mkdir()
            evaluate(
                metadata, urdf, calibration, validation, args.session, overlays
            )
        write_yaml(args.output / 'calibration.yaml', calibration)
        write_yaml(args.output / 'report.yaml', report)
        print(
            f"Calibration {report['status']}: {args.output / 'calibration.yaml'}"
        )
        for failure in report['failures']:
            print(f'  {failure}')
        return 0 if report['status'] == 'accepted' else 2
    except (ValueError, OSError, KeyError, cv2.error) as error:
        parser.exit(1, f'Calibration failed: {error}\n')


def validate_main(argv=None):
    parser = _parser(
        'Validate an existing mounting calibration on independent observations'
    )
    parser.add_argument('--calibration', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        metadata, urdf = load_session(args.session)
        calibration = load_yaml(args.calibration)
        if (
            calibration['urdf_sha256'] != fingerprint(urdf)
            or calibration['board'] != metadata['board']
            or not camera_models_match(
                calibration['source_calibration'], metadata['calibration']
            )
        ):
            raise ValueError(
                'Validation URDF, board or camera intrinsics differ from training'
            )
        samples = [s for s in metadata['samples'] if s['role'] == 'validation']
        if any(s['id'] in calibration['training_ids'] for s in samples):
            raise ValueError(
                'Validation observation overlaps the training set'
            )
        args.output.mkdir(parents=True, exist_ok=False)
        overlays = args.output / 'overlays'
        overlays.mkdir()
        report = evaluate(
            metadata, urdf, calibration, samples, args.session, overlays
        )
        write_yaml(args.output / 'report.yaml', report)
        print(f"Validation {report['status']}: {args.output / 'report.yaml'}")
        for failure in report['failures']:
            print(f'  {failure}')
        return 0 if report['status'] == 'accepted' else 2
    except (ValueError, OSError, KeyError, cv2.error) as error:
        parser.exit(1, f'Validation failed: {error}\n')


if __name__ == '__main__':
    command = sys.argv.pop(1) if len(sys.argv) > 1 else ''
    sys.exit({'solve': solve_main, 'validate': validate_main}[command]())
