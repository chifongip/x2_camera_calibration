"""Passive live checkerboard capture. No robot publishers or service clients."""

import argparse
from collections import deque
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import threading
import time
import uuid

import cv2
import numpy as np

from .geometry import HeadKinematics, board_points, project, unproject
from .ordering import PairOrdering
from .limits import CaptureLimits
from .session import (
    HEAD_JOINTS,
    camera_models_match,
    fingerprint,
    factory_extrinsics,
    load_yaml,
    stable_head,
    write_yaml,
)

HEAD_IMAGE = '/aima/hal/sensor/rgb_head_front_center/rgb_image/compressed'
HEAD_INFO = '/aima/hal/sensor/rgb_head_front_center/camera_info'
HEAD_STATE = '/aima/hal/joint/head/state'


def stamp_seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def detect(image, pattern):
    scale = min(1.0, 1280 / image.shape[1])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, None, fx=scale, fy=scale) if scale < 1 else gray
    found, corners = cv2.findChessboardCornersSB(
        small, pattern, flags=cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if not found:
        return None
    corners = (corners / scale).astype(np.float32)
    cv2.cornerSubPix(
        gray,
        corners,
        (5, 5),
        (-1, -1),
        (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 0.001),
    )
    return corners.reshape(-1, 2)


def _parser():
    parser = argparse.ArgumentParser(
        description='Capture passive X2/Odin checkerboard observations'
    )
    parser.add_argument(
        '--output', type=Path, required=True, help='New session directory'
    )
    parser.add_argument('--odin-calib', type=Path, required=True)
    parser.add_argument(
        '--urdf',
        type=Path,
        default=Path(
            '/home/ubuntu/x2_ws/src/x2_description/urdf/x2_ultra.urdf'
        ),
    )
    parser.add_argument('--head-image', default=HEAD_IMAGE)
    parser.add_argument('--head-info', default=HEAD_INFO)
    parser.add_argument('--head-state', default=HEAD_STATE)
    parser.add_argument('--odin-image', default='/odin1/image/compressed')
    parser.add_argument(
        '--odin-internal', help='Deprecated compatibility option; ignored'
    )
    parser.add_argument('--columns', type=int, default=8)
    parser.add_argument('--rows', type=int, default=6)
    parser.add_argument('--square-size', type=float, default=0.035)
    parser.add_argument(
        '--role', choices=['train', 'validation'], default='train'
    )
    parser.add_argument(
        '--headless',
        action='store_true',
        help='No GUI; requires --auto-capture',
    )
    parser.add_argument(
        '--auto-capture',
        action='store_true',
        help='For controlled recording/tests; capture eligible pairs at most once per second',
    )
    parser.add_argument(
        '--confirm-order',
        action='store_true',
        help='Assert the same physical corner ordering in headless captures',
    )
    parser.add_argument('--reverse-odin', action='store_true')
    parser.add_argument('--max-captures', type=int, default=0)
    parser.add_argument(
        '--max-train-captures',
        type=int,
        default=0,
        help='Training pair quota; zero is unlimited',
    )
    parser.add_argument(
        '--max-validation-captures',
        '--max-validation-capture',
        dest='max_validation_captures',
        type=int,
        default=0,
        help='Validation pair quota; zero is unlimited',
    )
    parser.add_argument(
        '--timeout',
        type=float,
        default=0,
        help='Exit after N seconds; zero is unlimited',
    )
    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    if args.headless and (not args.auto_capture or not args.confirm_order):
        parser.error('--headless requires --auto-capture and --confirm-order')
    if (
        min(
            args.max_captures,
            args.max_train_captures,
            args.max_validation_captures,
            args.timeout,
        )
        < 0
    ):
        parser.error('Capture limits must be nonnegative')
    if (
        args.headless
        and args.max_train_captures
        and args.max_validation_captures
    ):
        parser.error(
            'Combined stage quotas require a preview for T/V selection; '
            'use one role quota per headless session'
        )
    limits = CaptureLimits(
        args.max_train_captures,
        args.max_validation_captures,
        args.max_captures,
    )
    if (
        args.headless
        and any(limits.limits.values())
        and not limits.limits[args.role]
    ):
        parser.error(
            'Set a quota for the selected --role in a headless session'
        )
    try:
        objects = board_points(args.columns, args.rows, args.square_size)
        urdf = args.urdf.read_text(encoding='utf-8')
        kinematics = HeadKinematics(urdf)
        odin_file = load_yaml(args.odin_calib)
        odin = odin_file['cam_0']
        internal = factory_extrinsics(odin_file)
        project(
            odin, [[0, 0, 1]]
        )  # Fail before creating files for unsupported calibration.
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / 'images').mkdir()
        (args.output / 'robot.urdf').write_text(urdf, encoding='utf-8')
        (args.output / 'odin_calib.yaml').write_text(
            args.odin_calib.read_text(encoding='utf-8'), encoding='utf-8'
        )
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f'Cannot create capture session: {error}\n')

    # Import ROS only for live capture; solve/validate work without ROS imports.
    import rclpy
    from aimdk_msgs.msg import JointStateArray
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, CompressedImage

    rclpy.init(args=[])
    node = rclpy.create_node('x2_camera_calibration_capture')
    lock = threading.Lock()
    buffers = {'head': deque(maxlen=60), 'odin': deque(maxlen=30)}
    state = {'head': deque(maxlen=1000)}
    metadata = {
        'version': 1,
        'factory_internal': internal,
        'session_id': str(uuid.uuid4()),
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'board': {
            'columns': args.columns,
            'rows': args.rows,
            'square_size_m': args.square_size,
        },
        'max_pair_delta_sec': 0.030,
        'capture_limits': {
            'train': args.max_train_captures,
            'validation': args.max_validation_captures,
            'total': args.max_captures,
        },
        'urdf_sha256': fingerprint(urdf),
        'urdf_source': str(args.urdf.resolve()),
        'odin_calib_source': str(args.odin_calib.resolve()),
        'topics': {
            'head_image': args.head_image,
            'head_info': args.head_info,
            'head_state': args.head_state,
            'odin_image': args.odin_image,
        },
        'head_timestamp_policy': 'header.meas_stamp when nonzero, otherwise header.stamp',
        'samples': [],
    }

    def image_callback(camera, message):
        with lock:
            buffers[camera].append((message, time.monotonic()))

    def info_callback(message):
        model = {
            'K': [float(v) for v in message.k],
            'D': [float(v) for v in message.d],
            'model': message.distortion_model,
            'width': message.width,
            'height': message.height,
            'frame_id': message.header.frame_id,
        }
        with lock:
            state['head_model'] = model

    def head_callback(message):
        positions = {
            joint.name: joint.position
            for joint in message.joints
            if joint.name in HEAD_JOINTS
        }
        if positions.keys() != set(HEAD_JOINTS):
            return
        stamp = stamp_seconds(message.header.meas_stamp) or stamp_seconds(
            message.header.stamp
        )
        with lock:
            state['head'].append({'stamp': stamp, 'positions': positions})
            state['head_received'] = time.monotonic()

    subscriptions = [
        node.create_subscription(
            CompressedImage,
            args.head_image,
            lambda m: image_callback('head', m),
            qos_profile_sensor_data,
        ),
        node.create_subscription(
            CompressedImage,
            args.odin_image,
            lambda m: image_callback('odin', m),
            qos_profile_sensor_data,
        ),
        node.create_subscription(
            CameraInfo, args.head_info, info_callback, qos_profile_sensor_data
        ),
        node.create_subscription(
            JointStateArray,
            args.head_state,
            head_callback,
            qos_profile_sensor_data,
        ),
    ]
    stopping = threading.Event()

    def spin():
        while not stopping.is_set() and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)

    thread = threading.Thread(target=spin, daemon=True)
    thread.start()
    role = args.role
    reverse = args.reverse_odin
    confirmed = args.confirm_order
    start = time.monotonic()
    last_save = -float('inf')
    last_pair = None
    cached = {}
    ordering = PairOrdering(objects)
    last_message = None
    last_message_time = -float('inf')
    print(
        'Space: capture | R: cycle Odin order | C: confirm physical corner 0 | '
        'T: training | V: validation | Q: quit'
    )
    print(
        'Keep the board and robot stationary at capture. No robot commands are sent.'
    )
    try:
        while rclpy.ok():
            now = time.monotonic()
            if args.timeout and now - start >= args.timeout:
                break
            with lock:
                images = {
                    name: list(values) for name, values in buffers.items()
                }
                snapshot = deepcopy(state)
            previews = {}
            corners = {}
            messages = {}
            reason = None
            pair = None
            order_details = {}
            reversed_flags = {}
            ordering.pending = None
            # Failed detections must not leave an unbounded decoded-image cache.
            while len(cached) > 8:
                cached.pop(next(iter(cached)))
            try:
                if any(not values for values in images.values()):
                    raise ValueError(
                        'Waiting for both compressed image streams'
                    )
                # Use a buffered pair slightly behind newest data so head state can bracket it.
                candidates = []
                for odin_message, odin_received in reversed(images['odin']):
                    stamp_o = stamp_seconds(odin_message.header.stamp)
                    if odin_received < now - 0.5:
                        continue
                    head_message, head_received = min(
                        images['head'],
                        key=lambda item: abs(
                            stamp_seconds(item[0].header.stamp) - stamp_o
                        ),
                    )
                    stamp_h = stamp_seconds(head_message.header.stamp)
                    if (
                        abs(stamp_h - stamp_o) <= 0.030
                        and head_received >= now - 0.5
                        and snapshot['head']
                        and snapshot['head'][-1]['stamp']
                        >= max(stamp_h, stamp_o)
                    ):
                        candidates.append((head_message, odin_message))
                        break
                if not candidates:
                    raise ValueError(
                        'Waiting for image pair within 30 ms and bracketing head state'
                    )
                messages = dict(zip(('head', 'odin'), candidates[0]))
                stamps = {
                    name: stamp_seconds(m.header.stamp)
                    for name, m in messages.items()
                }
                if min(stamps.values()) <= 0:
                    raise ValueError('Image timestamps must be nonzero')
                pair = (stamps['head'], stamps['odin'])
                for name, message in messages.items():
                    cache_key = (name, stamp_seconds(message.header.stamp))
                    if cache_key not in cached:
                        image = cv2.imdecode(
                            np.frombuffer(message.data, np.uint8),
                            cv2.IMREAD_COLOR,
                        )
                        if image is None:
                            raise ValueError(f'{name}: JPEG decoding failed')
                        cached[cache_key] = (
                            image,
                            detect(image, (args.columns, args.rows)),
                        )
                    previews[name], corners[name] = cached[cache_key]
                detected = corners
                corners = {}
                corners, reversed_flags = ordering.stabilize(detected, stamps)
                if len(cached) > 8:
                    cached = {
                        key: value
                        for key, value in cached.items()
                        if key
                        in {('head', stamps['head']), ('odin', stamps['odin'])}
                    }
                if 'head_model' not in snapshot:
                    raise ValueError('Waiting for head CameraInfo')
                if now - snapshot.get('head_received', 0) > 0.5:
                    raise ValueError('Head state is stale')
                calibration = {
                    'head': snapshot['head_model'],
                    'odin': odin,
                    'internal': internal,
                }
                if 'calibration' in metadata and not camera_models_match(
                    metadata['calibration'], calibration
                ):
                    raise ValueError(
                        'Camera intrinsics changed; start a new session'
                    )
                if messages['head'].header.frame_id != 'rgb_head_front_center':
                    raise ValueError(
                        'Head image frame must be rgb_head_front_center'
                    )
                if (
                    snapshot['head_model']['frame_id']
                    != messages['head'].header.frame_id
                ):
                    raise ValueError('Head image and CameraInfo frames differ')
                if messages['odin'].header.frame_id not in ('', 'camera_0'):
                    raise ValueError(
                        'Unexpected Odin image frame; expected empty or camera_0 optical'
                    )
                for name, model in [
                    ('head', calibration['head']),
                    ('odin', odin),
                ]:
                    expected = (
                        (model['height'], model['width'])
                        if name == 'head'
                        else (model['image_height'], model['image_width'])
                    )
                    if previews[name].shape[:2] != tuple(expected):
                        raise ValueError(
                            f'{name}: image size does not match factory intrinsics'
                        )
                    if corners[name] is None:
                        raise ValueError(
                            f'{name}: all {args.columns * args.rows} corners are required'
                        )
                    unproject(model, corners[name])
                head_positions = stable_head(
                    snapshot['head'], (pair[0] + pair[1]) / 2
                )
                # Check each exposure, not just the midpoint.
                for stamp in pair:
                    stable_head(snapshot['head'], stamp)
                matched_reverse, order_details = ordering.match(
                    corners,
                    calibration,
                    kinematics.transform(head_positions),
                    pair,
                    reverse,
                )
                reversed_flags['head'] = (
                    order_details['head_corner_permutation'] == 'rotate_180'
                )
                reversed_flags['odin'] = (
                    order_details['odin_corner_permutation'] == 'rotate_180'
                )
                if confirmed and ordering.anchor is None:
                    ordering.confirm()
                if not confirmed:
                    raise ValueError(
                        'Match physical corner 0 in both images; press R as needed, then C'
                    )
                if not ordering.ready:
                    raise ValueError(ordering.wait_reason)
                if pair == last_pair:
                    raise ValueError('This image pair is already captured')
            except (ValueError, KeyError, cv2.error) as error:
                reason = str(error)
            if limits.pause_message(role):
                reason = limits.pause_message(role)

            key = -1
            if not args.headless:
                for name in ('head', 'odin'):
                    if name not in previews:
                        continue
                    image = previews[name].copy()
                    points = corners.get(name)
                    if points is not None:
                        cv2.drawChessboardCorners(
                            image,
                            (args.columns, args.rows),
                            points.astype(np.float32)[:, None],
                            True,
                        )
                        first = tuple(np.rint(points[0]).astype(int))
                        cv2.circle(image, first, 15, (0, 0, 255), 3)
                        cv2.putText(
                            image,
                            '0',
                            first,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            2,
                        )
                    image = cv2.resize(
                        image,
                        (960, int(image.shape[0] * 960 / image.shape[1])),
                    )
                    status = reason or 'READY (hold board still)'
                    line = f'{role} | {limits.progress()} | {status}'
                    cv2.putText(
                        image,
                        line[:110],
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 255, 0) if reason is None else (0, 100, 255),
                        1,
                    )
                    if snapshot['head']:
                        angles = snapshot['head'][-1]['positions']
                        line2 = ' '.join(
                            f'{name}={np.rad2deg(angles[name]):.2f}deg'
                            for name in HEAD_JOINTS
                        )
                        if pair:
                            line2 += f' | pair dt={abs(pair[0] - pair[1]) * 1000:.1f}ms'
                        cv2.putText(
                            image,
                            line2,
                            (10, 48),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (0, 255, 255),
                            1,
                        )
                    if order_details:
                        gap = order_details['order_separation_deg']
                        confidence = (
                            'unconfirmed'
                            if gap is None
                            else f'{gap:.0f}deg gap'
                        )
                        diagnostic = (
                            f'Order: {confidence} | pose delta (diagnostic only): '
                            f'{order_details["reference_translation_delta_m"] * 1000:.0f}mm '
                            f'{order_details["reference_rotation_delta_deg"]:.1f}deg'
                        )
                        cv2.putText(
                            image,
                            diagnostic,
                            (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (0, 255, 255),
                            1,
                        )
                    cv2.imshow(f'X2 calibration: {name}', image)
                key = cv2.waitKey(20) & 0xFF
            else:
                time.sleep(0.02)
            if key in (ord('q'), 27):
                break
            if key == ord('r'):
                reverse = (
                    (matched_reverse + 1) % 4
                    if ordering.pending is not None
                    else (int(reverse) + 1) % 4
                )
                confirmed = False
                ordering.clear_confirmation()
            if key == ord('c'):
                try:
                    ordering.confirm()
                    confirmed = True
                except ValueError as error:
                    print(str(error), flush=True)
            if key == ord('v'):
                role = 'validation'
            if key == ord('t'):
                role = 'train'
            if reason != last_message and now - last_message_time >= 1.0:
                print(
                    reason
                    or 'READY: hold board still; press Space to capture',
                    flush=True,
                )
                last_message = reason
                last_message_time = now
            capture_requested = key == 32 or (
                args.auto_capture and now - last_save >= 1
            )
            if (
                capture_requested
                and reason is None
                and limits.can_capture(role)
                and key not in (ord('r'), ord('c'), ord('t'), ord('v'))
            ):
                observation_id = (
                    f'{metadata["session_id"]}_{len(metadata["samples"]):04d}'
                )
                if 'calibration' not in metadata:
                    metadata['calibration'] = calibration
                    metadata['calibration_sha256'] = fingerprint(calibration)
                sample = {
                    'id': observation_id,
                    'role': role,
                    'stamps': stamps,
                    'image_frames': {
                        name: m.header.frame_id for name, m in messages.items()
                    },
                    'head_positions': head_positions,
                    'internal': internal,
                    'head_state_samples': [
                        s
                        for s in snapshot['head']
                        if min(pair) - 0.6 <= s['stamp'] <= max(pair) + 0.1
                    ],
                    'odin_order_reversed': reversed_flags['odin'],
                    'head_order_reversed': reversed_flags['head'],
                    'ordering': order_details,
                    'calibration_sha256': metadata['calibration_sha256'],
                }
                for name in ('head', 'odin'):
                    relative = f'images/{observation_id}_{name}.jpg'
                    (args.output / relative).write_bytes(
                        bytes(messages[name].data)
                    )
                    sample[f'{name}_corners'] = corners[name].tolist()
                    sample[f'{name}_image'] = relative
                metadata['samples'].append(sample)
                write_yaml(args.output / 'session.yaml', metadata)
                limits.record(role)
                last_save, last_pair = now, pair
                print(
                    f'Saved {role} pair {limits.counts[role]} (total {limits.total})',
                    flush=True,
                )
                if limits.finished:
                    print('Capture quotas complete.', flush=True)
                    break
    except KeyboardInterrupt:
        pass
    except (ValueError, OSError, cv2.error) as error:
        print(f'Capture stopped: {error}', flush=True)
        return 1
    finally:
        stopping.set()
        thread.join(timeout=2)
        for subscription in subscriptions:
            node.destroy_subscription(subscription)
        node.destroy_node()
        rclpy.shutdown()
        if not args.headless:
            cv2.destroyAllWindows()
        # Keep even an empty manifest to explain why no observations were acquired.
        write_yaml(args.output / 'session.yaml', metadata)
    return 0 if metadata['samples'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
