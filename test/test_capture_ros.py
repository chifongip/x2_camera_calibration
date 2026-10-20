"""Real DDS/subprocess capture integration, isolated from live robot traffic."""

import os
import random
import subprocess
import sys
import time

import cv2
import numpy as np
import pytest
import yaml

from x2_camera_calibration.session import load_session


@pytest.mark.parametrize('quota', ['total', 'train', 'validation'])
def test_passive_capture_with_synthetic_ros_inputs(
    tmp_path, monkeypatch, quota
):
    rclpy = pytest.importorskip('rclpy')
    from aimdk_msgs.msg import JointState, JointStateArray
    from nav_msgs.msg import Odometry
    from rclpy.context import Context
    from sensor_msgs.msg import CameraInfo, CompressedImage

    # Independent context and environment; never alter the parent process domain.
    domain = random.SystemRandom().randint(150, 220)
    environment = os.environ.copy()
    environment.update(
        ROS_DOMAIN_ID=str(domain),
        ROS_LOCALHOST_ONLY='1',
        ROS_LOG_DIR=str(tmp_path / 'ros_logs'),
    )
    environment.pop('FASTRTPS_DEFAULT_PROFILES_FILE', None)
    environment.pop('CYCLONEDDS_URI', None)
    monkeypatch.setenv('ROS_LOG_DIR', str(tmp_path / 'ros_logs'))
    monkeypatch.setenv('ROS_LOCALHOST_ONLY', '1')
    context = Context()
    rclpy.init(context=context, domain_id=domain)
    node = rclpy.create_node('synthetic_calibration_inputs', context=context)
    from test_solver import URDF

    (tmp_path / 'robot.urdf').write_text(URDF)
    fish = {
        'cam_model': 'FishPoly',
        'image_width': 640,
        'image_height': 480,
        'A11': 400.0,
        'A12': 0.0,
        'A22': 400.0,
        'u0': 320.0,
        'v0': 240.0,
        **{f'k{i}': 0.001 for i in range(2, 8)},
    }
    calib_path = tmp_path / 'calib.yaml'
    calib_path.write_text(yaml.safe_dump({'cam_0': fish}))
    image = np.full((480, 640, 3), 255, dtype=np.uint8)
    for row in range(7):
        for col in range(9):
            if (row + col) % 2 == 0:
                y0, y1 = 100 + row * 35, 100 + (row + 1) * 35
                x0, x1 = 140 + col * 35, 140 + (col + 1) * 35
                image[y0:y1, x0:x1] = 0
    jpeg = cv2.imencode('.jpg', image)[1].tobytes()
    head_topic = '/test/head/image'
    odin_topic = '/test/odin/image'
    info_topic = '/test/head/info'
    state_topic = '/test/head/state'
    internal_topic = '/test/odin/internal'
    head_pub = node.create_publisher(CompressedImage, head_topic, 10)
    odin_pub = node.create_publisher(CompressedImage, odin_topic, 10)
    info_pub = node.create_publisher(CameraInfo, info_topic, 10)
    state_pub = node.create_publisher(JointStateArray, state_topic, 10)
    internal_pub = node.create_publisher(Odometry, internal_topic, 10)
    output = tmp_path / 'session'
    command = [
        sys.executable,
        '-m',
        'x2_camera_calibration.capture',
        '--headless',
        '--auto-capture',
        '--confirm-order',
        '--max-captures',
        '2',
        '--timeout',
        '15',
        '--output',
        str(output),
        '--odin-calib',
        str(calib_path),
        '--urdf',
        str(tmp_path / 'robot.urdf'),
        '--head-image',
        head_topic,
        '--odin-image',
        odin_topic,
        '--head-info',
        info_topic,
        '--head-state',
        state_topic,
        '--odin-internal',
        internal_topic,
    ]
    if quota != 'total':
        command.remove('--max-captures')
        command.remove('2')
        command.extend([f'--max-{quota}-captures', '2'])
        if quota == 'validation':
            command.extend(['--role', 'validation'])
    log_path = tmp_path / 'capture.log'
    process = None
    try:
        with log_path.open('w') as log:
            process = subprocess.Popen(
                command, env=environment, stdout=log, stderr=subprocess.STDOUT
            )
            start = time.monotonic()
            while process.poll() is None and time.monotonic() - start < 20:
                stamp = node.get_clock().now().to_msg()
                head = CompressedImage()
                head.header.stamp = stamp
                head.header.frame_id = 'rgb_head_front_center'
                head.format, head.data = 'jpeg', jpeg
                odin = CompressedImage()
                odin.header.stamp = stamp
                odin.format, odin.data = 'jpeg', jpeg
                info = CameraInfo()
                info.header = head.header
                info.width, info.height = 640, 480
                info.distortion_model = 'plumb_bob'
                info.k = [400.0, 0.0, 320.0, 0.0, 400.0, 240.0, 0.0, 0.0, 1.0]
                info.d = [0.0] * 8
                state = JointStateArray()
                state.header.stamp = stamp
                state.header.meas_stamp = stamp
                state.joints = [
                    JointState(name=name, position=0.0)
                    for name in ('head_yaw_joint', 'head_pitch_joint')
                ]
                internal = Odometry()
                internal.header.stamp = stamp
                matrix = np.eye(4).ravel().tolist() + [0.0] * 20
                # /wiwc is refined online; this must not stop capture after pair 1.
                matrix[3] = 0.0001 * (time.monotonic() - start)
                internal.pose.covariance = matrix
                internal.twist.covariance = matrix
                state_pub.publish(state)
                info_pub.publish(info)
                internal_pub.publish(internal)
                head_pub.publish(head)
                odin_pub.publish(odin)
                time.sleep(0.03)
            assert process.poll() == 0, log_path.read_text()
        metadata, _ = load_session(output)
        assert len(metadata['samples']) == 2
        expected_role = 'validation' if quota == 'validation' else 'train'
        assert all(s['role'] == expected_role for s in metadata['samples'])
        assert all(
            len(sample['head_corners']) == 48 for sample in metadata['samples']
        )
        assert metadata['samples'][0]['head_state_samples']
        first, second = metadata['samples']
        assert (
            first['internal']['T_camera_lidar']
            != second['internal']['T_camera_lidar']
        )
        assert first['internal_stamp'] < second['internal_stamp']
        assert len(list((output / 'images').glob('*.jpg'))) == 4
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
        node.destroy_node()
        rclpy.shutdown(context=context)
