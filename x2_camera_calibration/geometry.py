"""Geometry uses T_A_B: transform coordinates in frame B into frame A.

FishPoly equations follow odin_ros_driver/include/polynomial_camera.hpp
(Manifold Tech Ltd., Apache-2.0), with guarded optical-axis limits and a
converged inverse instead of the SDK's fixed seven iterations.
"""

import math
import xml.etree.ElementTree as ET

import cv2
import numpy as np


def rigid(value):
    transform = np.asarray(value, dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError('Transform must be a finite 4 x 4 matrix')
    rotation = transform[:3, :3]
    if (
        not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-6)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3)
        or abs(np.linalg.det(rotation) - 1) > 2e-3
    ):
        raise ValueError('Invalid rigid transform rotation or homogeneous row')
    return transform


def inverse(value):
    transform = rigid(value)
    result = np.eye(4)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ transform[:3, 3]
    return result


def pose(rotation_vector, translation):
    result = np.eye(4)
    result[:3, :3] = cv2.Rodrigues(np.asarray(rotation_vector, dtype=float))[0]
    result[:3, 3] = np.asarray(translation).reshape(3)
    return rigid(result)


def transform_points(transform, points):
    return np.asarray(points) @ transform[:3, :3].T + transform[:3, 3]


def board_points(columns=8, rows=6, square_size=0.035):
    if (
        columns < 3
        or rows < 3
        or not np.isfinite(square_size)
        or square_size <= 0
    ):
        raise ValueError('Invalid checkerboard dimensions or square size')
    points = np.zeros((rows * columns, 3))
    points[:, :2] = np.mgrid[:columns, :rows].T.reshape(-1, 2) * square_size
    return points


def _head_model(model):
    k = np.asarray(model['K'], dtype=float).reshape(3, 3)
    d = np.asarray(model['D'], dtype=float)
    if (
        model.get('model') not in ('plumb_bob', 'rational_polynomial')
        or d.size not in (4, 5, 8, 12, 14)
        or not np.isfinite(k).all()
        or not np.isfinite(d).all()
        or k[0, 0] <= 0
        or k[1, 1] <= 0
        or not np.allclose(k[2], [0, 0, 1])
    ):
        raise ValueError('Unsupported or invalid head camera calibration')
    return k, d


def _fish_model(model):
    if model.get('cam_model') != 'FishPoly':
        raise ValueError('Odin calibration must use FishPoly')
    keys = ['A11', 'A12', 'A22', 'u0', 'v0'] + [f'k{i}' for i in range(2, 8)]
    values = np.asarray([model[key] for key in keys], dtype=float)
    if not np.isfinite(values).all() or values[0] <= 0 or values[2] <= 0:
        raise ValueError('Invalid FishPoly camera calibration')
    if model.get('p1', 0) != 0 or model.get('p2', 0) != 0:
        raise ValueError(
            'Nonzero FishPoly tangential distortion is unsupported by the SDK model'
        )
    return values[:5], values[5:]


def unproject(model, pixels):
    pixels = np.asarray(pixels, dtype=float).reshape(-1, 2)
    if not np.isfinite(pixels).all():
        raise ValueError('Non-finite image corners')
    if 'K' in model:
        k, d = _head_model(model)
        return cv2.undistortPointsIter(
            pixels[:, None],
            k,
            d,
            None,
            None,
            (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-12),
        ).reshape(-1, 2)
    (fx, skew, fy, cx, cy), coefficients = _fish_model(model)
    y = (pixels[:, 1] - cy) / fy
    x = (pixels[:, 0] - cx - y * skew) / fx
    radius = np.hypot(x, y)
    theta = radius.copy()
    for _ in range(40):
        polynomial = theta.copy()
        derivative = np.ones_like(theta)
        for power, coefficient in enumerate(coefficients, 2):
            polynomial += coefficient * theta**power
            derivative += power * coefficient * theta ** (power - 1)
        if np.any(derivative <= 0):
            raise ValueError(
                'Non-monotonic FishPoly projection in observed field of view'
            )
        step = (polynomial - radius) / derivative
        theta -= step
        if np.max(np.abs(step), initial=0) < 1e-12:
            break
    else:
        raise ValueError('FishPoly inverse did not converge')
    if np.any(theta < 0) or np.any(theta >= np.pi / 2 - 1e-4):
        raise ValueError(
            'Board corners must lie in the forward camera hemisphere'
        )
    scale = np.divide(
        np.tan(theta), radius, out=np.ones_like(theta), where=radius > 1e-12
    )
    return np.column_stack((x * scale, y * scale))


def project(model, points):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if not np.isfinite(points).all() or np.any(points[:, 2] <= 0):
        raise ValueError('Board points are behind the camera or non-finite')
    if 'K' in model:
        k, d = _head_model(model)
        return cv2.projectPoints(points, np.zeros(3), np.zeros(3), k, d)[
            0
        ].reshape(-1, 2)
    (fx, skew, fy, cx, cy), coefficients = _fish_model(model)
    radius = np.linalg.norm(points[:, :2], axis=1)
    theta = np.arctan2(radius, points[:, 2])
    distorted = theta.copy()
    for power, coefficient in enumerate(coefficients, 2):
        distorted += coefficient * theta**power
    scale = np.divide(
        distorted, radius, out=1 / points[:, 2], where=radius > 1e-12
    )
    xy = points[:, :2] * scale[:, None]
    return np.column_stack(
        (fx * xy[:, 0] + skew * xy[:, 1] + cx, fy * xy[:, 1] + cy)
    )


def board_pose(model, corners, objects):
    normalized = unproject(model, corners)
    ok, rotation, translation = cv2.solvePnP(
        np.asarray(objects, dtype=float),
        normalized,
        np.eye(3),
        None,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise ValueError('Checkerboard pose estimation failed')
    result = pose(rotation, translation)
    if np.any(transform_points(result, objects)[:, 2] <= 0):
        raise ValueError('Checkerboard pose is behind the camera')
    return result


def _origin(element):
    xyz = np.fromstring(element.get('xyz', '0 0 0'), sep=' ')
    rpy = np.fromstring(element.get('rpy', '0 0 0'), sep=' ')
    if xyz.size != 3 or rpy.size != 3:
        raise ValueError('Invalid URDF joint origin')
    rx = cv2.Rodrigues(np.array([rpy[0], 0, 0]))[0]
    ry = cv2.Rodrigues(np.array([0, rpy[1], 0]))[0]
    rz = cv2.Rodrigues(np.array([0, 0, rpy[2]]))[0]
    transform = np.eye(4)
    transform[:3, :3] = rz @ ry @ rx
    transform[:3, 3] = xyz
    return rigid(transform)


class HeadKinematics:
    def __init__(self, urdf_text):
        try:
            robot = ET.fromstring(urdf_text)
        except ET.ParseError as error:
            raise ValueError(f'Invalid URDF XML: {error}') from error
        by_child = {
            joint.find('child').get('link'): joint
            for joint in robot.findall('joint')
        }
        self.chain = []
        link = 'rgb_head_center'
        visited = set()
        while link != 'torso_link':
            if link in visited or link not in by_child:
                raise ValueError(
                    'URDF has no valid torso_link -> rgb_head_center chain'
                )
            visited.add(link)
            joint = by_child[link]
            if joint.get('type') not in ('fixed', 'revolute', 'continuous'):
                raise ValueError('Unsupported head joint type')
            self.chain.insert(0, joint)
            link = joint.find('parent').get('link')

    def transform(self, positions):
        result = np.eye(4)
        for joint in self.chain:
            origin = joint.find('origin')
            if origin is not None:
                result = result @ _origin(origin)
            if joint.get('type') != 'fixed':
                angle = positions[joint.get('name')]
                axis_element = joint.find('axis')
                axis = np.fromstring(
                    (
                        axis_element.get('xyz', '1 0 0')
                        if axis_element is not None
                        else '1 0 0'
                    ),
                    sep=' ',
                )
                if (
                    axis.size != 3
                    or not np.isfinite(axis).all()
                    or np.linalg.norm(axis) < 1e-9
                ):
                    raise ValueError('Invalid head joint axis')
                motion = pose(axis / np.linalg.norm(axis) * angle, np.zeros(3))
                result = result @ motion
        return rigid(result)


def mean_transform(transforms):
    transforms = [rigid(t) for t in transforms]
    if not transforms:
        raise ValueError('No mounting transforms')
    u, _, vt = np.linalg.svd(sum(t[:3, :3] for t in transforms))
    result = np.eye(4)
    result[:3, :3] = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
    result[:3, 3] = np.mean([t[:3, 3] for t in transforms], axis=0)
    return rigid(result)


def difference(reference, estimate):
    reference, estimate = rigid(reference), rigid(estimate)
    rotation = inverse(reference) @ rigid(estimate)
    return (
        float(np.linalg.norm(reference[:3, 3] - estimate[:3, 3])),
        math.degrees(
            float(np.linalg.norm(cv2.Rodrigues(rotation[:3, :3])[0]))
        ),
    )


def transform_record(transform, parent, child):
    transform = rigid(transform)
    # Eigenvector of the symmetric quaternion matrix handles rotations near pi.
    r = transform[:3, :3]
    matrix = (
        np.array(
            [
                [
                    r[0, 0] - r[1, 1] - r[2, 2],
                    r[1, 0] + r[0, 1],
                    r[2, 0] + r[0, 2],
                    r[2, 1] - r[1, 2],
                ],
                [
                    r[1, 0] + r[0, 1],
                    r[1, 1] - r[0, 0] - r[2, 2],
                    r[2, 1] + r[1, 2],
                    r[0, 2] - r[2, 0],
                ],
                [
                    r[2, 0] + r[0, 2],
                    r[2, 1] + r[1, 2],
                    r[2, 2] - r[0, 0] - r[1, 1],
                    r[1, 0] - r[0, 1],
                ],
                [
                    r[2, 1] - r[1, 2],
                    r[0, 2] - r[2, 0],
                    r[1, 0] - r[0, 1],
                    np.trace(r),
                ],
            ]
        )
        / 3
    )
    quaternion = np.linalg.eigh(matrix)[1][:, -1]
    if quaternion[3] < 0:
        quaternion = -quaternion
    return {
        'parent_frame': parent,
        'child_frame': child,
        'translation_m': transform[:3, 3].tolist(),
        'quaternion_xyzw': quaternion.tolist(),
        'matrix': transform.tolist(),
    }
