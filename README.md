# X2 / Odin camera mounting calibration

Passive capture and offline calibration for a torso-mounted Odin and the X2
head RGB camera. This package sends **no motion commands**, calls no device
services, and publishes no TF. It does not launch robot bringup or localization.
Factory intrinsics, URDF geometry, and tuned robot parameters remain unchanged.

This is a standalone ROS 2 package at `src/x2_camera_calibration`, alongside
`x2_description` and the other workspace source packages.

## Build

From the workspace root:

```bash
unset FASTRTPS_DEFAULT_PROFILES_FILE
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select x2_camera_calibration
source install/setup.bash
```

Dependencies are ROS Humble `rclpy`, `sensor_msgs`, `nav_msgs`, built AimDK
interfaces, NumPy, PyYAML, and OpenCV with `findChessboardCornersSB` and a GUI
backend. The offline commands do not import ROS or require SciPy. Keep the
robot's normal DDS profile for live capture; unsetting it above is only for
local builds/tests. Restore your normal robot runtime environment in the
terminal used for capture.

## Board and inputs

Use **8 columns x 6 rows of inner corners, 35 mm squares** (48 observations).
The inner-corner grid spans 245 x 175 mm; the surrounding squares are not part
of that inner-corner dimension. Mount the board flat on a rigid backing and
check the printed square size. Mark a physical corner outside the detected
grid so the same grid origin can be identified in both camera previews.

Default topics:

| Input | Topic |
| --- | --- |
| Head JPEG | `/aima/hal/sensor/rgb_head_front_center/rgb_image/compressed` |
| Head calibration | `/aima/hal/sensor/rgb_head_front_center/camera_info` |
| Head joints | `/aima/hal/joint/head/state` |
| Odin JPEG | `/odin1/image/compressed` |
| Odin internal calibration | `/odin1/wiwc` |

All topics can be overridden with `capture --help`. Images must have the same
dimensions as their calibration. Head `CameraInfo` uses all supplied distortion
coefficients, including its eight-coefficient rational model even if labeled
`plumb_bob`. Odin requires its original **FishPoly** `cam_0` calibration;
OpenCV's four-coefficient fisheye model is not interchangeable.

The head image frame must be `rgb_head_front_center`. The tool follows the
existing camera launch's identity alias from URDF `rgb_head_center`. Odin's
empty image frame is explicitly interpreted as `camera_0` optical coordinates.
The URDF camera axes are assumed to be optical: x right, y down, z forward.
These assumptions are included in every result for review.

## Capture

Open a terminal connected to the robot's ROS domain, source the workspace,
and use the **exact calibration belonging to the connected Odin**. Store local
sessions and results under the package's Git-ignored `data/` directory:

```bash
calibration_data=/home/ubuntu/x2_ws/src/x2_camera_calibration/data

ros2 run x2_camera_calibration capture \
  --output "$calibration_data/session_001" \
  --odin-calib /home/ubuntu/x2_ws/src/odin_ros_driver/config/calib.yaml \
  --urdf /home/ubuntu/x2_ws/src/x2_description/urdf/x2_ultra.urdf
```

The output directory must be new; capture creates it and any missing parent
directories. Use a different session name for each run. The package's `.gitignore`
excludes `data/`, including captures, calibration results, reports, and overlays.
Do not commit these generated files or device-specific calibration snapshots.
Back up important sessions separately; Git does not preserve ignored data.

1. Hold the robot and its head stationary. Both cameras must see the complete
   board with adequate lighting, focus, and no motion blur.
2. Inspect the red **0** corner in each preview. Press **R** to cycle Odin's
   grid ordering if necessary; press **C** to confirm the same physical corner is
   selected. The tool tracks corner numbering across frames and uses the
   confirmed torso-to-Odin-camera relationship to resolve later detector
   reversals, accounting for measured head angles. This reference selects corner
   order only; the offline solver still fits calibration from the full dataset.
3. Move the board between observations, then hold it still for at least one
   second. Press **Space** when READY. Collect **30 training pairs** at one head
   posture, varying board distance, tilt, and location across the shared view.
   Repeated captures of one pose do not provide adequate calibration diversity.
4. Press **V** to label subsequent captures as validation. Collect **10 or more
   independent board observations at three or more head postures**, with each
   posture separated from the others by more than one degree. Move the head
   using existing robot controls, then let it settle before capture. **T**
   switches back to training; keep all training at the original head posture.
5. Press **Q** or Escape to finish. Each observation is saved immediately.

The tool requires images within **30 ms** of one another, head-state samples
within **100 ms** on both sides of both exposures, and **0.5 s** of head history
with less than **0.2 degrees** of angular variation. Head timestamps use
`header.meas_stamp` if nonzero, otherwise `header.stamp`; no reception-time
substitution or arbitrary time offset is applied. Missing, stale, invalid, or
changed camera intrinsics are reported in the preview and block capture.
Odin may refine its `/odin1/wiwc` internal extrinsics while running. Each saved
observation includes its own internal matrices and their timestamp; small online
updates do not require restarting the session. Invalid matrices still block
capture. The solver uses each observation's matrices when converting the camera
calibration into the torso-to-IMU mounting transform. Older sessions without
per-observation matrices use their original session-wide snapshot.
Exposure synchronization is not guaranteed; stationary board capture is required.

The checkerboard detector can reverse row or column ordering. The tool tracks
all **four grid permutations** (identity, 180-degree reversal, row reversal, and
column reversal); **R** cycles these for initial confirmation. Preview and saved
observations use stabilized numbering. A brief missed detection blocks that pair
without discarding the confirmation or tracking history. Gaps longer than one
second trigger reacquisition.

After confirmation, the tool compares the four possible camera relationships.
The best ordering must be separated from the next-best by **45 degrees of
rotation disagreement** with the confirmed reference. The initial mounting pose
is approximate: translation/rotation deltas appear as diagnostics, with no
50 mm / 5 degree mounting-accuracy prerequisite for capture. Final accuracy is
assessed by the offline solver and independent validation.

Capture also requires **three distinct pairs spanning at least 0.3 seconds**
with quiet board corners. Motion above the larger of **2 pixels or 1% of the
observed board diagonal** restarts settling. Messages distinguish a moving board,
settling progress, and low corner-order confidence; console status is limited
to one update per second. Motion is checked in the image plane, so hold the
board still even if a small or distant motion passes this check.

A symmetric board cannot establish the first physical correspondence
automatically: inspect both previews before pressing **C**. After losing the
board, the visual origin may be reacquired at another end in both images;
ordering and settling are checked before saving. Use **R**, then recheck and
press **C**, to replace an incorrect initial reference. Saved observations record
the selected permutations and confidence diagnostics.

### Automatic capture

To save eligible image pairs automatically, at most once per second:

```bash
ros2 run x2_camera_calibration capture \
  --output "$calibration_data/session_002" \
  --odin-calib /home/ubuntu/x2_ws/src/odin_ros_driver/config/calib.yaml \
  --auto-capture \
  --max-train-captures 30 \
  --max-validation-captures 10
```

Verify matching physical corner **0** in both previews, press **R** if Odin's
ordering needs adjustment, then press **C** to enable capture. Move the board
between poses and hold it still for each observation. Detector reversals are
handled automatically after this initial confirmation. Image motion is checked,
but pose diversity is not; a stationary board will produce repeated
observations. After **30 training pairs**, saving pauses and prompts you to
prepare the validation head posture and press **V**. Collect the **10 validation
pairs across at least three stationary head postures**; capture then stops
automatically. Both roles are stored in the same session. Preview and console
show separate counts. Press **Q** to finish early.

`--max-train-captures` and `--max-validation-captures` default to **0** (unlimited).
The singular spelling `--max-validation-capture` is also accepted. With only one
role quota configured, reaching that quota ends capture. With both configured,
a full role pauses until you select the other with **T/V**, and capture ends
when both quotas are met. Switching to validation early does not discard the
remaining training quota. The legacy `--max-captures` still imposes a total cap
across both roles and can stop the run before its individual quotas are complete;
omit it for the staged workflow above.

`--headless --auto-capture --confirm-order --max-captures N --timeout SECONDS`
supports controlled recording and isolated tests. It asserts the operator has
already verified physical corner ordering. Image motion checks still apply;
scene quality and pose diversity require operator review. Normal interactive
capture is preferred on the robot.
Headless recording supports one role quota per session: for example,
`--role validation --max-validation-captures 10`. Combined training/validation
quotas require the preview so you can explicitly start the next stage.

## Solve and validate

```bash
calibration_data=/home/ubuntu/x2_ws/src/x2_camera_calibration/data

ros2 run x2_camera_calibration solve \
  --session "$calibration_data/session_001" \
  --output "$calibration_data/result_001"

ros2 run x2_camera_calibration validate \
  --session "$calibration_data/session_001" \
  --calibration "$calibration_data/result_001/calibration.yaml" \
  --output "$calibration_data/validation_001"
```

Output directories must be new. A later validation session can be captured with
`--role validation`; camera intrinsics and the URDF snapshot must match the
training session, and observation IDs must not overlap. Internal extrinsics are
recorded independently for each observation. Capture generates globally unique
IDs. Solve and validate exit **0** for accepted, **2** for requires-review, and
**1** for invalid input or processing failure. Candidate calibrations and reports
are written even when commissioning targets are missed.

Files include:

- `session.yaml`, original JPEG pairs, `robot.urdf`, and `odin_calib.yaml`:
  captured corners, roles, stamps, head-state history, per-observation internal
  extrinsics, and input fingerprints.
- `calibration.yaml`: matrices, translation in meters, **x/y/z/w** quaternions,
  frames, assumptions, retained/rejected training IDs, and source calibration.
- `report.yaml`: native-image reprojection RMS/p95 and per-observation mounting
  repeatability, acceptance status, and explicit failures.
- `overlays/`: validation images with observed corners in green and predicted
  corners as red crosses.

`T_A_B` maps coordinates **from B into A**. The solver converts both lens models
to a common virtual pinhole camera and uses fixed-intrinsic stereo calibration.
Robust initialization and reprojection rejection remove inconsistent pairs;
at least **20 training pairs** must remain. Training head angles must span no
more than 0.2 degrees. Validation estimates are excluded from fitting.

The two public results are `T_head_camera_odin_camera` and `T_torso_odin_imu`.
The mounting chain is:

```text
T_imu_camera = T_imu_lidar * inverse(T_camera_lidar)
T_torso_imu = T_torso_head_camera * T_head_camera_odin_camera * inverse(T_imu_camera)
```

`/odin1/wiwc` is used only as the driver's internal-calibration container: its
first 16 pose-covariance entries encode `T_camera_lidar`, and first 16
twist-covariance entries encode `T_imu_lidar`. Its pose is not robot odometry.
Head kinematics come directly from the specified URDF and measured joint angles;
the tool does not require another state publisher or controller manager.

Initial acceptance targets are **<=1.5 px RMS**, **<=3 px p95** for each camera,
and independent mounting estimates within **10 mm and 1 degree** of the fitted
mount at all validation observations. At least 10 validation pairs and three
head postures are required. These are commissioning targets, not an accuracy
guarantee. Inspect coverage and overlays even when all numeric checks pass.
Head-angle-dependent error suggests URDF camera geometry or joint offsets need
investigation; this tool does not silently adjust them.

Applying the mounting transform to TF or replacing FAST-LIO is a separate
integration step. Do not simply add another parent for Odin's existing `imu`
frame while its driver still publishes `odom -> imu`.

## Tests

```bash
unset FASTRTPS_DEFAULT_PROFILES_FILE
source /opt/ros/humble/setup.bash
colcon test --packages-select x2_camera_calibration --event-handlers console_direct+
colcon test-result --test-result-base build/x2_camera_calibration --verbose
```

Tests exercise mixed-model metric recovery with noise, independent head poses,
reversed corners/outliers, temporal corner-order stabilization and cross-camera
agreement, stale state and pairing, calibration fingerprints,
and a real-ROS headless capture subprocess on an isolated DDS domain. They
produce no robot commands. The source-workspace URDF check is skipped when the
description repository is absent.

FishPoly equations are adapted from Manifold Tech Ltd.'s Apache-2.0
`odin_ros_driver/include/polynomial_camera.hpp`; the inverse is iterated to
convergence and optical-axis limits are guarded.
