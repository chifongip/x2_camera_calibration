import pytest

from x2_camera_calibration.capture import _parser
from x2_camera_calibration.limits import CaptureLimits


def test_training_quota_pauses_until_validation_selected():
    limits = CaptureLimits(train=30, validation=10)
    for _ in range(30):
        assert limits.can_capture('train')
        limits.record('train')
    assert not limits.finished
    assert not limits.can_capture('train')
    assert 'press V' in limits.pause_message('train')
    assert limits.can_capture('validation')
    with pytest.raises(ValueError, match='limit reached'):
        limits.record('train')
    for _ in range(10):
        limits.record('validation')
    assert limits.finished
    assert limits.total == 40
    assert not limits.can_capture('validation')


def test_early_validation_does_not_skip_training_quota():
    limits = CaptureLimits(train=30, validation=10)
    for _ in range(10):
        limits.record('validation')
    assert not limits.finished
    assert 'press T' in limits.pause_message('validation')
    assert limits.can_capture('train')


@pytest.mark.parametrize('role', ['train', 'validation'])
def test_single_role_quota_stops_without_another_stage(role):
    limits = CaptureLimits(**{role: 2})
    limits.record(role)
    limits.record(role)
    assert limits.finished


def test_legacy_total_limit_and_unlimited_defaults():
    unlimited = CaptureLimits()
    for _ in range(50):
        unlimited.record('train')
    assert not unlimited.finished
    limited = CaptureLimits(train=30, validation=10, total=2)
    limited.record('train')
    limited.record('validation')
    assert limited.finished


def test_validation_flag_accepts_requested_singular_alias():
    args = _parser().parse_args(
        [
            '--output',
            '/tmp/not-created',
            '--odin-calib',
            '/tmp/not-read',
            '--max-train-captures',
            '30',
            '--max-validation-capture',
            '10',
        ]
    )
    assert args.max_train_captures == 30
    assert args.max_validation_captures == 10
