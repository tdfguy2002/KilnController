import sys
import os
import time as real_time
from unittest.mock import MagicMock, patch

# kiln-controller.py is not importable by name (hyphen), so load it via importlib.
import importlib.util
import unittest.mock as _mock

_kc_path = os.path.join(os.path.dirname(__file__), '..', 'kiln-controller.py')
_spec = importlib.util.spec_from_file_location("kiln_controller", _kc_path)
_mod = importlib.util.module_from_spec(_spec)

# kiln-controller.py runs module-level code on import:
#   - accesses config.kwh_rate, config.currency_type, config.simulate
#   - does `from oven import SimulatedOven, RealOven, Profile`
#   - does `from ovenWatcher import OvenWatcher`
#   - instantiates an oven and ovenWatcher
# Patch all of these before exec so no hardware is touched.
with _mock.patch.dict('sys.modules', {
    'bottle': _mock.MagicMock(),
    'gevent': _mock.MagicMock(),
    'gevent.pywsgi': _mock.MagicMock(),
    'geventwebsocket': _mock.MagicMock(),
    'geventwebsocket.handler': _mock.MagicMock(),
    'requests': _mock.MagicMock(),
    'digitalio': _mock.MagicMock(),
    'board': _mock.MagicMock(),
    'oven': _mock.MagicMock(),        # covers `from oven import ...`
    'ovenWatcher': _mock.MagicMock(), # covers `from ovenWatcher import OvenWatcher`
    'config': _mock.MagicMock(
        log_level='DEBUG',
        log_format='%(message)s',
        kiln_profiles_directory='storage/profiles',
        automatic_restart_state_file='/tmp/kiln_restart.json',
        simulate=True,          # real bool so `config.simulate == True` is True
        kwh_rate=0.15,          # accessed at module level in SETTINGS_DEFAULTS
        currency_type='USD',    # same
    ),
}):
    _spec.loader.exec_module(_mod)

delayed_run = _mod.delayed_run

# The module-level `gevent` reference used by delayed_run is the MagicMock
# that was injected via sys.modules at import time.  We patch against it directly.
# For `time.time`, delayed_run calls `time.time()` via the module's `time` attribute,
# which is the real `time` module.  We replace it with a MagicMock on the module.
_mod_gevent = _mod.gevent
_mock_time = MagicMock(wraps=real_time)
_mod.time = _mock_time


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_oven(state='SCHEDULED'):
    oven = MagicMock()
    oven.state = state
    oven.scheduled_start = 12345.0
    return oven


def make_profile():
    return MagicMock(name='test-fast')


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_schedule_calls_run_profile():
    """delayed_run calls oven.run_profile with the profile when state is SCHEDULED."""
    _mod_gevent.sleep.reset_mock()
    _mock_time.time.return_value = 1000.0

    oven = make_oven('SCHEDULED')
    profile = make_profile()
    watcher = MagicMock()
    start_at = 1000.0  # same as now → no sleep

    delayed_run(oven, start_at, profile, watcher)

    oven.run_profile.assert_called_once_with(profile)


def test_schedule_clears_scheduled_start():
    """delayed_run sets oven.scheduled_start = 0 before calling run_profile."""
    _mod_gevent.sleep.reset_mock()
    _mock_time.time.return_value = 1000.0

    cleared_before_run = []

    def check_state(*args, **kwargs):
        cleared_before_run.append(oven.scheduled_start)

    oven = make_oven('SCHEDULED')
    oven.run_profile.side_effect = check_state
    profile = make_profile()
    watcher = MagicMock()

    delayed_run(oven, 1000.0, profile, watcher)

    assert cleared_before_run == [0], (
        f"scheduled_start should be 0 when run_profile is called, got {cleared_before_run}"
    )


def test_cancel_prevents_run():
    """If oven.state is not SCHEDULED when delayed_run wakes, run_profile is never called."""
    _mod_gevent.sleep.reset_mock()
    _mock_time.time.return_value = 1000.0

    oven = make_oven('IDLE')  # cancelled before greenlet woke
    profile = make_profile()
    watcher = MagicMock()

    delayed_run(oven, 1000.0, profile, watcher)

    oven.run_profile.assert_not_called()
    watcher.record.assert_not_called()


def test_past_start_at_fires_immediately():
    """When start_at is in the past (delay <= 0), gevent.sleep is not called."""
    _mod_gevent.sleep.reset_mock()
    _mock_time.time.return_value = 1000.0

    oven = make_oven('SCHEDULED')
    profile = make_profile()
    watcher = MagicMock()
    start_at = 999.0  # 1 second in the past

    delayed_run(oven, start_at, profile, watcher)

    _mod_gevent.sleep.assert_not_called()
    oven.run_profile.assert_called_once_with(profile)


def test_future_start_at_sleeps():
    """When start_at is in the future, gevent.sleep is called with the correct duration."""
    _mod_gevent.sleep.reset_mock()
    _mock_time.time.return_value = 1000.0

    oven = make_oven('SCHEDULED')
    profile = make_profile()
    watcher = MagicMock()
    start_at = 1060.0  # 60 seconds from now

    delayed_run(oven, start_at, profile, watcher)

    _mod_gevent.sleep.assert_called_once_with(60.0)


def test_watcher_record_called():
    """watcher.record() is called with the profile after a successful run."""
    _mod_gevent.sleep.reset_mock()
    _mock_time.time.return_value = 1000.0

    oven = make_oven('SCHEDULED')
    profile = make_profile()
    watcher = MagicMock()

    delayed_run(oven, 1000.0, profile, watcher)

    watcher.record.assert_called_once_with(profile)
