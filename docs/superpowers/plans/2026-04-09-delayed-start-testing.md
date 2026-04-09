# Delayed Start Testing Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract `delayed_run` into a testable module-level function and add 6 unit tests covering its behaviour, plus document manual curl-based testing.

**Architecture:** The inner `delayed_run` closure in `kiln-controller.py` is lifted to a module-level function with explicit parameters (`oven`, `start_at`, `profile`, `watcher`). Tests in `Test/test_delayed_start.py` call this function synchronously with `time.time` and `gevent.sleep` patched to no-ops, using `MagicMock` for oven and watcher.

**Tech Stack:** Python 3, pytest, `unittest.mock` (stdlib), gevent

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `kiln-controller.py` | Modify lines ~223–243 | Lift `delayed_run` to module level |
| `Test/test_delayed_start.py` | Create | 6 unit tests for `delayed_run` |

---

### Task 1: Extract `delayed_run` into a module-level function

**Files:**
- Modify: `kiln-controller.py` (lines 223–243)

- [ ] **Step 1: Read the current schedule handler**

  Open `kiln-controller.py` and locate the `if bottle.request.json['cmd'] == 'schedule':` block (~line 223). Note the inner `delayed_run` closure and the `gevent.spawn(delayed_run)` call.

- [ ] **Step 2: Add the module-level function**

  Insert the following function **before** the `@app.post('/api')` decorator (i.e., at module level, not inside the handler). A good place is just above the `handle_api` function definition:

  ```python
  def delayed_run(oven, start_at, profile, watcher):
      delay = start_at - time.time()
      if delay > 0:
          gevent.sleep(delay)
      if oven.state != 'SCHEDULED':
          return
      oven.scheduled_start = 0
      oven.run_profile(profile)
      watcher.record(profile)
  ```

- [ ] **Step 3: Replace the call site**

  Inside the `schedule` handler, delete the inner `def delayed_run(): ...` definition and change the spawn line from:

  ```python
  gevent.spawn(delayed_run)
  ```

  to:

  ```python
  gevent.spawn(delayed_run, oven, start_at, scheduled_profile, ovenWatcher)
  ```

  The full `schedule` block should now look like:

  ```python
  if bottle.request.json['cmd'] == 'schedule':
      log.info("api schedule command received")
      wanted = bottle.request.json['profile']
      start_at = float(bottle.request.json['start_at'])
      profile = find_profile(wanted)
      if profile is None:
          return { "success": False, "error": "profile %s not found" % wanted }
      oven.scheduled_start = start_at
      oven.state = 'SCHEDULED'
      profile_json = json.dumps(profile)
      scheduled_profile = Profile(profile_json)
      gevent.spawn(delayed_run, oven, start_at, scheduled_profile, ovenWatcher)
  ```

- [ ] **Step 4: Verify the file has no syntax errors**

  ```bash
  cd /Users/donalddoyle/Documents/Kiln/kiln-controller
  source venv/bin/activate
  python3 -m py_compile kiln-controller.py && echo OK
  ```

  Expected: `OK`

- [ ] **Step 5: Commit the refactor**

  ```bash
  git add kiln-controller.py
  git commit -m "refactor: extract delayed_run to module-level function for testability"
  ```

---

### Task 2: Write the test file (all 6 tests)

**Files:**
- Create: `Test/test_delayed_start.py`

Write all tests **before** running them. Follow TDD: write the test, verify it fails, implement (already done — the function exists), verify it passes.

- [ ] **Step 1: Create `Test/test_delayed_start.py` with this content**

  ```python
  import sys
  import os
  import time
  from unittest.mock import MagicMock, patch, call

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

  @patch('gevent.sleep')
  @patch('time.time', return_value=1000.0)
  def test_schedule_calls_run_profile(mock_time, mock_sleep):
      """delayed_run calls oven.run_profile with the profile when state is SCHEDULED."""
      oven = make_oven('SCHEDULED')
      profile = make_profile()
      watcher = MagicMock()
      start_at = 1000.0  # same as now → no sleep

      delayed_run(oven, start_at, profile, watcher)

      oven.run_profile.assert_called_once_with(profile)


  @patch('gevent.sleep')
  @patch('time.time', return_value=1000.0)
  def test_schedule_clears_scheduled_start(mock_time, mock_sleep):
      """delayed_run sets oven.scheduled_start = 0 before calling run_profile."""
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


  @patch('gevent.sleep')
  @patch('time.time', return_value=1000.0)
  def test_cancel_prevents_run(mock_time, mock_sleep):
      """If oven.state is not SCHEDULED when delayed_run wakes, run_profile is never called."""
      oven = make_oven('IDLE')  # cancelled before greenlet woke
      profile = make_profile()
      watcher = MagicMock()

      delayed_run(oven, 1000.0, profile, watcher)

      oven.run_profile.assert_not_called()
      watcher.record.assert_not_called()


  @patch('gevent.sleep')
  @patch('time.time', return_value=1000.0)
  def test_past_start_at_fires_immediately(mock_time, mock_sleep):
      """When start_at is in the past (delay <= 0), gevent.sleep is not called."""
      oven = make_oven('SCHEDULED')
      profile = make_profile()
      watcher = MagicMock()
      start_at = 999.0  # 1 second in the past

      delayed_run(oven, start_at, profile, watcher)

      mock_sleep.assert_not_called()
      oven.run_profile.assert_called_once_with(profile)


  @patch('gevent.sleep')
  @patch('time.time', return_value=1000.0)
  def test_future_start_at_sleeps(mock_time, mock_sleep):
      """When start_at is in the future, gevent.sleep is called with the correct duration."""
      oven = make_oven('SCHEDULED')
      profile = make_profile()
      watcher = MagicMock()
      start_at = 1060.0  # 60 seconds from now

      delayed_run(oven, start_at, profile, watcher)

      mock_sleep.assert_called_once_with(60.0)


  @patch('gevent.sleep')
  @patch('time.time', return_value=1000.0)
  def test_watcher_record_called(mock_time, mock_sleep):
      """watcher.record() is called with the profile after a successful run."""
      oven = make_oven('SCHEDULED')
      profile = make_profile()
      watcher = MagicMock()

      delayed_run(oven, 1000.0, profile, watcher)

      watcher.record.assert_called_once_with(profile)
  ```

- [ ] **Step 2: Run the tests — expect them to fail (import issue or missing function)**

  ```bash
  cd /Users/donalddoyle/Documents/Kiln/kiln-controller
  source venv/bin/activate
  pytest Test/test_delayed_start.py -v
  ```

  If Task 1 is complete and `delayed_run` is at module level, tests should **pass** at this point. If any fail, debug the import shim first (the `importlib` block at the top of the test file).

- [ ] **Step 3: Fix any import issues and re-run until all 6 pass**

  Common issues:
  - `config` mock missing an attribute the module references at import time → add it to the `MagicMock` kwargs in the patch dict
  - `digitalio` or `board` imported by `lib/oven.py` at module level → add them to the patch dict too if needed

  Run after each fix:
  ```bash
  pytest Test/test_delayed_start.py -v
  ```

  Expected final output:
  ```
  Test/test_delayed_start.py::test_schedule_calls_run_profile PASSED
  Test/test_delayed_start.py::test_schedule_clears_scheduled_start PASSED
  Test/test_delayed_start.py::test_cancel_prevents_run PASSED
  Test/test_delayed_start.py::test_past_start_at_fires_immediately PASSED
  Test/test_delayed_start.py::test_future_start_at_sleeps PASSED
  Test/test_delayed_start.py::test_watcher_record_called PASSED
  6 passed
  ```

- [ ] **Step 4: Run the full test suite to confirm no regressions**

  ```bash
  pytest Test/ -v
  ```

  Expected: all existing `test_Profile.py` tests still pass alongside the 6 new tests.

- [ ] **Step 5: Commit the tests**

  ```bash
  git add Test/test_delayed_start.py
  git commit -m "test: add unit tests for delayed_run schedule logic"
  ```

---

### Task 3: Manual testing verification

**Files:** none (curl commands only)

**Prerequisite:** `simulate = True` must be set in `config.py`. Copy the test profile if needed:

```bash
cp Test/test-fast.json storage/profiles/test-fast.json
```

- [ ] **Step 1: Start the server in simulate mode**

  ```bash
  source venv/bin/activate
  ./kiln-controller.py
  ```

  Leave this running in a separate terminal for the steps below.

- [ ] **Step 2: Schedule a run 60 seconds from now**

  ```bash
  START=$(python3 -c "import time; print(time.time() + 60)")
  curl -s -X POST http://localhost:8081/api \
    -H 'Content-Type: application/json' \
    -d "{\"cmd\": \"schedule\", \"profile\": \"test-fast\", \"start_at\": $START}"
  ```

  Expected: `{"success": true}`

- [ ] **Step 3: Verify SCHEDULED state**

  ```bash
  curl -s http://localhost:8081/api/stats | python3 -m json.tool | grep -E '"state"|"scheduled_start"'
  ```

  Expected: `"state": "SCHEDULED"` and a non-zero `scheduled_start`.

- [ ] **Step 4: Cancel the schedule**

  ```bash
  curl -s -X POST http://localhost:8081/api \
    -H 'Content-Type: application/json' \
    -d '{"cmd": "cancel_schedule"}'
  curl -s http://localhost:8081/api/stats | python3 -m json.tool | grep -E '"state"|"scheduled_start"'
  ```

  Expected: `"state": "IDLE"`, `"scheduled_start": 0`.

- [ ] **Step 5: Schedule in the past (fires immediately)**

  ```bash
  START=$(python3 -c "import time; print(time.time() - 5)")
  curl -s -X POST http://localhost:8081/api \
    -H 'Content-Type: application/json' \
    -d "{\"cmd\": \"schedule\", \"profile\": \"test-fast\", \"start_at\": $START}"
  sleep 1
  curl -s http://localhost:8081/api/stats | python3 -m json.tool | grep '"state"'
  ```

  Expected: `"state": "RUNNING"`.

- [ ] **Step 6: Test invalid profile**

  ```bash
  START=$(python3 -c "import time; print(time.time() + 3600)")
  curl -s -X POST http://localhost:8081/api \
    -H 'Content-Type: application/json' \
    -d "{\"cmd\": \"schedule\", \"profile\": \"nonexistent\", \"start_at\": $START}"
  ```

  Expected: `{"success": false, "error": "profile nonexistent not found"}`

- [ ] **Step 7: Stop the server** (`Ctrl-C` in the server terminal)
