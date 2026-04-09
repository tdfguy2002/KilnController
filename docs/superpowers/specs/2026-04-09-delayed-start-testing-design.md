# Delayed Start Testing Design

**Date:** 2026-04-09
**Topic:** Testing the delayed start (scheduled run) feature
**Status:** Approved

---

## Overview

The delayed start feature allows a kiln run to be scheduled for a future time. A user selects a profile and a future datetime in the UI; the server sets `oven.state = 'SCHEDULED'` and spawns a gevent greenlet that sleeps until `start_at`, then fires `run_profile()`. The schedule can be cancelled before it fires.

Goals: catch regressions AND document expected behavior as living tests.

---

## Production Code Refactor

### Problem

The `delayed_run` logic is an inner closure inside the `schedule` command handler in `kiln-controller.py`. Inner closures cannot be imported or called directly by tests.

### Solution

Extract `delayed_run` into a module-level function with explicit parameters:

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

The `schedule` handler call site becomes:

```python
gevent.spawn(delayed_run, oven, start_at, scheduled_profile, ovenWatcher)
```

(`watcher` is the parameter name inside `delayed_run`; the caller passes `ovenWatcher`.)

`delayed_run` intentionally omits `startat` and `allow_seek` — scheduled runs always start from the beginning of the profile. Note: `allow_seek` defaults to `True` in `run_profile`, but since `oven.state` is `'SCHEDULED'` (not `'IDLE'`) when `delayed_run` calls it, the seek-start branch is skipped regardless.

This is the only production code change. It improves testability without altering behavior.

**Known limitation:** There is a narrow race between the `oven.state != 'SCHEDULED'` check and the `oven.run_profile()` call. In practice, gevent's cooperative scheduling makes this window negligible, and it is accepted as-is.

---

## Automated Tests

**File:** `Test/test_delayed_start.py`
**Framework:** pytest
**Mocking:** `unittest.mock.patch` for `time.time` and `gevent.sleep`; `MagicMock` for oven and watcher

### Test Cases

| # | Test name | What it verifies |
|---|-----------|-----------------|
| 1 | `test_schedule_calls_run_profile` | `oven.run_profile` is called with the profile when state is SCHEDULED |
| 2 | `test_schedule_clears_scheduled_start` | `oven.scheduled_start` is set to `0` before `run_profile` is called |
| 3 | `test_cancel_prevents_run` | If `oven.state != 'SCHEDULED'` before greenlet wakes, `run_profile` is never called |
| 4 | `test_past_start_at_fires_immediately` | When `start_at` is in the past, `gevent.sleep` is not called |
| 5 | `test_future_start_at_sleeps` | When `start_at` is in the future, `gevent.sleep` is called with the correct duration |
| 6 | `test_watcher_record_called` | `watcher.record()` is called with the profile after a successful run |

All tests use `MagicMock` for `oven` and `watcher`. Test 1 asserts `oven.run_profile.assert_called_once_with(profile)` rather than checking `oven.state` directly — `run_profile` setting state to `RUNNING` is production code behaviour that is verified by the `Oven` class's own tests, not duplicated here.

Note: the `schedule` handler's own act of setting `oven.scheduled_start = start_at` before spawning is not tested here — that is handler behaviour, not `delayed_run` behaviour, and is covered by manual testing.

### Key Design Decisions

- `time.time` is patched so tests control the perceived "current time" without real waiting
- `gevent.sleep` is patched to a no-op so no real sleeping occurs
- `delayed_run` is called synchronously in tests (not via `gevent.spawn`) — tests the function logic, not gevent scheduling
- Each test constructs its own mock oven and watcher for full isolation

---

## Manual Testing (curl)

**Prerequisite:** Set `simulate = True` in `config.py`, then start the server: `./kiln-controller.py`

A profile named `test-fast` must exist in `storage/profiles/`. Copy it from the test fixtures if needed:
```bash
cp Test/test-fast.json storage/profiles/test-fast.json
```

### 1. Schedule a run 60 seconds from now

```bash
START=$(python3 -c "import time; print(time.time() + 60)")
curl -s -X POST http://localhost:8081/api \
  -H 'Content-Type: application/json' \
  -d "{\"cmd\": \"schedule\", \"profile\": \"test-fast\", \"start_at\": $START}"
# Expected: {"success": true}, oven state → SCHEDULED
```

### 2. Verify SCHEDULED state and countdown

```bash
curl -s http://localhost:8081/api/stats | python3 -m json.tool | grep -E 'state|scheduled_start'
# Expected: "state": "SCHEDULED", "scheduled_start": <timestamp>
```

### 3. Cancel the schedule

```bash
curl -s -X POST http://localhost:8081/api \
  -H 'Content-Type: application/json' \
  -d '{"cmd": "cancel_schedule"}'
# Expected: state → IDLE, scheduled_start → 0, profile never runs
curl -s http://localhost:8081/api/stats | python3 -m json.tool | grep -E 'state|scheduled_start'
```

### 4. Schedule in the past (fires immediately)

```bash
START=$(python3 -c "import time; print(time.time() - 5)")
curl -s -X POST http://localhost:8081/api \
  -H 'Content-Type: application/json' \
  -d "{\"cmd\": \"schedule\", \"profile\": \"test-fast\", \"start_at\": $START}"
# Expected: fires immediately → state → RUNNING
```

### 5. Schedule with invalid profile

```bash
# start_at is irrelevant here — the error fires before it is consulted
START=$(python3 -c "import time; print(time.time() + 3600)")
curl -s -X POST http://localhost:8081/api \
  -H 'Content-Type: application/json' \
  -d "{\"cmd\": \"schedule\", \"profile\": \"nonexistent\", \"start_at\": $START}"
# Expected: {"success": false, "error": "profile nonexistent not found"}
```

---

## Files Changed

| File | Change |
|------|--------|
| `kiln-controller.py` | Extract `delayed_run` closure → module-level function |
| `Test/test_delayed_start.py` | New: 6 unit tests for `delayed_run` |

No other files are modified.
