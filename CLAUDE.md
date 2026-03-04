# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Setup
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt

# Run
source venv/bin/activate
./kiln-controller.py          # start the web server (port 8081)

# Tests
pytest Test/                  # run all tests
pytest Test/test_Profile.py   # run a single test file

# Hardware diagnostics
./test-thermocouple.py        # verify thermocouple reads
./test-output.py              # verify SSR GPIO output
./gpioreadall.py              # inspect all GPIO pin states

# PID autotuner (heats kiln to 400F, calculates Kp/Ki/Kd)
./kiln-tuner.py -t 400
./kiln-tuner.py -c            # calculate only from existing tuning.csv

# Remote logger (connects via WebSocket, writes CSV)
./kiln-logger.py --hostname localhost:8081 --csvfile /tmp/kilnstats.csv --pidstats --stdout

# Slack watcher (alerts when kiln deviates from schedule)
./watcher.py
```

## Architecture

The system is a Bottle web server (`kiln-controller.py`) that exposes a REST API and WebSocket endpoints. A background `Oven` thread runs the PID control loop, and `OvenWatcher` broadcasts state updates to connected WebSocket clients.

### Control loop

`Oven` (base class in `lib/oven.py`) runs a thread that every `sensor_time_wait` seconds:
1. Calls `update_runtime()` and `update_target_temp()` (interpolating from the profile)
2. Calls `heat_then_cool()` — runs PID, then turns SSR on for `heat_on` seconds and off for `heat_off` seconds within the duty cycle
3. Checks for emergency shutoff and schedule completion

`RealOven` and `SimulatedOven` both extend `Oven`. The simulator models thermal mass using `c_heat`, `c_oven`, `R_ho`, and `R_o_nocool` parameters.

### Temperature sensing

`TempSensorReal` (base) runs its own thread, calling `get_temperature()` in a loop, adding samples to `TempTracker` (sliding median window). `temperature()` returns the current median.

Two sensor drivers: `Max31855` and `Max31856`. Both support two paths:
- **spidev path** (`use_spidev_tc = True`): raw Linux SPI — required on Pi 5 / Debian Trixie where Blinka/lgpio conflicts arise
- **Blinka path**: Adafruit CircuitPython libraries via `board.SPI()`

`ThermocoupleError` normalizes fault strings from both drivers. Faults are mapped to canonical names (e.g. `"not connected"`, `"short circuit"`) and checked against `ignore_tc_*` config flags.

### Profile format

Profiles are JSON files in `storage/profiles/`:
```json
{"name": "cone-6", "data": [[0, 70], [3600, 200], [14400, 2250]], "temp_units": "c"}
```
`data` is a list of `[seconds, temperature]` waypoints. `Profile.get_target_temperature(t)` linearly interpolates between surrounding points. Profiles are always stored in °C internally; `normalize_temp_units()` converts on load for display.

### WebSocket API

| Route | Purpose |
|-------|---------|
| `/control` | Send `RUN`/`STOP`/`SIMULATE` commands |
| `/storage` | `GET`/`PUT`/`DELETE` profiles |
| `/status` | Subscribe to oven state updates |
| `/config` | Read display config (temp scale, kwh rate) |

REST: `POST /api` accepts `{"cmd": "run"|"pause"|"resume"|"stop"|"stats", ...}`. `GET /api/stats` returns PID stats.

### config.py

All hardware and runtime settings live here. Key flags:
- `simulate` — use `SimulatedOven` instead of `RealOven`
- `use_spidev_tc` — bypass Blinka SPI (required on Pi 5)
- `max31855` / `max31856` — select thermocouple chip
- `temp_scale` — `"f"` or `"c"` (must be lowercase; comparisons in kiln-controller.py are case-sensitive)
- `pid_kp`, `pid_ki`, `pid_kd` — PID gains (use `kiln-tuner.py` to determine these)
- `kiln_must_catch_up` — shifts schedule forward/back when kiln is outside `pid_control_window`

## Known Issues

- `oven.py` `get_avg_temp()` has an unused `chop=25` parameter — dead code from a previous implementation.
- `oven.py` `_decode_cj()` missing sign extension for `cjth` byte; returns wrong value for sub-zero cold junction temps (diagnostics only, doesn't affect PID).
- `ovenWatcher.py` / `kiln-controller.py` use bare `except:` in a few places — swallows `KeyboardInterrupt` and `SystemExit`.
