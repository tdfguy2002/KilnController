import threading
import time
import datetime
import logging
import json
import config
import os
import statistics

import digitalio  # still used for SSR output / other GPIO

log = logging.getLogger(__name__)


# ----------------------------
# Duplicate-log filter helper
# ----------------------------
class DupFilter(logging.Filter):
    """
    Simple duplicate-message filter: suppresses back-to-back identical log lines.
    """
    def __init__(self):
        super().__init__()
        self._last = None

    def filter(self, record):
        msg = record.getMessage()
        if msg == self._last:
            return False
        self._last = msg
        return True


class Duplogger:
    def __init__(self):
        self.log = logging.getLogger(f"{__name__}.dupfree")
        self.log.addFilter(DupFilter())

    def logref(self):
        return self.log


duplog = Duplogger().logref()


# ----------------------------
# GPIO Output (SSR control)
# ----------------------------
class Output:
    """
    GPIO output controlling SSR.

    Requires:
      config.gpio_heat
      config.gpio_heat_invert  (bool)  (if missing, defaults False)
    """
    def __init__(self):
        self.active = False
        self.heater = digitalio.DigitalInOut(config.gpio_heat)
        self.heater.direction = digitalio.Direction.OUTPUT

        invert = getattr(config, "gpio_heat_invert", False)
        self.off = invert
        self.on = not invert

    def heat(self, sleepfor: float):
        self.heater.value = self.on
        time.sleep(sleepfor)

    def cool(self, sleepfor: float):
        self.heater.value = self.off
        time.sleep(sleepfor)


# ----------------------------
# Board abstraction
# ----------------------------
class Board:
    def __init__(self):
        log.info("board: %s", self.name)
        self.temp_sensor.start()


class RealBoard(Board):
    def __init__(self):
        self.name = None
        self.load_libs()
        self.temp_sensor = self.choose_tempsensor()
        super().__init__()

    def load_libs(self):
        import board
        self.name = board.board_id

    def choose_tempsensor(self):
        # Support both naming styles seen in various forks/configs
        use_55 = getattr(config, "max31855", None)
        if use_55 is None:
            use_55 = getattr(config, "use_max31855", False)

        use_56 = getattr(config, "max31856", None)
        if use_56 is None:
            use_56 = getattr(config, "use_max31856", False)

        if use_55:
            return Max31855()
        if use_56:
            return Max31856()

        raise RuntimeError("No thermocouple selected in config (max31855/max31856)")


class SimulatedBoard(Board):
    def __init__(self):
        self.name = "simulated"
        self.temp_sensor = TempSensorSimulated()
        super().__init__()


# ----------------------------
# Temperature sensor threads
# ----------------------------
class TempSensor(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.time_step = getattr(config, "sensor_time_wait", 2)
        self.status = ThermocoupleTracker()


class TempSensorSimulated(TempSensor):
    def __init__(self):
        super().__init__()
        self.simulated_temperature = getattr(config, "sim_t_env", 25.0)

    def temperature(self):
        return self.simulated_temperature


class TempSensorReal(TempSensor):
    """
    Real sensor base class: averages multiple reads across each time_step.

    Requires:
      config.temperature_average_samples (int)
      config.temp_scale ("c" or "f")  (if missing, defaults "f")
    """
    def __init__(self):
        super().__init__()
        samples = float(getattr(config, "temperature_average_samples", 10))
        self.sleeptime = self.time_step / samples
        self.temptracker = TempTracker()

        # IMPORTANT:
        # If using spidev for thermocouple, do NOT create Blinka SPI or DigitalInOut CS here.
        # That avoids lgpio claiming SPI pins / knocking them out of ALT mode.
        if getattr(config, "use_spidev_tc", False):
            self.spi = None
            self.cs = None
            log.info("TempSensor: using spidev for thermocouple (skipping Blinka SPI + CS)")
        else:
            self.spi_setup()
            # Requires config.spi_cs in Blinka mode
            self.cs = digitalio.DigitalInOut(config.spi_cs)

    def spi_setup(self):
        """
        Configure SPI for thermocouple reads (Blinka path).

        If you ever disable config.use_spidev_tc, this function tries:
          - bitbangio.SPI if spi_sclk/spi_mosi/spi_miso exist
          - otherwise board.SPI()
        """
        if getattr(config, "use_spidev_tc", False):
            # Just in case someone calls this while spidev mode is on:
            log.info("SPI setup skipped (use_spidev_tc True)")
            self.spi = None
            return

        # Software SPI (bitbangio) if pins provided
        if (hasattr(config, "spi_sclk") and hasattr(config, "spi_mosi") and hasattr(config, "spi_miso")):
            import adafruit_bitbangio as bitbangio
            self.spi = bitbangio.SPI(config.spi_sclk, config.spi_mosi, config.spi_miso)
            log.info("Software SPI selected for reading thermocouple")
        else:
            import board
            self.spi = board.SPI()
            log.info("Hardware SPI selected for reading thermocouple")

    def get_temperature(self):
        """Read from the thermocouple and convert to configured scale."""
        try:
            temp_c = self.raw_temp()  # subclasses implement raw_temp() returning C
            scale = getattr(config, "temp_scale", "f").lower()
            if scale == "f":
                temp = (temp_c * 9 / 5) + 32
            else:
                temp = temp_c

            self.status.good()
            return temp

        except ThermocoupleError as tce:
            if tce.ignore:
                log.error("Problem reading temp (ignored) %s", tce.message)
                self.status.good()
            else:
                log.error("Problem reading temp %s", tce.message)
                self.status.bad()

        return None

    def temperature(self):
        """Average temp over a duty cycle."""
        return self.temptracker.get_avg_temp()

    def run(self):
        while True:
            temp = self.get_temperature()
            if temp is not None:
                self.temptracker.add(temp)
            time.sleep(self.sleeptime)


class TempTracker:
    """Sliding window of temperatures (median)."""
    def __init__(self):
        self.size = int(getattr(config, "temperature_average_samples", 10))
        self.temps = [0 for _ in range(self.size)]

    def add(self, temp):
        self.temps.append(temp)
        while len(self.temps) > self.size:
            del self.temps[0]

    def get_avg_temp(self, chop=25):
        # Using median is robust against spikes
        return statistics.median(self.temps)


class ThermocoupleTracker:
    """Tracks good/bad reads over last ~2 duty cycles."""
    def __init__(self):
        samples = int(getattr(config, "temperature_average_samples", 10))
        self.size = samples * 2
        self.status = [True for _ in range(self.size)]
        self.limit = 30  # percent

    def good(self):
        self.status.append(True)
        del self.status[0]

    def bad(self):
        self.status.append(False)
        del self.status[0]

    def error_percent(self):
        errors = sum(s is False for s in self.status)
        return (errors / self.size) * 100.0

    def over_error_limit(self):
        return self.error_percent() > self.limit


# ----------------------------
# Thermocouple errors (common)
# ----------------------------
class ThermocoupleError(Exception):
    """
    Normalizes/Maps errors from thermocouple libs and marks whether to ignore
    based on config flags.
    """
    def __init__(self, message):
        self.ignore = False
        self.message = message
        self.map_message()
        self.set_ignore()
        super().__init__(self.message)

    def set_ignore(self):
        # All these flags may or may not exist in config; default False
        def cfg(name):
            return bool(getattr(config, name, False))

        if self.message == "not connected" and cfg("ignore_tc_lost_connection"):
            self.ignore = True
        if self.message == "short circuit" and cfg("ignore_tc_short_errors"):
            self.ignore = True
        if self.message == "unknown" and cfg("ignore_tc_unknown_error"):
            self.ignore = True
        if self.message == "cold junction range fault" and cfg("ignore_tc_cold_junction_range_error"):
            self.ignore = True
        if self.message == "thermocouple range fault" and cfg("ignore_tc_range_error"):
            self.ignore = True
        if self.message == "cold junction temp too high" and cfg("ignore_tc_cold_junction_temp_high"):
            self.ignore = True
        if self.message == "cold junction temp too low" and cfg("ignore_tc_cold_junction_temp_low"):
            self.ignore = True
        if self.message == "thermocouple temp too high" and cfg("ignore_tc_temp_high"):
            self.ignore = True
        if self.message == "thermocouple temp too low" and cfg("ignore_tc_temp_low"):
            self.ignore = True
        if self.message == "voltage too high or low" and cfg("ignore_tc_voltage_error"):
            self.ignore = True

    def map_message(self):
        try:
            self.message = self.map[self.orig_message]
        except Exception:
            self.message = "unknown"


class Max31855_Error(ThermocoupleError):
    def __init__(self, message):
        self.orig_message = message
        # Map Adafruit MAX31855 RuntimeError strings to normalized ones
        self.map = {
            "thermocouple not connected": "not connected",
            "short circuit to ground": "short circuit",
            "short circuit to power": "short circuit",
        }
        super().__init__(message)


class Max31856_Error(ThermocoupleError):
    def __init__(self, message):
        self.orig_message = message
        self.map = {
            "cj_range": "cold junction range fault",
            "tc_range": "thermocouple range fault",
            "cj_high":  "cold junction temp too high",
            "cj_low":   "cold junction temp too low",
            "tc_high":  "thermocouple temp too high",
            "tc_low":   "thermocouple temp too low",
            "voltage":  "voltage too high or low",
            "open_tc":  "not connected",
        }
        super().__init__(message)


# ----------------------------
# MAX31855 via spidev or Blinka
# ----------------------------
class Max31855(TempSensorReal):
    """Thermocouple MAX31855 reader."""
    def __init__(self):
        super().__init__()
        log.info("thermocouple MAX31855")

        self._use_spidev = bool(getattr(config, "use_spidev_tc", False))
        self._spi = None
        self.thermocouple = None

        if self._use_spidev:
            import spidev
            bus = int(getattr(config, "spi_bus", 0))
            dev = int(getattr(config, "spi_dev", 0))
            hz = int(getattr(config, "spi_max_speed_hz", 500000))

            self._spi = spidev.SpiDev()
            self._spi.open(bus, dev)
            self._spi.max_speed_hz = hz
            self._spi.mode = 0
            log.info("MAX31855: using spidev%d.%d @ %d Hz", bus, dev, hz)
        else:
            import adafruit_max31855
            # Requires self.spi + self.cs from TempSensorReal (Blinka mode)
            self.thermocouple = adafruit_max31855.MAX31855(self.spi, self.cs)

    def _read_spidev_frame(self) -> int:
        b = self._spi.xfer2([0, 0, 0, 0])  # clock out 32 bits
        return (b[0] << 24) | (b[1] << 16) | (b[2] << 8) | b[3]

    @staticmethod
    def _decode_temp_c(v: int) -> float:
        """
        MAX31855 format:
          bits 31..18: signed 14-bit thermocouple temp, 0.25C/LSB
          bit 16: fault flag
          bits 2..0: fault code
        """
        fault_flag = (v >> 16) & 0x1
        fault_bits = v & 0x7
        if fault_flag:
            raise Max31855_Error(f"fault={fault_bits} raw=0x{v:08x}")

        tc = (v >> 18) & 0x3FFF
        if tc & 0x2000:  # sign
            tc -= 0x4000
        return tc * 0.25

    def raw_temp(self):
        try:
            if self._use_spidev:
                v = self._read_spidev_frame()
                return self._decode_temp_c(v)

            # Blinka path
            return self.thermocouple.temperature_NIST

        except RuntimeError as rte:
            # Adafruit MAX31855 raises RuntimeError strings
            if rte.args and rte.args[0]:
                raise Max31855_Error(rte.args[0])
            raise Max31855_Error("unknown")


# ----------------------------
# MAX31856 (kept for completeness)
# ----------------------------
#class Max31856(TempSensorReal):
#    def __init__(self):
#        super().__init__()
#        log.info("thermocouple MAX31856")
#        import adafruit_max31856
#
#        tc_type = getattr(config, "thermocouple_type", adafruit_max31856.ThermocoupleType.K)
#        self.thermocouple = adafruit_max31856.MAX31856(self.spi, self.cs, thermocouple_type=tc_type)
#
#        if getattr(config, "ac_freq_50hz", False):
#            self.thermocouple.noise_rejection = 50
#        else:
#            self.thermocouple.noise_rejection = 60
#
#    def raw_temp(self):
#        temp = self.thermocouple.temperature
#        for k, v in self.thermocouple.fault.items():
#            if v:
#                raise Max31856_Error(k)
#        return temp

class Max31856(TempSensorReal):
    """
    MAX31856 thermocouple reader using Linux spidev (reliable on Debian Trixie / Python 3.13).

    - Uses /dev/spidev<spi_bus>.<spi_dev>
    - Configures continuous conversion + 50/60Hz mains filter
    - Thermocouple type via config.thermocouple_type ("K","J","T","N","S","E","B","R"), default "K"
    - Returns a PID-friendly temperature via optional EMA smoothing (config.tc_ema_alpha)
    - Raises Max31856_Error with a fault key name (similar to the Adafruit fault dict)
    """

    # MAX31856 registers
    _CR0   = 0x00
    _CR1   = 0x01
    _MASK  = 0x02
    _SR    = 0x0F
    _CJTH  = 0x0A
    _CJTL  = 0x0B
    _LTCBH = 0x0C
    _LTCBM = 0x0D
    _LTCBL = 0x0E

    # Status register fault bits -> names
    _fault_bits = {
        0: "open_tc",
        1: "voltage",
        2: "tc_low",
        3: "tc_high",
        4: "cj_low",
        5: "cj_high",
        6: "tc_range",
        7: "cj_range",
    }

    # Thermocouple type mapping for CR1[3:0]
    _tc_type_map = {"K": 0, "J": 1, "T": 2, "N": 3, "S": 4, "E": 5, "B": 6, "R": 7}

    def __init__(self):
        # TempSensorReal sets up pin selections for Blinka mode; we still call it
        # to keep behavior consistent with the rest of the project.
        super().__init__()
        #log.info("thermocouple MAX31856 (spidev)")

        log.warning("### USING SPIDEV MAX31856 CLASS ###")

        import spidev

        self._spi = spidev.SpiDev()
        self._spi.open(getattr(config, "spi_bus", 0), getattr(config, "spi_dev", 0))
        self._spi.max_speed_hz = getattr(config, "spi_max_speed_hz", 500000)
        self._spi.mode = 0b01  # MAX31856 uses SPI mode 1

        # Continuous conversion + 50/60Hz filter (match the old adafruit noise_rejection knob)
        # CR0: bit7 CMODE=1 (continuous), bit0 FILTER (1=60Hz, 0=50Hz)
        filter_bit = 0x00 if getattr(config, "ac_freq_50hz", False) else 0x01
        cr0 = 0x80 | filter_bit

        # Thermocouple type in CR1[3:0]; default to K
        tc_type = getattr(config, "thermocouple_type", "K")
        if isinstance(tc_type, str):
            cr1 = self._tc_type_map.get(tc_type.upper(), 0)
        else:
            # If someone left an Adafruit enum here, fall back to K
            cr1 = 0

        # Unmask all faults so SR reports them
        self._w(self._MASK, [0x00])

        # Apply configuration
        self._w(self._CR0, [cr0])
        self._w(self._CR1, [cr1])

        # PID-friendly smoothing (optional)
        self._ema_alpha = getattr(config, "tc_ema_alpha", 0.2)
        self._t_filt = None

        # Clear any latched fault status by reading SR once
        _ = self._r(self._SR, 1)[0]

    def _r(self, reg, n=1):
        """Read n bytes starting at reg."""
        resp = self._spi.xfer2([reg & 0x7F] + [0x00] * n)
        return resp[1:]

    def _w(self, reg, data):
        """Write bytes to reg."""
        self._spi.xfer2([reg | 0x80] + list(data))

    @staticmethod
    def _decode_tc(b2, b1, b0):
        """
        Decode thermocouple temperature.
        MAX31856 LTC bytes are 24-bit; top 19 bits are signed with 1/128°C LSB.
        """
        raw = (b2 << 16) | (b1 << 8) | b0
        raw >>= 5  # drop lower 5 bits
        if raw & (1 << 18):  # sign bit for 19-bit value
            raw -= (1 << 19)
        return raw * 0.0078125  # 1/128

    @staticmethod
    def _decode_cj(cjth, cjtl):
        """Decode cold junction temperature (for debugging/optional use)."""
        return cjth + ((cjtl >> 4) * 0.0625)

    def _check_faults(self):
        """Raise Max31856_Error with a fault key if SR indicates a fault."""
        sr = self._r(self._SR, 1)[0]
        for bit, name in self._fault_bits.items():
            if sr & (1 << bit):
                raise Max31856_Error(name)

    def raw_temp(self):
        """
        Return the thermocouple temperature in °C (PID input).
        Applies optional EMA smoothing to reduce noise.
        """
        try:
            self._check_faults()

            b2, b1, b0 = self._r(self._LTCBH, 3)
            t = self._decode_tc(b2, b1, b0)

            # Optional smoothing
            if self._t_filt is None:
                self._t_filt = t
            else:
                a = float(self._ema_alpha)
                self._t_filt = a * t + (1.0 - a) * self._t_filt

            return self._t_filt

        except Max31856_Error:
            raise
        except Exception as e:
            raise Max31856_Error(str(e))

    def reference_temp(self):
        """
        Optional helper: cold-junction temperature in °C.
        Not required for PID, but useful for diagnostics.
        """
        try:
            cjth, cjtl = self._r(self._CJTH, 2)
            return self._decode_cj(cjth, cjtl)
        except Exception as e:
            raise Max31856_Error(str(e))

    def cleanup(self):
        """Optional cleanup hook if the project calls it."""
        try:
            if getattr(self, "_spi", None) is not None:
                self._spi.close()
        except Exception:
            pass

# ----------------------------
# Oven logic
# ----------------------------
class Oven(threading.Thread):
    """Common oven code for real or simulated ovens."""
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.temperature = 0
        self.time_step = getattr(config, "sensor_time_wait", 2)
        self.reset()

    def reset(self):
        self.cost = 0
        self.state = "IDLE"
        self.profile = None
        self.start_time = 0
        self.runtime = 0
        self.totaltime = 0
        self.target = 0
        self.heat = 0
        self.heat_rate = 0
        self.heat_rate_temps = []
        self.pid = PID(ki=getattr(config, "pid_ki", 1),
                       kd=getattr(config, "pid_kd", 1),
                       kp=getattr(config, "pid_kp", 1))
        self.catching_up = False

    @staticmethod
    def get_start_from_temperature(profile, temp):
        target_temp = profile.get_target_temperature(0)
        if temp > target_temp + 5:
            startat = profile.find_next_time_from_temperature(temp)
            log.info("seek_start is in effect, starting at: %s s, %s deg", round(startat), round(temp))
        else:
            startat = 0
        return startat

    def set_heat_rate(self, runtime, temp):
        numtemps = 60
        self.heat_rate_temps.append((runtime, temp))
        if len(self.heat_rate_temps) > numtemps:
            self.heat_rate_temps = self.heat_rate_temps[-numtemps:]

        time2 = self.heat_rate_temps[-1][0]
        time1 = self.heat_rate_temps[0][0]
        temp2 = self.heat_rate_temps[-1][1]
        temp1 = self.heat_rate_temps[0][1]
        if time2 > time1:
            self.heat_rate = ((temp2 - temp1) / (time2 - time1)) * 3600

    def run_profile(self, profile, startat=0, allow_seek=True):
        log.debug("run_profile run on thread %s", threading.current_thread().name)
        runtime = startat * 60
        if allow_seek and self.state == "IDLE" and getattr(config, "seek_start", False):
            temp = self.board.temp_sensor.temperature()
            runtime += self.get_start_from_temperature(profile, temp)

        self.reset()
        self.startat = startat * 60
        self.runtime = runtime
        self.start_time = datetime.datetime.now() - datetime.timedelta(seconds=self.startat)
        self.profile = profile
        self.totaltime = profile.get_duration()
        self.state = "RUNNING"
        log.info("Running schedule %s starting at %d minutes", profile.name, startat)
        log.info("Starting")

    def abort_run(self):
        self.reset()
        self.save_automatic_restart_state()

    def get_start_time(self):
        return datetime.datetime.now() - datetime.timedelta(milliseconds=self.runtime * 1000)

    def kiln_must_catch_up(self):
        if getattr(config, "kiln_must_catch_up", False):
            temp = self.board.temp_sensor.temperature() + getattr(config, "thermocouple_offset", 0)

            # added line below per chatGPT recommendation
            window = getattr(config, "pid_control_window", 25)  # degrees F
            #window = getattr(config, "pid_control_window", 0)
            if self.target - temp > window:
                log.info("kiln must catch up, too cold, shifting schedule")
                self.start_time = self.get_start_time()
                self.catching_up = True
                return

            if temp - self.target > window:
                log.info("kiln must catch up, too hot, shifting schedule")
                self.start_time = self.get_start_time()
                self.catching_up = True
                return

            self.catching_up = False

    def update_runtime(self):
        runtime_delta = datetime.datetime.now() - self.start_time
        if runtime_delta.total_seconds() < 0:
            runtime_delta = datetime.timedelta(0)
        self.runtime = runtime_delta.total_seconds()

    def update_target_temp(self):
        self.target = self.profile.get_target_temperature(self.runtime)

    def reset_if_emergency(self):
        offset = getattr(config, "thermocouple_offset", 0)
        emergency = getattr(config, "emergency_shutoff_temp", 99999)

        if self.board.temp_sensor.temperature() + offset >= emergency:
            log.info("emergency!!! temperature too high")
            if not getattr(config, "ignore_temp_too_high", False):
                self.abort_run()

        if self.board.temp_sensor.status.over_error_limit():
            log.info("emergency!!! too many errors in a short period")
            if not getattr(config, "ignore_tc_too_many_errors", False):
                self.abort_run()

    def reset_if_schedule_ended(self):
        if self.runtime > self.totaltime:
            log.info("schedule ended, shutting down")
            log.info("total cost = %s%.2f", getattr(config, "currency_type", "$"), self.cost)
            self.abort_run()

    def update_cost(self):
        if self.heat:
            cost = (getattr(config, "kwh_rate", 0) * getattr(config, "kw_elements", 0)) * (self.heat / 3600)
        else:
            cost = 0
        self.cost += cost

    def get_state(self):
        offset = getattr(config, "thermocouple_offset", 0)
        try:
            temp = self.board.temp_sensor.temperature() + offset
        except AttributeError:
            temp = 0

        self.set_heat_rate(self.runtime, temp)

        return {
            "cost": self.cost,
            "runtime": self.runtime,
            "temperature": temp,
            "target": self.target,
            "state": self.state,
            "heat": self.heat,
            "heat_rate": self.heat_rate,
            "totaltime": self.totaltime,
            "kwh_rate": getattr(config, "kwh_rate", 0),
            "currency_type": getattr(config, "currency_type", "$"),
            "profile": self.profile.name if self.profile else None,
            "pidstats": self.pid.pidstats,
            "catching_up": self.catching_up,
        }

    def save_state(self):
        with open(config.automatic_restart_state_file, "w", encoding="utf-8") as f:
            json.dump(self.get_state(), f, ensure_ascii=False, indent=4)

    def state_file_is_old(self):
        if os.path.isfile(config.automatic_restart_state_file):
            state_age = os.path.getmtime(config.automatic_restart_state_file)
            now = time.time()
            minutes = (now - state_age) / 60
            if minutes <= getattr(config, "automatic_restart_window", 15):
                return False
        return True

    def save_automatic_restart_state(self):
        if not getattr(config, "automatic_restarts", False):
            return False
        self.save_state()

    def should_i_automatic_restart(self):
        if not getattr(config, "automatic_restarts", False):
            return False
        if self.state_file_is_old():
            duplog.info("automatic restart not possible. state file does not exist or is too old.")
            return False

        with open(config.automatic_restart_state_file) as infile:
            d = json.load(infile)
        if d.get("state") != "RUNNING":
            duplog.info("automatic restart not possible. state = %s", d.get("state"))
            return False
        return True

    def automatic_restart(self):
        with open(config.automatic_restart_state_file) as infile:
            d = json.load(infile)

        startat = d["runtime"] / 60
        filename = f"{d['profile']}.json"
        profile_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "storage", "profiles", filename))

        log.info("automatically restarting profile = %s at minute = %d", profile_path, startat)

        with open(profile_path) as infile:
            profile_json = json.dumps(json.load(infile))

        profile = Profile(profile_json)
        self.run_profile(profile, startat=startat, allow_seek=False)
        self.cost = d.get("cost", 0)
        time.sleep(1)
        self.ovenwatcher.record(profile)

    def set_ovenwatcher(self, watcher):
        log.info("ovenwatcher set in oven class")
        self.ovenwatcher = watcher

    def run(self):
        while True:
            log.debug("Oven running on %s", threading.current_thread().name)

            if self.state == "IDLE":
                if self.should_i_automatic_restart():
                    self.automatic_restart()
                time.sleep(1)
                continue

            if self.state == "PAUSED":
                self.start_time = self.get_start_time()
                self.update_runtime()
                self.update_target_temp()
                self.heat_then_cool()
                self.reset_if_emergency()
                self.reset_if_schedule_ended()
                continue

            if self.state == "RUNNING":
                self.update_cost()
                self.save_automatic_restart_state()
                self.kiln_must_catch_up()
                self.update_runtime()
                self.update_target_temp()
                self.heat_then_cool()
                self.reset_if_emergency()
                self.reset_if_schedule_ended()


class SimulatedOven(Oven):
    def __init__(self):
        self.board = SimulatedBoard()

        self.t_env = getattr(config, "sim_t_env", 25.0)
        self.c_heat = getattr(config, "sim_c_heat", 1.0)
        self.c_oven = getattr(config, "sim_c_oven", 1.0)
        self.p_heat = getattr(config, "sim_p_heat", 1.0)

        self.R_o_nocool = getattr(config, "sim_R_o_nocool", 1.0)
        self.R_ho_noair = getattr(config, "sim_R_ho_noair", 1.0)
        self.R_ho = self.R_ho_noair

        self.speedup_factor = getattr(config, "sim_speedup_factor", 1)

        self.t = self.t_env
        self.t_h = self.t_env

        super().__init__()
        self.start_time = self.get_start_time()

        self.start()
        log.info("SimulatedOven started")

    def get_start_time(self):
        return datetime.datetime.now() - datetime.timedelta(milliseconds=self.runtime * 1000 / self.speedup_factor)

    def update_runtime(self):
        runtime_delta = datetime.datetime.now() - self.start_time
        if runtime_delta.total_seconds() < 0:
            runtime_delta = datetime.timedelta(0)
        self.runtime = runtime_delta.total_seconds() * self.speedup_factor

    def update_target_temp(self):
        self.target = self.profile.get_target_temperature(self.runtime)

    def heating_energy(self, pid):
        self.Q_h = self.p_heat * self.time_step * pid

    def temp_changes(self):
        self.t_h += self.Q_h / self.c_heat
        self.p_ho = (self.t_h - self.t) / self.R_ho
        self.t += self.p_ho * self.time_step / self.c_oven
        self.t_h -= self.p_ho * self.time_step / self.c_heat
        self.p_env = (self.t - self.t_env) / self.R_o_nocool
        self.t -= self.p_env * self.time_step / self.c_oven
        self.temperature = self.t
        self.board.temp_sensor.simulated_temperature = self.t

    def heat_then_cool(self):
        now_simulator = self.start_time + datetime.timedelta(milliseconds=self.runtime * 1000)
        pid = self.pid.compute(self.target, self.board.temp_sensor.temperature() + getattr(config, "thermocouple_offset", 0), now_simulator)

        heat_on = float(self.time_step * pid)
        heat_off = float(self.time_step * (1 - pid))

        self.heating_energy(pid)
        self.temp_changes()

        self.heat = heat_on if heat_on > 0 else 0.0
        time_left = self.totaltime - self.runtime

        try:
            log.info(
                "temp=%.2f, target=%.2f, error=%.2f, pid=%.2f, p=%.2f, i=%.2f, d=%.2f, heat_on=%.2f, heat_off=%.2f, run_time=%d, total_time=%d, time_left=%d",
                self.pid.pidstats["ispoint"],
                self.pid.pidstats["setpoint"],
                self.pid.pidstats["err"],
                self.pid.pidstats["pid"],
                self.pid.pidstats["p"],
                self.pid.pidstats["i"],
                self.pid.pidstats["d"],
                heat_on,
                heat_off,
                self.runtime,
                self.totaltime,
                time_left,
            )
        except KeyError:
            pass

        time.sleep(self.time_step / self.speedup_factor)


class RealOven(Oven):
    def __init__(self):
        self.board = RealBoard()
        self.output = Output()
        self.reset()

        super().__init__()
        self.start()

    def reset(self):
        super().reset()
        self.output.cool(0)

    def heat_then_cool(self):
        pid = self.pid.compute(self.target, self.board.temp_sensor.temperature() + getattr(config, "thermocouple_offset", 0), datetime.datetime.now())

        heat_on = float(self.time_step * pid)
        heat_off = float(self.time_step * (1 - pid))

        self.heat = 1.0 if heat_on > 0 else 0.0

        if heat_on:
            self.output.heat(heat_on)
        if heat_off:
            self.output.cool(heat_off)

        time_left = self.totaltime - self.runtime
        try:
            log.info(
                "temp=%.2f, target=%.2f, error=%.2f, pid=%.2f, p=%.2f, i=%.2f, d=%.2f, heat_on=%.2f, heat_off=%.2f, run_time=%d, total_time=%d, time_left=%d",
                self.pid.pidstats["ispoint"],
                self.pid.pidstats["setpoint"],
                self.pid.pidstats["err"],
                self.pid.pidstats["pid"],
                self.pid.pidstats["p"],
                self.pid.pidstats["i"],
                self.pid.pidstats["d"],
                heat_on,
                heat_off,
                self.runtime,
                self.totaltime,
                time_left,
            )
        except KeyError:
            pass


# ----------------------------
# Profile (schedule)
# ----------------------------
class Profile:
    def __init__(self, json_data):
        obj = json.loads(json_data)
        self.name = obj["name"]
        self.data = sorted(obj["data"])

    def get_duration(self):
        return max([t for (t, _x) in self.data])

    @staticmethod
    def find_x_given_y_on_line_from_two_points(y, point1, point2):
        if point1[0] > point2[0]:
            return 0
        if point1[1] >= point2[1]:
            return 0
        x = (y - point1[1]) * (point2[0] - point1[0]) / (point2[1] - point1[1]) + point1[0]
        return x

    def find_next_time_from_temperature(self, temperature):
        t = 0
        for index, point2 in enumerate(self.data):
            if point2[1] >= temperature:
                if index > 0:
                    if self.data[index - 1][1] <= temperature:
                        t = self.find_x_given_y_on_line_from_two_points(temperature, self.data[index - 1], point2)
                        if t == 0 and self.data[index - 1][1] == point2[1]:
                            t = self.data[index - 1][0]
                            break
        return t

    def get_surrounding_points(self, t):
        if t > self.get_duration():
            return (None, None)

        prev_point = None
        next_point = None
        for i in range(len(self.data)):
            if t < self.data[i][0]:
                prev_point = self.data[i - 1]
                next_point = self.data[i]
                break
        return (prev_point, next_point)

    def get_target_temperature(self, t):
        if t > self.get_duration():
            return 0

        prev_point, next_point = self.get_surrounding_points(t)
        incl = float(next_point[1] - prev_point[1]) / float(next_point[0] - prev_point[0])
        return prev_point[1] + (t - prev_point[0]) * incl


# ----------------------------
# PID controller
# ----------------------------
class PID:
    def __init__(self, ki=1, kp=1, kd=1):
        self.ki = ki
        self.kp = kp
        self.kd = kd
        self.lastNow = datetime.datetime.now()
        self.iterm = 0
        self.lastErr = 0
        self.pidstats = {}

    def compute(self, setpoint, ispoint, now):
        timeDelta = (now - self.lastNow).total_seconds()
        if timeDelta <= 0:
            timeDelta = 1e-6

        window_size = 100
        error = float(setpoint - ispoint)

        icomp = 0
        output = 0
        out4logs = 0
        dErr = 0

        pid_window = getattr(config, "pid_control_window", 0)

        if error < (-1 * pid_window):
            log.info("kiln outside pid control window, max cooling")
            output = 0
        elif error > (1 * pid_window):
            log.info("kiln outside pid control window, max heating")
            output = 1
            if getattr(config, "throttle_below_temp", None) and getattr(config, "throttle_percent", None):
                if setpoint <= config.throttle_below_temp:
                    output = config.throttle_percent / 100
                    log.info(
                        "max heating throttled at %d percent below %d degrees to prevent overshoot",
                        config.throttle_percent, config.throttle_below_temp
                    )
        else:
            icomp = (error * timeDelta * (1 / self.ki))
            self.iterm += icomp
            dErr = (error - self.lastErr) / timeDelta
            output = self.kp * error + self.iterm + self.kd * dErr
            output = sorted([-1 * window_size, output, window_size])[1]
            out4logs = output
            output = float(output / window_size)

        self.lastErr = error
        self.lastNow = now

        if output < 0:
            output = 0

        self.pidstats = {
            "time": time.mktime(now.timetuple()),
            "timeDelta": timeDelta,
            "setpoint": setpoint,
            "ispoint": ispoint,
            "err": error,
            "errDelta": dErr,
            "p": self.kp * error,
            "i": self.iterm,
            "d": self.kd * dErr,
            "kp": self.kp,
            "ki": self.ki,
            "kd": self.kd,
            "pid": out4logs,
            "out": output,
        }

        return output
