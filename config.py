import logging
import os

log_level = logging.INFO
log_format = '%(asctime)s %(levelname)s %(name)s: %(message)s'
listening_port = 8081

# ----- Thermocouple selection -----
max31855 = 1
max31856 = 0

# ----- Use spidev for thermocouple on Pi 5 -----
use_spidev_tc = True
spi_bus = 0
spi_dev = 0
spi_max_speed_hz = 100000

# ----- Runtime settings -----
simulate = False
sensor_time_wait = 2
sensor_read_retries = 3       # number of retry attempts on a failed temp read
sensor_retry_delay = 0.05     # seconds to wait between retries (helps with EMI glitches)

# ----- Outputs -----
import board
gpio_heat = board.D23   # SSR control
spi_cs = board.D8

# ----- Other settings -----
seek_start = True
stop_integral_windup = True
sim_speedup_factor = 1
kiln_must_catch_up = False
thermocouple_offset = 0
temperature_average_samples = 10
ac_freq_50hz = False
thermocouple_type = "K"
pid_control_window = 25   # degrees F
pid_kp = 3.316850653342161
pid_ki = 68.06106142939375
pid_kd = 187.19341216405226

# ----- EMA smoothing for MAX31856 -----
tc_ema_alpha = 0.2

# ----- Automatic restart on power loss -----
automatic_restarts = True
automatic_restart_state_file = os.path.abspath(os.path.join(os.path.dirname(__file__), 'state.json'))

# ----- Profiles directory -----
kiln_profiles_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "storage", "profiles"))

# ----- UI / display -----
temp_scale = "f"          # "f" or "c" (must be lowercase)

# ----- Time scaling for firing schedules -----
time_scale_slope = 1.0
time_scale_intercept = 0.0
time_scale_profile = False

# ----- Cost / energy display -----
kwh_rate = 0.209           # cost per kWh
kw_elements = 7.8          # Skutt 183-27/250: 7800W total
currency_type = "$"
