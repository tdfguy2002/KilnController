# Kiln Controller

Turns a Raspberry Pi into an inexpensive, web-enabled kiln controller.

## Features

- Supports Adafruit MAX31856 and MAX31855 thermocouple breakout boards
- Supports K, J, N, R, S, T, E, and B type thermocouples
- Dark industrial web UI — works from any browser on your network (phone, tablet, computer)
- Real-time sensor temperature, target temperature, heating rate, and cost display
- Live firing progress bar and status indicators (heating, cooling, air, hazard, door)
- **Schedule Builder** — create and edit firing schedules segment by segment, with rate or duration input per segment, hold times, and segment descriptions
- **Schedule Waypoints table** — view all waypoints with cumulative time, temperature, hold duration, and computed rate
- **Delayed Start** — schedule a firing to begin at a specific future date and time
- **Temperature Alarm** — get alerted when the kiln reaches a target temperature mid-run
- **Settings panel** — configure electricity rate, currency, watcher, ntfy topic, and TC error alerts from the UI without editing config files
- **Run log viewer** — browse and download CSV logs of past firings directly from the UI
- Real-time firing cost estimate (configurable kWh rate and currency symbol)
- Real-time heating rate in degrees per hour
- PID control with configurable Kp, Ki, Kd parameters
- Automatic schedule advancement — shifts schedule forward or back when kiln is outside the PID control window
- Prevents PID integral windup when temperature is far from setpoint
- Continues monitoring temperature after schedule completes
- Automatic restart after power outage
- **Watcher** — background process that sends push notifications via [ntfy.sh](https://ntfy.sh) when the kiln deviates from schedule, encounters thermocouple errors, or hits a user-defined temperature alarm
- REST + WebSocket API for programmatic control
- Accurate kiln simulation mode for testing without hardware
- Raspberry Pi 5 / Debian Trixie compatible via direct spidev path (no Blinka/lgpio conflicts)
- Remote CSV logger via WebSocket


## Screenshots

**Main Dashboard**

![Main Page](https://github.com/tdfguy2002/KilnController/blob/main/public/assets/images/Main%20Page.png)

**Schedule Builder**

![Schedule Builder](https://github.com/tdfguy2002/KilnController/blob/main/public/assets/images/Edit%20Menu.png)

**Settings**

![Settings](https://github.com/tdfguy2002/KilnController/blob/main/public/assets/images/Settings.png)

**Log Viewer**

![Log](https://github.com/tdfguy2002/KilnController/blob/main/public/assets/images/Kiln%20Controller%20Log.png)


## Hardware

### Parts

| Image | Hardware | Description |
|-------|----------|-------------|
| ![Image](https://github.com/jbruce12000/kiln-controller/blob/main/public/assets/images/rpi.png) | [Raspberry Pi Zero 2W](https://www.raspberrypi.com/products/raspberry-pi-zero-2-w/) | Tested on a Raspberry Pi Zero 2W. Any Pi with SPI and Wi-Fi should work. Raspberry Pi 5 requires setting `use_spidev_tc = True` in config due to Blinka/lgpio conflicts. |
| ![Image](https://github.com/jbruce12000/kiln-controller/blob/main/public/assets/images/max31855.png) | [Adafruit MAX31855](https://www.adafruit.com/product/269) or [Adafruit MAX31856](https://www.adafruit.com/product/3263) | Thermocouple breakout board. MAX31855 supports K-type only; MAX31856 supports K, J, N, R, S, T, E, B. |
| ![Image](https://github.com/jbruce12000/kiln-controller/blob/main/public/assets/images/k-type-thermocouple.png) | [Thermocouple](https://www.auberins.com/index.php?main_page=product_info&cPath=20_3&products_id=39) | Use a heavy-duty ceramic thermocouple rated for kilns. S-type is common with MAX31856. |
| ![Image](https://github.com/jbruce12000/kiln-controller/blob/main/public/assets/images/breadboard.png) | Breadboard | Breadboard, ribbon cable, GPIO connector, and jumper wires. |
| ![Image](https://github.com/jbruce12000/kiln-controller/blob/main/public/assets/images/ssr.png) | Solid State Relay | Zero-crossing SSR rated for your kiln's current. A single [3-phase SSR](https://www.auberins.com/index.php?main_page=product_info&cPath=2_30&products_id=331) works for 220V kilns. Always use a heat sink. |
| ![Image](https://github.com/jbruce12000/kiln-controller/blob/main/public/assets/images/ks-1018.png) | Electric Kiln | Any electric kiln without digital controls. Works with 110V or 220V (choose appropriate SSR). |

### Schematic

Three GPIO pins connect the Pi to the thermocouple board (data, chip select, clock). A transistor driven by a fourth GPIO pin switches 5V to the SSR. See `config.py` for pin assignments.

**WARNING** — This project involves high voltages and currents. Ensure your build meets local electrical codes.

![Image](https://github.com/jbruce12000/kiln-controller/blob/main/public/assets/images/schematic.png)


## Installation

```bash
sudo apt-get update && sudo apt-get dist-upgrade
git clone https://github.com/jbruce12000/kiln-controller
cd kiln-controller
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Enable SPI (hardware deployment)

```bash
sudo raspi-config
# Interfacing Options → SPI → Yes → Reboot
```

### Raspberry Pi 5 / Debian Trixie

This project is tested on a **Raspberry Pi Zero 2W**. If you use a Pi 5 and encounter Blinka/lgpio conflicts, set `use_spidev_tc = True` in `config.py` to use the raw Linux SPI path instead.


## Configuration

All hardware and runtime settings are in `config.py`. Key options:

| Variable | Default | Description |
|----------|---------|-------------|
| `simulate` | `True` | Use simulated oven for testing — set to `False` for real hardware |
| `temp_scale` | `f` | Temperature units: `f` for Fahrenheit, `c` for Celsius (must be lowercase) |
| `use_spidev_tc` | `False` | Use raw spidev SPI instead of Blinka — required on Pi 5 |
| `max31855` / `max31856` | | Select your thermocouple chip |
| `pid_kp`, `pid_ki`, `pid_kd` | | PID gains — use `kiln-tuner.py` to calculate |
| `sensor_time_wait` | `2` | Duty cycle in seconds — increase for mechanical relays |
| `kiln_must_catch_up` | `True` | Shift schedule when kiln is outside the PID control window |

Runtime settings (kWh rate, currency, ntfy topic, watcher settings) can also be adjusted from the **Settings** panel in the UI.


## Usage

### Start the server

```bash
source venv/bin/activate
./kiln-controller.py
```

Open `http://<pi-ip>:8081` in any browser on your network.

### Autostart on boot

```bash
/home/pi/kiln-controller/start-on-boot
```

### Simulation

Set `simulate = True` in `config.py`, start the server, select a profile, and click **Start**. The simulator models your kiln's thermal mass in near real time.

### Creating and editing schedules

Click the **list icon** (⊞) next to the profile selector to open the **Schedule Builder**. Add segments by specifying:
- **Target** temperature
- **Rate** (°/hr) — or click the computed duration to switch to entering duration directly
- **Hold** time (minutes) at that temperature
- Optional **description** for each segment

To edit an existing schedule, select it in the dropdown and click the **edit icon** (✎). The builder reconstructs all segments from the stored waypoints.

### Starting a run

Select a profile and click **Start**. Before firing begins you can optionally:
- **Delayed Start** — pick a future date/time for the firing to begin automatically
- **Temperature Alarm** — set a temperature at which you want a push notification

### Delayed start

Check **Delayed Start** in the start dialog and pick a date/time. The schedule will begin automatically at that time. A **Cancel Schedule** button appears on the main dashboard to abort the pending start.

### Watcher & push notifications

The watcher monitors a running kiln and sends push notifications via [ntfy.sh](https://ntfy.sh) when:
- The kiln temperature deviates from the schedule by more than a configurable limit
- A thermocouple read error is detected
- A temperature alarm threshold is reached

Configure your ntfy topic, deviation limit, and toggle options in the **Settings** panel (⚙ in the nav bar). To run the watcher standalone:

```bash
source venv/bin/activate
./watcher.py
```

### Run logs

Click **Runs** in the nav bar to browse past firing logs. Each completed run is saved as a CSV file that can be downloaded for analysis.

### Remote logger

Log a live or completed run to CSV from any machine on the network:

```bash
./kiln-logger.py --hostname <pi-ip>:8081 --csvfile /tmp/kilnstats.csv --pidstats --stdout
```


## Diagnostics & Tuning

```bash
./test-thermocouple.py   # verify thermocouple reads
./test-output.py         # verify SSR GPIO output
./gpioreadall.py         # inspect all GPIO pin states
```

### PID autotuner

```bash
./kiln-tuner.py -t 400   # heat to 400°F and calculate Kp/Ki/Kd
./kiln-tuner.py -c       # calculate from existing tuning.csv
```

Copy the resulting values into `config.py`. See the [PID Tuning Guide](https://github.com/jbruce12000/kiln-controller/blob/main/docs/pid_tuning.md) for manual tuning. A live PID stats view is available at `/state`.

### Tests

```bash
pytest Test/                   # run all tests
pytest Test/test_Profile.py    # run a specific test file
```


## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/control` | WebSocket | Send `RUN` / `STOP` / `SIMULATE` commands |
| `/storage` | WebSocket | `GET` / `PUT` / `DELETE` profiles |
| `/status` | WebSocket | Subscribe to real-time oven state updates |
| `/config` | WebSocket | Read display config (temp scale, kWh rate) |
| `/api` | POST | `{"cmd": "run"|"pause"|"resume"|"stop"|"stats"}` |
| `/api/stats` | GET | Current PID stats |
| `/api/runs` | GET | List of saved run log files |
| `/log` | GET | Live log viewer |


## License

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.


## Support

Please use the [issue tracker](https://github.com/jbruce12000/kiln-controller/issues) for project-related issues. For hardware troubleshooting, see the [troubleshooting guide](https://github.com/jbruce12000/kiln-controller/blob/main/docs/troubleshooting.md).


## Origin

Originally forked from [apollo-ng/picoReflow](https://github.com/apollo-ng/picoReflow) — substantially rewritten and extended.
