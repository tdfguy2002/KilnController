#!/usr/bin/env python3
import time
import datetime
import spidev

def decode_max31855(v: int):
    # Fault bits are valid when bit 16 is 1 (fault flag)
    fault_flag = (v >> 16) & 0x1
    fault_bits = v & 0x7

    # Thermocouple temperature: bits 31..18, 14-bit signed, 0.25C/LSB
    tc = (v >> 18) & 0x3FFF
    if tc & 0x2000:
        tc -= 0x4000
    tc_c = tc * 0.25

    # Internal (cold junction) temperature: bits 15..4, 12-bit signed, 0.0625C/LSB
    cj = (v >> 4) & 0x0FFF
    if cj & 0x800:
        cj -= 0x1000
    cj_c = cj * 0.0625

    return tc_c, cj_c, fault_flag, fault_bits

def main():
    # Pi 5 header SPI for your wiring is spidev0.0 (RP1 SPI0 CE0)
    bus, dev = 0, 0

    spi = spidev.SpiDev()
    spi.open(bus, dev)
    spi.max_speed_hz = 500000
    spi.mode = 0

    print(f"Reading MAX31855 via spidev{bus}.{dev}...")
    print("Degrees displayed in F")

    try:
        while True:
            b = spi.xfer2([0, 0, 0, 0])
            v = (b[0] << 24) | (b[1] << 16) | (b[2] << 8) | b[3]

            tc_c, cj_c, fault_flag, fault_bits = decode_max31855(v)
            tc_f = tc_c * 9 / 5 + 32

            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if fault_flag:
                print(f"{ts} TC={tc_c:.2f}C {tc_f:.2f}F  CJ={cj_c:.2f}C  FAULT={fault_bits}  RAW={hex(v)} bytes={b}")
            else:
                print(f"{ts} TC={tc_c:.2f}C {tc_f:.2f}F  CJ={cj_c:.2f}C  RAW={hex(v)} bytes={b}")

            time.sleep(1)

    except KeyboardInterrupt:
        pass
    finally:
        spi.close()

if __name__ == "__main__":
    main()
