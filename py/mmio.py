from periphery import MMIO
import sys

path = sys.argv[1]

# Open am335x real-time clock subsystem page
rtc_mmio = MMIO(0x44E3E000, 0x1000, path=path)

# Read current time
rtc_secs = rtc_mmio.read32(0x00)
rtc_mins = rtc_mmio.read32(0x04)
rtc_hrs = rtc_mmio.read32(0x08)

print("hours: {:02x} minutes: {:02x} seconds: {:02x}".format(rtc_hrs, rtc_mins, rtc_secs))

rtc_mmio.close()

# Open am335x control module page
ctrl_mmio = MMIO(0x44E10000, 0x1000)

# Read MAC address
mac_id0_lo = ctrl_mmio.read32(0x630)
mac_id0_hi = ctrl_mmio.read32(0x634)

print("MAC address: {:04x}{:08x}".format(mac_id0_lo, mac_id0_hi))

ctrl_mmio.close()
