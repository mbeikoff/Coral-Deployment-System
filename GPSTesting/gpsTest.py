import serial
import pynmea2
import time
from datetime import datetime
from collections import deque

# Config
SERIAL_PORT = '/dev/ttyUSB0'  # Your port
BAUD_RATE = 115200
TIMEOUT = 1
ROLLING_HISTORY = 5  # Last N fixes in summary
MIN_QUAL = 2  # >=2: DGPS/RTK (bump to 4 once corrections flow)

# Serial setup
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)

# Rolling history for summary
history = deque(maxlen=ROLLING_HISTORY)

print("Starting GNSS reader... (Outdoors + NTRIP for RTK; Ctrl+C to stop)")
print(f"Logging Qual >= {MIN_QUAL} (DGPS/RTK)...")
fix_count = 0
try:
    while True:
        line = ser.readline().decode('ascii', errors='replace').strip()
        if line.startswith('$GNGGA') or line.startswith('$GGA'):
            try:
                msg = pynmea2.parse(line)
                if isinstance(msg, pynmea2.GGA):
                    qual = int(msg.gps_qual)  # 0=No, 1=Auton, 2=DGPS, 4=RTK Float, 5=Fixed
                    if qual >= MIN_QUAL:
                        lat = msg.latitude if msg.latitude else None
                        lon = msg.longitude if msg.longitude else None
                        # Fixed: Use HDOP as proxy for acc (lower = better; real prec from GST)
                        acc_proxy = float(msg.horizontal_dil) if hasattr(msg, 'horizontal_dil') and msg.horizontal_dil else 999
                        sats = int(msg.num_sats) if msg.num_sats else 0
                        hdop = float(msg.horizontal_dil) if hasattr(msg, 'horizontal_dil') and msg.horizontal_dil else 999
                        if lat and lon:  # Valid pos
                            ts = datetime.now().strftime("%H:%M:%S")
                            ns = 'N' if msg.lat_dir == 'N' else 'S'
                            ew = 'E' if msg.lon_dir == 'E' else 'W'
                            qual_str = {2: "DGPS", 4: "RTK Float", 5: "RTK Fixed"}.get(qual, f"Qual {qual}")
                            print(f"\n[{ts}] Fix #{fix_count + 1}:")
                            print(f"  Lat: {lat:.6f} {ns}")
                            print(f"  Lon: {lon:.6f} {ew}")
                            print(f"  Qual: {qual_str} | HDOP (acc proxy): {acc_proxy:.2f} | Sats: {sats} | HDOP: {hdop:.1f}")
                            
                            # Add to history
                            history.append({
                                'ts': ts,
                                'lat': f"{lat:.6f} {ns}",
                                'lon': f"{lon:.6f} {ew}",
                                'acc': acc_proxy,
                                'sats': sats,
                                'hdop': hdop,
                                'qual': qual_str
                            })
                            fix_count += 1
                            
                            # Rolling summary
                            if len(history) > 0:
                                print(f"\nLast {len(history)} Fixes Summary:")
                                print("-" * 60)
                                for i, h in enumerate(history, 1):
                                    print(f"{i:2d}. [{h['ts']}] {h['lat']:15s} | {h['lon']:15s} | {h['qual']:10s} | HDOP: {h['acc']:.2f} | Sats: {h['sats']:2d} | HDOP: {h['hdop']:.1f}")
                                print("-" * 60)
                        else:
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] Partial fix (no pos): Qual {qual}")
                    else:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] Non-{MIN_QUAL}+: Qual {qual} (add NTRIP for RTK)")
            except pynmea2.ParseError as e:
                print(f"Parse error: {line[:50]}...")  # Debug bad lines
        time.sleep(0.1)  # ~10Hz poll
except KeyboardInterrupt:
    print("\nStopped by user.")
finally:
    ser.close()
    print(f"\nTotal fixes logged: {fix_count}")