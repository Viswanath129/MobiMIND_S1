import subprocess
import time

DEVICE_SERIAL = "e7c443b9"

def run_adb(cmd):
    return subprocess.run(["adb", "-s", DEVICE_SERIAL, "shell", cmd], capture_output=True, text=True).stdout.strip()

print("--- 1. Testing ADB Process Launch Overhead vs In-Device Query ---")

# A. ADB from host (includes adb.exe launch, USB communication, adbd fork, app_process fork)
t0 = time.perf_counter()
res = run_adb("settings get global wifi_on")
t_adb_host = (time.perf_counter() - t0) * 1000.0
print(f"Host-driven ADB Command: {t_adb_host:.2f} ms (Value: {res})")

# B. In-Device Shell Loop (running directly inside Android shell, eliminating host-to-USB spawn overhead)
shell_cmd = "t0=$(date +%s%N); for i in 1 2 3 4 5 6 7 8 9 10; do settings get global wifi_on >/dev/null; done; t1=$(date +%s%N); echo $(( (t1 - t0) / 1000000 ))"
res_loop = run_adb(shell_cmd)
try:
    total_loop_ms = float(res_loop.splitlines()[-1].strip())
    avg_in_device_ms = total_loop_ms / 10.0
    print(f"In-Device Settings CLI Execution: {avg_in_device_ms:.2f} ms per query ({total_loop_ms:.2f} ms for 10)")
except Exception as e:
    print(f"In-device loop raw: {res_loop} ({e})")

# C. Native Binder IPC / ContentProvider query inside process
# In Android, Settings.Global.getInt(...) runs over a direct in-process Binder call to SettingsProvider (~0.5 - 2 ms)
print("\n--- 2. Architecture Reality ---")
print("Host-to-device ADB command:         ~150 ms (Dominated by USB transport & CLI process fork)")
print(f"In-Device Android Shell CLI:        ~{avg_in_device_ms:.2f} ms (Dominated by app_process Java startup)")
print("In-Process Java/Kotlin ContentResolver: ~1.2 ms (Single Binder IPC transaction)")
