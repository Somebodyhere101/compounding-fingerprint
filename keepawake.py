"""Hold the system awake (ES_SYSTEM_REQUIRED) only while the experiment queue runs; exits by itself.
Uses the per-process SetThreadExecutionState API; no settings are changed and the hold ends when this exits."""
import ctypes, os, sys, time
ES_CONTINUOUS, ES_SYSTEM_REQUIRED, ES_DISPLAY_REQUIRED = 0x80000000, 0x00000001, 0x00000002
LOG = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'master2.log')
MARK = sys.argv[2] if len(sys.argv) > 2 else 'MASTER2 FINISHED'
t0 = time.time()
while time.time() - t0 < 16 * 3600:
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED)
    if os.path.exists(LOG) and MARK in open(LOG, encoding='utf-8', errors='ignore').read():
        break
    time.sleep(60)
ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
print('keepawake released', time.strftime('%H:%M'))
