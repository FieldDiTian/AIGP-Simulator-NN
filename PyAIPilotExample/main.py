#
# Sample Python client for the AI GP controller
#

import time
import threading

from setup import setup_components

# Modify these properties if you want to run the server remotely for example
SIM_SERVER_UDP_IP = "127.0.0.1"
SIM_SERVER_UDP_PORT = 14550

# time since sim started ms
system_boot_ms = int(time.time() * 1000)

# arbitrary shared data between the various components
shared_data = {"_lock": threading.RLock()}

# setup components
components = setup_components(shared_data, system_boot_ms, SIM_SERVER_UDP_IP, SIM_SERVER_UDP_PORT)
controller = components['controller']
ts_loop = components['ts_loop']
mavlink_rx = components['mavlink_rx']
vision_rx = components['vision_rx']

print("Arming drone...", flush=True)
controller.arm()
print("Starting control loop...", flush=True)
is_running = True
try:
    while is_running:
        is_running = controller.update()
except KeyboardInterrupt:
    print("\nControl loop interrupted.", flush=True)
finally:
    controller.shutdown()

    # exit
    for component in (ts_loop, mavlink_rx, vision_rx):
        if component is None:
            continue
        thread = component.get_thread_for_join()
        if thread is not None:
            thread.join(timeout=1.0)

print("Client exited!", flush=True)
