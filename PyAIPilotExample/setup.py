from pymavlink import mavutil
from timesync import TimeSync
from vision_rx import VisionRX
from mavlink_rx import MAVLinkRX
from controller import Controller
from keyboard_control import KeyboardControlConsole

def setup_components(
        shared_data,
        system_boot_ms,
        server_ip,
        server_udp_port,
        command_source=None,
        logger=None,
        start_vision=True,
        heartbeat_timeout_s=None):
    # -------------------------------
    # Mavlink Connection
    # -------------------------------
    # Start a connection listening on a UDP port
    sim_conn = mavutil.mavlink_connection('udpin:%s:%s' % (server_ip, server_udp_port,))
    print("Waiting for heartbeat...", flush=True)
    heartbeat = sim_conn.wait_heartbeat(timeout=heartbeat_timeout_s)
    if heartbeat is None:
        raise TimeoutError("Timed out waiting for MAVLink heartbeat.")
    print(f"Connected to system: {sim_conn.target_system}", flush=True)

    # -------------------------------
    # Setup Mavlink msg receiver
    # -------------------------------
    print("Setting up MAVLink rx...", flush=True)
    mavlink_rx = MAVLinkRX.create_mavlink_rx(sim_conn, shared_data)

    # -------------------------------
    # Timesync request Loop
    # -------------------------------
    print("Setting up Timesync loop...", flush=True)
    ts_loop = TimeSync.create_timesync(sim_conn, shared_data)

    # -------------------------------
    # Connect Vision receiver
    # -------------------------------
    vision_rx = VisionRX(shared_data) if start_vision else None

    # -------------------------------
    # Main control loop
    # -------------------------------
    if command_source is None:
        command_source = KeyboardControlConsole()
    controller = Controller(sim_conn, shared_data, system_boot_ms, command_source, logger=logger)

    return {
        'vision_rx': vision_rx,
        'mavlink_rx': mavlink_rx,
        'ts_loop': ts_loop,
        'sim_conn': sim_conn,
        'controller': controller
    }
