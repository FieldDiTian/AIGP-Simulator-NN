import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from pymavlink import mavutil


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXE = ROOT / "AIGP_3364" / "FlightSim.exe"


def wait_heartbeat(host, port, timeout_s):
    conn = mavutil.mavlink_connection(f"udpin:{host}:{port}")
    msg = conn.wait_heartbeat(timeout=timeout_s)
    conn.close()
    return msg is not None


def wait_telemetry(host, port, timeout_s):
    conn = mavutil.mavlink_connection(f"udpin:{host}:{port}")
    try:
        heartbeat = conn.wait_heartbeat(timeout=timeout_s)
        if heartbeat is None:
            return False
        deadline = time.time() + timeout_s
        telemetry_types = {"ATTITUDE", "ODOMETRY", "HIGHRES_IMU", "LOCAL_POSITION_NED"}
        while time.time() < deadline:
            msg = conn.recv_match(blocking=True, timeout=1.0)
            if msg is None:
                continue
            if msg.get_type() in telemetry_types:
                return True
        return False
    finally:
        conn.close()


def heartbeat_once(host, port):
    return wait_heartbeat(host, port, 0.1)


def ready_once(host, port, ready_signal):
    if ready_signal == "heartbeat":
        return wait_heartbeat(host, port, 0.1)
    return wait_telemetry(host, port, 0.1)


def launch_process(exe_path):
    if not exe_path.exists():
        raise FileNotFoundError(f"FlightSim executable not found: {exe_path}")
    return subprocess.Popen([str(exe_path)], cwd=str(exe_path.parent))


def terminate_process_tree(proc):
    if os.name == "nt":
        if proc is not None:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        for image_name in ("DCGame-Win64-Shipping.exe", "FlightSim.exe"):
            subprocess.run(
                ["taskkill", "/IM", image_name, "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        return

    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def save_screenshot(path):
    try:
        import pyautogui
    except ImportError:
        return False
    image = pyautogui.screenshot()
    image.save(str(path))
    return True


def window_screenshot(pyautogui, window):
    return pyautogui.screenshot(region=(window.left, window.top, window.width, window.height))


def color_ratio(image, rel_box, color):
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("ui-auto Flight HUD detection requires numpy.") from exc

    width, height = image.size
    x1, y1, x2, y2 = rel_box
    crop = image.crop((
        int(width * x1),
        int(height * y1),
        int(width * x2),
        int(height * y2),
    )).convert("RGB")
    arr = np.asarray(crop)
    if color == "yellow":
        mask = (arr[:, :, 0] > 170) & (arr[:, :, 1] > 150) & (arr[:, :, 2] < 110)
    elif color == "orange":
        mask = (arr[:, :, 0] > 180) & (arr[:, :, 1] > 40) & (arr[:, :, 1] < 140) & (arr[:, :, 2] < 90)
    else:
        raise ValueError(f"Unknown color mask: {color}")
    return float(mask.mean())


def flight_hud_visible(pyautogui, window, screenshot_dir=None, screenshot_name=None):
    image = window_screenshot(pyautogui, window)
    if screenshot_dir and screenshot_name:
        image.save(str(screenshot_dir / screenshot_name))

    # The true in-race view has yellow "FLIGHT MODE / ACRO" text in the upper
    # right and a yellow speed/camera HUD around the lower center. Event menus
    # may already emit MAVLink telemetry, so telemetry alone is not sufficient.
    top_right_yellow = color_ratio(image, (0.78, 0.03, 0.96, 0.16), "yellow")
    bottom_hud_yellow = color_ratio(image, (0.35, 0.80, 0.67, 0.96), "yellow")
    return top_right_yellow > 0.006 and bottom_hud_yellow > 0.006


def find_game_window(pyautogui, timeout_s):
    deadline = time.time() + timeout_s
    last_titles = []
    while time.time() < deadline:
        candidates = []
        last_titles = []
        for window in pyautogui.getAllWindows():
            title = (window.title or "").strip()
            if title:
                last_titles.append(title)
            if title == "AI-GP" and window.width > 500 and window.height > 300 and window.left > -1000:
                candidates.append(window)

        if candidates:
            candidates.sort(key=lambda w: w.width * w.height, reverse=True)
            return candidates[0]
        time.sleep(0.25)

    seen = ", ".join(sorted(set(last_titles))[:12])
    raise RuntimeError(f"Could not find visible AI-GP game window. Visible titles: {seen}")


def load_input_driver(pyautogui):
    try:
        import pydirectinput
    except ImportError:
        return pyautogui
    pydirectinput.PAUSE = 0.05
    return pydirectinput


def send_click(input_driver, x, y):
    input_driver.moveTo(int(x), int(y))
    time.sleep(0.05)
    input_driver.click()


def send_press(input_driver, key):
    input_driver.press(key)


def focus_window(pyautogui, input_driver, window):
    hwnd = getattr(window, "_hWnd", None)
    if hwnd:
        try:
            import win32con
            import win32gui

            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            win32gui.SetWindowPos(
                hwnd,
                win32con.HWND_TOPMOST,
                0,
                0,
                0,
                0,
                win32con.SWP_NOMOVE | win32con.SWP_NOSIZE,
            )
            win32gui.SetWindowPos(
                hwnd,
                win32con.HWND_NOTOPMOST,
                0,
                0,
                0,
                0,
                win32con.SWP_NOMOVE | win32con.SWP_NOSIZE,
            )
            win32gui.BringWindowToTop(hwnd)
            win32gui.SetForegroundWindow(hwnd)
        except Exception:
            pass
    try:
        if getattr(window, "isMinimized", False):
            window.restore()
    except Exception:
        pass
    try:
        window.activate()
    except Exception:
        # Some Unreal windows reject Win32 activation; a title-bar click is enough to focus it.
        pass
    send_click(input_driver, window.left + min(80, window.width / 2), window.top + min(20, window.height / 2))
    time.sleep(0.2)


def click_relative(input_driver, window, x_ratio, y_ratio):
    x = window.left + window.width * x_ratio
    y = window.top + window.height * y_ratio
    send_click(input_driver, x, y)


def dismiss_transient_overlays(input_driver, window):
    # Discord's game overlay can appear above the simulator and intercept clicks.
    # This relative point is the overlay close button when it is present; otherwise
    # it lands in an inert upper-right area of the AI-GP content region.
    click_relative(input_driver, window, 0.935, 0.15)
    time.sleep(0.25)


def click_relative_many(input_driver, window, x_ratio, y_ratio, clicks=1, interval_s=0.2):
    for click_index in range(clicks):
        if click_index:
            time.sleep(interval_s)
        click_relative(input_driver, window, x_ratio, y_ratio)


def default_aigp_ui_sequence():
    return [
        {
            "name": "press_any_button",
            "action": "click_relative",
            "x": 0.50,
            "y": 0.36,
            "clicks": 2,
            "interval_s": 0.4,
            "wait_s": 2.0,
            "description": "click the splash screen prompt",
        },
        {
            "name": "press_any_button_fallback",
            "action": "press",
            "key": "enter",
            "wait_s": 3.0,
            "description": "keyboard fallback for the splash/login screen",
        },
        {
            "name": "submit_login",
            "action": "click_relative",
            "x": 0.76,
            "y": 0.68,
            "wait_s": 6.0,
            "description": "submit pre-filled simulator credentials",
        },
        {
            "name": "select_available_event",
            "action": "click_relative",
            "x": 0.86,
            "y": 0.26,
            "wait_s": 6.0,
            "description": "click the AVAILABLE event entry",
        },
        {
            "name": "start_race",
            "action": "click_relative",
            "x": 0.84,
            "y": 0.80,
            "wait_s": 8.0,
            "description": "enter the active simulator/race screen",
        },
    ]


def load_ui_sequence(path):
    if path is None:
        return default_aigp_ui_sequence()

    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("UI sequence JSON must be a list of step objects.")
    return data


def execute_ui_step(input_driver, window, step):
    action = step.get("action")
    if action == "press":
        send_press(input_driver, step["key"])
        return
    if action == "click_relative":
        click_relative_many(
            input_driver,
            window,
            float(step["x"]),
            float(step["y"]),
            clicks=int(step.get("clicks", 1)),
            interval_s=float(step.get("interval_s", 0.1)),
        )
        return
    raise ValueError(f"Unsupported UI step action: {action}")


def wait_for_flight_ui(pyautogui, input_driver, deadline, host, port, screenshot_dir):
    attempt = 0
    while time.time() < deadline:
        window = find_game_window(pyautogui, min(5.0, max(0.5, deadline - time.time())))
        focus_window(pyautogui, input_driver, window)
        dismiss_transient_overlays(input_driver, window)
        if flight_hud_visible(pyautogui, window, screenshot_dir, f"ui_auto_flight_check_{attempt:02d}.png"):
            if wait_telemetry(host, port, min(5.0, max(0.1, deadline - time.time()))):
                return True

        attempt += 1
        print(f"ui-auto flight attempt {attempt}: advance any current screen toward RACE.", flush=True)
        click_relative_many(input_driver, window, 0.50, 0.36, clicks=2, interval_s=0.35)
        send_press(input_driver, "enter")
        time.sleep(1.0)
        dismiss_transient_overlays(input_driver, window)
        if screenshot_dir:
            window_screenshot(pyautogui, window).save(str(screenshot_dir / f"ui_auto_attempt_{attempt:02d}_splash_or_login.png"))

        click_relative_many(input_driver, window, 0.76, 0.68, clicks=1)
        time.sleep(2.0)
        dismiss_transient_overlays(input_driver, window)
        if screenshot_dir:
            window_screenshot(pyautogui, window).save(str(screenshot_dir / f"ui_auto_attempt_{attempt:02d}_submit.png"))

        click_relative_many(input_driver, window, 0.86, 0.26, clicks=2, interval_s=0.35)
        time.sleep(2.0)
        dismiss_transient_overlays(input_driver, window)
        if screenshot_dir:
            window_screenshot(pyautogui, window).save(str(screenshot_dir / f"ui_auto_attempt_{attempt:02d}_available.png"))

        click_relative_many(input_driver, window, 0.84, 0.80, clicks=2, interval_s=0.35)
        time.sleep(5.0)
        if screenshot_dir:
            window_screenshot(pyautogui, window).save(str(screenshot_dir / f"ui_auto_attempt_{attempt:02d}_race.png"))

    return False


def run_anchor_clicks(pyautogui, anchor_dir, screenshot_dir, timeout_s, host, port, ready_signal):
    anchor_dir = Path(anchor_dir) if anchor_dir else None
    if anchor_dir is None or not anchor_dir.exists():
        return False

    anchors = sorted(anchor_dir.glob("*.png"))
    if not anchors:
        return False

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if ready_once(host, port, ready_signal):
            return True
        for anchor in anchors:
            try:
                pos = pyautogui.locateCenterOnScreen(str(anchor), confidence=0.85)
            except TypeError:
                pos = pyautogui.locateCenterOnScreen(str(anchor))
            except Exception:
                pos = None
            if pos is not None:
                print(f"ui-auto anchor click: {anchor.name}", flush=True)
                pyautogui.click(pos.x, pos.y)
                time.sleep(1.0)
                break
        else:
            if screenshot_dir:
                screenshot_dir.mkdir(parents=True, exist_ok=True)
                save_screenshot(screenshot_dir / f"ui_auto_anchor_{int(time.time())}.png")
            time.sleep(1.0)
    return False


def run_ui_auto(anchor_dir, screenshot_dir, timeout_s, host, port, ready_signal, ui_sequence_path):
    try:
        import pyautogui
    except ImportError as exc:
        raise RuntimeError("ui-auto requires pyautogui. Install it or use --mode manual-ready.") from exc

    pyautogui.FAILSAFE = False
    input_driver = load_input_driver(pyautogui)
    print(f"ui-auto: input driver={input_driver.__name__}", flush=True)

    deadline = time.time() + timeout_s
    if screenshot_dir:
        screenshot_dir.mkdir(parents=True, exist_ok=True)

    if anchor_dir:
        print("ui-auto: trying anchor image clicks first.", flush=True)
        if run_anchor_clicks(pyautogui, anchor_dir, screenshot_dir, min(10.0, timeout_s), host, port, ready_signal):
            return True

    sequence = load_ui_sequence(ui_sequence_path)
    window_timeout = min(30.0, max(1.0, deadline - time.time()))
    window = find_game_window(pyautogui, window_timeout)
    print(
        f"ui-auto: using window '{window.title.strip()}' "
        f"at x={window.left} y={window.top} w={window.width} h={window.height}",
        flush=True,
    )

    for index, step in enumerate(sequence, start=1):
        if time.time() >= deadline:
            break
        window = find_game_window(pyautogui, min(5.0, max(0.5, deadline - time.time())))
        focus_window(pyautogui, input_driver, window)
        dismiss_transient_overlays(input_driver, window)
        print(
            f"ui-auto step {index}/{len(sequence)}: {step.get('name', '<unnamed>')} "
            f"({step.get('description', step.get('action'))})",
            flush=True,
        )
        execute_ui_step(input_driver, window, step)
        time.sleep(float(step.get("wait_s", 1.0)))
        window = find_game_window(pyautogui, min(5.0, max(0.5, deadline - time.time())))
        dismiss_transient_overlays(input_driver, window)
        if screenshot_dir:
            save_screenshot(screenshot_dir / f"ui_auto_{index:02d}_{step.get('name', 'step')}.png")

    remaining = max(0.1, deadline - time.time())
    if ready_signal == "heartbeat":
        print("ui-auto: waiting for MAVLink heartbeat after UI sequence.", flush=True)
        return wait_heartbeat(host, port, remaining)

    print("ui-auto: waiting for Flight HUD plus MAVLink telemetry after UI sequence.", flush=True)
    return wait_for_flight_ui(pyautogui, input_driver, time.time() + remaining, host, port, screenshot_dir)


def main():
    parser = argparse.ArgumentParser(description="Launch FlightSim and wait for MAVLink heartbeat.")
    parser.add_argument("--mode", choices=["manual-ready", "ui-auto"], default="manual-ready")
    parser.add_argument("--exe", default=str(DEFAULT_EXE))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=14550)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--no-start", action="store_true", help="Do not start the exe; only wait for heartbeat.")
    parser.add_argument("--close-after-ready", action="store_true")
    parser.add_argument("--anchor-dir", default=None, help="Optional PNG anchors for ui-auto mode.")
    parser.add_argument("--ui-sequence", default=None, help="Optional JSON UI step sequence for ui-auto mode.")
    parser.add_argument("--save-screenshots", action="store_true")
    parser.add_argument("--ready-signal", choices=["heartbeat", "telemetry"], default="telemetry")
    args = parser.parse_args()

    proc = None
    screenshot_dir = ROOT / "logs" / "ui_screenshots" if args.save_screenshots else None
    try:
        if not args.no_start:
            proc = launch_process(Path(args.exe))
            print(f"Started FlightSim pid={proc.pid}", flush=True)

        if args.mode == "ui-auto":
            ready = run_ui_auto(
                args.anchor_dir,
                screenshot_dir,
                args.timeout_s,
                args.host,
                args.port,
                args.ready_signal,
                args.ui_sequence,
            )
        else:
            print("manual-ready mode: put FlightSim into the flyable simulator screen.", flush=True)
            if args.ready_signal == "heartbeat":
                ready = wait_heartbeat(args.host, args.port, args.timeout_s)
            else:
                ready = wait_telemetry(args.host, args.port, args.timeout_s)
        if not ready:
            if screenshot_dir:
                screenshot_dir.mkdir(parents=True, exist_ok=True)
                save_screenshot(screenshot_dir / "heartbeat_timeout.png")
            raise TimeoutError(f"Timed out waiting for MAVLink {args.ready_signal}.")

        print(f"MAVLink {args.ready_signal} ready.", flush=True)
        return 0
    finally:
        if args.close_after_ready:
            terminate_process_tree(proc)


if __name__ == "__main__":
    sys.exit(main())
