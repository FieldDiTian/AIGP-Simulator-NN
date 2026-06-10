import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from pymavlink import mavutil


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXE = ROOT / "AIGP_3364" / "FlightSim.exe"


class Rect(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


class MouseInput(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class KeyBdInput(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class HardwareInput(ctypes.Structure):
    _fields_ = [
        ("uMsg", ctypes.c_ulong),
        ("wParamL", ctypes.c_short),
        ("wParamH", ctypes.c_ushort),
    ]


class InputUnion(ctypes.Union):
    _fields_ = [
        ("mi", MouseInput),
        ("ki", KeyBdInput),
        ("hi", HardwareInput),
    ]


class Input(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_ulong),
        ("union", InputUnion),
    ]


INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_ABSOLUTE = 0x8000


def set_process_dpi_aware():
    if os.name != "nt":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return
    except Exception:
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


set_process_dpi_aware()


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
    left, top, width, height = window_region(window)
    return pyautogui.screenshot(region=(left, top, width, height))


def window_region(window):
    hwnd = getattr(window, "_hWnd", None)
    if hwnd and os.name == "nt":
        rect = Rect()
        try:
            dwmapi = ctypes.windll.dwmapi
            # DWMWA_EXTENDED_FRAME_BOUNDS returns the visible window bounds.
            result = dwmapi.DwmGetWindowAttribute(
                ctypes.c_void_p(hwnd),
                ctypes.c_uint(9),
                ctypes.byref(rect),
                ctypes.sizeof(rect),
            )
            if result == 0:
                return (
                    int(rect.left),
                    int(rect.top),
                    int(rect.right - rect.left),
                    int(rect.bottom - rect.top),
                )
        except Exception:
            pass
        try:
            ctypes.windll.user32.GetWindowRect(ctypes.c_void_p(hwnd), ctypes.byref(rect))
            return (
                int(rect.left),
                int(rect.top),
                int(rect.right - rect.left),
                int(rect.bottom - rect.top),
            )
        except Exception:
            pass
    return int(window.left), int(window.top), int(window.width), int(window.height)


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
            left, top, width, height = window_region(window)
            if title == "AI-GP" and width > 500 and height > 300 and left > -1000:
                candidates.append(window)

        if candidates:
            candidates.sort(key=lambda w: window_region(w)[2] * window_region(w)[3], reverse=True)
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
    x = int(x)
    y = int(y)
    try:
        send_click_with_sendinput(x, y)
    except Exception:
        pass
    try:
        send_click_with_win32(x, y)
    except Exception:
        pass
    input_driver.moveTo(x, y)
    time.sleep(0.05)
    input_driver.click()
    try:
        import pyautogui

        pyautogui.click(x, y)
    except Exception:
        pass


def send_click_with_sendinput(x, y):
    if os.name != "nt":
        return
    abs_x, abs_y = screen_to_absolute(x, y)
    extra = ctypes.c_ulong(0)
    inputs = (Input * 3)(
        Input(
            type=INPUT_MOUSE,
            union=InputUnion(mi=MouseInput(
                dx=abs_x,
                dy=abs_y,
                mouseData=0,
                dwFlags=MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE,
                time=0,
                dwExtraInfo=ctypes.pointer(extra),
            )),
        ),
        Input(
            type=INPUT_MOUSE,
            union=InputUnion(mi=MouseInput(
                dx=abs_x,
                dy=abs_y,
                mouseData=0,
                dwFlags=MOUSEEVENTF_LEFTDOWN | MOUSEEVENTF_ABSOLUTE,
                time=0,
                dwExtraInfo=ctypes.pointer(extra),
            )),
        ),
        Input(
            type=INPUT_MOUSE,
            union=InputUnion(mi=MouseInput(
                dx=abs_x,
                dy=abs_y,
                mouseData=0,
                dwFlags=MOUSEEVENTF_LEFTUP | MOUSEEVENTF_ABSOLUTE,
                time=0,
                dwExtraInfo=ctypes.pointer(extra),
            )),
        ),
    )
    ctypes.windll.user32.SendInput(len(inputs), ctypes.byref(inputs), ctypes.sizeof(Input))


def screen_to_absolute(x, y):
    user32 = ctypes.windll.user32
    left = user32.GetSystemMetrics(76)
    top = user32.GetSystemMetrics(77)
    width = max(1, user32.GetSystemMetrics(78))
    height = max(1, user32.GetSystemMetrics(79))
    abs_x = int((int(x) - left) * 65535 / max(1, width - 1))
    abs_y = int((int(y) - top) * 65535 / max(1, height - 1))
    return abs_x, abs_y


def send_click_with_win32(x, y):
    if os.name != "nt":
        return
    user32 = ctypes.windll.user32
    user32.SetCursorPos(int(x), int(y))
    time.sleep(0.05)
    user32.mouse_event(0x0002, 0, 0, 0, 0)
    time.sleep(0.05)
    user32.mouse_event(0x0004, 0, 0, 0, 0)


def send_press(input_driver, key):
    try:
        send_key_with_sendinput(key)
    except Exception:
        pass
    input_driver.press(key)
    try:
        import pyautogui

        pyautogui.press(key)
    except Exception:
        pass
    try:
        send_key_with_win32(key)
    except Exception:
        pass


def send_key_with_sendinput(key):
    if os.name != "nt":
        return
    vk_map = {
        "enter": 0x0D,
        "space": 0x20,
        "esc": 0x1B,
    }
    vk = vk_map.get(str(key).lower())
    if vk is None:
        return
    scan = ctypes.windll.user32.MapVirtualKeyW(vk, 0)
    extra = ctypes.c_ulong(0)
    inputs = (Input * 2)(
        Input(
            type=INPUT_KEYBOARD,
            union=InputUnion(ki=KeyBdInput(
                wVk=0,
                wScan=scan,
                dwFlags=KEYEVENTF_SCANCODE,
                time=0,
                dwExtraInfo=ctypes.pointer(extra),
            )),
        ),
        Input(
            type=INPUT_KEYBOARD,
            union=InputUnion(ki=KeyBdInput(
                wVk=0,
                wScan=scan,
                dwFlags=KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP,
                time=0,
                dwExtraInfo=ctypes.pointer(extra),
            )),
        ),
    )
    ctypes.windll.user32.SendInput(len(inputs), ctypes.byref(inputs), ctypes.sizeof(Input))


def send_key_with_win32(key):
    if os.name != "nt":
        return
    vk_map = {
        "enter": 0x0D,
        "space": 0x20,
        "esc": 0x1B,
    }
    vk = vk_map.get(str(key).lower())
    if vk is None:
        return
    user32 = ctypes.windll.user32
    user32.keybd_event(vk, 0, 0, 0)
    time.sleep(0.05)
    user32.keybd_event(vk, 0, 2, 0)


def focus_window(pyautogui, input_driver, window):
    hwnd = getattr(window, "_hWnd", None)
    if hwnd:
        try:
            import win32con
            import win32gui
            import win32process

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
            foreground = win32gui.GetForegroundWindow()
            current_thread = win32api_get_current_thread_id()
            target_thread, _ = win32process.GetWindowThreadProcessId(hwnd)
            foreground_thread, _ = win32process.GetWindowThreadProcessId(foreground)
            try:
                ctypes.windll.user32.AllowSetForegroundWindow(-1)
            except Exception:
                pass
            try:
                ctypes.windll.user32.AttachThreadInput(current_thread, target_thread, True)
                ctypes.windll.user32.AttachThreadInput(current_thread, foreground_thread, True)
                win32gui.SetForegroundWindow(hwnd)
                win32gui.SetActiveWindow(hwnd)
                win32gui.SetFocus(hwnd)
            finally:
                ctypes.windll.user32.AttachThreadInput(current_thread, target_thread, False)
                ctypes.windll.user32.AttachThreadInput(current_thread, foreground_thread, False)
            win32gui.BringWindowToTop(hwnd)
            win32gui.SetForegroundWindow(hwnd)
            try:
                ctypes.windll.user32.SwitchToThisWindow(ctypes.c_void_p(hwnd), True)
            except Exception:
                pass
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
    left, top, width, height = window_region(window)
    send_click(input_driver, left + min(80, width / 2), top + min(20, height / 2))
    send_click(input_driver, left + width * 0.50, top + height * 0.50)
    time.sleep(0.2)


def win32api_get_current_thread_id():
    return ctypes.windll.kernel32.GetCurrentThreadId()


def click_relative(input_driver, window, x_ratio, y_ratio):
    left, top, width, height = window_region(window)
    x = left + width * x_ratio
    y = top + height * y_ratio
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


def press_any_button_burst(input_driver, window):
    left, top, width, height = window_region(window)
    focus_x = left + width * 0.50
    focus_y = top + height * 0.36
    for _ in range(3):
        send_click(input_driver, focus_x, focus_y)
        time.sleep(0.2)
    for key in ("enter", "space", "enter"):
        send_press(input_driver, key)
        time.sleep(0.25)


def default_aigp_ui_sequence():
    return [
        {
            "name": "press_any_button",
            "action": "press_any_button_burst",
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
    if action == "press_any_button_burst":
        press_any_button_burst(input_driver, window)
        return
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
        press_any_button_burst(input_driver, window)
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
        f"at x={window_region(window)[0]} y={window_region(window)[1]} "
        f"w={window_region(window)[2]} h={window_region(window)[3]}",
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
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--restart-delay-s", type=float, default=2.0)
    parser.add_argument("--no-start", action="store_true", help="Do not start the exe; only wait for heartbeat.")
    parser.add_argument("--close-after-ready", action="store_true")
    parser.add_argument("--anchor-dir", default=None, help="Optional PNG anchors for ui-auto mode.")
    parser.add_argument("--ui-sequence", default=None, help="Optional JSON UI step sequence for ui-auto mode.")
    parser.add_argument("--save-screenshots", action="store_true")
    parser.add_argument("--ready-signal", choices=["heartbeat", "telemetry"], default="telemetry")
    args = parser.parse_args()

    attempts = max(1, int(args.attempts))
    base_screenshot_dir = ROOT / "logs" / "ui_screenshots" if args.save_screenshots else None
    last_error = None
    for attempt in range(1, attempts + 1):
        proc = None
        screenshot_dir = attempt_screenshot_dir(base_screenshot_dir, attempt, attempts)
        try:
            if not args.no_start:
                proc = launch_process(Path(args.exe))
                print(f"Started FlightSim pid={proc.pid} attempt={attempt}/{attempts}", flush=True)

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
                print(
                    f"manual-ready mode attempt={attempt}/{attempts}: "
                    "put FlightSim into the flyable simulator screen.",
                    flush=True,
                )
                if args.ready_signal == "heartbeat":
                    ready = wait_heartbeat(args.host, args.port, args.timeout_s)
                else:
                    ready = wait_telemetry(args.host, args.port, args.timeout_s)
            if ready:
                print(f"MAVLink {args.ready_signal} ready.", flush=True)
                if args.close_after_ready:
                    terminate_process_tree(proc)
                return 0

            if screenshot_dir:
                screenshot_dir.mkdir(parents=True, exist_ok=True)
                save_screenshot(screenshot_dir / "heartbeat_timeout.png")
            raise TimeoutError(f"Timed out waiting for MAVLink {args.ready_signal}.")
        except Exception as exc:
            last_error = exc
            print(f"FlightSim launch attempt {attempt}/{attempts} failed: {exc}", flush=True)
            terminate_process_tree(proc)
            if attempt < attempts:
                time.sleep(max(0.0, args.restart_delay_s))
        finally:
            if args.close_after_ready:
                terminate_process_tree(proc)

    raise TimeoutError(f"FlightSim was not ready after {attempts} attempt(s): {last_error}")


def attempt_screenshot_dir(base_dir, attempt, attempts):
    if base_dir is None:
        return None
    if attempts <= 1:
        return base_dir
    return base_dir / f"attempt_{attempt:02d}_{time.strftime('%Y%m%d_%H%M%S')}"


if __name__ == "__main__":
    sys.exit(main())
