# AI Grand Prix Virtual Qualifier Analysis

Based on the uploaded AI Grand Prix Virtual Qualifier technical specification,
the core task is not simply "write an AI model." The task is to build a
complete **autonomous drone flight system**: using a forward-facing camera and
MAVLink telemetry, the aircraft must autonomously pass through the start gate,
intermediate gates, and finish gate in simulation, with no human intervention
during the submitted timed run.

## 1. Competition Objective

The goal of Round One is to verify whether the submitted software can navigate
the full course. The course structure includes:

| Item | Content |
| ---- | ------- |
| Start | start gate |
| Middle course | intermediate gates / sequential race gates |
| Finish | finish gate |
| Maximum time | 8 minutes |
| Submission requirement | no human intervention during the timed run; otherwise immediate disqualification |

The document does not provide an explicit scoring formula, such as whether
ranking is by total time, whether there are collision penalties, missed-gate
penalties, or reset penalties. Therefore the only reliable conclusion is:
**the system must complete the course autonomously; a high-score strategy
should first guarantee stable completion, then optimize speed.**

---

## 2. Hard Constraints

### 2.1 No Manual Takeover

Any human interaction during the submitted timed run leads to immediate
disqualification. This means perception, planning, control, and recovery must
all be completed automatically in software.

### 2.2 Runtime Environment

The specification says the simulator runs on Windows 11 on a standard PC, with
roughly 8 GB of GPU memory. Linux is not currently supported. Competitors may
use Python; the document notes that Python 3.14.2 works, and other environments
are also allowed.

Engineering implication: do not keep the main development environment only on
Linux. Validate dependencies, networking, MAVLink, image-stream decoding, and
control frequency on Windows 11 early.

### 2.3 Control Frequency

The simulator physics rate is 120 Hz, but control-command frequency must be
below 100 Hz, and heartbeat must be at least 2 Hz.

Recommended setup:

```text
control loop: 50-80 Hz
vision processing: 30 Hz
```

The control layer should predict state between image frames; it should not send
commands only when a new image arrives.

### 2.4 No GPS or Global Position

The simulator internally uses local Cartesian coordinates, but the document
explicitly says there is no GPS simulation and no exposed absolute global
position.

This is critical. The autonomy stack must not rely on "knowing its global
position on the track." It should primarily rely on:

1. forward camera gate detection
2. IMU, attitude, velocity, and related telemetry
3. vision-based relative localization
4. deterministic track knowledge as feed-forward information when allowed

---

## 3. Available Technical Information

### 3.1 Simulation Environment

The environment includes a start gate, sequential race gates, a finish gate,
vertical and horizontal obstacles, boundaries, terrain, and scene structure.
Course geometry, physics parameters, and environmental conditions are identical
for all teams and deterministic.

This means repeated testing is valuable. The same track can be used for
parameter tuning, trajectory optimization, and failure-case logging.

### 3.2 Aircraft and Gate Dimensions

Drone dimensions:

| Dimension | Value |
| --------- | ----: |
| Width | 280 mm |
| Length | 280 mm |
| Height | 160 mm |

Gate outer boundary:

```text
2700 x 2700 x 260 mm
```

Gate inner opening:

```text
1500 x 1500 x 260 mm
```

Estimated theoretical safety margin:

| Direction | Inner opening | Body size | One-sided theoretical margin |
| --------- | ------------: | --------: | ---------------------------: |
| Lateral | 1500 mm | 280 mm | about 610 mm |
| Vertical | 1500 mm | 160 mm | about 670 mm |

The theoretical margin is large, but high-speed flight consumes margin through
attitude tilt, control delay, vision error, and collision-model effects. The
practical strategy should not skim the gate edge. A reasonable target is to
keep gate-center error within **0.3-0.4 m**, and preferably below 0.5 m even in
fast sections.

### 3.3 Coordinate Frames and Camera

MAVLink frame convention is NED:

| Frame | Meaning |
| ----- | ------- |
| `MAV_FRAME_LOCAL_NED` | origin is a ground-fixed point, usually the drone's arming position |
| `MAV_FRAME_BODY_NED` | origin is the body; X forward, Y right, Z down |

The camera and body share the same origin. The camera is pitched upward by
20 deg relative to the body. IMU-to-body is an identity map. The camera is a
pinhole model with no distortion, resolution 640 x 360, principal point
`[320, 180]`, and `fx = fy = 320`.

Important note: the document states `VFoV = 90 deg`, but with `fx = fy = 320`
and resolution 640 x 360, the standard pinhole calculation gives horizontal
FOV of about 90 deg and vertical FOV of about 58.7 deg. This may be a
`VFoV/HFoV` labeling inconsistency. Before implementing projection, PnP, or
visual servoing, validate intrinsics using simulator images instead of trusting
a single field blindly.

### 3.4 Communication Interface

Communication uses a MAVLink 2 / MAVSDK-compatible interface over UDP.
Supported messages include:

| Message | Direction | Use |
| ------- | --------- | --- |
| `HEARTBEAT` | Simulator -> Client | connection state |
| `ATTITUDE` | Simulator -> Client | attitude |
| `HIGHRES_IMU` | Simulator -> Client | vehicle state / measurement |
| `SET_POSITION_TARGET_LOCAL_NED` | Client -> Simulator | control interface |
| `SET_ATTITUDE_TARGET` | Client -> Simulator | control interface |
| `TIMESYNC` | Simulator -> Client | time sync |

The document does not say which fields or type masks in
`SET_POSITION_TARGET_LOCAL_NED` are actually used by the simulator. It also
does not specify the exact simulator semantics of `SET_ATTITUDE_TARGET`
thrust, attitude, or yaw-rate fields. Therefore interface experiments should be
done early: send velocity, position, acceleration, attitude, and thrust
commands separately, record the aircraft response, and identify the real
control channel.

### 3.5 Image Stream

The camera image stream runs at 30 Hz, resolution 640 x 360, default UDP port
5600. Each JPEG image is split into multiple packets. Each packet has a
24-byte metadata header containing `frame_id`, `chunk_id`, `total_chunks`,
`jpeg_size`, `payload_size`, and `sim_time_ns`.

Engineering requirements:

1. Reassemble packets by `frame_id`.
2. Handle out-of-order, lost, and duplicate packets.
3. Drop incomplete frames; do not feed corrupted images into perception.
4. Align image and telemetry using `sim_time_ns`.
5. Log image latency; otherwise high-speed flight will miss gates.

---

## 3.6 Current Codebase Mapping

The current folder is not a complete source-code project. It is a packaged
Windows simulator plus a Python autonomy client template. The internal Unreal /
FlightSim logic is packaged; the main editable part is `PyAIPilotExample`.

| Path | Purpose | Main edit target |
| ---- | ------- | ---------------- |
| `AIGP_3364/FlightSim.exe` | simulator launcher | no |
| `AIGP_3364/FlightSim/Binaries/Win64/DCGame-Win64-Shipping.exe` | actual packaged Unreal executable | no |
| `AIGP_3364/FlightSim/Content/Paks/FlightSim-WindowsNoEditor.pak` | packaged track, models, materials, etc. | usually no |
| `AIGP_3364/FlightSim/Content/FMOD/Desktop/*.bank` | FMOD audio assets | no |
| `AIGP_3364/Engine/*` | Unreal Engine runtime dependencies | no |
| `PyAIPilotExample/main.py` | Python client entry point; sets MAVLink address/port, creates components, arms, enters control loop | yes |
| `PyAIPilotExample/setup.py` | creates MAVLink connection, MAVLink receiver, vision receiver, and controller | yes |
| `PyAIPilotExample/mavlink_rx.py` | receives and parses MAVLink telemetry, race status, track gate information, and collision information | yes |
| `PyAIPilotExample/vision_rx.py` | receives UDP image fragments, reassembles JPEG, decodes with OpenCV | yes |
| `PyAIPilotExample/controller.py` | sends arm/reset/motor/attitude/position control commands | yes |
| `PyAIPilotExample/timesync.py` | defines TIMESYNC request loop | yes |
| `PyAIPilotExample/requirements.txt` | Python dependencies: pymavlink, opencv-python, numpy, matplotlib, keyboard | depends |

### 3.6.1 Startup and Connection Flow

The current template starts like this:

```text
start AIGP_3364/FlightSim.exe
-> enter the simulator and let it start sending MAVLink / vision data
-> python PyAIPilotExample/main.py
-> setup_components(...)
-> mavutil.mavlink_connection("udpin:127.0.0.1:14550")
-> wait_heartbeat()
-> start MAVLinkRX receive thread
-> start TimeSync request thread
-> start VisionRX image receive thread
-> create Controller
-> controller.arm()
-> while True: controller.update()
```

The Python client listens for MAVLink UDP on `127.0.0.1:14550`. The image
receiver binds to `0.0.0.0:5600`. If the simulator and Python client run on
different machines, modify the MAVLink address in `main.py` and confirm the
simulator's UDP output target.

### 3.6.2 Actual Data Flow in Code

MAVLink state flow:

```text
Simulator
-> UDP 14550
-> setup.py / mavutil.mavlink_connection(...)
-> mavlink_rx.py / recv_match(blocking=False)
-> on_heartbeat / on_attitude / on_odometry / on_highres_imu / on_collision / ...
-> shared_data or logging system
-> controller.py
```

Image flow:

```text
Simulator
-> UDP 5600
-> vision_rx.py / sock.recvfrom(...)
-> 24-byte header: frame_id, chunk_id, total_chunks, jpeg_size, payload_size, sim_time_ns
-> reassemble JPEG by frame_id and chunk_id
-> cv2.imdecode(...)
-> process_frame(frame_id, image)
-> gate detection / pose estimation
-> shared_data or logging system
-> controller.py
```

Control-command flow:

```text
controller.py
-> arm: MAV_CMD_COMPONENT_ARM_DISARM
-> reset: custom MAVLINK_CMD_SIM_RESET = 31000
-> set_actuator_control_target_send(...)
-> set_attitude_target_send(...)
-> set_position_target_local_ned_send(...)
-> Simulator
```

Three control interfaces are already provided:

| Function | MAVLink message | Current meaning |
| -------- | --------------- | --------------- |
| `update_motor_control()` | `SET_ACTUATOR_CONTROL_TARGET` | directly sends an 8-channel actuator/motor array |
| `update_attitude_flight_control()` | `SET_ATTITUDE_TARGET` | sends roll/pitch/yaw rate and thrust |
| `update_position_flight_control()` | `SET_POSITION_TARGET_LOCAL_NED` | example NED velocity target |

Currently, `Controller.update()` reads a `FlightCommand` from
`KeyboardControlConsole`, then calls `update_attitude_flight_control()` to send
body-rate and thrust targets. This is still a manual keyboard-control example,
not an autonomous baseline suitable for a timed run.

### 3.6.3 MAVLink Parsing and Available Information

`mavlink_rx.py` currently recognizes these message types:

| Message | Handler | Available information |
| ------- | ------- | --------------------- |
| `HEARTBEAT` | `on_heartbeat()` | whether armed |
| `TIMESYNC` | `on_timesync()` | request/response timestamps |
| `ATTITUDE` | `on_attitude()` | roll, pitch, yaw, angular rates |
| `LOCAL_POSITION_NED` | `on_local_position_ned()` | NED position and velocity |
| `ODOMETRY` | `on_odometry()` | position, quaternion, velocity, angular velocity, reset count |
| `HIGHRES_IMU` | `on_highres_imu()` | acceleration and gyroscope |
| `ACTUATOR_OUTPUT_STATUS` | `on_actuator_output_status()` | four motor outputs |
| `COLLISION` | `on_collision()` | gate/environment collision ID, threat level, impulse |
| `ENCAPSULATED_DATA` | `on_encapsulated_data()` | race status or track data |
| `DATA_TRANSMISSION_HANDSHAKE` | receive-loop handling | track-data fragment count |

`ENCAPSULATED_DATA` contains two custom payloads:

| data_type | Meaning | Fields |
| --------- | ------- | ------ |
| `1` | race status | `sim_boot_time_ms`, `race_start_boot_time_ms`, `race_finish_time_ns`, `active_gate_index`, `last_gate_race_time` |
| `2` | track info | `gate_id`, gate NED position, quaternion orientation, width, height |

The Python template therefore exposes more than the short technical summary:
camera and basic telemetry, plus active gate index, track gate geometry,
collision events, and actuator output. These fields can greatly reduce
development difficulty, but whether official submissions may rely on track
info for track-specific behavior should be confirmed with the organizers.

### 3.6.4 Python Receive-Side File Input and Output

Here, "input/output" means the actual data processed by Python files, not the
competition-level problem statement. The template is not a pure functional
pipeline; most files communicate through threads, sockets, MAVLink connections,
and side effects on `shared_data`.

Basis for the following statements:

| Content | Basis | Verified from live simulator output |
| ------- | ----- | ----------------------------------- |
| MAVLink message types, read fields, handlers | `PyAIPilotExample/mavlink_rx.py` source | no; from code |
| Image UDP header format `"<IHHIIQ"` | `PyAIPilotExample/vision_rx.py` source | no; from code |
| race status / track info `struct.unpack_from(...)` formats | `PyAIPilotExample/mavlink_rx.py` source | no; from code |
| suggested `shared_data[...]` output structure | recommendation based on current code gaps | no; recommendation |
| numeric examples below | constructed examples to explain format | no; not packet captures |

Receive-side Python I/O overview:

| Python file | Input | Processing | Current output | Output to add |
| ----------- | ----- | ---------- | -------------- | ------------- |
| `main.py` | constants `SIM_SERVER_UDP_IP`, `SIM_SERVER_UDP_PORT`; empty `shared_data`; local `system_boot_ms` | calls `setup_components()`, starts components, arms, enters control loop | component objects, control loop, command side effects | config loading, log path, run state, exit code |
| `setup.py` | `shared_data`, `system_boot_ms`, MAVLink IP/port | creates MAVLink UDP connection, waits for heartbeat, creates MAVLinkRX, TimeSync, VisionRX, Controller | returns dict: `vision_rx`, `mavlink_rx`, `ts_loop`, `sim_conn`, `controller` | shared lock, logger, config object |
| `mavlink_rx.py` | MAVLink messages from `mavlink_connection.recv_match()` | dispatches by message type to `on_heartbeat()`, `on_attitude()`, `on_track_data()`, etc. | mostly local variables; track fragments stored in `track_chunks` | write to `shared_data["telemetry"]`, `shared_data["race"]`, `shared_data["track"]`, `shared_data["events"]` |
| `vision_rx.py` | UDP `5600` image fragment packets | parses 24-byte header, reassembles JPEG by `frame_id/chunk_id`, decodes with OpenCV | calls empty `process_frame(frame_id, image)`; no saved result | write `shared_data["vision"]` with `frame_id`, `sim_time_ns`, image, gate detections |
| `timesync.py` | local `time.time_ns()` and MAVLink connection | sends TIMESYNC request at 10 Hz | sends `TIMESYNC` MAVLink messages | combine with `mavlink_rx.py/on_timesync()` to compute and store clock offset |

The simulator sends data over at least two channels: MAVLink UDP `14550` and
vision UDP `5600`. MAVLink is received by `mavlink_rx.py` using
`recv_match(blocking=False)`. Images are received by `vision_rx.py` using a UDP
socket bound to `0.0.0.0:5600`.

More specific `mavlink_rx.py` I/O:

| Handler | Input message | Read fields | Current Python output |
| ------- | ------------- | ----------- | --------------------- |
| `on_heartbeat(msg)` | `HEARTBEAT` | `msg.base_mode` | local `armed`, not saved |
| `on_timesync(msg)` | `TIMESYNC` | `msg.ts1`, `msg.tc1` | local `request_time`, `response_time`, not saved |
| `on_attitude(msg)` | `ATTITUDE` | `time_boot_ms`, roll/pitch/yaw, rollspeed/pitchspeed/yawspeed | local attitude variables, not saved |
| `on_local_position_ned(msg)` | `LOCAL_POSITION_NED` | `x/y/z`, `vx/vy/vz`, `time_boot_ms` | local position/velocity variables, not saved |
| `on_odometry(msg)` | `ODOMETRY` | `x/y/z`, `q`, `vx/vy/vz`, angular rates, `time_usec`, `reset_counter` | local odometry variables, not saved |
| `on_highres_imu(msg)` | `HIGHRES_IMU` | `xacc/yacc/zacc`, `xgyro/ygyro/zgyro`, `time_usec` | local IMU variables, not saved |
| `on_actuator_output_status(msg)` | `ACTUATOR_OUTPUT_STATUS` | `actuator[0:4]`, `time_usec` | local motor-feedback variables, not saved |
| `on_collision(msg)` | `COLLISION` | `id`, `threat_level`, `horizontal_minimum_delta` | local collision variables, not saved |
| `on_race_status(msg)` | `ENCAPSULATED_DATA` type `1` | custom race-status payload | local race-status variables, not saved |
| `on_track_data_packet(msg)` | `ENCAPSULATED_DATA` type `2` fragment | `data_type`, `transfer_id`, `msg.seqnr`, payload | fragments written to `self.track_chunks`; calls `on_track_data()` when complete |
| `on_track_data(payload)` | assembled track payload | `num_gates` and each gate's position/orientation/width/height | local gate variables, not saved |

In short, `mavlink_rx.py` already has a rich input stream, but most parsed
values are not written to shared state. The next step is to convert each
`on_*()` local variable group into a structured dictionary and write it to
`self.data`, i.e. `shared_data`.

Suggested output format:

```python
shared_data["telemetry"]["attitude"] = {
    "time_boot_ms": time_boot_ms,
    "roll": roll,
    "pitch": pitch,
    "yaw": yaw,
    "rollspeed": roll_speed,
    "pitchspeed": pitch_speed,
    "yawspeed": yaw_speed,
}
```

Constructed example values, not a live packet capture:

```python
# mavlink_rx.py input: an ATTITUDE MAVLink message
msg.time_boot_ms = 15320
msg.roll = 0.03
msg.pitch = -0.08
msg.yaw = 1.57
msg.rollspeed = 0.10
msg.pitchspeed = -0.20
msg.yawspeed = 0.05

# mavlink_rx.py should write this to shared_data
shared_data["telemetry"]["attitude"] = {
    "time_boot_ms": 15320,
    "roll": 0.03,
    "pitch": -0.08,
    "yaw": 1.57,
    "rollspeed": 0.10,
    "pitchspeed": -0.20,
    "yawspeed": 0.05,
}
```

More specific `vision_rx.py` I/O:

| Stage | Input | Processing | Output |
| ----- | ----- | ---------- | ------ |
| UDP receive | `packet` returned by `sock.recvfrom(65536)` | split into `header` and `payload` | raw fragment |
| header parse | `packet[:24]` | `struct.unpack("<IHHIIQ", header)` | `frame_id`, `chunk_id`, `total_chunks`, `jpeg_size`, `payload_size`, `sim_time_ns` |
| fragment cache | `frame_id`, `chunk_id`, `payload` | write to `frames[frame_id]["chunks"][chunk_id]` | in-memory fragment table |
| JPEG reassembly | all chunks for one frame | concatenate chunks from `0..total_chunks-1` into `jpeg_bytes` | complete JPEG bytes |
| image decode | `jpeg_bytes` | `np.frombuffer()` + `cv2.imdecode(..., cv2.IMREAD_COLOR)` | OpenCV BGR image `image` |
| perception | `frame_id`, `image` | current `process_frame()` is empty | currently no output; should output gate detection |

Suggested final output format for `vision_rx.py`:

```python
shared_data["vision"] = {
    "frame_id": frame_id,
    "sim_time_ns": sim_time_ns,
    "image_shape": image.shape,
    "gate_detection": {
        "bbox_xyxy": [x1, y1, x2, y2],
        "center_px": [cx, cy],
        "confidence": confidence,
        "corners_px": [[x0, y0], [x1, y1], [x2, y2], [x3, y3]],
    },
}
```

Constructed example values, not a live packet capture:

```python
# vision_rx.py input: parsed UDP image-fragment header
frame_id = 42
chunk_id = 0
total_chunks = 3
jpeg_size = 84211
payload_size = 30000
sim_time_ns = 123456789000

# after all three chunks are assembled and decoded, assume this gate detection
shared_data["vision"] = {
    "frame_id": 42,
    "sim_time_ns": 123456789000,
    "image_shape": (360, 640, 3),
    "gate_detection": {
        "bbox_xyxy": [250, 90, 390, 230],
        "center_px": [320, 160],
        "confidence": 0.92,
        "corners_px": [[255, 95], [385, 92], [390, 228], [250, 230]],
    },
}
```

The image UDP packet format is hard-coded in `vision_rx.py` as little-endian:

```python
header_format = "<IHHIIQ"
```

The 24-byte header layout:

| Offset | Field | Type | Meaning |
| -----: | ----- | ---- | ------- |
| 0 | `frame_id` | `uint32` | image frame ID |
| 4 | `chunk_id` | `uint16` | fragment ID, starting at `0` |
| 6 | `total_chunks` | `uint16` | total number of JPEG fragments for this frame |
| 8 | `jpeg_size` | `uint32` | full JPEG byte length |
| 12 | `payload_size` | `uint32` | JPEG fragment byte length in this UDP packet |
| 16 | `sim_time_ns` | `uint64` | server-side simulator timestamp for the image frame, in ns |

The header is followed by `payload_size` bytes of JPEG fragment data. The
receiver groups by `frame_id`, concatenates chunks in `chunk_id =
0..total_chunks-1` order, then decodes:

```python
img_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
```

Therefore `process_frame()` receives an OpenCV BGR image, not RGB. The current
signature is `process_frame(frame_id, image)` and does not pass `sim_time_ns`
forward. It should be changed to `process_frame(frame_id, sim_time_ns, image)`
or otherwise write the timestamp to `shared_data`.

Race-status payload format:

```python
struct.unpack_from("<BQqqIq", raw_payload)
```

Field order:

| Field | Type | Meaning |
| ----- | ---- | ------- |
| `data_type` | `uint8` | fixed to `1`, race status |
| `sim_boot_time_ms` | `uint64` | server time since simulator boot, ms |
| `race_start_boot_time_ms` | `int64` | boot time when race started; negative means not started |
| `race_finish_time_ns` | `int64` | race finish timestamp relative to simulator boot, ns; negative means still running |
| `active_gate_index` | `uint32` | current target gate index |
| `last_gate_race_time` | `int64` | race time corresponding to the most recent gate; code comment says seconds |

Track information is fragmented. `DATA_TRANSMISSION_HANDSHAKE` establishes the
fragment context, and multiple `ENCAPSULATED_DATA` messages are concatenated:

1. `DATA_TRANSMISSION_HANDSHAKE.width` is used as `transfer_id`.
2. `DATA_TRANSMISSION_HANDSHAKE.packets` is the number of track-data fragments.
3. Each `ENCAPSULATED_DATA` payload starts with `struct.unpack_from("<BH", raw_payload)` to read `data_type` and `transfer_id`.
4. `data_type = 2` means track info; remaining bytes are stored by `msg.seqnr` in `track_chunks[transfer_id]`.
5. After all `packets` fragments arrive, chunks are concatenated by `seqnr = 0..packets-1` into `full_payload`.

Assembled track payload:

```python
num_gates, = struct.unpack_from("<H", payload)
gate = struct.unpack_from("<Hfffffffff", payload)
```

The payload starts with 2-byte `num_gates`. Each gate then occupies 38 bytes:

| Field | Type | Meaning |
| ----- | ---- | ------- |
| `gate_id` | `uint16` | gate ID |
| `position_ned_x` | `float32` | gate center NED X, m |
| `position_ned_y` | `float32` | gate center NED Y, m |
| `position_ned_z` | `float32` | gate center NED Z, m |
| `orientation_ned_w` | `float32` | gate orientation quaternion w |
| `orientation_ned_x` | `float32` | gate orientation quaternion x |
| `orientation_ned_y` | `float32` | gate orientation quaternion y |
| `orientation_ned_z` | `float32` | gate orientation quaternion z |
| `width` | `float32` | gate width, m |
| `height` | `float32` | gate height, m |

`ODOMETRY.q` and gate orientation in track info are stored as quaternion
`w, x, y, z`. In `mavlink_rx.py`, the code maps `msg.q[1]`, `msg.q[2]`,
`msg.q[3]`, and `msg.q[0]` to local variables `qx, qy, qz, qw`.

Most important conclusion: **the code template can obtain every gate's NED
position, orientation, and size, and it can obtain the current active gate
index.** This enables not only pure visual servoing, but also a hybrid approach
using track information plus visual correction.

Caveats:

1. `mavlink_rx.py` currently unpacks fields into local variables but does not write them to `shared_data`, so the controller cannot use them yet.
2. `vision_rx.py` parses `sim_time_ns` but does not propagate it; high-speed flight needs this timestamp for image-delay compensation.
3. `LOCAL_POSITION_NED`, `ODOMETRY`, and track info look like simulator ground truth, while the technical spec emphasizes no GPS / absolute global position. Whether official submissions may depend on these fields should be confirmed. They can be used for development, logging, and validation first, then gradually removed or de-emphasized.

Suggested unified `shared_data` structure:

```python
shared_data = {
    "telemetry": {
        "attitude": {...},
        "local_position_ned": {...},
        "odometry": {...},
        "imu": {...},
        "actuator_output": {...},
    },
    "race": {
        "active_gate_index": None,
        "race_start_boot_time_ms": None,
        "race_finish_time_ns": None,
        "last_gate_race_time": None,
    },
    "track": {
        "gates": [],
    },
    "vision": {
        "frame_id": None,
        "sim_time_ns": None,
        "image": None,
        "gate_detection": None,
    },
    "events": {
        "last_collision": None,
    },
}
```

### 3.6.5 Python Control-Side File Input and Output

Control-side Python files convert "control input" into MAVLink commands sent to
the simulator. The current template receives control input from the keyboard.
The official autonomous version should instead compute inputs from
`shared_data`: vision, attitude, velocity, race status, and track info.

Basis for the following statements:

| Content | Basis | Verified from live simulator output |
| ------- | ----- | ----------------------------------- |
| `FlightCommand` fields | `PyAIPilotExample/flight_command.py` source | not simulator output |
| keyboard-to-control mapping | `PyAIPilotExample/keyboard_control.py` source | not simulator output |
| arm/reset/attitude/position/timesync call formats | `PyAIPilotExample/controller.py`, `timesync.py` source | no; from code |
| `AutonomyCommandSource` | suggested structure for replacing keyboard input | no; recommendation |
| numeric control examples below | constructed to explain format | no; not live flight data |

Control-side Python I/O overview:

| Python file | Input | Processing | Current output | Autonomous version should output |
| ----------- | ----- | ---------- | -------------- | -------------------------------- |
| `keyboard_control.py` | keyboard state: `A/D/W/S/Q/E/Space/Esc` | maps keys to angular rates and thrust | `FlightCommand` object | should not be used in an official timed run |
| `flight_command.py` | no runtime input; defines data structure | defines `FlightCommand` dataclass | unified control-command format | can remain the output format for the autonomy controller |
| `controller.py` | `sim_conn`, `shared_data`, `system_boot_ms`, `command_source` | calls `command_source.read_command()` each frame, then sends MAVLink command | default `SET_ATTITUDE_TARGET`; `update()` returns whether to keep running | compute `FlightCommand` automatically from `shared_data` |
| `timesync.py` | local time, MAVLink connection | builds TIMESYNC requests at 10 Hz | `TIMESYNC` MAVLink messages | also write sync offset for logging / delay compensation |
| `main.py` | component objects, user interrupt | arms, loops over `controller.update()`, joins threads on exit | `MAV_CMD_COMPONENT_ARM_DISARM` and continuous control commands | batch testing, auto reset, log flush |

The control input interface itself is not MAVLink; it is the `command_source`
object used by `Controller`. The object only needs a minimal interface:

```python
class CommandSource:
    def read_command(self) -> FlightCommand:
        ...

    def close(self) -> None:
        ...
```

`close()` is optional; `Controller.shutdown()` checks it via `getattr()`.
`read_command()` is the required part. It returns one `FlightCommand` every
control cycle.

Current default chain:

```text
main.py
-> setup_components(...)
-> KeyboardControlConsole()
-> Controller(...)
-> controller.arm()
-> while is_running: controller.update()
-> command_source.read_command()
-> update_attitude_flight_control(...)
-> SET_ATTITUDE_TARGET
```

The current template is therefore a keyboard-control sample, not autonomous
control. The command object is defined in `flight_command.py`:

```python
@dataclass(frozen=True)
class FlightCommand:
    roll_rate: float
    pitch_rate: float
    yaw_rate: float
    thrust: float
    exit_requested: bool = False
```

Keyboard mapping:

| Key | Field change | Current magnitude |
| --- | ------------ | ----------------- |
| `A` / `D` | `roll_rate` negative / positive | `1.0 rad/s` |
| `W` / `S` | `pitch_rate` negative / positive | `1.0 rad/s` |
| `Q` / `E` | `yaw_rate` negative / positive | `1.0 rad/s` |
| `Space` | `thrust = 0.6`, otherwise `0.0` | normalized thrust |
| `Esc` | `exit_requested = True` | exits control loop |

From a file-I/O perspective, `keyboard_control.py` takes key states as input
and outputs:

```python
FlightCommand(
    roll_rate=roll_axis * 1.0,
    pitch_rate=pitch_axis * 1.0,
    yaw_rate=yaw_axis * 1.0,
    thrust=0.6 if space_pressed else 0.0,
    exit_requested=esc_pressed,
)
```

For autonomous flight, replace `KeyboardControlConsole` with something like:

```python
class AutonomyCommandSource:
    def __init__(self, shared_data):
        self.shared_data = shared_data

    def read_command(self):
        detection = self.shared_data["vision"]["gate_detection"]
        attitude = self.shared_data["telemetry"]["attitude"]
        return FlightCommand(
            roll_rate=...,
            pitch_rate=...,
            yaw_rate=...,
            thrust=...,
            exit_requested=False,
        )
```

Constructed example values, not live flight data:

```python
# controller.py input: detection from vision_rx.py
shared_data["vision"]["gate_detection"] = {
    "center_px": [370, 160],
    "confidence": 0.90,
}

# image center is [320, 180], so the gate is 50 px right and 20 px up
error_x = 370 - 320
error_y = 160 - 180

# a simple proportional controller could output:
command = FlightCommand(
    roll_rate=0.0,
    pitch_rate=-0.2,
    yaw_rate=0.5,
    thrust=0.55,
    exit_requested=False,
)
```

This `FlightCommand` then enters `update_attitude_flight_control()`. The actual
simulator output command is:

```python
set_attitude_target_send(
    time_boot_ms,
    target_system,
    target_component,
    ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE,
    [1, 0, 0, 0],
    0.0,   # roll_rate
    -0.2,  # pitch_rate
    0.5,   # yaw_rate
    0.55,  # thrust
)
```

The control-loop frequency is `CONTROL_HZ = 60`, which satisfies the spec's
requirement that control commands stay below 100 Hz.

| Control mode | Code function | MAVLink message / command | Suitable use | Risk |
| ------------ | ------------- | ------------------------- | ------------ | ---- |
| Arm vehicle | `arm()` | `MAV_CMD_COMPONENT_ARM_DISARM`, param1 = `1` | required before takeoff | only arms; does not stabilize flight |
| Reset simulator | `send_sim_reset_command()` | custom command `31000` | automated testing, repeated trials | whether allowed in official timed run must be confirmed |
| Direct motor / actuator | `update_motor_control()` | `SET_ACTUATOR_CONTROL_TARGET` | low-level control, response research | hardest to tune; can lose control |
| Attitude / body-rate / thrust | `update_attitude_flight_control()` | `SET_ATTITUDE_TARGET` | visual servoing, racing control | requires a stable outer loop |
| Local position / velocity target | `update_position_flight_control()` | `SET_POSITION_TARGET_LOCAL_NED` | low-speed baseline, interface experiments | exact supported fields must be tested |
| Time-sync request | `timesync_send()` | `TIMESYNC` | align local time and simulator time | not a flight-control command |

From a dynamics I/O perspective, we can confirm the control inputs and observed
outputs, but not the full dynamics equations. The simulator's internal mass,
inertia, thrust curve, motor response, drag, and inner-loop flight-controller
gains are not exposed in the Python template.

Treat the simulator as a black box:

```text
x(t + dt), y(t) = simulator_dynamics(x(t), u(t))
```

where `u(t)` is the control sent to FlightSim and `y(t)` is telemetry observed
by Python.

Known I/O:

| Control mode | Input control `u(t)` | Input frame / unit | Internal simulator meaning | Observable output `y(t)` |
| ------------ | -------------------- | ------------------ | -------------------------- | ------------------------ |
| body-rate + thrust, current default | `roll_rate`, `pitch_rate`, `yaw_rate`, `thrust` | body frame; rad/s; `thrust` 0..1 normalized | sent through `SET_ATTITUDE_TARGET` to simulator / inner-loop controller; quaternion attitude ignored | `ATTITUDE`; `ODOMETRY` / `LOCAL_POSITION_NED`; `HIGHRES_IMU` |
| local velocity target | `vx`, `vy`, `vz` | `MAV_FRAME_LOCAL_NED`; m/s; N forward, E right, D down | sent through `SET_POSITION_TARGET_LOCAL_NED` to high-level position/velocity controller; current mask enables only velocity | same telemetry; can evaluate velocity-command response |
| direct actuator | `controls[0:8]`, currently first four used | actuator floats; normalization not specified in template | direct low-level actuator/motor channels via `SET_ACTUATOR_CONTROL_TARGET` | `ACTUATOR_OUTPUT_STATUS`, attitude, velocity, IMU, collision |

The closest current relation to "what do I input, and what does the aircraft
output" is:

```text
Input:
FlightCommand(
    roll_rate=p_cmd,
    pitch_rate=q_cmd,
    yaw_rate=r_cmd,
    thrust=T_cmd,
)

Sent:
SET_ATTITUDE_TARGET(
    body_roll_rate=p_cmd,
    body_pitch_rate=q_cmd,
    body_yaw_rate=r_cmd,
    thrust=T_cmd,
)

Output:
ATTITUDE:
    roll, pitch, yaw, rollspeed, pitchspeed, yawspeed
LOCAL_POSITION_NED / ODOMETRY:
    x, y, z, vx, vy, vz
HIGHRES_IMU:
    xacc, yacc, zacc, xgyro, ygyro, zgyro
ACTUATOR_OUTPUT_STATUS:
    actuator[0:4]
```

At this stage, do not write a trusted white-box model such as:

```text
m * a = R * thrust - m * g - drag
I * omega_dot = torque - omega x I omega
```

Those are common quadrotor model forms, but `m`, `I`, thrust coefficient,
torque coefficient, drag, and inner-loop controller parameters are not present
in the current materials. A usable dynamics model requires system
identification: apply step/sweep control inputs, record telemetry outputs, and
fit the response.

Specific control-message formats follow.

**Arm:**

```python
command_long_send(
    target_system,
    target_component,
    MAV_CMD_COMPONENT_ARM_DISARM,
    0,  # confirmation
    1,  # param1: arm
    0, 0, 0, 0, 0, 0
)
```

To disarm, `param1` is usually `0`.

**Reset simulator:**

```python
command_long_send(
    target_system,
    target_component,
    31000,  # MAVLINK_CMD_SIM_RESET
    0,
    0, 0, 0, 0, 0, 0, 0
)
```

This is suitable for automated testing and repeated trials. Whether it is
allowed in an official timed run must be confirmed.

**Time-sync request:**

```python
timesync_send(
    int(time.time_ns()),  # tc1: client time ns
    0                    # ts1: 0 means request
)
```

`timesync.py` sends requests at `TIMESYNC_REQUEST_HZ = 10`. `setup.py` already
calls `TimeSync.create_timesync(...)` to start the thread.

Main flight-control modes:

1. **Direct actuator control**

   `update_motor_control()` sends an 8-channel actuator array. The current code
   approximately uses:

   ```python
   motor_rpms = [front_left, front_right, back_left, back_right, 0, 0, 0, 0]
   ```

   MAVLink call:

   ```python
   set_actuator_control_target_send(
       time_usec,
       target_system,
       target_component,
       0,          # group_mlx
       controls    # 8 floats
   )
   ```

   The current example constants are `front_left=0`, `front_right=1`,
   `back_left=0`, `back_right=0`; they are only useful to verify that the
   interface is connected. This interface requires implementing motor mixing,
   attitude stabilization, and thrust allocation. It is not suitable as the
   first autonomous gate-passing solution.

2. **Attitude target / body-rate target control**

   `update_attitude_flight_control()` currently ignores quaternion attitude and
   sends body roll rate, pitch rate, yaw rate, and thrust:

   ```text
   roll_rate, pitch_rate, yaw_rate, thrust
   ```

   MAVLink call:

   ```python
   set_attitude_target_send(
       time_boot_ms,
       target_system,
       target_component,
       ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE,  # usually 128
       [1, 0, 0, 0],  # quaternion, ignored by type_mask
       roll_rate,     # rad/s
       pitch_rate,    # rad/s
       yaw_rate,      # rad/s
       thrust         # 0..1 normalized
   )
   ```

   `time_boot_ms` is computed as `int(time.time() * 1000) - system_boot_ms`.
   `system_boot_ms` is the client-side local time baseline recorded by
   `main.py`. Strictly speaking, it is not simulator boot time; it simply gives
   the MAVLink message a monotonically increasing timestamp.

   This mode fits visual servoing:

   ```text
   gate left/right error -> adjust yaw_rate or roll_rate
   gate up/down error -> adjust pitch_rate or thrust
   gate appears larger / closer -> reduce forward speed or prepare to switch gate
   ```

   If FlightSim responds stably to `SET_ATTITUDE_TARGET`, this is the most
   promising control mode for racing.

3. **Local position / velocity target control**

   `update_position_flight_control()` sends `SET_POSITION_TARGET_LOCAL_NED`.
   The current type mask ignores position, acceleration, yaw, and yaw_rate; it
   uses only velocity:

   ```text
   frame = MAV_FRAME_LOCAL_NED
   type_mask = ignore x/y/z + ignore ax/ay/az + ignore yaw/yaw_rate
   vx = 2.0 m/s
   vy = 0.0
   vz = 0.0
   ```

   MAVLink call:

   ```python
   set_position_target_local_ned_send(
       time_boot_ms,
       target_system,
       target_component,
       MAV_FRAME_LOCAL_NED,
       VELOCITY_POSITION_MASK,  # usually 3527, uses only vx/vy/vz
       0.0, 0.0, 0.0,    # x, y, z ignored
       2.0, 0.0, 0.0,    # vx, vy, vz used
       0.0, 0.0, 0.0,    # ax, ay, az ignored
       0.0,              # yaw ignored
       0.0               # yaw_rate ignored
   )
   ```

   This is useful for an early low-speed baseline: send a velocity target and
   verify that the aircraft moves forward stably. It can later be extended to:

   ```text
   generate a velocity vector from the current gate's NED position
   or correct vx / vy / vz from visual error
   ```

   However, because the specification emphasizes no GPS / absolute global
   position, using NED gate truth in the final strategy requires a rules check.

Control-related points already aligned:

1. `CONTROL_HZ = 60`, within the recommended 50-80 Hz range and below the 100 Hz limit.
2. `setup.py` already calls `TimeSync.create_timesync(...)`, so the timesync request loop runs.
3. `main.py` supports `Ctrl+C` exit and calls `get_thread_for_join()` to stop TimeSync, MAVLink RX, and Vision RX threads.

Core remaining task: replace `KeyboardControlConsole` with an autonomous command
source, or make `Controller.update()` compute `FlightCommand` directly from
`shared_data`.

Recommended first control chain:

```text
vision_rx.py detects gate center
-> writes shared_data["vision"]["gate_detection"]
-> mavlink_rx.py writes attitude, velocity, active_gate_index
-> controller.py reads shared_data
-> computes yaw_rate / pitch_rate / thrust from image gate-center error
-> sends SET_ATTITUDE_TARGET to the simulator
```

For a more stable early interface test, use `SET_POSITION_TARGET_LOCAL_NED` for
low-speed velocity control first, then migrate to `SET_ATTITUDE_TARGET`.

### 3.6.6 Current Template Gaps

The template is not yet a complete autonomous system. Main gaps:

1. `vision_rx.py` has an empty `process_frame()`; there is no gate detection, image logging, or perception output.
2. `mavlink_rx.py` parses telemetry fields but mostly keeps them as local variables; it does not write them to `shared_data`, so the controller cannot read state.
3. `shared_data` is a plain dict without locks; once receive threads and control thread share data, add `threading.Lock` or a thread-safe queue.
4. `timesync.py` can send TIMESYNC requests, but `on_timesync()` does not compute clock offset or store sync results.
5. Default control comes from `KeyboardControlConsole`, so manual key input is still required; it is not a timed-run autonomous baseline.
6. `update_motor_control()` has only one nonzero motor in the example constants; switching to actuator control can easily lose control.
7. There is no client heartbeat sender. If the official interface requires client heartbeat at least 2 Hz, add it.
8. Image reception can handle out-of-order fragments but does not time out or clean lost incomplete frames, which can leak memory over long runs.
9. `main.py` has `Ctrl+C` and thread join, but lacks exception protection, log flush, automatic reset, and batch trial flow.
10. There is no complete data logging, replay, automated testing, or tuning framework in the original template.

### 3.6.7 Files to Modify First

Suggested minimum viable baseline order:

| Order | File | Add |
| ----: | ---- | --- |
| 1 | `mavlink_rx.py` | write attitude, odometry, IMU, race status, track info, and collision to `shared_data` |
| 2 | `vision_rx.py` | implement gate detection in `process_frame()` and output bbox/center/confidence/timestamp |
| 3 | `controller.py` | read `shared_data`; replace keyboard `FlightCommand` with gate-center servoing or trajectory-control commands |
| 4 | `setup.py` | add shared lock, logger, and timesync offset writing |
| 5 | `main.py` | add config, exception handling, cleanup, optional reset, and log paths |

After these five steps, the abstract modules from the spec - communication and
logging, gate detection, state estimation, and controller - will align with the
real codebase.

---

## 4. Most Likely High-Score Logic

The document does not provide the official scoring formula, so it is not safe
to say "high score equals shortest time." But from drone racing and the
qualification description, a reasonable objective order is:

1. Complete the course legally.
2. Use no human intervention.
3. Do not miss gates, hit gates, or leave bounds.
4. Minimize time once stable completion is achieved.

Strategy priority:

| Priority | Goal | Reason |
| -------: | ---- | ------ |
| 1 | 100% autonomous completion | without completion or with human intervention, score is meaningless |
| 2 | stable gate passage | gates are the core task; collision or missed gates are the biggest risk |
| 3 | reduce visual loss | at high speed, losing the gate from view often leads directly to failure |
| 4 | optimize path and speed | speed matters only after stability |
| 5 | exploit deterministic track structure | repeated testing can find a faster racing line |

---

## 5. Recommended Technical Paths

### Path A: Fastest Reliable Version

Best for getting a valid score quickly.

Architecture:

```text
UDP image reassembly
-> gate detection
-> gate center-error estimation
-> visual servo control
-> speed / yaw / pitch / roll commands
-> after passing a gate, search for the next gate
```

Core approach:

1. Use classical vision or a lightweight model to detect gates.
   - If gate colors are stable, use HSV thresholding, edges, contours, and rectangle detection.
   - If lighting varies, train a lightweight YOLO or segmentation model.
2. Use the gate inner-opening center as the target.
3. If the gate center is left/right in the image, correct yaw/roll.
4. If the gate center is up/down, correct pitch/thrust.
5. A larger apparent gate size means the aircraft is closer; use this as coarse distance.
6. Fly slowly at first to ensure completion.
7. Log image, attitude, velocity, and commands for every failure.

Advantages: fast to develop, no full 3D map dependency.

Disadvantages: limited at high speed; turns and occlusions are fragile.

### Path B: Higher-Score Version

Best after reliable completion is achieved.

Architecture:

```text
image + telemetry
-> gate corner detection / segmentation
-> PnP relative-pose estimation
-> EKF / state estimation
-> gate-to-gate trajectory planning
-> speed-profile optimization
-> MPC / cascaded PID control
```

Core approach:

1. Detect the four inner-corner or outer-frame keypoints of the gate.
2. Use known gate dimensions and camera intrinsics with PnP to estimate gate pose relative to the camera.
3. Correct for the camera's 20 deg upward pitch relative to the body.
4. Use IMU, attitude, and velocity for short-term prediction between 30 Hz image frames.
5. Assign three waypoints per gate:
   - pre-gate approach point
   - gate center
   - post-gate exit point
6. Generate a racing line from gate orientation and next-gate direction.
7. Accelerate on straights and decelerate before turns.
8. Switch to conservative mode when visual confidence is low, the gate is partial, or the aircraft is too close.

This path has a better chance of scoring well because it does not merely "fly
toward whatever gate is visible"; it plans attitude, speed, and next-gate entry
angle ahead of time.

---

## 6. Useful Competition Properties

### 6.1 Deterministic Course

The document explicitly says course geometry, physics parameters, and
environmental conditions are deterministic.

This enables:

1. repeated runs on the same track
2. a known gate order and relative-pose table
3. automated speed-parameter sweeps
4. Bayesian optimization / CMA-ES / grid search over segment parameters
5. saved best approach angle for each gate
6. special strategies for specific turns

But confirm whether the organizers allow pre-recorded track information. The
document does not forbid it, but it also does not explicitly authorize it.

### 6.2 Known Gate Size

The gate inner opening is known to be 1.5 m x 1.5 m. Use this for:

1. PnP distance estimation
2. gate-alignment checks
3. gate-crossing time estimation
4. speed limit selection
5. safety corridor computation

### 6.3 No Camera Distortion

The document says the camera is a standard pinhole model with no lens
distortion.

This reduces visual-localization difficulty. OpenCV `solvePnP`, homography,
and keypoint reprojection error can be used directly for pose and confidence
estimation.

---

## 7. Most Important Engineering Modules

### 7.1 Communication and Logging

This is the first priority, not the vision model.

Must implement:

1. MAVLink connection
2. heartbeat maintenance
3. command stream
4. telemetry parser
5. TIMESYNC processing
6. vision UDP packet reassembly
7. full logging

Suggested log fields:

| Type | Fields |
| ---- | ------ |
| time | `sim_time_ns`, local time, latency |
| image | `frame_id`, completeness, JPEG size |
| telemetry | attitude, IMU, velocity, system status |
| perception | gate bbox, corners, confidence, relative pose |
| planning | current target gate, waypoint, target speed |
| control | sent attitude / position target / thrust |
| events | gate passed, collision, lost tracking, recovery |

Without logs, speed optimization is impossible.

### 7.2 Gate Detection

Suggested stages:

First version: classical CV.

- color thresholding
- edge detection
- contour filtering
- rectangle / frame detection
- gate-center servoing

Second version: deep learning.

- YOLO gate detection
- segmentation of gate frame
- corner/keypoint model for four corners
- data augmentation for lighting and blur

For high score, keypoint/segmentation is better than bbox-only detection.
BBoxes can roughly align the drone, but keypoints can support real pose
estimation.

### 7.3 State Estimation

Because there is no GPS or global position, state estimation should focus on
"relative pose to the gate."

Recommended structure:

```text
IMU / attitude / velocity prediction
+ camera-based gate pose update
+ confidence-based measurement weighting
```

If visual confidence is high, use gate pose to correct the state. If the gate is
briefly lost, use IMU and velocity prediction for 0.5-1 second. If loss exceeds
a threshold, slow down and search for the gate.

### 7.4 Controller

Use two layers:

1. outer layer: trajectory and speed planning
2. inner layer: attitude / velocity / yaw control

If the simulator supports `SET_ATTITUDE_TARGET` stably, racing performance will
usually be better than with high-level position commands. If time is limited,
prototype with `SET_POSITION_TARGET_LOCAL_NED` or velocity targets, then move
to attitude targets.

Control goals:

1. low lateral error when passing through a gate
2. yaw aligned with gate normal
3. no aggressive roll near the gate plane
4. quick turn toward the next gate after passage
5. slow before turns, accelerate after turns

---

## 8. High-Score Strategy: Reliable First, Fast Later

### Stage 1: Get a Completing Baseline

Goal: slow, no collision, full course completion.

Suggested parameters:

| Item | Recommendation |
| ---- | -------------- |
| Speed | conservative fixed speed |
| Control frequency | 50-80 Hz |
| Vision | 30 Hz gate-center servoing |
| Safety | slow down when the gate is lost |
| Gate passage | continue forward briefly after passing a gate, then search for the next gate |

### Stage 2: Improve Reliability

Goal: repeated-run success rate near 100%.

Add:

1. gate detection confidence
2. dropped-frame handling
3. visual-delay compensation
4. speed limiting before turns
5. recovery logic
6. automated regression tests

After each change, run at least 20 trials and track:

| Metric | Target |
| ------ | ------ |
| completion rate | >95% |
| gate collisions | near 0 |
| gate lost count | as low as possible |
| average lateral gate-passage error | <0.3-0.4 m |
| maximum lateral gate-passage error | <0.5 m |
| control latency | stable and predictable |

### Stage 3: Optimize Speed

Goal: reduce total time without significantly hurting completion rate.

Optimize:

1. maximum speed per segment
2. approach angle for each gate
3. whether to turn early after gate passage
4. straight-line acceleration
5. braking point before turns
6. yaw alignment near gates
7. fallback speed when visual confidence is low

### Stage 4: Track-Specific Optimization

Because the environment is deterministic, tune parameters per gate:

```text
gate 1: start acceleration
gate 2: conservative passage
gate 3: early right turn
gate 4: slow down before entry
...
```

More advanced automated search:

```text
parameters = [target speed per segment, pre-gate distance, post-gate distance, yaw gain, lateral gain]
objective = finish time + collision penalty + missed-gate penalty + lateral-error penalty
optimizer = Bayesian optimization / CMA-ES / grid search
```

---

## 9. Questions the Document Does Not Answer

These directly affect high-score strategy:

| Question | Why it matters |
| -------- | -------------- |
| What is the scoring formula? | decides whether to take speed risks or prioritize conservative completion |
| Does collision immediately fail the run? | determines safety margin |
| How is a missed gate detected? | determines gate-crossing logic |
| Is pre-recorded track information allowed? | determines whether deterministic racing lines can be used |
| Are offline training, RL, or model training allowed? | affects AI approach |
| Are external libraries, model size, or GPU usage limited? | affects vision model |
| What exact MAVLink control fields are supported? | affects controller design |
| Are MAVLink endpoints, ports, and launch flow configurable? | template defaults are MAVLink UDP 14550 and vision UDP 5600, but official remote configuration still needs confirmation |
| Is there a reset API or multiple-attempt mechanism? | affects testing and competition strategy |
| Will the track change, or is Virtual Qualifier 1 fixed? | decides whether track-specific optimization is viable |

---

## 10. Recommended Final System

Most reasonable high-score architecture:

```text
Vision UDP Receiver
-> JPEG Frame Reassembler
-> Gate Detection / Keypoint Estimation
-> Gate Relative Pose Estimation
-> Telemetry Fusion
-> Local State Estimator
-> Gate Sequence Manager
-> Trajectory Planner
-> Speed Profiler
-> Attitude / Velocity Controller
-> MAVLink Command Sender
-> Logger + Auto Tuner
```

Most important elements:

1. stable image reassembly
2. reliable gate detection
3. correct coordinate transforms
4. accurate gate-passage detection
5. conservative speed profile first, then gradual speed increase
6. every failure reproducible from logs

---

## 11. Concrete Task List

### Week 1: Bring Up Interfaces

1. Run the simulator on Windows 11.
2. Establish UDP MAVLink connection with MAVSDK / pymavlink.
3. Receive `HEARTBEAT`, `ATTITUDE`, and `HIGHRES_IMU`.
4. Implement heartbeat sending.
5. Implement test scripts for `SET_POSITION_TARGET_LOCAL_NED` and `SET_ATTITUDE_TARGET`.
6. Implement vision UDP 5600 image reassembly.
7. Save video and telemetry logs.

### Week 2: Low-Speed Autonomous Gate Passage

1. Implement gate detection.
2. Implement gate-center servoing.
3. Pass through a single gate at low speed.
4. Implement gate-passed detection.
5. Extend to multiple gates.
6. Fly the full course without optimizing speed.

### Week 3: Reliability

1. Add image packet-loss handling.
2. Add visual confidence.
3. Add recovery when a gate is lost.
4. Add speed reduction before turns.
5. Add attitude limits.
6. Run 20-50 automated trials and classify failures.

### Week 4: Speed

1. Set different speeds for each gate-to-gate segment.
2. Add PnP pose estimation.
3. Add trajectory feed-forward.
4. Optimize pre-gate and post-gate waypoints.
5. Use automated parameter search.
6. Lock one "fastest" and one "most reliable" configuration.

---

## 12. Conclusion

From the specification, high score depends on full-system engineering, not a
single algorithm:

1. **Build reliable completion first**, because speed is meaningless without completion.
2. **Use gate size, camera intrinsics, and deterministic course structure**.
3. **Use visual relative localization instead of GPS/global-position dependency**.
4. **Keep control below 100 Hz, but run the control layer faster than the vision layer**.
5. **Treat logging, automated testing, and tuning as core modules**.
6. **Use track-specific racing lines and speed profiles later for time advantage**.

Recommended path: first build a classical-vision + visual-servo baseline, then
upgrade to gate keypoints/PnP + state estimation + trajectory planning +
automatic tuning.
