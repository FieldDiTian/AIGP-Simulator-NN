| Item | Result |
|---|---|
| Engine | Unreal Engine 4 |
| Version | **UE 4.27 family**; pak metadata contains `EngineVersion: 4.27.0` |
| Build type | Win64 `Shipping` |
| Cooked platform | `WindowsNoEditor` |
| Launcher | `AIGP_3364/FlightSim.exe` |
| Actual UE executable | `AIGP_3364/FlightSim/Binaries/Win64/DCGame-Win64-Shipping.exe` |
| Asset package | `FlightSim-WindowsNoEditor.pak` |

Dependency versions:

| Component | Version / Evidence |
|---|---|
| PhysX | `PhysX 3.4.0.0` |
| APEX | `1.4.0` |
| PhysXVehicles | pak contains `PhysXVehicles.uplugin` |
| PhysXCooking | pak contains `PhysXCooking.uplugin` |
| FMOD | `2.2.3` |
| OpenXR Loader | `1.0.17.0` |
| XAudio2.9 redist | `1.0.0.1` |
| OpenSSL | `1.1.1k` |
| Intel Open Image Denoise | `1.4.0` |
| PGOS SDK | `0.19.0.1038` |
| GME Engine | `2.9.6.534` |
| PlayFab plugin | pak contains `Unreal Engine 4.27 Current API version: 1.62.210820` |

Important conclusion:

```text
This is a UE4.27 / PhysX 3.4 project.
It is not a UE5 / Chaos project.
```

Still unknown:

- Epic changelist / exact branch
- Whether it is stock UE4.27.0 or a custom fork
- Project source code, flight-control source code, and map source code
- Project-level physical parameters such as drone mass, inertia, and thrust curves

Most accurate summary: **The AI-GP simulator is a UE4.27-family Win64 Shipping / WindowsNoEditor packaged project. Its physics runtime includes PhysX 3.4, APEX 1.4, and the PhysXVehicles/PhysXCooking plugins. Audio uses FMOD 2.2.3.**
