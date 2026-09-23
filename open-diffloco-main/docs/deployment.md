# Go2 Deployment

This repository deploys trained Go2 policies with the C++ Unitree low-level
controller under `src/envs/go2/variants/<variant>/deploy_cpp`. Python is used
only to export training checkpoints and to provide optional ROS2 command
publishers.

## Accepted policies

The currently accepted real-robot candidates are:

| Method | Training run | Checkpoint | Deployment policy |
| --- | --- | --- | --- |
| SHAC | `training_runs/shac_20260820_002033` | `policy_best_tracking.pkl` | `policy_best_tracking_deploy.npz` |
| JAVE | `training_runs/jave_20260820_151021` | `policy_best_tracking.pkl` | `policy_best_tracking_deploy.npz` |

Both policies use the `blind_nolinvel_nokinref` Go2 variant, 50 Hz policy
inference, 500 Hz low-level motor commands, and command ranges exported in the
`.npz`: `vx=[-1.5, 1.5] m/s`, `vy=[-1.0, 1.0] m/s`, `yaw=[-1.5, 1.5] rad/s`.

## 1. Prepare the deployment computer

Use a computer connected to the Go2 low-level DDS network, for example the
onboard PC or an external NUC over Ethernet.

Install runtime/build dependencies on that computer:

- Unitree C++ SDK2 with CMake package discovery enabled.
- Eigen3, zlib, CMake, and a C++17 compiler.
- Python environment from this repository for policy export.
- ROS2 only if you want `/velocity_command` command-topic control.

Identify the robot network interface:

```bash
ip -br link
```

Common real-robot interfaces are `eth0`, `enp3s0`, or another wired adapter.
Use `lo` only for simulator/loopback tests.

## 2. Export the accepted policies

From the repository root:

```bash
cd /home/a/文档/diffloco/open-diffloco-main
PATH="$PWD/.conda/bin:$PATH" XLA_PYTHON_CLIENT_PREALLOCATE=false \
  ./.conda/bin/python -m src.deploy.accepted_go2_deploy export all
```

This produces:

```text
training_runs/shac_20260820_002033/policy_best_tracking_deploy.npz
training_runs/jave_20260820_151021/policy_best_tracking_deploy.npz
```

The deployment `.npz` contains actor weights, observation-normalizer statistics,
default joint angles, action scale, actuator gains, policy rate metadata, and
velocity command ranges. Do not deploy the `.pkl` file directly.

## 3. Build the C++ deployment executable

Build without ROS2 if you will use the built-in terminal or wireless command
source:

```bash
cd /home/a/文档/diffloco/open-diffloco-main
cmake -S . -B build -DOPEN_DIFFLOCO_ENABLE_ROS2=OFF
cmake --build build -j"$(nproc)"
```

Equivalent launcher command:

```bash
PATH="$PWD/.conda/bin:$PATH" ./.conda/bin/python \
  -m src.deploy.accepted_go2_deploy build --build-dir build
```

Build with ROS2 only if you need command messages on `/velocity_command`:

```bash
cmake -S . -B build-ros2 -DOPEN_DIFFLOCO_ENABLE_ROS2=ON
cmake --build build-ros2 -j"$(nproc)"
```

The executable used by both accepted policies is:

```text
build/src/envs/go2/variants/blind_nolinvel_nokinref/deploy_cpp/deploy_blind_nolinvel_nokinref
```

## 4. Dry-run the exact launch command

Check the command without sending DDS motor commands:

```bash
PATH="$PWD/.conda/bin:$PATH" ./.conda/bin/python \
  -m src.deploy.accepted_go2_deploy run shac \
  --interface eth0 --command-source terminal --dry-run

PATH="$PWD/.conda/bin:$PATH" ./.conda/bin/python \
  -m src.deploy.accepted_go2_deploy run jave \
  --interface eth0 --command-source terminal --dry-run
```

Replace `eth0` with the real Go2 network interface from `ip -br link`.

## 5. Launch on the real Go2

Before running, keep the physical emergency stop ready, clear the area around
the robot, start with zero commands, and test small speed increments first.

Run SHAC:

```bash
PATH="$PWD/.conda/bin:$PATH" ./.conda/bin/python \
  -m src.deploy.accepted_go2_deploy run shac \
  --interface eth0 --command-source terminal
```

Run JAVE:

```bash
PATH="$PWD/.conda/bin:$PATH" ./.conda/bin/python \
  -m src.deploy.accepted_go2_deploy run jave \
  --interface eth0 --command-source terminal
```

If the executable has not been built yet, add `--build-if-missing` to let the
launcher run CMake first.

Direct C++ commands are also supported:

```bash
./build/src/envs/go2/variants/blind_nolinvel_nokinref/deploy_cpp/deploy_blind_nolinvel_nokinref \
  --policy training_runs/shac_20260820_002033/policy_best_tracking_deploy.npz \
  --interface eth0 \
  --command-source terminal

./build/src/envs/go2/variants/blind_nolinvel_nokinref/deploy_cpp/deploy_blind_nolinvel_nokinref \
  --policy training_runs/jave_20260820_151021/policy_best_tracking_deploy.npz \
  --interface eth0 \
  --command-source terminal
```

## 6. Operate the Go2 with terminal control

The C++ controller starts in `IDLE` with zero torque. Terminal input is
line-based: type the key and press Enter, or press Enter on an empty line.

| Input | Effect |
| --- | --- |
| empty Enter from `IDLE` | stand up to the policy default pose |
| empty Enter from `READY` | enter `WALKING` with zero velocity command |
| `w` + Enter | increase forward `vx` by `+0.1 m/s` |
| `s` + Enter | decrease `vx` by `-0.1 m/s` for backward walking |
| `a` + Enter | increase lateral `vy` left by `+0.1 m/s` |
| `d` + Enter | decrease lateral `vy` right by `-0.1 m/s` |
| `q` + Enter | increase yaw rate left by `+0.1 rad/s` |
| `e` + Enter | decrease yaw rate right by `-0.1 rad/s` |
| `0` + Enter | zero all velocity commands |
| `x` + Enter | emergency stop and hold current joints |
| `Ctrl-C` | graceful sit-down and exit |

Recommended first real-robot sequence:

```text
Enter      # IDLE -> STANDUP -> READY
Enter      # READY -> WALKING, command is still zero
w Enter    # vx = +0.1 m/s
0 Enter    # zero command
s Enter    # vx = -0.1 m/s
0 Enter    # zero command
Ctrl-C     # sit down and exit
```

Only increase commands after the robot is stable. For this policy, stay within
the exported command ranges: `vx +/-1.5`, `vy +/-1.0`, `yaw +/-1.5`.

## 7. Optional wireless or ROS2 control

Built-in wireless command source reads the Unitree remote from
`LowState.wireless_remote` while the same keyboard state machine handles
stand-up and walking transitions:

```bash
PATH="$PWD/.conda/bin:$PATH" ./.conda/bin/python \
  -m src.deploy.accepted_go2_deploy run shac \
  --interface eth0 --command-source wireless
```

Wireless stick mapping:

- Left stick Y: forward/backward `vx`.
- Left stick X: lateral `vy`.
- Right stick X: yaw rate.
- Release sticks to return command toward zero.

For ROS2 command-topic control, build with ROS2 and run:

```bash
PATH="$PWD/.conda/bin:$PATH" ./.conda/bin/python \
  -m src.deploy.accepted_go2_deploy run shac \
  --interface eth0 --command-source ros2 --cmd-topic /velocity_command
```

In another terminal, publish commands with the helper controller:

```bash
PATH="$PWD/.conda/bin:$PATH" ./.conda/bin/python -m src.deploy.terminal_command --control diffloco
```

The Python terminal helper uses arrow keys for `vx/vy`, `a/d` for yaw, Space to
zero commands, and `q` to quit. The Python wireless helper is available for
ROS2 command publishing:

```bash
PATH="$PWD/.conda/bin:$PATH" ./.conda/bin/python \
  -m src.deploy.wireless_command --net eth0 --control diffloco --topic /velocity_command
```

## 8. Safety and troubleshooting

- If launch prints `No state after 10s`, verify Go2 is powered on, the network
  interface is correct, and DDS domain ID matches the robot setup.
- If Unitree sport mode is active, the controller attempts to release it before
  low-level control. Do not run another motion controller at the same time.
- If the robot oscillates during stand-up, stop with physical e-stop or `x`, then
  inspect joint order, network delay, and PD gain overrides.
- If command response is clipped, verify the `.npz` was exported from the chosen
  checkpoint and that the C++ startup log prints the expected command ranges.
- Use `--kp` and `--kd` only for controlled troubleshooting; the default comes
  from the exported policy actuator metadata.
