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
| AHAC | `training_runs/ahac_20260828_180341` | `policy_best_tracking.pkl` | `policy_best_tracking_deploy.npz` |

All three policies use the `blind_nolinvel_nokinref` Go2 variant, 50 Hz policy
inference, 500 Hz low-level motor commands, and command ranges exported in the
`.npz`: `vx=[-1.5, 1.5] m/s`, `vy=[-1.0, 1.0] m/s`, `yaw=[-1.5, 1.5] rad/s`.

## Exporting a trained policy

The C++ deployment executables consume a `.npz` file, not the pickled
training checkpoints. Export a checkpoint with:

```bash
python -m src.deploy.export_policy training_runs/<run>/policy_best.pkl
```

This writes `training_runs/<run>/policy_best_deploy.npz` containing the actor
weights, observation-normalizer statistics, actuator gains, and environment
metadata. The environment variant is read from the run's `hparams.json`, so
variant-specific data (e.g. the kinematic gait reference for
`blind_linvel_kinref`) is included automatically.

The current C++ deployment reads command ranges from the `.npz`. Legacy exports
without range metadata fall back to `vx=+/-1.5`, `vy=+/-1.0`, `yaw=+/-1.5`.

Export the accepted AHAC policy:

```bash
cd /home/a/文档/diffloco/open-diffloco-test
PATH="$PWD/.conda/bin:$PATH" XLA_PYTHON_CLIENT_PREALLOCATE=false \
  ./.conda/bin/python -m src.deploy.accepted_go2_deploy export ahac
```

## Building the C++ deployment

C++ deployment code is co-located with each Go2 variant and can be built from the repository root.

Build without ROS2:

```bash
cmake -S . -B build -DOPEN_DIFFLOCO_ENABLE_ROS2=OFF
cmake --build build
```

Build with ROS2 command-message support:

```bash
cmake -S . -B build-ros2 -DOPEN_DIFFLOCO_ENABLE_ROS2=ON
cmake --build build-ros2
```

The `OPEN_DIFFLOCO_ENABLE_ROS2` option enables ROS2 velocity-command input.
Terminal and wireless command sources are available in the ROS2-free build.

Command sources:

- `terminal`: keyboard command input inside the C++ deployment executable.
- `wireless`: command input from `LowState.wireless_remote`.
- `ros2`: command input from the configured ROS2 velocity command topic,
  available only with `OPEN_DIFFLOCO_ENABLE_ROS2=ON`.

Velocity command publishers are available under `src/deploy/`:

```bash
python -m src.deploy.terminal_command --control diffloco
python -m src.deploy.wireless_command --net lo --control diffloco --topic /velocity_command
```

They publish command messages for the C++ deployment `ros2` command source.

## AHAC quick launch

Build without ROS2 for terminal or wireless control:

```bash
cd /home/a/文档/diffloco/open-diffloco-test
cmake -S . -B build -DOPEN_DIFFLOCO_ENABLE_ROS2=OFF
cmake --build build -j"$(nproc)"
```

Dry-run the real-robot command before sending motor commands:

```bash
PATH="$PWD/.conda/bin:$PATH" ./.conda/bin/python \
  -m src.deploy.accepted_go2_deploy run ahac \
  --interface eth0 --command-source terminal --dry-run
```

Launch AHAC on the real Go2 after replacing `eth0` with the actual DDS network
interface:

```bash
PATH="$PWD/.conda/bin:$PATH" ./.conda/bin/python \
  -m src.deploy.accepted_go2_deploy run ahac \
  --interface eth0 --command-source terminal
```

Terminal control is line-based: press Enter once to stand up, press Enter again
to enter walking, then use `w/s` for `vx`, `a/d` for `vy`, `q/e` for yaw, `0` to
zero commands, `x` for estop, and `Ctrl-C` for graceful sit-down and exit.
