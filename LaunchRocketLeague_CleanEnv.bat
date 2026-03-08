@echo off
REM Launch Rocket League with a clean environment (no LibTorch/CUDA/Python in PATH).
REM Use this if RL crashes due to DLL conflicts from GigaLearnCPP dev setup.

REM Clear dev-related env vars that can cause DLL conflicts
set "CUDA_PATH="
set "CUDNN_PATH="
set "LIBTORCH_PATH="

REM Minimal PATH - only Windows essentials (no LibTorch/CUDA/Python)
set "PATH=C:\Windows\System32;C:\Windows;C:\Windows\System32\Wbem;C:\Windows\System32\WindowsPowerShell\v1.0"

REM Epic Games default install path - change if your RL is elsewhere
set "RL_PATH=C:\Program Files\Epic Games\rocketleague\Binaries\Win64\RocketLeague.exe"

if not exist "%RL_PATH%" (
    echo Rocket League not found at: %RL_PATH%
    echo Edit this .bat file and set RL_PATH to your install location.
    pause
    exit /b 1
)

start "" "%RL_PATH%"
