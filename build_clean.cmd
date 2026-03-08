@echo off
REM Full clean + rebuild (use when you need a clean slate; normal builds use build.cmd for speed).
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
if exist "%ROOT%\build" (
    echo Removing build folder for full rebuild...
    rmdir /s /q "%ROOT%\build"
)
call "%~dp0build.cmd" %*
