@echo off
REM Build GigaLearn with CUDA. Copies CUDA VS integration if missing (needs Admin once).
REM Paths: LIBTORCH_PATH, CUDA_ROOT, VCVARS - or defaults below.

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

REM LibTorch: env var, or ./libtorch, or C:\libtorch
if defined LIBTORCH_PATH (
    set "LIBTORCH=%LIBTORCH_PATH%"
) else if exist "%ROOT%\libtorch" (
    set "LIBTORCH=%ROOT%\libtorch"
) else if exist "%ROOT%\GigaLearnCPP\libtorch" (
    set "LIBTORCH=%ROOT%\GigaLearnCPP\libtorch"
) else (
    set "LIBTORCH=C:\libtorch"
)

REM CUDA: env var or default
if defined CUDA_ROOT (
    set "CUDA_ROOT_CMAKE=%CUDA_ROOT:\=/%"
) else (
    set "CUDA_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
    set "CUDA_ROOT_CMAKE=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0"
)

REM VS: env var or default (D:\vs = Visual Studio install)
if defined VCVARS (
    set "VCVARS_PATH=%VCVARS%"
) else (
    set "VCVARS_PATH=D:\vs\VC\Auxiliary\Build\vcvars64.bat"
)
set "VS_BUILD_CUSTOM=D:\vs\MSBuild\Microsoft\VC\v180\BuildCustomizations"
set "CUDA_VS_SRC=%CUDA_ROOT%\extras\visual_studio_integration\MSBuildExtensions"

REM Ensure CUDA 13.0 VS integration is in VS BuildCustomizations (required for CUDA build)
if not exist "%VS_BUILD_CUSTOM%\CUDA 13.0.props" (
    echo CUDA 13.0.props not found in VS. Copying from CUDA installer.
    if not exist "%CUDA_VS_SRC%\CUDA 13.0.props" (
        echo Error: CUDA VS integration not found at %CUDA_VS_SRC%
        echo Install CUDA Toolkit or fix CUDA_ROOT in this script.
        pause
        exit /b 1
    )
    copy /Y "%CUDA_VS_SRC%\CUDA 13.0.props" "%VS_BUILD_CUSTOM%\" 2>nul
    copy /Y "%CUDA_VS_SRC%\CUDA 13.0.targets" "%VS_BUILD_CUSTOM%\" 2>nul
    copy /Y "%CUDA_VS_SRC%\CUDA 13.0.Version.props" "%VS_BUILD_CUSTOM%\" 2>nul
    copy /Y "%CUDA_VS_SRC%\CUDA 13.0.xml" "%VS_BUILD_CUSTOM%\" 2>nul
    copy /Y "%CUDA_VS_SRC%\Nvda.Build.CudaTasks.v13.0.dll" "%VS_BUILD_CUSTOM%\" 2>nul
    if not exist "%VS_BUILD_CUSTOM%\CUDA 13.0.props" (
        echo Copy failed - need Administrator. Right-click build.cmd -^> "Run as administrator"
        echo Or run copy_cuda_vs_integration.cmd as Administrator once.
        pause
        exit /b 1
    )
    echo CUDA VS integration copied.
)

if not exist "%LIBTORCH%" (
    echo LibTorch not found at %LIBTORCH%. Set LIBTORCH_PATH or put libtorch in %ROOT%\libtorch
    pause
    exit /b 1
)
if not exist "%VCVARS_PATH%" (
    echo vcvars64.bat not found at %VCVARS_PATH%. Set VCVARS env or edit this .cmd.
    pause
    exit /b 1
)

REM Keep build folder for incremental builds (no full rebuild each time)
if not exist "%ROOT%\build\CMakeCache.txt" (
    echo First-time configure
) else (
    echo Reusing build cache - incremental build
)

echo Configuring (CUDA)...
call "%VCVARS_PATH%"
cd /d "%ROOT%"
REM CUDA 13.0.targets reads CudaToolkitDir from CUDA_PATH_V13_0 (used by TryCompile during configure)
set "CUDA_PATH_V13_0=%CUDA_ROOT%"
REM Skip CUDA compiler test (uses VS 2026 which nvcc does not list as supported) and allow nvcc to use it anyway
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="%LIBTORCH%" -DCUDA_TOOLKIT_ROOT_DIR="%CUDA_ROOT_CMAKE%" -T host=x64,cuda=13.0 -DCMAKE_VS_PLATFORM_TOOLSET_CUDA_CUSTOM_DIR="%CUDA_ROOT_CMAKE%" -DCMAKE_CUDA_COMPILER_WORKS=1 -DCMAKE_CUDA_FLAGS="--allow-unsupported-compiler" -Wno-dev
if errorlevel 1 (
    echo.
    echo If CUDA 13.0.props was not found, run copy_cuda_vs_integration.cmd as Administrator once.
    pause
    exit /b 1
)

echo Building... (GigaLearnCPP.dll can take several minutes; lines below show progress)
REM /m = parallel projects, /v:n = normal verbosity. CudaToolkitDir needs trailing slash (else v13.0+bin=v13.0bin).
set "CUDA_DIR_MSBUILD=%CUDA_ROOT_CMAKE%/"
cmake --build build --config Release -- /m /v:n /p:CudaToolkitDir="%CUDA_DIR_MSBUILD%"
if errorlevel 1 (
    pause
    exit /b 1
)

echo Build succeeded.
pause
