@echo off
REM Run this ONCE as Administrator (right-click -> Run as administrator)
REM So Visual Studio can find CUDA 13.0 when building GigaLearn.

set "SRC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\extras\visual_studio_integration\MSBuildExtensions"
set "DST=D:\vs\MSBuild\Microsoft\VC\v180\BuildCustomizations"

if not exist "%SRC%\CUDA 13.0.props" (
    echo CUDA VS integration not found at %SRC%
    pause
    exit /b 1
)
if not exist "%DST%" (
    echo VS BuildCustomizations not found at %DST%
    echo Edit this .cmd file if your Visual Studio is installed elsewhere.
    pause
    exit /b 1
)

copy /Y "%SRC%\CUDA 13.0.props" "%DST%\"
copy /Y "%SRC%\CUDA 13.0.targets" "%DST%\"
copy /Y "%SRC%\CUDA 13.0.Version.props" "%DST%\"
copy /Y "%SRC%\CUDA 13.0.xml" "%DST%\"
copy /Y "%SRC%\Nvda.Build.CudaTasks.v13.0.dll" "%DST%\"

echo Done. You can now run build.cmd to compile GigaLearn.
pause
