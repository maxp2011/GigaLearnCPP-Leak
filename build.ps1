# Build GigaLearn with CUDA (LibTorch GPU). Uses Visual Studio and CUDA 13.0.
# Optional env: LIBTORCH_PATH, CUDA_ROOT, VCVARS - or defaults below.

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

$libtorch = if ($env:LIBTORCH_PATH) { $env:LIBTORCH_PATH }
            elseif (Test-Path "$root\libtorch") { "$root\libtorch" }
            elseif (Test-Path "$root\GigaLearnCPP\libtorch") { "$root\GigaLearnCPP\libtorch" }
            else { "C:\libtorch" }

$cudaRoot = if ($env:CUDA_ROOT) { $env:CUDA_ROOT }
            else { "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0" }

$vcvars = if ($env:VCVARS) { $env:VCVARS }
          else { "D:\vs\VC\Auxiliary\Build\vcvars64.bat" }

if (-not (Test-Path $libtorch)) {
    Write-Error "LibTorch not found at $libtorch. Set LIBTORCH_PATH or put libtorch in $root\libtorch"
    exit 1
}
if (-not (Test-Path $vcvars)) {
    Write-Error "vcvars64.bat not found at $vcvars. Set VCVARS env or adjust path."
    exit 1
}

# Clean and configure with CUDA toolset
Remove-Item -Recurse -Force "$root\build" -ErrorAction SilentlyContinue

$cmakeArgs = @(
    "-B", "build",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_PREFIX_PATH=$libtorch",
    "-DCUDA_TOOLKIT_ROOT_DIR=$cudaRoot",
    "-T", "host=x64,cuda=13.0"
)
if (Test-Path "$cudaRoot\extras\visual_studio_integration\MSBuildExtensions\CUDA 13.0.props") {
    $cmakeArgs += "-DCMAKE_VS_PLATFORM_TOOLSET_CUDA_CUSTOM_DIR=$cudaRoot"
}

Write-Host "Configuring (CUDA)..."
& cmd /c "call `"$vcvars`" && cd /d `"$root`" && cmake $($cmakeArgs -join ' ')"
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "If you see 'CUDA 13.0.props was not found', run copy_cuda_vs_integration.ps1 as Administrator once."
    exit $LASTEXITCODE
}

Write-Host "Building..."
& cmd /c "call `"$vcvars`" && cd /d `"$root`" && cmake --build build --config Release"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Build succeeded. Output: $root\build\Release\"
