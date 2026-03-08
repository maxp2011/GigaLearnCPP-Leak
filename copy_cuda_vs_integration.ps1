# Run this script ONCE as Administrator so Visual Studio can find CUDA 13.0 when building.
# Without this, CMake fails with: "The imported project CUDA 13.0.props was not found."
#
# Usage: Right-click PowerShell -> Run as Administrator, then:
#   cd "C:\Users\Maxph\Downloads\GigaLearnCPP-Leak"
#   .\copy_cuda_vs_integration.ps1

$cudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
$src = "$cudaPath\extras\visual_studio_integration\MSBuildExtensions"
$dst = "D:\vs\MSBuild\Microsoft\VC\v180\BuildCustomizations"

if (-not (Test-Path $src)) {
    Write-Error "CUDA VS integration not found at $src"
    exit 1
}
if (-not (Test-Path $dst)) {
    Write-Error "VS BuildCustomizations not found at $dst (adjust path if your VS is elsewhere)"
    exit 1
}

$files = @(
    "CUDA 13.0.props",
    "CUDA 13.0.targets",
    "CUDA 13.0.Version.props",
    "CUDA 13.0.xml",
    "Nvda.Build.CudaTasks.v13.0.dll"
)
foreach ($f in $files) {
    Copy-Item "$src\$f" "$dst\$f" -Force
    Write-Host "Copied $f"
}
Write-Host "Done. You can now run build.ps1 to compile GigaLearn with CUDA."
