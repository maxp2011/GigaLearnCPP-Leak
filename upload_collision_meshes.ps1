# Upload collision meshes to https://github.com/maxp2011/Collision_Meshes
# Run from PowerShell on your Windows machine (where the meshes exist)

$sourcePath = "C:\Users\Maxph\Downloads\rl sim vis (1)\rlsimviscpp\build\yes yes yesy\RocketSim-main\collision_meshes"
$repoUrl = "https://github.com/maxp2011/Collision_Meshes.git"
$workDir = "$env:TEMP\Collision_Meshes_upload"

if (-not (Test-Path $sourcePath)) {
    Write-Error "Source path not found: $sourcePath"
    exit 1
}

Write-Host "Creating work dir: $workDir"
Remove-Item -Recurse -Force $workDir -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path $workDir | Out-Null

Set-Location $workDir

# Clone or init repo
if (Test-Path ".git") { Remove-Item -Recurse -Force .git }
git clone $repoUrl .
if ($LASTEXITCODE -ne 0) { Write-Error "git clone failed"; exit 1 }

# Copy meshes (preserve soccar/hoops structure)
Write-Host "Copying collision meshes..."
Copy-Item -Path "$sourcePath\*" -Destination "." -Recurse -Force

# Add, commit, push
git add -A
$count = (git status --short | Measure-Object -Line).Lines
if ($count -eq 0) {
    Write-Host "Nothing to commit - meshes may already be up to date."
    exit 0
}
git commit -m "Add RocketSim collision meshes (soccar, hoops)"
git push origin main

Write-Host "Done! Meshes uploaded to $repoUrl"
Set-Location $PSScriptRoot
