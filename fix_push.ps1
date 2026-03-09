# Fix push: remove large zip from commit, add to gitignore, push
# Run from project root: .\fix_push.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# 1. Undo last commit (keep changes)
git reset --soft HEAD~1

# 2. Unstage the large zip (wherever it is)
git reset HEAD -- "*.zip" 2>$null
git reset HEAD -- "63711562314.zip" 2>$null
git reset HEAD -- "**/63711562314.zip" 2>$null

# 3. Add *.zip to .gitignore if not present
$gitignore = ".gitignore"
$content = Get-Content $gitignore -Raw -ErrorAction SilentlyContinue
if ($content -and $content -notmatch "^\*\.zip\s*$" -and $content -notmatch "`n\*\.zip\s*`n") {
    Add-Content $gitignore "`n# Large archives (GitHub 100MB limit)`n*.zip"
}

# 4. Stage everything except zips
git add -A
git reset HEAD -- "*.zip" 2>$null

# 5. Commit and push
git status
git commit -m "Fix LibTorch build (TMPDIR, ninja -j4, USE_FLASH_ATTENTION=OFF)"
git push origin main
Write-Host "Done."
