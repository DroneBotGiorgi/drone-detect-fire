$ErrorActionPreference = "Stop"

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $projectRoot

New-Item -ItemType Directory -Force -Path "tools", "tmp" | Out-Null

Write-Host "[1/3] Download Android platform-tools (adb)..."
$platformToolsZip = Join-Path $projectRoot "tmp/platform-tools-latest-windows.zip"
Invoke-WebRequest -Uri "https://dl.google.com/android/repository/platform-tools-latest-windows.zip" -OutFile $platformToolsZip
if (Test-Path "tools/platform-tools") {
    Remove-Item -Recurse -Force "tools/platform-tools"
}
Expand-Archive -Path $platformToolsZip -DestinationPath "tools" -Force

Write-Host "[2/3] Download FFmpeg (essentials build)..."
$ffmpegZip = Join-Path $projectRoot "tmp/ffmpeg-release-essentials.zip"
Invoke-WebRequest -Uri "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip" -OutFile $ffmpegZip
if (Test-Path "tools/ffmpeg") {
    Remove-Item -Recurse -Force "tools/ffmpeg"
}
Expand-Archive -Path $ffmpegZip -DestinationPath "tmp" -Force
$ffmpegExtracted = Get-ChildItem "tmp" -Directory |
    Where-Object { $_.Name -like "ffmpeg-*essentials_build" } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if (-not $ffmpegExtracted) {
    throw "FFmpeg archive extracted folder not found."
}
Move-Item -Path $ffmpegExtracted.FullName -Destination "tools/ffmpeg"

Write-Host "[3/3] Download scrcpy..."
$scrcpyVersion = "v3.3.1"
$scrcpyZip = Join-Path $projectRoot "tmp/scrcpy-win64-$scrcpyVersion.zip"
Invoke-WebRequest -Uri "https://github.com/Genymobile/scrcpy/releases/download/$scrcpyVersion/scrcpy-win64-$scrcpyVersion.zip" -OutFile $scrcpyZip
if (Test-Path "tools/scrcpy") {
    Remove-Item -Recurse -Force "tools/scrcpy"
}
Expand-Archive -Path $scrcpyZip -DestinationPath "tools/scrcpy" -Force

Write-Host "Done. Local tools installed under tools/."
Write-Host "adb: tools/platform-tools/adb.exe"
Write-Host "ffmpeg: tools/ffmpeg/bin/ffmpeg.exe"
Write-Host "scrcpy: tools/scrcpy/scrcpy-win64-v3.3.1/scrcpy.exe"
