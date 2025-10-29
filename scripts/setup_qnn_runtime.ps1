$ErrorActionPreference = 'Stop'

param(
    [string]$QnnSdkRoot = 'C:\Users\rawat\Downloads\v2.28.0.241029'
)

Write-Host "Using QNN SDK root:" $QnnSdkRoot
if (!(Test-Path $QnnSdkRoot)) { throw "QNN SDK root not found: $QnnSdkRoot" }

$projRoot = (Resolve-Path "$PSScriptRoot\..\").Path
$jniDir   = Join-Path $projRoot 'app\src\main\jniLibs\arm64-v8a'
$hexDir   = Join-Path $projRoot 'app\src\main\assets\hexagon-v79'
$ctxDir   = Join-Path $projRoot 'app\src\main\assets\models\clip\context_binaries'

New-Item -ItemType Directory -Force $jniDir | Out-Null
New-Item -ItemType Directory -Force $hexDir | Out-Null
New-Item -ItemType Directory -Force $ctxDir | Out-Null

# Copy core runtime shared libs
$runtime = @(
  'libQnnSystem.so','libQnnHtp.so','libQnnHtpV79Stub.so',
  'libQnnOsUtils.so','libcdsprpc.so','libadsp_default_listener.so'
)
foreach ($name in $runtime) {
  $file = Get-ChildItem -Path $QnnSdkRoot -Recurse -File -Filter $name -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($file) { Copy-Item $file.FullName $jniDir -Force; Write-Host "Copied" $name }
  else { Write-Warning "Missing $name in SDK" }
}

# Copy Hexagon v79 artifacts
$hexFiles = Get-ChildItem -Path $QnnSdkRoot -Recurse -File -ErrorAction SilentlyContinue |
  Where-Object { $_.Name -like 'libQnnHtpV79Skel.so' -or $_.Name -like 'libhexagon*.so' -or $_.Name -like '*.mbn' -or $_.Name -like '*.bin' -or $_.FullName -match 'v79' }
foreach ($f in $hexFiles) { Copy-Item $f.FullName $hexDir -Force }

# Copy a context.bin if present
$ctx = Get-ChildItem -Path $QnnSdkRoot,$env:USERPROFILE'\Downloads' -Recurse -File -Filter 'context.bin' -ErrorAction SilentlyContinue | Select-Object -First 1
if ($ctx) { Copy-Item $ctx.FullName (Join-Path $ctxDir 'context.bin') -Force; Write-Host "Copied context.bin" }
else { Write-Warning "No context.bin found; CLIP will run in mock mode until provided" }

Write-Host "--- Verify jniLibs/arm64-v8a ---"
Get-ChildItem $jniDir -File | Select-Object Name,Length | Format-Table -AutoSize

Write-Host "--- Verify assets/hexagon-v79 ---"
Get-ChildItem $hexDir -File | Select-Object Name,Length | Format-Table -AutoSize

Write-Host "--- Verify clip/context_binaries ---"
Get-ChildItem $ctxDir -File -ErrorAction SilentlyContinue | Select-Object Name,Length | Format-Table -AutoSize

Write-Host "Done. Now run: .\\gradlew.bat assembleDebug && .\\gradlew.bat installDebug"


