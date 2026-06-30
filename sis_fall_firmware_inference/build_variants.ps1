<#
.SYNOPSIS
    Build 2 ban firmware test: ESP-NN ON (sdkconfig.on) va ESP-NN OFF (sdkconfig.off)
    vao 2 build dir rieng -> doi qua lai chi can flash, khong recompile.

.DESCRIPTION
    Da tich hop fix cho cac loi hay gap khi build ESP-IDF tren Windows:
      - Build dir ngan (build_on/build_off) de ne loi long-path (>250 ky tu).
      - Kill tien trinh build con sot (ninja/gcc/ccache) de ne loi file-lock
        "The process cannot access the file because it is being used by another process".
      - Tro dung file SDKCONFIG (tranh go nham ten).

    YEU CAU: chay trong moi truong ESP-IDF (ESP-IDF PowerShell, hoac da chay export.ps1)
    de lenh `idf.py` co tren PATH.

.PARAMETER Clean
    Xoa build dir truoc khi build (build lai tu dau). Mac dinh: build incremental.

.PARAMETER NoCcache
    Tat ccache (IDF_CCACHE_ENABLE=0) cho lan chay nay - dung khi van dinh loi file-lock.

.PARAMETER Only
    Chi build 1 ban: 'on' hoac 'off'. Mac dinh build ca hai.

.PARAMETER Flash
    Sau khi build xong, flash 1 ban: 'on' hoac 'off' (khong flash ca hai cung luc).

.PARAMETER Port
    Cong COM de flash (vd COM5). De trong thi idf.py tu tim.

.EXAMPLE
    .\build_variants.ps1                      # build ca ON va OFF
    .\build_variants.ps1 -Clean               # xoa build dir roi build lai ca hai
    .\build_variants.ps1 -Only on -Flash on   # build ESP-NN ON roi flash luon
    .\build_variants.ps1 -NoCcache            # build ca hai, tat ccache (khi dinh file-lock)
#>

[CmdletBinding()]
param(
    [switch]$Clean,
    [switch]$NoCcache,
    [ValidateSet('on', 'off', 'both')][string]$Only = 'both',
    [ValidateSet('on', 'off', 'none')][string]$Flash = 'none',
    [string]$Port = ''
)

$ErrorActionPreference = 'Stop'

# --- Thu muc project = noi chua script nay ---
$ProjectDir = $PSScriptRoot
Set-Location $ProjectDir
Write-Host "[*] Project: $ProjectDir" -ForegroundColor Cyan

# --- Kiem tra idf.py co san khong ---
if (-not (Get-Command idf.py -ErrorAction SilentlyContinue)) {
    Write-Host "[LOI] Khong tim thay 'idf.py' tren PATH." -ForegroundColor Red
    Write-Host "      Hay mo 'ESP-IDF PowerShell' hoac chay export.ps1 truoc khi chay script nay." -ForegroundColor Yellow
    exit 1
}

# --- Dinh nghia 2 bien the (build dir NGAN de ne long-path) ---
$variants = @(
    [pscustomobject]@{ Name = 'ON';  Key = 'on';  BuildDir = 'build_on';  Sdkconfig = 'sdkconfig.on' }
    [pscustomobject]@{ Name = 'OFF'; Key = 'off'; BuildDir = 'build_off'; Sdkconfig = 'sdkconfig.off' }
)
if ($Only -ne 'both') {
    $variants = $variants | Where-Object { $_.Key -eq $Only }
}

# --- Kiem tra file sdkconfig.on/.off ton tai ---
foreach ($v in $variants) {
    if (-not (Test-Path $v.Sdkconfig)) {
        Write-Host "[LOI] Khong thay file $($v.Sdkconfig). Tao no truoc (cp sdkconfig $($v.Sdkconfig) roi sua phan ESP-NN)." -ForegroundColor Red
        exit 1
    }
}

# --- Tat ccache neu yeu cau ---
if ($NoCcache) {
    $env:IDF_CCACHE_ENABLE = '0'
    Write-Host "[*] Da tat ccache (IDF_CCACHE_ENABLE=0) cho lan chay nay." -ForegroundColor Yellow
}

# --- Kill tien trinh build con sot de ne file-lock ---
function Stop-StrayBuildProcesses {
    $names = 'ninja', 'ccache', 'xtensa-esp32s3-elf-gcc', 'cc1', 'cc1plus', 'xtensa-esp32s3-elf-g++'
    $procs = Get-Process -Name $names -ErrorAction SilentlyContinue
    if ($procs) {
        Write-Host "[*] Kill $($procs.Count) tien trinh build con sot (ne file-lock)..." -ForegroundColor Yellow
        $procs | Stop-Process -Force -ErrorAction SilentlyContinue
        Start-Sleep -Milliseconds 500
    }
}

# --- Build tung bien the ---
$results = @()
foreach ($v in $variants) {
    Write-Host ""
    Write-Host ("=" * 60) -ForegroundColor Green
    Write-Host "[BUILD] ESP-NN $($v.Name)  ->  -B $($v.BuildDir)  -D SDKCONFIG=$($v.Sdkconfig)" -ForegroundColor Green
    Write-Host ("=" * 60) -ForegroundColor Green

    Stop-StrayBuildProcesses

    if ($Clean -and (Test-Path $v.BuildDir)) {
        Write-Host "[*] Xoa $($v.BuildDir) (--Clean)..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force $v.BuildDir -ErrorAction SilentlyContinue
    }

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & idf.py -B $v.BuildDir -D "SDKCONFIG=$($v.Sdkconfig)" build
    $code = $LASTEXITCODE
    $sw.Stop()

    $ok = ($code -eq 0)
    $results += [pscustomobject]@{
        Variant  = "ESP-NN $($v.Name)"
        BuildDir = $v.BuildDir
        Status   = if ($ok) { 'OK' } else { "FAILED (exit $code)" }
        Minutes  = [math]::Round($sw.Elapsed.TotalMinutes, 1)
    }
    if (-not $ok) {
        Write-Host "[LOI] Build ESP-NN $($v.Name) that bai (exit $code)." -ForegroundColor Red
        Write-Host "      Neu loi 'file used by another process': chay lai voi -NoCcache, hoac dong VS Code/monitor." -ForegroundColor Yellow
    }
}

# --- Tom tat ---
Write-Host ""
Write-Host ("=" * 60) -ForegroundColor Cyan
Write-Host "TOM TAT BUILD" -ForegroundColor Cyan
Write-Host ("=" * 60) -ForegroundColor Cyan
$results | Format-Table -AutoSize

$anyFail = $results | Where-Object { $_.Status -ne 'OK' }
if ($anyFail) {
    Write-Host "[!] Co ban build that bai - xem log o tren." -ForegroundColor Red
    exit 1
}

# --- Flash neu yeu cau ---
if ($Flash -ne 'none') {
    $fv = $variants | Where-Object { $_.Key -eq $Flash } | Select-Object -First 1
    if (-not $fv) {
        Write-Host "[!] Khong build ban '$Flash' nen khong flash duoc (kiem tra -Only)." -ForegroundColor Yellow
    }
    else {
        Write-Host ""
        Write-Host "[FLASH] ESP-NN $($fv.Name) (build dir $($fv.BuildDir))..." -ForegroundColor Green
        Stop-StrayBuildProcesses
        $flashArgs = @('-B', $fv.BuildDir, '-D', "SDKCONFIG=$($fv.Sdkconfig)")
        if ($Port) { $flashArgs += @('-p', $Port) }
        $flashArgs += 'flash'
        & idf.py @flashArgs
        if ($LASTEXITCODE -eq 0) {
            Write-Host ""
            Write-Host "[OK] Da flash ESP-NN $($fv.Name). Buoc tiep theo (dong terminal nay truoc khi chay tool python):" -ForegroundColor Cyan
            $mn = "v30_opt_espnn_$($fv.Name)"
            Write-Host "     python ..\firmware_test_tool\test_inference_uart.py --pipeline v30 --model-name $mn -n 200" -ForegroundColor Gray
        }
    }
}

Write-Host ""
Write-Host "[XONG] Doi qua lai giua 2 ban CHI can flash, khong build lai:" -ForegroundColor Cyan
Write-Host "       idf.py -B build_on  -D SDKCONFIG=sdkconfig.on  flash monitor" -ForegroundColor Gray
Write-Host "       idf.py -B build_off -D SDKCONFIG=sdkconfig.off flash monitor" -ForegroundColor Gray
