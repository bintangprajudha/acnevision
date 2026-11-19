# AcneVision - jalankan server API (scripts/serve.py) + Cloudflare quick tunnel (HTTPS).
# WebView Capacitor memakai skema https, jadi aplikasi HP WAJIB memakai URL https dari tunnel
# (HTTP lewat LAN akan diblokir sebagai mixed content).
#
# Jalankan dari mana saja:
#   powershell -ExecutionPolicy Bypass -File scripts\run_server.ps1              # server + tunnel
#   powershell -ExecutionPolicy Bypass -File scripts\run_server.ps1 -NoTunnel    # server lokal saja
# Berhenti: Ctrl+C (server dan tunnel ikut dihentikan).
# Log: runs\serve\server.log (stdout) + runs\serve\server.err.log (log aplikasi/uvicorn), runs\serve\cloudflared.log
# Prasyarat tunnel: cloudflared  ->  winget install --id Cloudflare.cloudflared
param([int]$Port = 8000, [switch]$NoTunnel)
$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
$weights = Join-Path $root "runs\detect\yolov8l_960_tiles2\weights\best.pt"
$logDir = Join-Path $root "runs\serve"
New-Item -ItemType Directory -Force $logDir | Out-Null

if (-not (Test-Path $py)) { Write-Host "Python venv tidak ditemukan: $py" -ForegroundColor Red; exit 1 }
if (-not (Test-Path $weights)) { Write-Host "Bobot E9 tidak ditemukan: $weights" -ForegroundColor Red; exit 1 }
if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    Write-Host "Port $Port sudah dipakai proses lain. Hentikan dulu atau pakai -Port." -ForegroundColor Red; exit 1
}

$cf = $null
if (-not $NoTunnel) {
    $cmd = Get-Command cloudflared -ErrorAction SilentlyContinue
    if ($cmd) { $cf = $cmd.Source }
    else {
        $cand = @("$env:ProgramFiles\cloudflared\cloudflared.exe", "${env:ProgramFiles(x86)}\cloudflared\cloudflared.exe",
                  "$env:LOCALAPPDATA\Microsoft\WinGet\Links\cloudflared.exe")
        $cf = $cand | Where-Object { Test-Path $_ } | Select-Object -First 1
    }
    if (-not $cf) {
        Write-Host "cloudflared belum terpasang. Pasang dengan:" -ForegroundColor Red
        Write-Host "    winget install --id Cloudflare.cloudflared" -ForegroundColor Yellow
        Write-Host "lalu buka terminal baru. Atau jalankan dengan -NoTunnel untuk server lokal saja." -ForegroundColor Red
        exit 1
    }
}

# API key TETAP untuk endpoint internal /predict (header X-API-Key), disimpan di .env.server (di-ignore git).
# Dibuat otomatis sekali; berikan ke tim backend lewat jalur privat. Untuk mengganti key: hapus file lalu jalankan ulang.
$secretFile = Join-Path $root ".env.server"
if (-not $env:ACNEVISION_API_KEY) {
    if (-not (Test-Path $secretFile)) {
        $bytes = New-Object byte[] 32
        [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
        $newKey = [Convert]::ToBase64String($bytes).TrimEnd("=").Replace("+", "-").Replace("/", "_")
        @("# Rahasia server model AcneVision - JANGAN di-commit / dibagikan di grup.",
          "# Key endpoint internal /predict (header X-API-Key). Berikan ke tim backend lewat jalur privat.",
          "ACNEVISION_API_KEY=$newKey") | Set-Content -Path $secretFile -Encoding ascii
        Write-Host "API key tetap baru dibuat di $secretFile" -ForegroundColor Yellow
    }
    foreach ($line in Get-Content $secretFile) {
        if ($line -match '^\s*([A-Z_]+)\s*=\s*(.+?)\s*$') { Set-Item -Path "env:$($Matches[1])" -Value $Matches[2] }
    }
}

$server = $null; $tunnel = $null
try {
    Write-Host "[$(Get-Date -Format HH:mm:ss)] Memulai server (model E9 + warm-up)..." -ForegroundColor Cyan
    $server = Start-Process -FilePath $py -ArgumentList @("scripts\serve.py", "--port", "$Port") -NoNewWindow -PassThru `
        -RedirectStandardOutput (Join-Path $logDir "server.log") -RedirectStandardError (Join-Path $logDir "server.err.log")
    $health = $null
    for ($i = 0; $i -lt 180 -and -not $health; $i++) {
        if ($server.HasExited) {
            Write-Host "Server berhenti saat startup. Log terakhir:" -ForegroundColor Red
            Get-Content (Join-Path $logDir "server.err.log") -Tail 20; exit 1
        }
        try { $health = Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 2 } catch { Start-Sleep -Seconds 1 }
    }
    if (-not $health) { Write-Host "Server tidak merespons /health dalam 180 detik." -ForegroundColor Red; exit 1 }
    Write-Host "[$(Get-Date -Format HH:mm:ss)] Server siap: $($health.model.name) sha256 $($health.model.weights_sha256), GPU $($health.gpu.device)" -ForegroundColor Green
    Write-Host "  Lokal : http://127.0.0.1:$Port/api/v1  (health: /health)"
    Write-Host "  API key /predict: tetap, tersimpan di .env.server (buka file itu untuk menyalinnya)"

    if (-not $NoTunnel) {
        Write-Host "[$(Get-Date -Format HH:mm:ss)] Membuka Cloudflare quick tunnel..." -ForegroundColor Cyan
        $cfLog = Join-Path $logDir "cloudflared.log"
        $url = $null
        # Quick tunnel kadang ditolak Cloudflare ("provisioning failed with status 429"): coba ulang dengan jeda.
        foreach ($wait in @(0, 15, 30, 60)) {
            if ($wait -gt 0) { Write-Host "  Mencoba lagi dalam $wait detik..." -ForegroundColor Yellow; Start-Sleep -Seconds $wait }
            if (Test-Path $cfLog) { Remove-Item $cfLog }
            $tunnel = Start-Process -FilePath $cf -ArgumentList @("tunnel", "--url", "http://127.0.0.1:$Port", "--no-autoupdate") `
                -NoNewWindow -PassThru -RedirectStandardOutput (Join-Path $logDir "cloudflared.out.log") -RedirectStandardError $cfLog
            for ($i = 0; $i -lt 60 -and -not $url -and -not $tunnel.HasExited; $i++) {
                Start-Sleep -Seconds 1
                $m = Select-String -Path $cfLog -Pattern "https://[a-z0-9-]+\.trycloudflare\.com" -ErrorAction SilentlyContinue |
                    Select-Object -First 1
                if ($m) { $url = $m.Matches[0].Value }
            }
            if ($url) { break }
            $last = Get-Content $cfLog -Tail 1 -ErrorAction SilentlyContinue
            Write-Host "  cloudflared gagal: $last" -ForegroundColor Red
            if (-not $tunnel.HasExited) { Stop-Process -Id $tunnel.Id -Force -ErrorAction SilentlyContinue }
            $tunnel = $null
        }
        if (-not $url) {
            Write-Host "Tunnel tidak berhasil dibuat. Bila status 429: Cloudflare membatasi quick tunnel dari IP ini;" -ForegroundColor Red
            Write-Host "tunggu beberapa menit lalu ulangi, atau pakai named tunnel (docs/API.md). Log: $cfLog" -ForegroundColor Red
            exit 1
        }
        Write-Host ""
        Write-Host "Isi .env aplikasi mobile (lalu build ulang / restart Vite):" -ForegroundColor Green
        Write-Host "    VITE_USE_MOCK=false" -ForegroundColor Yellow
        Write-Host "    VITE_API_BASE_URL=$url/api/v1" -ForegroundColor Yellow
        Write-Host "  Cek dari HP/browser: $url/health"
        Write-Host "  URL quick tunnel BERUBAH setiap kali skrip ini dijalankan (lihat docs/API.md untuk named tunnel)."
    }
    Write-Host ""
    Write-Host "Berjalan. Tekan Ctrl+C untuk berhenti." -ForegroundColor Cyan
    while (-not $server.HasExited) {
        Start-Sleep -Seconds 2
        if ($tunnel -and $tunnel.HasExited) { Write-Host "Peringatan: cloudflared berhenti (lihat log)." -ForegroundColor Red; $tunnel = $null }
    }
    Write-Host "Server berhenti (exit $($server.ExitCode)). Lihat runs\serve\server.err.log" -ForegroundColor Red
}
finally {
    foreach ($p in @($tunnel, $server)) {
        if ($p -and -not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
    }
    Write-Host "[$(Get-Date -Format HH:mm:ss)] Server dan tunnel dihentikan." -ForegroundColor Cyan
}
