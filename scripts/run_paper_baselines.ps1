# AcneVision - pembanding terkontrol untuk paper (Tingkat 1; jalankan SETELAH B1 selesai).
#   e12    : evaluasi TEST YOLOv8l @ 1280 (E12, bobot sudah ada; setup mirip AcneAI)   ±10 menit
#   rtdetr : B2 RT-DETR-l @ 960 (smoke -> training -> evaluasi plain)                  ±6-12 jam (perkiraan)
#   seeds  : E9 seed 7 dan 123 (training -> evaluasi plain + SAHI 1728)                ±11 jam PER seed
# Jalankan dari root proyek, mis.:
#   powershell -ExecutionPolicy Bypass -File scripts\run_paper_baselines.ps1 -Parts e12,rtdetr
#   powershell -ExecutionPolicy Bypass -File scripts\run_paper_baselines.ps1 -Parts seeds
# Progres ringkas: runs\chain.log ; log tiap langkah: runs\chain_<langkah>.log
param([string[]]$Parts = @("e12", "rtdetr"))
$ErrorActionPreference = "Continue"
# `powershell -File` meneruskan "e12,rtdetr" sebagai SATU string (bukan array) -> pecah manual.
$Parts = @($Parts | ForEach-Object { $_ -split "," } | ForEach-Object { $_.Trim().ToLower() } | Where-Object { $_ })
$unknown = @($Parts | Where-Object { $_ -notin @("e12", "rtdetr", "seeds") })
if ($unknown.Count -gt 0) { Write-Host "Bagian tidak dikenal: $($unknown -join ', ') (pilihan: e12, rtdetr, seeds)" -ForegroundColor Red; exit 1 }
Write-Host "Bagian yang dijalankan: $($Parts -join ', ')" -ForegroundColor Cyan
Set-Location $PSScriptRoot
$py = "..\.venv\Scripts\python.exe"
$log = "..\runs\chain.log"
$smoke = Join-Path (Resolve-Path "..\runs").Path "smoke"

function Step($name, [string[]]$a) {
    $msg = "[$(Get-Date -Format HH:mm:ss)] START $name"
    Write-Host $msg -ForegroundColor Cyan; $msg | Add-Content $log
    & $py @a 2>&1 | Tee-Object -FilePath "..\runs\chain_$name.log"
    $code = $LASTEXITCODE
    $msg = "[$(Get-Date -Format HH:mm:ss)] END $name exit=$code"
    Write-Host $msg -ForegroundColor Cyan; $msg | Add-Content $log
    if ($code -ne 0) { Write-Host "GAGAL di langkah $name, berhenti." -ForegroundColor Red; exit $code }
}

function Guard($w) {
    if (Test-Path $w) { Write-Host "Bobot sudah ada ($w); hapus/rename dulu bila ingin melatih ulang." -ForegroundColor Red; exit 1 }
}

if ($Parts -contains "e12") {
    # E12 sebelumnya hanya dinilai di val (screening). Untuk paper: satu kali evaluasi test, protokol standar.
    Step "eval_e12_test" @("evaluate.py", "--weights", "..\runs\detect\yolov8l_1280\weights\best.pt")
}

if ($Parts -contains "rtdetr") {
    $cfg = "..\configs\rtdetr_l_960.yaml"
    $w = "..\runs\detect\rtdetr_l_960\weights\best.pt"
    Guard $w
    Step "b2_smoke" @("train.py", "--cfg", $cfg, "--epochs", "1", "--fraction", "0.02", "--project", $smoke, "--name", "b2_smoke")
    Step "b2_train" @("train.py", "--cfg", $cfg)
    Step "eval_b2"  @("evaluate.py", "--weights", $w)
}

if ($Parts -contains "seeds") {
    foreach ($s in @(7, 123)) {
        $cfg = "..\configs\yolov8l_960_tiles2_s$s.yaml"
        $w = "..\runs\detect\yolov8l_960_tiles2_s$s\weights\best.pt"
        Guard $w
        Step "e9s${s}_train"     @("train.py", "--cfg", $cfg)
        Step "eval_e9s$s"        @("evaluate.py", "--weights", $w)
        Step "eval_e9s${s}_sahi" @("evaluate.py", "--weights", $w, "--mode", "sahi", "--slice", "1728")
    }
}
"[$(Get-Date -Format HH:mm:ss)] ALL DONE (run_paper_baselines: $($Parts -join ','))" | Add-Content $log
Write-Host "ALL DONE" -ForegroundColor Green
