# AcneVision - Tahap 1 (screening ide paper, HANYA train + val; test tidak disentuh):
#   E10 head P2 (Wen 2022) | E11 noise + kecerahan/kontras (Zhang 2024) | E12 imgsz 1280 (AcneAI 2024)
# Ketiga smoke test dijalankan lebih dulu agar OOM (P2 / 1280) ketahuan dalam ±10 menit.
# Lalu training E11 -> E12 -> E10, kemudian evaluate.py --val-only (plain + SAHI 1728) untuk E3, E10, E11, E12.
# Jalankan dari root proyek:  powershell -ExecutionPolicy Bypass -File scripts\run_stage1.ps1
# Lewati smoke test (bila sudah lolos):  ... -File scripts\run_stage1.ps1 -SkipSmoke
# Progres ringkas: runs\chain.log ; log tiap langkah: runs\chain_<langkah>.log
# Estimasi (RTX 4060): E11 ±2,5 jam, E12 ±4-4,5 jam, E10 ±3,5-4,5 jam, eval val ±30 menit.
param([switch]$SkipSmoke)
$ErrorActionPreference = "Continue"
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

# Training dilewati bila bobot sudah ada (tidak menimpa), sehingga script aman dijalankan ulang
function Train($id, $cfg, $run) {
    if (Test-Path "..\runs\detect\$run\weights\best.pt") { Write-Host "Lewati $id ($run sudah ada)" -ForegroundColor Yellow; return }
    Step "${id}_train" @("train.py", "--cfg", $cfg)
}

function EvalVal($id, $run) {
    $w = "..\runs\detect\$run\weights\best.pt"
    Step "val_$id"      @("evaluate.py", "--weights", $w, "--val-only")
    Step "val_${id}_sahi" @("evaluate.py", "--weights", $w, "--val-only", "--mode", "sahi", "--slice", "1728")
}

$exp = @(
    @("e11", "..\configs\yolov8l_960_augnoise.yaml", "yolov8l_960_augnoise"),
    @("e12", "..\configs\yolov8l_1280.yaml", "yolov8l_1280"),
    @("e10", "..\configs\yolov8l_960_p2.yaml", "yolov8l_960_p2")
)

if (-not $SkipSmoke) {
    foreach ($e in $exp) {
        Step "$($e[0])_smoke" @("train.py", "--cfg", $e[1], "--epochs", "1", "--fraction", "0.02", "--project", $smoke, "--name", "$($e[0])_smoke")
    }
}
foreach ($e in $exp) { Train $e[0] $e[1] $e[2] }

# Screening di val: baseline E3 + tiga ide
EvalVal "e3" "yolov8l_960"
foreach ($e in $exp) { EvalVal $e[0] $e[2] }
Step "bootstrap_val" @("bootstrap.py", "--split", "val")
"[$(Get-Date -Format HH:mm:ss)] ALL DONE (run_stage1)" | Add-Content $log
Write-Host "ALL DONE" -ForegroundColor Green
