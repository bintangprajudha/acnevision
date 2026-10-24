# AcneVision - Rantai evaluasi + fine-tune E4 (lihat docs/EXPERIMENTS.md)
# Jalankan dari root proyek:  powershell -ExecutionPolicy Bypass -File scripts\run_experiments.ps1
# Progres ringkas: runs\chain.log ; log tiap langkah: runs\chain_<langkah>.log
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$py = "..\.venv\Scripts\python.exe"
$log = "..\runs\chain.log"

function Step($name, [string[]]$a) {
    $msg = "[$(Get-Date -Format HH:mm:ss)] START $name"
    Write-Host $msg -ForegroundColor Cyan; $msg | Add-Content $log
    & $py @a 2>&1 | Tee-Object -FilePath "..\runs\chain_$name.log"
    $msg = "[$(Get-Date -Format HH:mm:ss)] END $name exit=$LASTEXITCODE"
    Write-Host $msg -ForegroundColor Cyan; $msg | Add-Content $log
}

$b0 = "..\runs\detect\yolov8l_960_default\weights\best.pt"
$e3 = "..\runs\detect\yolov8l_960\weights\best.pt"
$e4 = "..\runs\detect\yolov8l_960_tiles2_ft\weights\best.pt"
$cfg = "..\configs\yolov8l_960_tiles2_ft.yaml"

Step "eval_b0"      @("evaluate.py", "--weights", $b0)
Step "eval_e3"      @("evaluate.py", "--weights", $e3)
Step "eval_e3_tta"  @("evaluate.py", "--weights", $e3, "--mode", "tta")
Step "eval_e3_sahi" @("evaluate.py", "--weights", $e3, "--mode", "sahi", "--slice", "1728")
Step "eval_b0_sahi" @("evaluate.py", "--weights", $b0, "--mode", "sahi", "--slice", "1728")
Step "smoke_e4"     @("train.py", "--cfg", $cfg, "--epochs", "1", "--fraction", "0.02", "--project", "..\runs\smoke")
Step "train_e4"     @("train.py", "--cfg", $cfg)          # auto-resume dari last.pt bila terhenti
Step "eval_e4"      @("evaluate.py", "--weights", $e4)
Step "eval_e4_sahi" @("evaluate.py", "--weights", $e4, "--mode", "sahi", "--slice", "1728")
"[$(Get-Date -Format HH:mm:ss)] ALL DONE" | Add-Content $log
Write-Host "ALL DONE" -ForegroundColor Green
