# AcneVision - Evaluasi ulang semua varian agar metrics.json memuat mAP@50-95 internal
# (map50_95_test_internal; berlaku juga untuk SAHI yang tidak punya mAP Ultralytics).
# Jalankan dari root proyek:  powershell -ExecutionPolicy Bypass -File scripts\rerun_evals.ps1
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

Step "eval_b0"      @("evaluate.py", "--weights", $b0)
Step "eval_e3"      @("evaluate.py", "--weights", $e3)
Step "eval_e3_tta"  @("evaluate.py", "--weights", $e3, "--mode", "tta")
Step "eval_e4"      @("evaluate.py", "--weights", $e4)
Step "eval_e3_sahi" @("evaluate.py", "--weights", $e3, "--mode", "sahi", "--slice", "1728")
Step "eval_e4_sahi" @("evaluate.py", "--weights", $e4, "--mode", "sahi", "--slice", "1728")
Step "eval_b0_sahi" @("evaluate.py", "--weights", $b0, "--mode", "sahi", "--slice", "1728")
Step "report"       @("report.py")
"[$(Get-Date -Format HH:mm:ss)] ALL DONE (rerun_evals)" | Add-Content $log
Write-Host "ALL DONE" -ForegroundColor Green
