$ErrorActionPreference = "Stop"

$repoRoot = "C:\Workspace\RacingSim\assetto_corsa_gym"
$pythonExe = "C:\Workspace\RacingSim\.venv-acgym\Scripts\python.exe"
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$watchDir = Join-Path $repoRoot ("outputs\live_watch_" + $timestamp)
$smokeLog = Join-Path $watchDir "smoke.log"
$trainLog = Join-Path $watchDir "train.log"
$statusFile = Join-Path $watchDir "status.txt"

New-Item -ItemType Directory -Force -Path $watchDir | Out-Null
Push-Location $repoRoot

function Write-Status([string]$message) {
    $line = ("[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $message)
    $line | Tee-Object -FilePath $statusFile -Append
}

function Test-TcpPort([string]$HostName, [int]$Port) {
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $async = $client.BeginConnect($HostName, $Port, $null, $null)
        $ok = $async.AsyncWaitHandle.WaitOne(1000, $false)
        if (-not $ok) {
            return $false
        }
        $client.EndConnect($async) | Out-Null
        return $true
    } catch {
        return $false
    } finally {
        $client.Close()
    }
}

Write-Status "Waiting for Assetto Corsa gym plugin on localhost:2347."
while (-not (Test-TcpPort -HostName "127.0.0.1" -Port 2347)) {
    Start-Sleep -Seconds 2
}

Write-Status "Plugin detected. Running smoke_random.py."
& $pythonExe -u "C:\Workspace\RacingSim\assetto_corsa_gym\smoke_random.py" *>&1 | Tee-Object -FilePath $smokeLog
if ($LASTEXITCODE -ne 0) {
    Write-Status "Smoke test failed with exit code $LASTEXITCODE."
    Pop-Location
    exit $LASTEXITCODE
}

Write-Status "Smoke test passed. Starting live SAC training."
& $pythonExe -u "C:\Workspace\RacingSim\assetto_corsa_gym\train.py" `
    "disable_wandb=True" `
    "Agent.num_steps=20000" `
    "Agent.memory_size=50000" `
    "Agent.offline_buffer_size=0" `
    "Agent.start_steps=1000" `
    "Agent.batch_size=64" `
    "AssettoCorsa.track=monza" `
    "AssettoCorsa.car=ks_mazda_miata" `
    *>&1 | Tee-Object -FilePath $trainLog

Write-Status "Training exited with code $LASTEXITCODE."
Pop-Location
exit $LASTEXITCODE
