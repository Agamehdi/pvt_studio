param(
    [int]$PreferredPort = 8501
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

function Test-PythonCandidate {
    param(
        [string]$Executable,
        [string[]]$PrefixArguments = @()
    )
    try {
        $arguments = @($PrefixArguments) + @(
            "-c",
            "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"
        )
        & $Executable @arguments 2>$null | Out-Null
        return ($LASTEXITCODE -eq 0)
    }
    catch {
        return $false
    }
}

function Find-Python {
    $launcher = Get-Command "py.exe" -ErrorAction SilentlyContinue
    if ($null -ne $launcher -and (Test-PythonCandidate $launcher.Source @("-3"))) {
        return [PSCustomObject]@{ Executable = $launcher.Source; Prefix = @("-3") }
    }

    $pathPython = Get-Command "python.exe" -ErrorAction SilentlyContinue
    if ($null -ne $pathPython -and (Test-PythonCandidate $pathPython.Source)) {
        return [PSCustomObject]@{ Executable = $pathPython.Source; Prefix = @() }
    }

    $candidates = @(
        (Join-Path $env:USERPROFILE "anaconda3\python.exe"),
        (Join-Path $env:USERPROFILE "miniconda3\python.exe"),
        (Join-Path $env:LOCALAPPDATA "anaconda3\python.exe"),
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python313\python.exe"),
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"),
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python311\python.exe"),
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python310\python.exe"),
        (Join-Path $env:ProgramData "Anaconda3\python.exe"),
        (Join-Path $env:ProgramData "Miniconda3\python.exe")
    ) | Select-Object -Unique

    foreach ($candidate in $candidates) {
        if ((Test-Path -LiteralPath $candidate) -and (Test-PythonCandidate $candidate)) {
            return [PSCustomObject]@{ Executable = $candidate; Prefix = @() }
        }
    }
    return $null
}

function Test-PortAvailable {
    param([int]$Port)
    $listener = $null
    try {
        $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $Port)
        $listener.Start()
        return $true
    }
    catch {
        return $false
    }
    finally {
        if ($null -ne $listener) {
            $listener.Stop()
        }
    }
}

try {
    Write-Host ""
    Write-Host "PVT Studio launcher" -ForegroundColor Cyan
    Write-Host "Project: $PSScriptRoot"

    if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot "app.py"))) {
        throw "app.py is missing. Extract the complete ZIP archive before starting the app."
    }
    if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot "requirements.txt"))) {
        throw "requirements.txt is missing. Extract the complete ZIP archive before starting the app."
    }

    $venvDirectory = Join-Path $PSScriptRoot ".venv"
    $venvPython = Join-Path $venvDirectory "Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $venvPython)) {
        $basePython = Find-Python
        if ($null -eq $basePython) {
            throw "Python 3.10 or newer was not found. Install Python, Anaconda, or Miniconda and run this launcher again."
        }
        Write-Host "Creating the local Python environment..." -ForegroundColor Yellow
        $venvArguments = @($basePython.Prefix) + @("-m", "venv", $venvDirectory)
        & $basePython.Executable @venvArguments
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $venvPython)) {
            throw "Python was found, but the local environment could not be created."
        }
    }

    & $venvPython -c "import streamlit, numpy, pandas, plotly" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Installing PVT Studio packages (first run only)..." -ForegroundColor Yellow
        & $venvPython -m pip install --disable-pip-version-check -r (Join-Path $PSScriptRoot "requirements.txt")
        if ($LASTEXITCODE -ne 0) {
            throw "Package installation failed. Check the internet/proxy connection and try again."
        }
    }

    $port = $null
    foreach ($candidatePort in $PreferredPort..($PreferredPort + 20)) {
        if (Test-PortAvailable $candidatePort) {
            $port = $candidatePort
            break
        }
    }
    if ($null -eq $port) {
        throw "No free local port was found between $PreferredPort and $($PreferredPort + 20)."
    }

    $logDirectory = Join-Path $PSScriptRoot "logs"
    New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $stdoutLog = Join-Path $logDirectory "streamlit_$stamp.log"
    $stderrLog = Join-Path $logDirectory "streamlit_$($stamp)_error.log"
    $streamlitArguments = @(
        "-m", "streamlit", "run", "app.py",
        "--server.address", "127.0.0.1",
        "--server.port", "$port",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false"
    )

    Write-Host "Starting Streamlit on http://127.0.0.1:$port ..." -ForegroundColor Yellow
    $server = Start-Process `
        -FilePath $venvPython `
        -ArgumentList $streamlitArguments `
        -WorkingDirectory $PSScriptRoot `
        -RedirectStandardOutput $stdoutLog `
        -RedirectStandardError $stderrLog `
        -WindowStyle Minimized `
        -PassThru

    $healthUrl = "http://127.0.0.1:$port/_stcore/health"
    $appUrl = "http://127.0.0.1:$port"
    $healthy = $false
    for ($attempt = 0; $attempt -lt 120; $attempt++) {
        if ($server.HasExited) {
            break
        }
        try {
            $response = Invoke-WebRequest -Uri $healthUrl -UseBasicParsing -TimeoutSec 2
            if ($response.StatusCode -eq 200) {
                $healthy = $true
                break
            }
        }
        catch {
            Start-Sleep -Milliseconds 500
        }
    }

    if (-not $healthy) {
        if (-not $server.HasExited) {
            Stop-Process -Id $server.Id -Force
        }
        $details = ""
        if (Test-Path -LiteralPath $stderrLog) {
            $details = (Get-Content -LiteralPath $stderrLog -Tail 25) -join [Environment]::NewLine
        }
        throw "Streamlit did not become healthy within 60 seconds.`n$details`nLogs: $logDirectory"
    }

    Start-Process $appUrl
    Write-Host "PVT Studio is running and the browser has been opened." -ForegroundColor Green
    Write-Host "URL: $appUrl"
    Write-Host "Logs: $logDirectory"
    exit 0
}
catch {
    Write-Host ""
    Write-Host "STARTUP ERROR" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}
