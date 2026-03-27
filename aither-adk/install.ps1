# AitherADK Universal Installer (Windows PowerShell)
# Usage:
#   irm https://aitherium.com/install.ps1 | iex
#   irm https://aitherium.com/install.ps1 | iex -- -Full
#   irm https://aitherium.com/install.ps1 | iex -- -WithOpenClaw

param(
    [switch]$Full,
    [switch]$WithOpenClaw,
    [switch]$NonInteractive,
    [switch]$Help
)

$ErrorActionPreference = "Stop"
$Version = "0.9.0"
$AdkPackage = "aither-adk"

if ($Help) {
    Write-Host "AitherADK Installer"
    Write-Host ""
    Write-Host "Usage: irm https://aitherium.com/install.ps1 | iex"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -Full             Install ADK + local AitherOS stack"
    Write-Host "  -WithOpenClaw     Auto-integrate with OpenClaw if detected"
    Write-Host "  -NonInteractive   No prompts (auto-accept defaults)"
    exit 0
}

Write-Host ""
Write-Host "AitherADK Installer v$Version" -ForegroundColor Cyan
Write-Host "=============================" -ForegroundColor Cyan
Write-Host ""

# ── Check Python ──────────────────────────────────────────────
$Python = $null
foreach ($candidate in @("python", "python3", "py")) {
    try {
        $ver = & $candidate -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        if ($ver) {
            $parts = $ver.Split(".")
            if ([int]$parts[0] -ge 3 -and [int]$parts[1] -ge 10) {
                $Python = $candidate
                break
            }
        }
    } catch { }
}

if (-not $Python) {
    Write-Host "[!!] Python 3.10+ required but not found." -ForegroundColor Red
    Write-Host ""
    Write-Host "Install Python:"
    Write-Host "  winget install Python.Python.3.12"
    Write-Host "  Or: https://python.org/downloads/"
    exit 1
}

$pyVer = & $Python --version
Write-Host "[OK] $pyVer" -ForegroundColor Green

# ── Check pip/pipx ────────────────────────────────────────────
$Installer = $null
if (Get-Command pipx -ErrorAction SilentlyContinue) {
    $Installer = "pipx"
    Write-Host "[OK] pipx available (recommended)" -ForegroundColor Green
} elseif (Get-Command pip -ErrorAction SilentlyContinue) {
    $Installer = "pip"
    Write-Host "[OK] pip available (pipx recommended: pip install pipx)" -ForegroundColor Yellow
} else {
    Write-Host "[!!] No pip/pipx found." -ForegroundColor Red
    Write-Host "  Run: $Python -m ensurepip --upgrade"
    exit 1
}

# ── Install ADK ───────────────────────────────────────────────
Write-Host ""
Write-Host "Installing AitherADK..." -ForegroundColor White

if ($Installer -eq "pipx") {
    try {
        pipx install $AdkPackage
    } catch {
        pipx upgrade $AdkPackage
    }
} else {
    & $Installer install --upgrade $AdkPackage
}

# Verify
if (Get-Command aither -ErrorAction SilentlyContinue) {
    Write-Host "[OK] AitherADK installed" -ForegroundColor Green
} else {
    Write-Host "[!!] 'aither' not in PATH. Restart your terminal." -ForegroundColor Yellow
}

# ── Detect environment ────────────────────────────────────────
Write-Host ""
Write-Host "Detecting environment..." -ForegroundColor White

# GPU
try {
    $gpu = & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1
    if ($gpu) {
        Write-Host "[OK] GPU: $gpu" -ForegroundColor Green
    }
} catch { }

# Ollama
if (Get-Command ollama -ErrorAction SilentlyContinue) {
    Write-Host "[OK] Ollama installed" -ForegroundColor Green
} else {
    Write-Host "[--] Ollama not found (optional: https://ollama.com)" -ForegroundColor Yellow
}

# OpenClaw
$OpenClawDir = Join-Path $HOME ".openclaw"
if (Test-Path $OpenClawDir) {
    Write-Host "[OK] OpenClaw detected at $OpenClawDir" -ForegroundColor Green

    $Integrated = $false
    $ocConfigPath = Join-Path $OpenClawDir "openclaw.json"
    if (Test-Path $ocConfigPath) {
        $content = Get-Content $ocConfigPath -Raw
        if ($content -match "aither") {
            $Integrated = $true
            Write-Host "[OK] OpenClaw already integrated with AitherOS" -ForegroundColor Green
        }
    }

    if (-not $Integrated) {
        Write-Host ""
        Write-Host "OpenClaw detected! Connect to AitherOS agent fleet?" -ForegroundColor Cyan
        Write-Host "  This gives OpenClaw access to 29 agents, 100+ tools, swarm coding."
        Write-Host ""

        $integrate = "y"
        if (-not $NonInteractive -and -not $WithOpenClaw) {
            $integrate = Read-Host "  Integrate? [Y/n]"
            if (-not $integrate) { $integrate = "y" }
        }

        if ($integrate.ToLower() -eq "y") {
            Write-Host ""
            try {
                & aither integrate openclaw
            } catch {
                Write-Host "  Run 'aither integrate openclaw' after setup" -ForegroundColor Yellow
            }
        }
    }
}

# ── Full stack install ────────────────────────────────────────
if ($Full) {
    Write-Host ""
    Write-Host "Installing full AitherOS stack..." -ForegroundColor White

    if (Get-Command docker -ErrorAction SilentlyContinue) {
        Write-Host "[OK] Docker available" -ForegroundColor Green
        Write-Host "  Full stack install requires the AitherOS repo."
        Write-Host "  git clone https://github.com/Aitherium/aither && cd aither"
        Write-Host "  docker compose -f docker-compose.aitheros.yml up -d"
    } elseif (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Host "[!!] Docker not found. Installing..." -ForegroundColor Yellow
        Write-Host "  winget install Docker.DockerDesktop"
    } else {
        Write-Host "[!!] Docker not found. Install from: https://docs.docker.com/get-docker/"
    }
}

# ── Summary ───────────────────────────────────────────────────
Write-Host ""
Write-Host "Installation complete!" -ForegroundColor Green
Write-Host ""
Write-Host "  Quick start:"
Write-Host "    aither register          # Create account (free)"
Write-Host "    aither init my-agent     # Scaffold a project"
Write-Host "    cd my-agent"
Write-Host "    aither run               # Start your agent"
Write-Host ""
Write-Host "  More commands:"
Write-Host "    aither connect           # Connect to cloud + detect backends"
Write-Host "    aither onboard           # Full interactive setup wizard"
Write-Host "    aither integrate         # Connect external tools"
Write-Host "    aither publish           # Submit to Elysium marketplace"
Write-Host "    aither aeon              # Multi-agent group chat"
Write-Host ""
Write-Host "  Docs: https://github.com/Aitherium/aither#readme"
Write-Host ""
