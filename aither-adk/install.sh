#!/usr/bin/env bash
# AitherADK Universal Installer
# Usage:
#   curl -fsSL https://aitherium.com/install.sh | sh
#   curl -fsSL https://aitherium.com/install.sh | sh -s -- --full
#   curl -fsSL https://aitherium.com/install.sh | sh -s -- --with-openclaw

set -euo pipefail

BOLD="\033[1m"
CYAN="\033[96m"
GREEN="\033[92m"
YELLOW="\033[93m"
RED="\033[91m"
RESET="\033[0m"

VERSION="0.9.0"
ADK_PACKAGE="aither-adk"
FULL_STACK=false
WITH_OPENCLAW=false
NONINTERACTIVE=false

# ── Parse args ────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --full)        FULL_STACK=true; shift ;;
        --with-openclaw) WITH_OPENCLAW=true; shift ;;
        --non-interactive|-y) NONINTERACTIVE=true; shift ;;
        --help|-h)
            echo "AitherADK Installer"
            echo ""
            echo "Usage: curl -fsSL https://aitherium.com/install.sh | sh [-- OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --full            Install ADK + local AitherOS stack"
            echo "  --with-openclaw   Auto-integrate with OpenClaw if detected"
            echo "  --non-interactive No prompts (auto-accept defaults)"
            echo "  --help            Show this help"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo ""
echo -e "${BOLD}${CYAN}AitherADK Installer v${VERSION}${RESET}"
echo -e "${CYAN}=============================${RESET}"
echo ""

# ── Check Python ──────────────────────────────────────────────
PYTHON=""
for candidate in python3 python; do
    if command -v "$candidate" &>/dev/null; then
        py_version=$("$candidate" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
        major=$(echo "$py_version" | cut -d. -f1)
        minor=$(echo "$py_version" | cut -d. -f2)
        if [[ "$major" -ge 3 ]] && [[ "$minor" -ge 10 ]]; then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [[ -z "$PYTHON" ]]; then
    echo -e "${RED}Python 3.10+ required but not found.${RESET}"
    echo ""
    echo "Install Python:"
    echo "  macOS:  brew install python@3.12"
    echo "  Ubuntu: sudo apt install python3.12"
    echo "  Fedora: sudo dnf install python3.12"
    echo "  Other:  https://python.org/downloads/"
    exit 1
fi

echo -e "${GREEN}[OK]${RESET} Python: $($PYTHON --version)"

# ── Check pipx/pip ────────────────────────────────────────────
INSTALLER=""
if command -v pipx &>/dev/null; then
    INSTALLER="pipx"
    echo -e "${GREEN}[OK]${RESET} pipx available (recommended)"
elif command -v pip &>/dev/null; then
    INSTALLER="pip"
    echo -e "${YELLOW}[OK]${RESET} pip available (pipx recommended: pip install pipx)"
elif command -v pip3 &>/dev/null; then
    INSTALLER="pip3"
    echo -e "${YELLOW}[OK]${RESET} pip3 available"
else
    echo -e "${RED}No pip/pipx found. Install pip first:${RESET}"
    echo "  $PYTHON -m ensurepip --upgrade"
    exit 1
fi

# ── Install ADK ───────────────────────────────────────────────
echo ""
echo -e "${BOLD}Installing AitherADK...${RESET}"

if [[ "$INSTALLER" == "pipx" ]]; then
    pipx install "$ADK_PACKAGE" || pipx upgrade "$ADK_PACKAGE"
else
    "$INSTALLER" install --upgrade "$ADK_PACKAGE"
fi

# Verify installation
if command -v aither &>/dev/null; then
    echo -e "${GREEN}[OK]${RESET} AitherADK installed: $(which aither)"
else
    echo -e "${YELLOW}[!!]${RESET} 'aither' command not in PATH"
    echo "  Add to PATH: export PATH=\"\$HOME/.local/bin:\$PATH\""
fi

# ── Detect environment ────────────────────────────────────────
echo ""
echo -e "${BOLD}Detecting environment...${RESET}"

# GPU
if command -v nvidia-smi &>/dev/null; then
    GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "")
    if [[ -n "$GPU" ]]; then
        echo -e "${GREEN}[OK]${RESET} GPU: $GPU"
    fi
fi

# Ollama
if command -v ollama &>/dev/null; then
    echo -e "${GREEN}[OK]${RESET} Ollama installed"
else
    echo -e "${YELLOW}[--]${RESET} Ollama not found (optional: https://ollama.com)"
fi

# OpenClaw
OPENCLAW_DIR="$HOME/.openclaw"
if [[ -d "$OPENCLAW_DIR" ]]; then
    echo -e "${GREEN}[OK]${RESET} OpenClaw detected at $OPENCLAW_DIR"

    # Check if already integrated
    INTEGRATED=false
    if [[ -f "$OPENCLAW_DIR/openclaw.json" ]]; then
        if grep -qi "aither" "$OPENCLAW_DIR/openclaw.json" 2>/dev/null; then
            INTEGRATED=true
            echo -e "${GREEN}[OK]${RESET} OpenClaw already integrated with AitherOS"
        fi
    fi

    if [[ "$INTEGRATED" == "false" ]]; then
        if [[ "$WITH_OPENCLAW" == "true" ]] || [[ "$NONINTERACTIVE" == "false" ]]; then
            echo ""
            echo -e "${CYAN}OpenClaw detected! Connect to AitherOS agent fleet?${RESET}"
            echo "  This gives OpenClaw access to 29 agents, 100+ tools, swarm coding."
            echo ""

            INTEGRATE="y"
            if [[ "$NONINTERACTIVE" == "false" ]] && [[ "$WITH_OPENCLAW" != "true" ]]; then
                read -r -p "  Integrate? [Y/n] " INTEGRATE
                INTEGRATE=${INTEGRATE:-y}
            fi

            if [[ "${INTEGRATE,,}" == "y" ]]; then
                echo ""
                aither integrate openclaw 2>/dev/null || echo "  Run 'aither integrate openclaw' after setup"
            fi
        fi
    fi
fi

# ── Full stack install ────────────────────────────────────────
if [[ "$FULL_STACK" == "true" ]]; then
    echo ""
    echo -e "${BOLD}Installing full AitherOS stack...${RESET}"

    if command -v docker &>/dev/null; then
        echo -e "${GREEN}[OK]${RESET} Docker available"
        # Download and run docker-compose
        echo "  Pulling AitherOS containers..."
        echo "  docker compose -f docker-compose.aitheros.yml up -d"
        echo ""
        echo -e "${YELLOW}[!!]${RESET} Full stack install requires the AitherOS repo."
        echo "  git clone https://github.com/Aitherium/aither && cd aither"
        echo "  docker compose -f docker-compose.aitheros.yml up -d"
    else
        echo -e "${YELLOW}[!!]${RESET} Docker not found. Install Docker first:"
        echo "  https://docs.docker.com/get-docker/"
    fi
fi

# ── Summary ───────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}Installation complete!${RESET}"
echo ""
echo "  Quick start:"
echo "    aither register          # Create account (free)"
echo "    aither init my-agent     # Scaffold a project"
echo "    cd my-agent"
echo "    aither run               # Start your agent"
echo ""
echo "  More commands:"
echo "    aither connect           # Connect to cloud + detect backends"
echo "    aither onboard           # Full interactive setup wizard"
echo "    aither integrate         # Connect external tools"
echo "    aither publish           # Submit to Elysium marketplace"
echo "    aither aeon              # Multi-agent group chat"
echo ""
echo "  Docs: https://github.com/Aitherium/aither#readme"
echo ""
