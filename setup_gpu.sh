#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------
# CONFIG (EDIT AS NEEDED)
# ---------------------------------------------
REPO_URL="https://github.com/raxgvv/GeoNLI.git"
REPO_NAME="GeoNLI"
# Set your own git identity before running this script:
# GIT_USER="Your Name"
# GIT_EMAIL="you@example.com"

CONDA_PREFIX="$HOME/miniconda"
CONDA_ENV="py310"
NVM_VERSION="v0.39.7"
NODE_VERSION="20"
# ---------------------------------------------

log() { echo -e "\n[SETUP] $*\n"; }

# ---------------------------------------------
# 1. UPDATE + BASICS
# ---------------------------------------------
log "Updating apt and installing essentials..."
apt update -y
apt install -y curl wget git build-essential ca-certificates unzip

# ---------------------------------------------
# 2. GIT CONFIG
# ---------------------------------------------
log "Configuring git..."
if [ -n "${GIT_USER:-}" ] && [ -n "${GIT_EMAIL:-}" ]; then
    git config --global user.name "$GIT_USER"
    git config --global user.email "$GIT_EMAIL"
else
    log "GIT_USER/GIT_EMAIL not set; using your existing global git identity."
fi

# ---------------------------------------------
# 3. INSTALL MINICONDA
# ---------------------------------------------
if [ ! -d "$CONDA_PREFIX" ]; then
    log "Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$CONDA_PREFIX"
    rm /tmp/miniconda.sh
else
    log "Miniconda already exists at $CONDA_PREFIX"
fi

# load conda
eval "$($CONDA_PREFIX/bin/conda shell.bash hook)"
conda init

# ---------------------------------------------
# 3.1 Accept Anaconda Terms (fix ToS error)
# ---------------------------------------------
log "Accepting Anaconda Terms of Service..."
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

# ---------------------------------------------
# 4. CREATE CONDA ENV
# ---------------------------------------------
if conda env list | grep -q "^$CONDA_ENV "; then
    log "Conda env $CONDA_ENV already exists."
else
    log "Creating conda env $CONDA_ENV..."
    conda create -y -n "$CONDA_ENV" python=3.10
fi

# ---------------------------------------------
# 5. INSTALL NVM + NODE
# ---------------------------------------------
if [ ! -d "$HOME/.nvm" ]; then
    log "Installing nvm..."
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/$NVM_VERSION/install.sh | bash
fi

export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

log "Installing Node $NODE_VERSION..."
nvm install "$NODE_VERSION"
nvm alias default "$NODE_VERSION"

log "Node: $(node -v), npm: $(npm -v)"

# ---------------------------------------------
# 6. CLONE REPO (if not already present)
# ---------------------------------------------
if [ ! -d "$HOME/$REPO_NAME" ] && [ ! -d "/data/$REPO_NAME" ]; then
    log "Cloning repo from $REPO_URL..."
    git clone "$REPO_URL" "$HOME/$REPO_NAME"
else
    log "Repo already exists. Skipping git clone."
fi

# Determine project path
if [ -d "$HOME/$REPO_NAME" ]; then PROJECT_DIR="$HOME/$REPO_NAME";
elif [ -d "/data/$REPO_NAME" ]; then PROJECT_DIR="/data/$REPO_NAME";
else
    echo "ERROR: Repo not found"; exit 1;
fi
log "Using PROJECT_DIR = $PROJECT_DIR"

# ---------------------------------------------
# 7. FRONTEND INSTALL
# ---------------------------------------------
FRONTEND_DIR="$PROJECT_DIR/ui/frontend"
if [ -d "$FRONTEND_DIR" ]; then
    log "Installing frontend dependencies..."
    cd "$FRONTEND_DIR"
    if [ -f package-lock.json ]; then npm ci; else npm install; fi
else
    log "Frontend directory not found! ($FRONTEND_DIR)"
fi

# ---------------------------------------------
# 8. BACKEND INSTALL
# ---------------------------------------------
BACKEND_DIR="$PROJECT_DIR/backend"
conda activate "$CONDA_ENV"

if [ -f "$BACKEND_DIR/requirements.txt" ]; then
    log "Installing backend Python deps..."
    python -m pip install --upgrade pip
    pip install -r "$BACKEND_DIR/requirements.txt"
else
    log "Backend requirements.txt not found!"
fi

conda deactivate

# ---------------------------------------------
# 9. GENERATE HELPER RUN SCRIPTS
# ---------------------------------------------
SCRIPTS_DIR="$PROJECT_DIR/_instance_scripts"
mkdir -p "$SCRIPTS_DIR"

# ---- frontend runner ----
cat > "$SCRIPTS_DIR/start_frontend.sh" <<'EOF'
#!/usr/bin/env bash
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
nvm use default

FRONTEND="$HOME/Inter-IIT/ui/frontend"
[ -d "/data/Inter-IIT/ui/frontend" ] && FRONTEND="/data/Inter-IIT/ui/frontend"

cd "$FRONTEND"
npm run dev -- --host 0.0.0.0 --port 3000
EOF
chmod +x "$SCRIPTS_DIR/start_frontend.sh"

# ---- backend runner ----
cat > "$SCRIPTS_DIR/start_backend.sh" <<'EOF'
#!/usr/bin/env bash
CONDA_PREFIX="$HOME/miniconda"
eval "$($CONDA_PREFIX/bin/conda shell.bash hook)"
conda activate py310

PROJECT_DIR="$HOME/Inter-IIT"
[ -d "/data/Inter-IIT" ] && PROJECT_DIR="/data/Inter-IIT"
cd "$PROJECT_DIR"
python -m backend.app
EOF
chmod +x "$SCRIPTS_DIR/start_backend.sh"

log "Created start scripts in: $SCRIPTS_DIR"

# ---------------------------------------------
# DONE
# ---------------------------------------------
echo -e "\n🎉 SETUP COMPLETE 🎉

To start the frontend (Terminal 1):
  cd <project-dir>/_instance_scripts
  ./start_frontend.sh

To start the backend (Terminal 2):
  cd <project-dir>/_instance_scripts
  ./start_backend.sh

Project directory: $PROJECT_DIR
"
