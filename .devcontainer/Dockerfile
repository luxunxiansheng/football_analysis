FROM rayproject/ray:latest-gpu

# Switch to root user for installation and setup
USER root

# Copy environment.yml (if found) to a temp location so we update the environment. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
COPY environment.yml* .devcontainer/noop.txt /tmp/conda-tmp/
RUN chmod -R 755 /tmp/conda-tmp && \
    if [ -f "/tmp/conda-tmp/environment.yml" ]; then umask 0002 && /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml; fi && \
    rm -rf /tmp/conda-tmp

# Add Chinese mirror sources for faster downloads - detect Ubuntu version from /etc/os-release
RUN UBUNTU_CODENAME=$(grep VERSION_CODENAME /etc/os-release | cut -d= -f2 | tr -d '"' || echo "jammy") && \
    echo "Using Ubuntu codename: ${UBUNTU_CODENAME}" && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ ${UBUNTU_CODENAME} main restricted universe multiverse" > /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ ${UBUNTU_CODENAME}-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ ${UBUNTU_CODENAME}-backports main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ ${UBUNTU_CODENAME}-security main restricted universe multiverse" >> /etc/apt/sources.list

# Install required system dependencies for matplotlib and other Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libfreetype6-dev \
    libpng-dev \
    libqhull-dev \
    libagg-dev \
    libcairo2-dev \
    libffi-dev \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Configure conda with Chinese mirror and use Python 3.9 - find conda installation first
RUN CONDA_PATH=$(which conda 2>/dev/null | xargs dirname | xargs dirname || echo "") && \
    if [ -n "$CONDA_PATH" ] && [ -d "$CONDA_PATH" ]; then \
        echo "Found conda at: $CONDA_PATH" && \
        conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main && \
        conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free && \
        conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge && \
        conda config --set show_channel_urls yes && \
        conda install -y python=3.9; \
    else \
        echo "Conda not found, installing Miniconda..." && \
        apt-get update && apt-get install -y wget && \
        wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
        bash /tmp/miniconda.sh -b -p /opt/conda && \
        rm /tmp/miniconda.sh && \
        /opt/conda/bin/conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main && \
        /opt/conda/bin/conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free && \
        /opt/conda/bin/conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge && \
        /opt/conda/bin/conda config --set show_channel_urls yes && \
        /opt/conda/bin/conda install -y python=3.9; \
    fi

# Try to install packages via conda first, then fallback to pip for unavailable packages
COPY requirements.txt* .devcontainer/noop.txt /tmp/pip-tmp/
RUN if [ -f "/tmp/pip-tmp/requirements.txt" ]; then \
        # Find conda and pip paths
        CONDA_CMD=$(which conda 2>/dev/null || echo "/opt/conda/bin/conda") && \
        PIP_CMD=$(which pip 2>/dev/null || echo "/opt/conda/bin/pip") && \
        # First try to install as many packages as possible with conda
        $CONDA_CMD install -y --file /tmp/pip-tmp/requirements.txt || true && \
        # Then install remaining packages with pip (conda-aware)
        $PIP_CMD install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple && \
        $PIP_CMD install --no-cache-dir -r /tmp/pip-tmp/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple; \
    fi \
    && rm -rf /tmp/pip-tmp

# Set up conda environment for development
RUN CONDA_PATH=$(which conda 2>/dev/null | xargs dirname | xargs dirname || echo "/opt/conda") && \
    if [ -d "$CONDA_PATH" ]; then \
        echo "Conda found at $CONDA_PATH" && \
        echo "export PATH=\"$CONDA_PATH/bin:\$PATH\"" >> /root/.bashrc && \
        echo "source $CONDA_PATH/etc/profile.d/conda.sh" >> /root/.bashrc; \
    else \
        echo "Warning: Conda directory not found at $CONDA_PATH"; \
    fi

# Stay as root user for development (simpler and avoids permission issues)
# USER root is already active


