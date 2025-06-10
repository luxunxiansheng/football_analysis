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

# install required Python packages from requirements.txt
COPY requirements.txt /tmp/requirements.txt
RUN if [ -f "/tmp/requirements.txt" ]; then \
    pip install --no-cache-dir -r /tmp/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple; \
    fi && \
    rm -rf /tmp/requirements.txt
# Install additional Python packages