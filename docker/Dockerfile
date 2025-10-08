# Base image for Jetson Orin Nano with latest JetPack (Ubuntu 22.04)
ARG L4T_IMG_TAG=r36.4.0
# Use the L4T ML image which includes Ubuntu 22.04 and CUDA support
FROM dustynv/l4t-ml:${L4T_IMG_TAG}

# Setup timezone
RUN echo 'Etc/UTC' > /etc/timezone && \
    rm -f /etc/localtime && \
    ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    apt-get update --fix-missing && \
    apt-get install -q -y --no-install-recommends --fix-missing tzdata && \
    rm -rf /var/lib/apt/lists/*

# Install essential packages
RUN apt-get update --fix-missing && apt-get install -q -y --no-install-recommends --fix-missing \
    dirmngr \
    gnupg2 \
    lsb-release \
    wget \
    curl \
    zlib1g \
    busybox \
    software-properties-common \
    python3-dev \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    libgpiod2 \
    python3-libgpiod \
    && rm -rf /var/lib/apt/lists/*

# Add NVIDIA JetPack repository for L4T 36.4 (JetPack 6.1/6.2)
RUN echo "deb https://repo.download.nvidia.com/jetson/common r36.4 main" > /etc/apt/sources.list.d/nvidia-l4t-apt-source.list && \
    echo "deb https://repo.download.nvidia.com/jetson/t234 r36.4 main" >> /etc/apt/sources.list.d/nvidia-l4t-apt-source.list && \
    apt-key adv --fetch-keys https://repo.download.nvidia.com/jetson/jetson-ota-public.asc

# Setup environment
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Install Cudnn
RUN apt-get update --fix-missing && apt-get install -y --no-install-recommends --fix-missing \
    libcudnn9-cuda-12 \
    libcudnn9-dev-cuda-12 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages (Jetson.GPIO via pip for latest version, override index to avoid timeouts)
RUN apt-get update --fix-missing && apt-get install -y --no-install-recommends --fix-missing \
    python3-pip \
    python3-argcomplete \
    bash \
    sudo \
    && rm -rf /var/lib/apt/lists/* && \
    PIP_INDEX_URL=https://pypi.org/simple pip3 install --timeout 300 --upgrade pip && \
    PIP_INDEX_URL=https://pypi.org/simple pip3 install --timeout 300 Jetson.GPIO

# Upgrade pip and install requirements (handle conflicts with --ignore-installed, override index)
COPY app/requirements.txt /tmp/requirements.txt
RUN PIP_INDEX_URL=https://pypi.org/simple pip3 install --timeout 300 --ignore-installed -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Create app directory
WORKDIR /app
COPY app /app

# Make sure runner.sh is executable
RUN chmod +x /app/runner.sh

# Entry point
ENTRYPOINT ["/app/runner.sh"]