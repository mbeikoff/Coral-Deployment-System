# Coral Deployment System

## Quick Start

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/Coral-Deployment-System.git
    cd Coral-Deployment-System
    ```

2. **Build the Docker image:**
    ```bash
    docker build -t coraldeploy-local -f docker/Dockerfile .
    ```

3. **Run the Docker container:**
    ```bash
    sudo docker run -it --rm \
      --runtime=nvidia --gpus all \
      --network=host --privileged \
      --device=/dev/gpiochip0 --device=/dev/gpiochip1 --device=/dev/gpiochip2 \
      -v /proc/device-tree/compatible:/proc/device-tree/compatible \
      -v /proc/device-tree/chosen:/proc/device-tree/chosen \
      -v /sys/devices/:/sys/devices/ \
      -v /sys/firmware/devicetree:/sys/firmware/devicetree \
      -v $PWD/output:/app/output \
      -e JETSON_MODEL_NAME=JETSON_ORIN_NANO \
      coraldeploy-local
    ```
    Adjust volume mounts as needed for your outputs/uploads.

4. **Set up your MapTiler API key:**
    ```bash
    python3 api_key_template.py
    ```
    Follow the prompt to enter your API key. This will create an `api_key.py` file (which is git-ignored).

5. **Open your browser and go to:**
    [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## Docker Installation (Required Version)

This project requires Docker version **27.5.1** and containerd **1.7.24** for full compatibility with Jetson hardware and GPIO access. Follow these steps to install the correct Docker version:

1. **Clean up any existing Docker installation:**
    ```bash
    sudo systemctl stop docker docker.socket containerd
    sudo apt-get purge -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    sudo apt-get autoremove -y --purge
    sudo apt-get autoclean
    sudo rm -rf /var/lib/docker /etc/docker
    sudo rm -f /etc/apt/sources.list.d/docker.list
    ```
2. **Add the Docker repository:**
    ```bash
    sudo apt-get update
    sudo apt-get install -y ca-certificates curl gnupg lsb-release
    sudo mkdir -p /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update
    ```
3. **Download the required .deb packages:**
    ```bash
    cd ~  # Or any temp dir
    wget https://download.docker.com/linux/ubuntu/dists/jammy/pool/stable/arm64/containerd.io_1.7.24-1_arm64.deb
    wget https://download.docker.com/linux/ubuntu/dists/jammy/pool/stable/arm64/docker-ce-cli_27.5.1-1~ubuntu.22.04~jammy_arm64.deb
    wget https://download.docker.com/linux/ubuntu/dists/jammy/pool/stable/arm64/docker-ce_27.5.1-1~ubuntu.22.04~jammy_arm64.deb
    # Optional: Rootless extras
    wget https://download.docker.com/linux/ubuntu/dists/jammy/pool/stable/arm64/docker-ce-rootless-extras_27.5.1-1~ubuntu.22.04~jammy_arm64.deb
    ```
4. **Install the packages:**
    ```bash
    sudo dpkg -i containerd.io_1.7.24-1_arm64.deb
    sudo dpkg -i docker-ce-cli_27.5.1-1~ubuntu.22.04~jammy_arm64.deb
    sudo dpkg -i docker-ce_27.5.1-1~ubuntu.22.04~jammy_arm64.deb
    # Optional: sudo dpkg -i docker-ce-rootless-extras_27.5.1-1~ubuntu.22.04~jammy_arm64.deb
    sudo apt-get install -f
    sudo apt-get install -y docker-buildx-plugin docker-compose-plugin
    ```
5. **Start and verify Docker:**
    ```bash
    sudo systemctl start docker
    sudo systemctl enable docker
    docker --version  # Should show Docker version 27.5.1
    sudo docker run hello-world
    ```
    If you see cgroup warnings (common on Jetson), edit `/boot/extlinux/extlinux.conf` (add `systemd.unified_cgroup_hierarchy=0` to the APPEND line), then reboot.

6. **Prevent auto-upgrades (recommended):**
    ```bash
    sudo apt-mark hold docker-ce docker-ce-cli containerd.io
    ```

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Docker Installation (Required Version)](#docker-installation-required-version)
3. [Overview](#overview)
4. [API Key Setup](#api-key-setup)
5. [Running the App](#running-the-app)
6. [Using the Web App](#using-the-web-app)
7. [Ultrasonic Sensor Integration](#ultrasonic-sensor-integration)
8. [Cluster Management & GPX Export](#cluster-management--gpx-export)
9. [Requirements](#requirements)
10. [Troubleshooting](#troubleshooting)

---

## Overview

The Coral Deployment System is a Flask-based web application for visualizing and managing coral patch deployment data. It allows users to upload CSV files, cluster deployment points, and interact with a live map interface for field operations. The dashboard provides real-time stats, GPS tracking, and live hardware integration.

## API Key Setup

This app uses MapTiler for satellite map tiles. You must provide your own MapTiler API key:

1. Run `python3 api_key_template.py` in the project directory.
2. Enter your API key when prompted. This creates `api_key.py` (ignored by git).

## Running the App

1. Install Python 3 and pip if not already installed.
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Set up your API key as above.
4. Start the app:
    ```bash
    python3 app.py
    ```
5. Open your browser to [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Using the Web App

1. **Landing Page:** Start at the landing page.
2. **New Session:** Click "New Session" to upload your CSV data file. Adjust clustering parameters as needed.
3. **Dashboard:** After upload, you'll be redirected to the dashboard with an interactive map. The map shows patch points and clusters. Your GPS location (if enabled) will be tracked and compared to deployment zones.
4. **Live Ultrasonic Sensor Data:**
    - The dashboard displays live readings from an HC-SR04 ultrasonic sensor in the stats panel.
    - When the sensor detects a distance below the deploy threshold (default 15cm), a real-time popup notification appears: "Coral sample deployed!"
    - Sensor code is integrated directly in `app.py` and uses Jetson.GPIO for hardware access.
    - Sensor initialization occurs after file upload/session start, preventing resource conflicts.
5. **Cluster Management & GPX Export:**
    - Clusters are shown in a sidebar with color badges and stats.
    - You can toggle cluster visibility on the map and download selected clusters as GPX files for field navigation.
6. **Session Summary:** End your session to view a summary of your deployment activity.
7. **History/Help:** Use the navigation bar for session history or help.

### File Upload Format

Your CSV should include columns: `patch_id`, `patch_lat`, `patch_lon`, `patch_decision`, `ping_depth`, etc. Only rows with `patch_decision == 2` are clustered for deployment.

## Ultrasonic Sensor Integration

- The app uses Jetson.GPIO to read from an HC-SR04 ultrasonic sensor.
- Sensor initialization and monitoring are managed in a background thread, started after session creation.
- Live readings are sent to the dashboard via Flask-SocketIO.
- If the sensor detects a distance below the deploy threshold, a popup notification is triggered in real time.

## Cluster Management & GPX Export

- Clusters are color-coded and listed in the sidebar.
- You can toggle cluster visibility and download selected clusters as GPX files for use in GPS devices.
- GPX export is available via the "Download Selected GPX" button.

## Requirements

See `requirements.txt` for all dependencies. Main packages:

- Flask
- pandas
- folium
- scikit-learn
- numpy
- shapely
- scipy
- geopandas
- flask-session
- flask-socketio
- eventlet
- gpxpy
- pyserial
- pynmea2

## Troubleshooting

- If you see a map tile error, check your MapTiler API key.
- For missing packages, re-run `pip install -r requirements.txt`.
- If the app doesn't start, ensure Python 3 is installed and you're in the correct directory.
- For ultrasonic sensor issues, ensure your hardware is connected and Jetson.GPIO is installed.
- If the dashboard freezes, ensure only one ultrasonic thread is running and the app is started with `eventlet` monkey patching.
- For Docker issues, ensure you are using Docker version 27.5.1 and containerd 1.7.24 as described above.

## Jetson GPIO Pin Setup

If you are running on Jetson hardware, you must configure the TRIG pin as an output before running the app. Run the following command in your terminal:

```bash
sudo busybox devmem 0x2448030 w 0xA
```

This sets the TRIG GPIO pin to output mode (required for HC-SR04 operation).

---

For more details, see comments in `app.py` or open an issue.