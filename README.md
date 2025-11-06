# Coral Deployment System

## Quick Start

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/Coral-Deployment-System.git
    cd Coral-Deployment-System
    ```

2. **Set up your MapTiler API key:**
    ```bash
    python3 api_key_template.py
    ```
    Follow the prompt to enter your API key. This will create an `api_key.py` file (which is git-ignored).

3. **Build the Docker image:**
    ```bash
    docker build -t coraldeploy-local -f docker/Dockerfile .
    ```

4. **Run the Docker container:**
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
        -v $PWD/storage:/app/storage \
        -e JETSON_MODEL_NAME=JETSON_ORIN_NANO \
        coraldeploy-local
    ```
    Adjust volume mounts as needed for your outputs/uploads.

5. **Open your browser and go to:**
    [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## Important New Features

### Map Tile Caching
The system now includes intelligent map tile caching for offline operation:

- **How it works:**
  - Map tiles are automatically cached when viewing areas in the map
  - Cached tiles are stored in `storage/tile_cache/` organized by zoom/x/y coordinates
  - Once cached, tiles remain available even without internet connection
  - Supports zoom levels 5-22 for detailed satellite imagery

- **To cache map areas:**
  1. Start a new session
  2. Pan and zoom around all areas you plan to work in
  3. Try different zoom levels you might need
  4. The system automatically saves these tiles for offline use

- **Best practices:**
  - Cache areas before field deployment
  - Explore target areas at various zoom levels
  - Verify cached areas by testing offline
  - Remember to cache surrounding areas for navigation

### Other Improvements
- Enhanced GPS tracking with configurable path width
- Real-time ultrasonic sensor feedback
- Improved cluster visualization with convex hulls
- Session resume capability
- Comprehensive data export
- Better offline operation support

---

## IMPORTANT: API Key Setup Before Building Docker

Before building the Docker container, you must set up your MapTiler API key. This ensures the container has access to the required API key for map tiles.

1. Run the following command in your project directory:
    ```bash
    python3 api_key_template.py
    ```
2. Enter your MapTiler API key when prompted. This will create an `api_key.py` file (which is git-ignored).

**Only after this step should you proceed to build the Docker image.**

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

You can run the Coral Deployment System in two ways:

### 1. Through Docker (Recommended for Production)

- This is the preferred method for production deployments.
- All dependencies and hardware access are managed inside the container.
- Follow the Docker build and run instructions above.

### 2. Standalone (Recommended for Development)

- Useful for development, debugging, or rapid iteration.
- You must manually install all dependencies:
    ```bash
    pip install -r requirements.txt
    ```
- You must run the app with `sudo` to access GPIO hardware:
    ```bash
    sudo python3 app.py
    ```
- Before starting, set up the GPIO pin for the ultrasonic sensor:
    ```bash
    sudo busybox devmem 0x2448030 w 0xA
    ```
    (This command configures the TRIG pin as output. The setup code is now located in `flask_session/gpio_setup.py`.)
- Make sure your MapTiler API key is set up by running:
    ```bash
    python3 api_key_template.py
    ```

## System Features

### 1. Data Management
- Upload and process CSV or GPX files
- Automatic clustering of deployment points
- Session management with resume capability
- Comprehensive data export

### 2. Map Interface
- Interactive satellite map display
- Offline map tile caching
- Cluster visualization with convex hulls
- Real-time GPS tracking
- Configurable path width for better visibility
- Multiple zoom levels (5-22)

### 3. Hardware Integration
- Real-time ultrasonic sensor readings
- Automated deployment detection
- GPS position tracking and logging
- Depth data integration

### 4. Clustering & Navigation
- DBSCAN-based point clustering
- Convex hull boundary generation
- GPX export for navigation
- Cluster-specific statistics

### 5. Session Management
- Live session tracking
- Pausable/resumable sessions
- Historical session viewing
- Comprehensive session data export

## File Upload Format

### CSV Format
Your CSV should include columns:
- `patch_id`: Unique identifier
- `patch_lat`: Latitude
- `patch_lon`: Longitude
- `patch_decision`: Decision code (0=reject, 1=maybe, 2=deploy)
- `ping_depth`: Depth reading

### GPX Format
- Supports routes with waypoints
- Each route becomes a cluster
- Points automatically converted to deployment targets

## Requirements

See `requirements.txt` for all dependencies. Main packages:

- Flask & Flask extensions
- pandas
- folium
- scikit-learn
- numpy
- shapely
- scipy
- geopandas
- gpxpy
- pyserial
- pynmea2
- eventlet

## Hardware Setup

### Ultrasonic Sensor
- Using HC-SR04 sensor
- TRIG_PIN = 7 (Physical)
- ECHO_PIN = 15 (Physical)
- Run this command before starting:
    ```bash
    sudo busybox devmem 0x2448030 w 0xA
    ```

### GPS Module
- Supports NMEA protocol
- Configurable update rate
- Provides position, speed, and quality metrics

## Troubleshooting

### Map Issues
- If tiles don't load online: Check MapTiler API key
- If tiles don't load offline: Ensure area was previously cached
- To force cache refresh: Clear `storage/tile_cache` directory

### Hardware Issues
- For ultrasonic errors: Check pin configuration
- For GPS issues: Verify serial connection
- For Docker access: Verify device permissions

### Data Issues
- For CSV problems: Verify column names and format
- For GPX issues: Ensure valid routes/waypoints
- For session errors: Check storage permissions

### System Issues
- For memory errors: Reduce cached area size
- For performance issues: Optimize zoom levels cached
- For Docker issues: Verify version compatibility

---

For more details, see the source code or open an issue.