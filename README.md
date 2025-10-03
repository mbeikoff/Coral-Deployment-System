# Coral Deployment System

## Quick Start

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/Coral-Deployment-System.git
    cd Coral-Deployment-System
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Set up your MapTiler API key:**
    ```bash
    python3 api_key_template.py
    ```
    Follow the prompt to enter your API key. This will create an `api_key.py` file (which is git-ignored).

4. **Run the web application:**
    ```bash
    python3 app.py
    ```

5. **Open your browser and go to:**
    [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Overview](#overview)
3. [API Key Setup](#api-key-setup)
4. [Running the App](#running-the-app)
5. [Using the Web App](#using-the-web-app)
6. [Ultrasonic Sensor Integration](#ultrasonic-sensor-integration)
7. [Cluster Management & GPX Export](#cluster-management--gpx-export)
8. [Requirements](#requirements)
9. [Troubleshooting](#troubleshooting)

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

## Troubleshooting

- If you see a map tile error, check your MapTiler API key.
- For missing packages, re-run `pip install -r requirements.txt`.
- If the app doesn't start, ensure Python 3 is installed and you're in the correct directory.
- For ultrasonic sensor issues, ensure your hardware is connected and Jetson.GPIO is installed.
- If the dashboard freezes, ensure only one ultrasonic thread is running and the app is started with `eventlet` monkey patching.

---

For more details, see comments in `app.py` or open an issue.