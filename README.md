
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
6. [Requirements](#requirements)
7. [Troubleshooting](#troubleshooting)

---

## Overview

The Coral Deployment System is a Flask-based web application for visualizing and managing coral patch deployment data. It allows users to upload CSV files, cluster deployment points, and interact with a live map interface for field operations.

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
4. **Session Summary:** End your session to view a summary of your deployment activity.
5. **History/Help:** Use the navigation bar for session history or help.

### File Upload Format

Your CSV should include columns: `patch_id`, `patch_lat`, `patch_lon`, `patch_decision`, `ping_depth`, etc. Only rows with `patch_decision == 2` are clustered for deployment.

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

## Troubleshooting

- If you see a map tile error, check your MapTiler API key.
- For missing packages, re-run `pip install -r requirements.txt`.
- If the app doesn't start, ensure Python 3 is installed and you're in the correct directory.

---

For more details, see comments in `app.py` or open an issue.