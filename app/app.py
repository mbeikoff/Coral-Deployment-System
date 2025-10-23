import eventlet
eventlet.monkey_patch()

from flask import Flask, render_template, request, jsonify, session, redirect, Response, abort, flash
import pandas as pd
import folium
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import numpy as np
from shapely.geometry import Point
from shapely.ops import unary_union
from scipy.spatial import ConvexHull
import geopandas as gpd
import os
import serial
import pynmea2
import json
from datetime import datetime, timedelta
import uuid
from flask_sqlalchemy import SQLAlchemy
from flask_session import Session  # pip install flask-session
from folium import Element
from folium.plugins import MarkerCluster, LocateControl, Realtime
from folium import JsCode
import io

# pip install gpxpy  # Add this for GPX export
import gpxpy
import gpxpy.gpx

# Disable verbose logging
import logging
logging.getLogger('socketio').setLevel(logging.ERROR)
logging.getLogger('engineio').setLevel(logging.ERROR)
logging.getLogger('eventlet').setLevel(logging.ERROR)
logging.getLogger('eventletwebsocket.handler').setLevel(logging.ERROR)

# --- Ultrasonic sensor setup (Jetson HC-SR04) ---
import time

TRIG_PIN = 7   # Physical pin 7
ECHO_PIN = 15  # Physical pin 15
GPIO = None
ULTRASONIC_INITIALIZED = False
ultrasonic_thread = None

def get_ultrasonic_distance(debug=False):
    global GPIO, ULTRASONIC_INITIALIZED
    if not ULTRASONIC_INITIALIZED or GPIO is None:
        return -1
    try:
        # Send 10us pulse to trigger
        GPIO.output(TRIG_PIN, True)
        time.sleep(0.00001)
        GPIO.output(TRIG_PIN, False)
        if debug:
            print("Trigger sent. Waiting for ECHO high...")
        # Wait for echo to go high
        pulse_start = time.time()
        timeout = pulse_start + 0.01  # Shorter 10ms timeout
        while GPIO.input(ECHO_PIN) == 0:
            pulse_start = time.time()
            if pulse_start > timeout:
                if debug:
                    print("Timeout waiting for ECHO high")
                return -1
        if debug:
            print("ECHO high detected. Waiting for low...")
        # Wait for echo to go low
        pulse_end = time.time()
        timeout = pulse_end + 0.01  # Shorter 10ms timeout
        while GPIO.input(ECHO_PIN) == 1:
            pulse_end = time.time()
            if pulse_end > timeout:
                if debug:
                    print("Timeout: ECHO never went low (stuck high - no object or wiring issue)")
                return -1
        pulse_duration = pulse_end - pulse_start
        distance = pulse_duration * 17150  # cm
        if debug:
            print(f"Pulse duration: {pulse_duration*1e6:.1f} µs")
            print(f"Distance: {distance} cm")
        return round(distance, 2)
    except Exception as e:
        if debug:
            print(f"Ultrasonic error: {e}")
        return -1

from flask_socketio import SocketIO, emit
import threading
from api_key import API_KEY
import io
from folium import Element
from folium.plugins import MarkerCluster, LocateControl, Realtime
from folium import JsCode


app = Flask(__name__)
app.secret_key = 'reefscan_secret'  # Change for prod
app.config['SESSION_TYPE'] = 'filesystem'

# Get the directory where the script lives
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Create storage folder in the script directory
STORAGE_FOLDER = os.path.join(SCRIPT_DIR, 'storage')
os.makedirs(STORAGE_FOLDER, exist_ok=True)

# Database in storage with absolute path
DB_PATH = os.path.join(STORAGE_FOLDER, 'reefscan.db')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_PATH}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

Session(app)
db = SQLAlchemy(app)
socketio = SocketIO(app)  # Removed engineio_logger to avoid the TypeError

UPLOAD_FOLDER = os.path.join(STORAGE_FOLDER, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Global for demo (use DB later)
clusters_data = {}  # {session_id: {'blobs': [...], 'start_time': ..., 'deploy_count': 0, ...}}

CURRENT_SESSION_ID = None
CURRENT_IN_ZONE = False

# --- Real-time ultrasonic sensor thread ---
ultrasonic_distance = 0.0
DEPLOY_THRESHOLD = 15.0

def ultrasonic_monitor():
    global ultrasonic_distance, ULTRASONIC_INITIALIZED, CURRENT_SESSION_ID, CURRENT_IN_ZONE, gps_lat, gps_lon
    while ULTRASONIC_INITIALIZED:
        dist = get_ultrasonic_distance()
        ultrasonic_distance = dist
        # Add callback to queue emit async
        socketio.emit('ultrasonic_update', {'distance': dist}, callback=lambda: None)
        if dist > 0 and dist < DEPLOY_THRESHOLD and CURRENT_IN_ZONE and CURRENT_SESSION_ID in clusters_data:
            sess_data = clusters_data[CURRENT_SESSION_ID]
            sess_data['deploy_count'] += 1
            # Find nearest cluster
            min_dist_to_blob = float('inf')
            nearest_cluster_id = None
            for i, blob in enumerate(sess_data['blobs']):
                blob_dist = haversine_dist(gps_lat or 0, gps_lon or 0, blob['lat'], blob['lon'])
                if blob_dist < min_dist_to_blob:
                    min_dist_to_blob = blob_dist
                    nearest_cluster_id = sess_data['valid_clusters'][i]
            # Log to DB
            try:
                sid = int(CURRENT_SESSION_ID)
                deploy = Deployment(
                    session_id=sid,
                    timestamp=datetime.now(),
                    lat=float(gps_lat) if gps_lat else None,
                    lon=float(gps_lon) if gps_lon else None,
                    ultrasonic_distance=dist,
                    cluster_id=nearest_cluster_id
                )
                db.session.add(deploy)
                sess = db.session.get(ReefSession, sid)
                sess.deploy_count = sess_data['deploy_count']
                db.session.commit()
            except Exception as e:
                print(f"Deployment log error: {e}")
            socketio.emit('deploy_event', {'message': 'Coral sample deployed!', 'distance': dist, 'deploy_count': sess_data['deploy_count']}, callback=lambda: None)
        time.sleep(0.05)  # 20Hz polling

def haversine_dist(lat1, lon1, lat2, lon2):
    R = 6371000
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

# Database Models
class ReefSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_key = db.Column(db.String(50), unique=True)
    start_time = db.Column(db.DateTime)
    end_time = db.Column(db.DateTime, nullable=True)
    total_distance = db.Column(db.Float, default=0.0)
    deploy_count = db.Column(db.Integer, default=0)
    total_patches = db.Column(db.Integer)
    csv_filename = db.Column(db.String(200))
    df_json = db.Column(db.Text)  # New: Store full DF as JSON for resilience
    eps = db.Column(db.Float)
    min_samples = db.Column(db.Integer)
    min_cluster_size = db.Column(db.Integer)
    hide_no_deploy = db.Column(db.Boolean, default=True)
    status = db.Column(db.String(20), default='ongoing')
    clusters_json = db.Column(db.Text)
    gps_logs = db.relationship('GPSLog', backref='session', lazy=True, cascade='all, delete-orphan')
    deployments = db.relationship('Deployment', backref='session', lazy=True, cascade='all, delete-orphan')

class GPSLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('reef_session.id'))
    timestamp = db.Column(db.DateTime)
    lat = db.Column(db.Float)
    lon = db.Column(db.Float)
    speed = db.Column(db.Float, default=0.0)
    depth = db.Column(db.Float, default=20.0)
    qual = db.Column(db.Integer)
    sats = db.Column(db.Integer)
    hdop = db.Column(db.Float)

class Deployment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('reef_session.id'))
    timestamp = db.Column(db.DateTime)
    lat = db.Column(db.Float, nullable=True)
    lon = db.Column(db.Float, nullable=True)
    ultrasonic_distance = db.Column(db.Float)
    cluster_id = db.Column(db.Integer, nullable=True)

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/new', methods=['GET', 'POST'])
def new_session():
    global GPIO, ULTRASONIC_INITIALIZED, ultrasonic_thread, CURRENT_SESSION_ID, CURRENT_IN_ZONE
    if request.method == 'POST':
        file = request.files['file']
        if file.filename == '':
            return render_template('new.html', error='No file selected')
        
        eps = float(request.form.get('eps', 50.0))  # Default 50m for deploy threshold
        min_samples = int(request.form.get('min_samples', 2))
        min_cluster_size = int(request.form.get('min_cluster_size', 2))
        hide_no_deploy = request.form.get('hide_no_deploy', 'on') == 'on'  # Checkbox default on (hide)
        
        try:
            df = pd.read_csv(file)
            if df.empty:
                return render_template('new.html', error='Uploaded CSV is empty')
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            
            # Clustering (only on decision 2)
            deploy_df = df[df['patch_decision'] == 2].copy()
            if len(deploy_df) > 0:
                coords = deploy_df[['patch_lon', 'patch_lat']].values
                dbscan = DBSCAN(eps=eps / 6371000, min_samples=min_samples, metric='haversine').fit(np.radians(coords))
                deploy_df['cluster'] = dbscan.labels_
                cluster_sizes = deploy_df['cluster'].value_counts()
                valid_clusters = [c for c in cluster_sizes[cluster_sizes >= min_cluster_size].index if c != -1]
                
                # Store labeled deploy_df and valid_clusters to avoid recompute
                labeled_deploy_df = deploy_df.copy()  # Retain labels/sizes
                
                # Prep blobs as list of [centre_lat, centre_lon, radius=eps] for JS checks
                blobs = []
                for cid in valid_clusters:
                    cluster_pts = deploy_df[deploy_df['cluster'] == cid][['patch_lat', 'patch_lon']].values
                    centre = cluster_pts.mean(axis=0)
                    blobs.append({'lat': centre[0], 'lon': centre[1], 'radius': eps})  # For distance calc
                
                # Create session in DB
                new_sess = ReefSession(
                    session_key=str(uuid.uuid4()),
                    start_time=datetime.now(),
                    total_distance=0.0,
                    deploy_count=0,
                    total_patches=len(df),
                    csv_filename=file.filename,
                    df_json=df.to_json(orient='records'),  # Store full DF as JSON
                    eps=eps,
                    min_samples=min_samples,
                    min_cluster_size=min_cluster_size,
                    hide_no_deploy=hide_no_deploy,
                    status='ongoing',
                    clusters_json=json.dumps({
                        'blobs': blobs,
                        'labeled_deploy_df': labeled_deploy_df.to_json(orient='records'),
                        'valid_clusters': valid_clusters
                    })
                )
                db.session.add(new_sess)
                db.session.commit()
                
                session_id = str(new_sess.id)
                session['session_id'] = session_id
                
                # Start session in memory
                clusters_data[session_id] = {
                    'blobs': blobs,
                    'start_time': time.time(),
                    'deploy_count': 0,
                    'total_distance': 0.0,
                    'prev_pos': None,
                    'df': df,
                    'eps': eps,
                    'min_samples': min_samples,
                    'min_cluster_size': min_cluster_size,
                    'hide_no_deploy': hide_no_deploy,
                    'labeled_deploy_df': labeled_deploy_df,
                    'valid_clusters': valid_clusters
                }
                CURRENT_SESSION_ID = session_id
                CURRENT_IN_ZONE = False

                # --- Ultrasonic sensor initialization ---
                if not ULTRASONIC_INITIALIZED:
                    import Jetson.GPIO as _GPIO
                    GPIO = _GPIO
                    GPIO.setmode(GPIO.BOARD)
                    GPIO.setup(TRIG_PIN, GPIO.OUT)
                    GPIO.setup(ECHO_PIN, GPIO.IN)
                    GPIO.output(TRIG_PIN, False)
                    ULTRASONIC_INITIALIZED = True
                    time.sleep(0.05)
                    if ultrasonic_thread is None:
                        ultrasonic_thread = threading.Thread(target=ultrasonic_monitor, daemon=True)
                        ultrasonic_thread.start()

                return redirect('/dashboard')
            else:
                return render_template('new.html', error='No deploy (2) points found!')
        except Exception as e:
            return render_template('new.html', error=f'Error: {str(e)}')
    
    return render_template('new.html')

@app.route('/resume/<int:sid>')
def resume_session(sid):
    global CURRENT_SESSION_ID, CURRENT_IN_ZONE
    sess = db.session.get(ReefSession, sid)
    if sess is None:
        flash('Session not found.', 'error')
        return redirect('/history')
    if sess.status == 'completed':
        flash('This session is completed and cannot be resumed.', 'error')
        return redirect('/history')
    if sess.end_time:
        sess.end_time = None
        db.session.commit()
    session['session_id'] = str(sid)
    # Reconstruct data
    try:
        # Prefer DF from JSON (resilient to file issues)
        if sess.df_json:
            df = pd.read_json(sess.df_json, orient='records')
        else:
            # Fallback to file
            csv_path = os.path.join(UPLOAD_FOLDER, sess.csv_filename)
            if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
                raise ValueError("CSV file missing or empty")
            df = pd.read_csv(csv_path)
        if df.empty:
            raise ValueError("Loaded data is empty")
    except Exception as e:
        flash(f"Error resuming session: Unable to load data - {str(e)}. Please re-upload if needed.", 'error')
        return redirect('/history')
    
    cluster_dict = json.loads(sess.clusters_json or '{}')
    try:
        labeled_deploy_df = pd.read_json(cluster_dict['labeled_deploy_df'], orient='records')
    except (KeyError, ValueError):
        # Fallback if no labeled data (old session?)
        labeled_deploy_df = pd.DataFrame()
        valid_clusters = []
    else:
        valid_clusters = cluster_dict.get('valid_clusters', [])
    data = {
        'blobs': cluster_dict.get('blobs', []),
        'start_time': sess.start_time.timestamp(),
        'deploy_count': sess.deploy_count,
        'total_distance': sess.total_distance,
        'prev_pos': None,
        'df': df,
        'eps': sess.eps,
        'min_samples': sess.min_samples,
        'min_cluster_size': sess.min_cluster_size,
        'hide_no_deploy': sess.hide_no_deploy,
        'labeled_deploy_df': labeled_deploy_df,
        'valid_clusters': valid_clusters
    }
    # Set prev_pos from last log
    last_log = GPSLog.query.filter_by(session_id=sid).order_by(GPSLog.timestamp.desc()).first()
    if last_log:
        data['prev_pos'] = (last_log.lat, last_log.lon)
        # Set initial zone
        min_dist = float('inf')
        prev_lat, prev_lon = data['prev_pos']
        for blob in data['blobs']:
            dist = haversine_dist(prev_lat, prev_lon, blob['lat'], blob['lon'])
            min_dist = min(min_dist, dist)
        CURRENT_IN_ZONE = min_dist <= data['eps']
    clusters_data[str(sid)] = data
    CURRENT_SESSION_ID = str(sid)
    return redirect('/dashboard')

@app.route('/view/<int:sid>')
def view_session(sid):
    sess = db.session.get(ReefSession, sid)
    if sess is None:
        abort(404)
    
    # Reconstruct data similar to resume
    try:
        if sess.df_json:
            df = pd.read_json(sess.df_json, orient='records')
        else:
            csv_path = os.path.join(UPLOAD_FOLDER, sess.csv_filename)
            if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
                raise ValueError("CSV file missing or empty")
            df = pd.read_csv(csv_path)
        if df.empty:
            raise ValueError("Loaded data is empty")
    except Exception as e:
        flash(f"Error loading session data: {str(e)}", 'error')
        return redirect('/history')
    
    cluster_dict = json.loads(sess.clusters_json or '{}')
    try:
        labeled_deploy_df = pd.read_json(cluster_dict['labeled_deploy_df'], orient='records')
    except (KeyError, ValueError):
        labeled_deploy_df = pd.DataFrame()
        valid_clusters = []
    else:
        valid_clusters = cluster_dict.get('valid_clusters', [])
    
    data = {
        'blobs': cluster_dict.get('blobs', []),
        'deploy_count': sess.deploy_count,
        'total_distance': sess.total_distance,
        'df': df,
        'eps': sess.eps or 50.0,
        'min_samples': sess.min_samples or 2,
        'min_cluster_size': sess.min_cluster_size or 2,
        'hide_no_deploy': sess.hide_no_deploy,
        'labeled_deploy_df': labeled_deploy_df,
        'valid_clusters': valid_clusters
    }
    
    # Load historical data
    gps_logs = GPSLog.query.filter_by(session_id=sid).order_by(GPSLog.timestamp).all()
    deployments = Deployment.query.filter_by(session_id=sid).all()
    
    # Generate map similar to dashboard, but static
    df_view = data['df'].copy()
    if data.get('hide_no_deploy', True):
        df_view = df_view[df_view['patch_decision'] != 0].copy()
    centre_lat = df_view['patch_lat'].mean() if not df_view.empty else 0
    centre_lon = df_view['patch_lon'].mean() if not df_view.empty else 0
    
    # Prepare points data for JS
    points_data = []
    def get_colour(decision):
        if decision == 0: return 'red'
        if decision == 1: return 'green'
        if decision == 2: return 'yellow'
        return 'blue'
    
    for _, row in df_view.iterrows():
        popup_escaped = str(row.get('patch_id', '')).replace("'", "\\'") + r"<br>Decision: " + str(row.get('patch_decision', '')).replace("'", "\\'") + r"<br>Depth: " + str(row.get('ping_depth', '')).replace("'", "\\'")
        points_data.append({
            'lat': row['patch_lat'],
            'lon': row['patch_lon'],
            'popup': popup_escaped,
            'color': get_colour(row['patch_decision'])
        })
    
    # Clusters
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    cluster_info = []
    cluster_layers_js = []
    for i, cid in enumerate(valid_clusters):
        color = colors[i % len(colors)]
        cluster_pts = labeled_deploy_df[labeled_deploy_df['cluster'] == cid][['patch_lat', 'patch_lon']].values
        size = len(cluster_pts)
        if size > 0:
            center_lat, center_lon = cluster_pts.mean(axis=0)
            if size >= 3:
                hull_input = cluster_pts[:, [1, 0]]  # lon, lat
                hull = ConvexHull(hull_input)
                vertices = cluster_pts[hull.vertices]
                hull_pts = [Point(v[1], v[0]) for v in vertices]  # lat, lon
                hull_poly = unary_union(hull_pts).convex_hull
                gdf = gpd.GeoDataFrame({'geometry': [hull_poly]}, crs='EPSG:4326')
                geojson = gdf.to_json()
                geojson_escaped = geojson.replace("'", "\\'")
                cluster_layers_js.append(f"""
                var cluster{cid} = L.geoJSON(JSON.parse('{geojson_escaped}'), {{
                    style: {{fillColor: '{color}', color: '{color}', weight: 3, fillOpacity: 0.4}},
                    onEachFeature: function(feature, layer) {{
                        layer.bindPopup("Cluster {cid}<br>Size: {size} points");
                    }}
                }}).addTo(map);
                """)
                area_m2 = round(gdf.to_crs('EPSG:3857').area.iloc[0], 2)
            else:
                r_m = data['eps'] / 2
                area_m2 = round(np.pi * r_m**2, 2)
                r_deg = data['eps'] / 111000 * 57.3
                cluster_layers_js.append(f"""
                var cluster{cid} = L.circle([{center_lat}, {center_lon}], {{radius: {r_deg}, color: '{color}', weight: 3, fillColor: '{color}', fillOpacity: 0.4}})
                    .bindPopup("Cluster {cid}<br>Size: {size} points")
                    .addTo(map);
                """)
            
            cluster_info.append({
                'cid': cid,
                'size': size,
                'center_lat': round(float(center_lat), 6),
                'center_lon': round(float(center_lon), 6),
                'area_m2': area_m2,
                'color': color
            })
    
    # Historical trail
    historical_trail_js = ""
    if gps_logs:
        points_js = chr(10).join([f"[{log.lat}, {log.lon}]," for log in gps_logs])
        historical_trail_js = f"""
        var historicalTrail = L.polyline([
        {points_js}
        ], {{color: 'gray', weight: 2, dashArray: '5,5', opacity: 0.6}}).addTo(map);
        """
    
    # Past deployments
    past_deploys_js = ""
    if deployments:
        dep_markers_js = chr(10).join([f"""
        var depMarker{dep.id} = L.marker([{dep.lat}, {dep.lon}], {{
            icon: L.divIcon({{
                className: 'deploy-marker',
                html: '<div style="background: orange; width: 16px; height: 16px; border-radius: 50%; border: 2px solid white;"></div>'
            }})
        }}).bindPopup('Past Deployment<br>Time: {dep.timestamp.strftime("%H:%M:%S")}<br>Dist: {dep.ultrasonic_distance} cm{"<br>Cluster: " + str(dep.cluster_id) if dep.cluster_id else ""}');
        pastDeploysLayer.addLayer(depMarker{dep.id});
        """ for dep in deployments])
        past_deploys_js = f"""
        var pastDeploysLayer = L.layerGroup();
        {dep_markers_js}
        pastDeploysLayer.addTo(map);
        """
    
    # Build static map
    map_html = f"""
    <div id="map" style="height: 60vh; width: 100%;"></div>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        var map = L.map('map').setView([{centre_lat}, {centre_lon}], 15);
        L.tileLayer('https://api.maptiler.com/maps/satellite/{{z}}/{{x}}/{{y}}.png?key={API_KEY}', {{
            attribution: '© MapTiler',
            maxZoom: 22
        }}).addTo(map);
        
        // Points layer
        var pointsLayer = L.layerGroup();
        {chr(10).join([f"L.circleMarker([{p['lat']}, {p['lon']}], {{radius: 3, color: '{p['color']}', fill: true, fillOpacity: 0.7}}).bindPopup('{p['popup']}').addTo(pointsLayer);" for p in points_data])}
        pointsLayer.addTo(map);
        
        // Cluster layers
        {chr(10).join(cluster_layers_js)}
        
        // Historical trail
        {historical_trail_js}
        
        // Past deployments
        {past_deploys_js}
        
        window.map = map;
    </script>
    <style>.deploy-marker {{ background: transparent; border: none; }}</style>
    """
    
    # Prepare data for tables
    patches_data = df.to_dict('records')
    gps_data = [{'timestamp': log.timestamp.strftime('%Y-%m-%d %H:%M:%S'), 'lat': log.lat, 'lon': log.lon, 'speed': log.speed, 'depth': log.depth, 'qual': log.qual, 'sats': log.sats, 'hdop': log.hdop} for log in gps_logs]
    deploys_data = [{'timestamp': dep.timestamp.strftime('%Y-%m-%d %H:%M:%S'), 'lat': dep.lat, 'lon': dep.lon, 'ultrasonic_distance': dep.ultrasonic_distance, 'cluster_id': dep.cluster_id} for dep in deployments]
    
    # Stats
    mission_time = str(timedelta(seconds=int((sess.end_time - sess.start_time).total_seconds()))) if sess.end_time else 'N/A'
    stats = {
        'mission_time': mission_time,
        'deploy_count': sess.deploy_count,
        'total_distance': f"{sess.total_distance:.2f} km",
        'total_patches': sess.total_patches,
        'blobs': len(cluster_info)
    }
    
    return render_template('view.html', map_html=map_html, stats=stats, cluster_info=cluster_info, 
                           patches_data=patches_data, gps_data=gps_data, deploys_data=deploys_data, sid=sid)

@app.route('/export/<int:sid>')
def export_session(sid):
    sess = db.session.get(ReefSession, sid)
    if sess is None:
        abort(404)
    
    try:
        if sess.df_json:
            df = pd.read_json(sess.df_json, orient='records')
        else:
            csv_path = os.path.join(UPLOAD_FOLDER, sess.csv_filename)
            df = pd.read_csv(csv_path)
        
        gps_logs = GPSLog.query.filter_by(session_id=sid).all()
        gps_df = pd.DataFrame([
            {'timestamp': log.timestamp, 'lat': log.lat, 'lon': log.lon, 'speed': log.speed, 'depth': log.depth, 'qual': log.qual, 'sats': log.sats, 'hdop': log.hdop}
            for log in gps_logs
        ])
        
        deployments = Deployment.query.filter_by(session_id=sid).all()
        deploys_df = pd.DataFrame([
            {'timestamp': dep.timestamp, 'lat': dep.lat, 'lon': dep.lon, 'ultrasonic_distance': dep.ultrasonic_distance, 'cluster_id': dep.cluster_id}
            for dep in deployments
        ])
        
        # Combine into one DF with session info
        df['session_id'] = sid
        df['session_start'] = sess.start_time
        df['session_end'] = sess.end_time
        gps_df['session_id'] = sid
        gps_df['session_start'] = sess.start_time
        gps_df['session_end'] = sess.end_time
        deploys_df['session_id'] = sid
        deploys_df['session_start'] = sess.start_time
        deploys_df['session_end'] = sess.end_time
        
        combined_df = pd.concat([df, gps_df, deploys_df], ignore_index=True, sort=False)
        combined_df = combined_df.fillna('')
        
        output = io.StringIO()
        combined_df.to_csv(output, index=False)
        csv_content = output.getvalue()
        
        filename = f"session_{sid}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
        
        return Response(
            csv_content,
            mimetype="text/csv",
            headers={"Content-disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        abort(500, description=str(e))

# API endpoint for live ultrasonic sensor data
@app.route('/api/ultrasonic')
def ultrasonic_api():
    try:
        distance = get_ultrasonic_distance()
        return jsonify({'distance': distance})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download_gpx')
def download_gpx():
    session_id = request.args.get('session_id') or session.get('session_id')
    if not session_id:
        return redirect('/new')
    
    selected_str = request.args.get('selected', None)
    try:
        sid = int(session_id)
        sess = db.session.get(ReefSession, sid)
        if sess is None:
            return redirect('/new')
        
        cluster_dict = json.loads(sess.clusters_json or '{}')
        labeled_deploy_df = pd.read_json(cluster_dict['labeled_deploy_df'], orient='records')
        valid_clusters = cluster_dict['valid_clusters']
        
        # Filter by selected if provided
        if selected_str:
            try:
                selected = [int(c.strip()) for c in selected_str.split(',') if c.strip()]
                valid_clusters = [c for c in valid_clusters if c in selected]
            except ValueError:
                pass  # Invalid, fall back to all
        
        print(f"DEBUG: Starting GPX export for {len(valid_clusters)} clusters (selected: {selected_str})")  # Console log
        
        gpx = gpxpy.gpx.GPX()
        route_count = 0
        
        for cid in valid_clusters:
            try:
                cluster_pts = labeled_deploy_df[labeled_deploy_df['cluster'] == cid][['patch_lat', 'patch_lon']].values  # [lat, lon]
                route_name = f"Cluster {cid} (size: {len(cluster_pts)})"
                route = gpxpy.gpx.GPXRoute(name=route_name, description=f"Convex hull boundary for cluster {cid}")
                
                if len(cluster_pts) >= 3:
                    hull_input = cluster_pts[:, [1, 0]]  # [lon, lat] for ConvexHull
                    hull = ConvexHull(hull_input)
                    ordered_cluster_pts = cluster_pts[hull.vertices]  # ordered [lat, lon]
                    for lat, lon in ordered_cluster_pts:
                        rtept = gpxpy.gpx.GPXRoutePoint(latitude=lat, longitude=lon)
                        route.points.append(rtept)
                    # Close the route for a polygon-like boundary
                    if len(route.points) > 0:
                        route.points.append(route.points[0])
                else:
                    # For small clusters, add all points (no hull)
                    for lat, lon in cluster_pts:
                        rtept = gpxpy.gpx.GPXRoutePoint(latitude=lat, longitude=lon)
                        route.points.append(rtept)
                
                gpx.routes.append(route)
                route_count += 1
                print(f"DEBUG: Added route for cluster {cid} ({len(route.points)} points)")
            except Exception as e:
                print(f"DEBUG: Skipped cluster {cid} due to error: {e}")
                continue  # Skip bad cluster but continue with others
        
        print(f"DEBUG: GPX export complete with {route_count} routes")  # Final log
        
        xml = gpx.to_xml()
        
        # FIXED: Use the provided filename query param, default to timestamp if empty
        filename = request.args.get('filename', None)
        if not filename:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"clusters_{timestamp}"
        filename += '.gpx'
        
        return Response(
            xml,
            mimetype="application/gpx+xml",
            headers={"Content-disposition": f"attachment; filename={filename}"}
        )
    except ValueError:
        return redirect('/new')

@app.route('/api/delete_session/<int:sid>', methods=['DELETE', 'POST'])
def delete_session(sid):
    sess = db.session.get(ReefSession, sid)
    if not sess:
        return jsonify({'error': 'Session not found'}), 404
    session_id = str(sid)
    if session_id in clusters_data:
        del clusters_data[session_id]
    if session.get('session_id') == session_id:
        session.pop('session_id')
    global CURRENT_SESSION_ID, CURRENT_IN_ZONE
    if CURRENT_SESSION_ID == session_id:
        CURRENT_SESSION_ID = None
        CURRENT_IN_ZONE = False
    db.session.delete(sess)
    db.session.commit()
    return jsonify({'status': 'deleted'})

@app.route('/end_session')
def end_session():
    global CURRENT_SESSION_ID, CURRENT_IN_ZONE
    mode = request.args.get('mode', 'save')
    if 'session_id' not in session:
        return redirect('/')
    session_id = session['session_id']
    data = clusters_data.pop(session_id, None)
    sid = int(session_id)
    sess = db.session.get(ReefSession, sid)
    if not sess:
        return redirect('/')
    if mode == 'pause':
        sess.end_time = None
        sess.status = 'paused'
        db.session.commit()
        session.pop('session_id')
        CURRENT_SESSION_ID = None
        CURRENT_IN_ZONE = False
        return redirect('/history')
    else:
        # save/completed
        sess.end_time = datetime.now()
        sess.status = 'completed'
        if data:
            sess.total_distance = data.get('total_distance', sess.total_distance)
            sess.deploy_count = data.get('deploy_count', sess.deploy_count)
            sess.clusters_json = json.dumps({
                'blobs': data['blobs'],
                'labeled_deploy_df': data['labeled_deploy_df'].to_json(orient='records'),
                'valid_clusters': data['valid_clusters']
            })
        db.session.commit()
        session.pop('session_id')
        CURRENT_SESSION_ID = None
        CURRENT_IN_ZONE = False
        
        # Reconstruct data for summary view
        try:
            if sess.df_json:
                df = pd.read_json(sess.df_json, orient='records')
            else:
                csv_path = os.path.join(UPLOAD_FOLDER, sess.csv_filename)
                df = pd.read_csv(csv_path)
            if df.empty:
                raise ValueError("Loaded data is empty")
        except Exception as e:
            flash(f"Error loading session data for summary: {str(e)}", 'error')
            return redirect('/history')
        
        cluster_dict = json.loads(sess.clusters_json or '{}')
        try:
            labeled_deploy_df = pd.read_json(cluster_dict['labeled_deploy_df'], orient='records')
        except (KeyError, ValueError):
            labeled_deploy_df = pd.DataFrame()
            valid_clusters = []
        else:
            valid_clusters = cluster_dict.get('valid_clusters', [])
        
        view_data = {
            'blobs': cluster_dict.get('blobs', []),
            'deploy_count': sess.deploy_count,
            'total_distance': sess.total_distance,
            'df': df,
            'eps': sess.eps or 50.0,
            'min_samples': sess.min_samples or 2,
            'min_cluster_size': sess.min_cluster_size or 2,
            'hide_no_deploy': sess.hide_no_deploy,
            'labeled_deploy_df': labeled_deploy_df,
            'valid_clusters': valid_clusters
        }
        
        # Load historical data
        gps_logs = GPSLog.query.filter_by(session_id=sid).order_by(GPSLog.timestamp).all()
        deployments = Deployment.query.filter_by(session_id=sid).all()
        
        # Generate map similar to dashboard, but static
        df_view = view_data['df'].copy()
        if view_data.get('hide_no_deploy', True):
            df_view = df_view[df_view['patch_decision'] != 0].copy()
        centre_lat = df_view['patch_lat'].mean() if not df_view.empty else 0
        centre_lon = df_view['patch_lon'].mean() if not df_view.empty else 0
        
        # Prepare points data for JS
        points_data = []
        def get_colour(decision):
            if decision == 0: return 'red'
            if decision == 1: return 'green'
            if decision == 2: return 'yellow'
            return 'blue'
        
        for _, row in df_view.iterrows():
            popup_escaped = str(row.get('patch_id', '')).replace("'", "\\'") + r"<br>Decision: " + str(row.get('patch_decision', '')).replace("'", "\\'") + r"<br>Depth: " + str(row.get('ping_depth', '')).replace("'", "\\'")
            points_data.append({
                'lat': row['patch_lat'],
                'lon': row['patch_lon'],
                'popup': popup_escaped,
                'color': get_colour(row['patch_decision'])
            })
        
        # Clusters
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        cluster_info = []
        cluster_layers_js = []
        for i, cid in enumerate(valid_clusters):
            color = colors[i % len(colors)]
            cluster_pts = labeled_deploy_df[labeled_deploy_df['cluster'] == cid][['patch_lat', 'patch_lon']].values
            size = len(cluster_pts)
            if size > 0:
                center_lat, center_lon = cluster_pts.mean(axis=0)
                if size >= 3:
                    hull_input = cluster_pts[:, [1, 0]]  # lon, lat
                    hull = ConvexHull(hull_input)
                    vertices = cluster_pts[hull.vertices]
                    hull_pts = [Point(v[1], v[0]) for v in vertices]  # lat, lon
                    hull_poly = unary_union(hull_pts).convex_hull
                    gdf = gpd.GeoDataFrame({'geometry': [hull_poly]}, crs='EPSG:4326')
                    geojson = gdf.to_json()
                    geojson_escaped = geojson.replace("'", "\\'")
                    cluster_layers_js.append(f"""
                    var cluster{cid} = L.geoJSON(JSON.parse('{geojson_escaped}'), {{
                        style: {{fillColor: '{color}', color: '{color}', weight: 3, fillOpacity: 0.4}},
                        onEachFeature: function(feature, layer) {{
                            layer.bindPopup("Cluster {cid}<br>Size: {size} points");
                        }}
                    }}).addTo(map);
                    """)
                    area_m2 = round(gdf.to_crs('EPSG:3857').area.iloc[0], 2)
                else:
                    r_m = view_data['eps'] / 2
                    area_m2 = round(np.pi * r_m**2, 2)
                    r_deg = view_data['eps'] / 111000 * 57.3
                    cluster_layers_js.append(f"""
                    var cluster{cid} = L.circle([{center_lat}, {center_lon}], {{radius: {r_deg}, color: '{color}', weight: 3, fillColor: '{color}', fillOpacity: 0.4}})
                        .bindPopup("Cluster {cid}<br>Size: {size} points")
                        .addTo(map);
                    """)
                
                cluster_info.append({
                    'cid': cid,
                    'size': size,
                    'center_lat': round(float(center_lat), 6),
                    'center_lon': round(float(center_lon), 6),
                    'area_m2': area_m2,
                    'color': color
                })
        
        # Historical trail
        historical_trail_js = ""
        if gps_logs:
            points_js = chr(10).join([f"[{log.lat}, {log.lon}]," for log in gps_logs])
            historical_trail_js = f"""
            var historicalTrail = L.polyline([
            {points_js}
            ], {{color: 'gray', weight: 2, dashArray: '5,5', opacity: 0.6}}).addTo(map);
            """
        
        # Past deployments
        past_deploys_js = ""
        if deployments:
            dep_markers_js = chr(10).join([f"""
            var depMarker{dep.id} = L.marker([{dep.lat}, {dep.lon}], {{
                icon: L.divIcon({{
                    className: 'deploy-marker',
                    html: '<div style="background: orange; width: 16px; height: 16px; border-radius: 50%; border: 2px solid white;"></div>'
                }})
            }}).bindPopup('Past Deployment<br>Time: {dep.timestamp.strftime("%H:%M:%S")}<br>Dist: {dep.ultrasonic_distance} cm{"<br>Cluster: " + str(dep.cluster_id) if dep.cluster_id else ""}');
            pastDeploysLayer.addLayer(depMarker{dep.id});
            """ for dep in deployments])
            past_deploys_js = f"""
            var pastDeploysLayer = L.layerGroup();
            {dep_markers_js}
            pastDeploysLayer.addTo(map);
            """
        
        # Build static map
        map_html = f"""
        <div id="map" style="height: 60vh; width: 100%;"></div>
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <script>
            var map = L.map('map').setView([{centre_lat}, {centre_lon}], 15);
            L.tileLayer('https://api.maptiler.com/maps/satellite/{{z}}/{{x}}/{{y}}.png?key={API_KEY}', {{
                attribution: '© MapTiler',
                maxZoom: 22
            }}).addTo(map);
            
            // Points layer
            var pointsLayer = L.layerGroup();
            {chr(10).join([f"L.circleMarker([{p['lat']}, {p['lon']}], {{radius: 3, color: '{p['color']}', fill: true, fillOpacity: 0.7}}).bindPopup('{p['popup']}').addTo(pointsLayer);" for p in points_data])}
            pointsLayer.addTo(map);
            
            // Cluster layers
            {chr(10).join(cluster_layers_js)}
            
            // Historical trail
            {historical_trail_js}
            
            // Past deployments
            {past_deploys_js}
            
            window.map = map;
        </script>
        <style>.deploy-marker {{ background: transparent; border: none; }}</style>
        """
        
        # Prepare data for tables
        patches_data = df.to_dict('records')
        gps_data = [{'timestamp': log.timestamp.strftime('%Y-%m-%d %H:%M:%S'), 'lat': log.lat, 'lon': log.lon, 'speed': log.speed, 'depth': log.depth, 'qual': log.qual, 'sats': log.sats, 'hdop': log.hdop} for log in gps_logs]
        deploys_data = [{'timestamp': dep.timestamp.strftime('%Y-%m-%d %H:%M:%S'), 'lat': dep.lat, 'lon': dep.lon, 'ultrasonic_distance': dep.ultrasonic_distance, 'cluster_id': dep.cluster_id} for dep in deployments]
        
        # Stats
        mission_time = str(timedelta(seconds=int((datetime.now() - sess.start_time).total_seconds())))
        final_stats = {
            'mission_time': mission_time,
            'deploy_count': sess.deploy_count,
            'total_distance': f"{sess.total_distance:.2f} km",
            'total_patches': sess.total_patches,
            'blobs': len(cluster_info)
        }
        
        return render_template('summary.html', map_html=map_html, stats=final_stats, cluster_info=cluster_info, 
                               patches_data=patches_data, gps_data=gps_data, deploys_data=deploys_data, sid=sid)

@app.route('/dashboard')
def dashboard():
    if 'session_id' not in session:
        return redirect('/new')
    
    session_id = session['session_id']
    sid = int(session_id)
    sess = db.session.get(ReefSession, sid)
    if sess is None:
        return redirect('/new')
    
    data = clusters_data.get(session_id, {})
    if not data:
        return redirect('/new')
    
    # Sync from DB if needed
    data['deploy_count'] = sess.deploy_count
    data['total_distance'] = sess.total_distance
    
    # Generate map as Leaflet HTML/JS string (no Folium)
    df = data['df']
    if data.get('hide_no_deploy', True):
        df = df[df['patch_decision'] != 0].copy()  # Filter out no-deploy points
    centre_lat = df['centre_lat'].mean() if 'centre_lat' in df else df['center_lat'].mean()
    centre_lon = df['centre_lon'].mean() if 'centre_lon' in df else df['center_lon'].mean()
    
    eps = data.get('eps', 50.0)
    min_samples = data.get('min_samples', 2)
    min_cluster_size = data.get('min_cluster_size', 2)
    
    # Prepare points data for JS
    points_data = []
    def get_colour(decision):
        if decision == 0: return 'red'      # Don't deploy
        if decision == 1: return 'green'    # Deploy (now green)
        if decision == 2: return 'yellow'   # Coral (now yellow)
        return 'blue'
    
    for _, row in df.iterrows():
        popup_escaped = str(row['patch_id']).replace("'", "\\'") + r"<br>Decision: " + str(row['patch_decision']).replace("'", "\\'") + r"<br>Depth: " + str(row['ping_depth']).replace("'", "\\'")
        points_data.append({
            'lat': row['patch_lat'],
            'lon': row['patch_lon'],
            'popup': popup_escaped,
            'color': get_colour(row['patch_decision'])
        })
    
    # OPTIMIZED: Use stored labeled data (no DBSCAN refit)
    labeled_deploy_df = data['labeled_deploy_df']
    valid_clusters = data['valid_clusters']
    
    # Predefined colors for clusters
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Prepare cluster_info for template
    cluster_info = []
    cluster_layers_js = []  # JS to add cluster layers
    for i, cid in enumerate(valid_clusters):
        color = colors[i % len(colors)]
        cluster_pts = labeled_deploy_df[labeled_deploy_df['cluster'] == cid][['patch_lat', 'patch_lon']].values
        size = len(cluster_pts)
        if size > 0:
            center_lat, center_lon = cluster_pts.mean(axis=0)
            # Calc area
            if size >= 3:
                hull_input = cluster_pts[:, [1, 0]]  # lon, lat
                hull = ConvexHull(hull_input)
                vertices = cluster_pts[hull.vertices]
                hull_pts = [Point(v[1], v[0]) for v in vertices]  # lat, lon
                hull_poly = unary_union(hull_pts).convex_hull
                gdf = gpd.GeoDataFrame({'geometry': [hull_poly]}, crs='EPSG:4326')
                # GeoJSON for polygon
                geojson = gdf.to_json()
                geojson_escaped = geojson.replace("'", "\\'")
                cluster_layers_js.append(f"""
                var cluster{cid} = L.geoJSON(JSON.parse('{geojson_escaped}'), {{
                    style: {{fillColor: '{color}', color: '{color}', weight: 3, fillOpacity: 0.4}},
                    onEachFeature: function(feature, layer) {{
                        layer.bindPopup("Cluster {cid}<br>Size: {size} points");
                    }}
                }}).addTo(map);
                window.overlayLayers['Cluster {cid}'] = cluster{cid};
                """)
                area_m2 = round(gdf.to_crs('EPSG:3857').area.iloc[0], 2)
            else:
                # Approx circle area for small clusters
                r_m = eps / 2
                area_m2 = round(np.pi * r_m**2, 2)
                # Circle for small
                r_deg = eps / 111000 * 57.3  # Approx degrees to meters conversion
                cluster_layers_js.append(f"""
                var cluster{cid} = L.circle([{center_lat}, {center_lon}], {{radius: {r_deg}, color: '{color}', weight: 3, fillColor: '{color}', fillOpacity: 0.4}})
                    .bindPopup("Cluster {cid}<br>Size: {size} points")
                    .addTo(map);
                window.overlayLayers['Cluster {cid}'] = cluster{cid};
                """)
            
            cluster_info.append({
                'cid': cid,
                'size': size,
                'center_lat': round(float(center_lat), 6),
                'center_lon': round(float(center_lon), 6),
                'area_m2': area_m2,
                'color': color
            })
    
    # Load historical data for map
    gps_logs = GPSLog.query.filter_by(session_id=sid).order_by(GPSLog.timestamp).all()
    deployments = Deployment.query.filter_by(session_id=sid).all()
    
    # Build conditional JS parts
    historical_trail_js = ""
    if gps_logs:
        points_js = chr(10).join([f"[{log.lat}, {log.lon}]," for log in gps_logs])
        historical_trail_js = f"""
        var historicalTrail = L.polyline([
        {points_js}
        ], {{color: 'gray', weight: 2, dashArray: '5,5', opacity: 0.6}}).addTo(map);
        window.overlayLayers['Historical Trail'] = historicalTrail;
        """
    
    past_deploys_js = ""
    if deployments:
        dep_markers_js = chr(10).join([f"""
        var depMarker{dep.id} = L.marker([{dep.lat}, {dep.lon}], {{
            icon: L.divIcon({{
                className: 'deploy-marker',
                html: '<div style="background: orange; width: 16px; height: 16px; border-radius: 50%; border: 2px solid white;"></div>'
            }})
        }}).bindPopup('Past Deployment<br>Time: {dep.timestamp.strftime("%H:%M:%S")}<br>Dist: {dep.ultrasonic_distance} cm{"<br>Cluster: " + str(dep.cluster_id) if dep.cluster_id else ""}');
        pastDeploysLayer.addLayer(depMarker{dep.id});
        """ for dep in deployments])
        past_deploys_js = f"""
        var pastDeploysLayer = L.layerGroup();
        {dep_markers_js}
        pastDeploysLayer.addTo(map);
        window.overlayLayers['Past Deployments'] = pastDeploysLayer;
        """
    
    # Build Leaflet map HTML/JS
    map_html = f"""
    <div id="map" style="height: 80vh; width: 100%;"></div>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        var map = L.map('map').setView([{centre_lat}, {centre_lon}], 15);
        L.tileLayer('https://api.maptiler.com/maps/satellite/{{z}}/{{x}}/{{y}}.png?key={API_KEY}', {{
            attribution: '© MapTiler',
            maxZoom: 22
        }}).addTo(map);
        
        // Points layer
        var pointsLayer = L.layerGroup();
        {chr(10).join([f"L.circleMarker([{p['lat']}, {p['lon']}], {{radius: 3, color: '{p['color']}', fill: true, fillOpacity: 0.7}}).bindPopup('{p['popup']}').addTo(pointsLayer);" for p in points_data])}
        pointsLayer.addTo(map);
        window.overlayLayers = {{'Points': pointsLayer}};
        
        // Cluster layers
        {chr(10).join(cluster_layers_js)}
        
        // Historical trail
        {historical_trail_js}
        
        // Past deployments
        {past_deploys_js}
        
        // GPS marker (direct Leaflet marker)
        var gpsMarker = L.marker([{centre_lat}, {centre_lon}], {{
            icon: L.divIcon({{
                className: 'gps-marker',
                html: '<div style="background: blue; width: 12px; height: 12px; border-radius: 50%; border: 2px solid white; box-shadow: 0 0 5px rgba(0,0,255,0.5);"></div>'
            }})
        }}).bindPopup('Current GPS Position').addTo(map);
        
        // GPS layer for control
        var gpsLayer = L.layerGroup([gpsMarker]);
        gpsLayer.addTo(map);
        window.overlayLayers['GPS'] = gpsLayer;
        
        window.map = map;
        window.gpsMarker = gpsMarker;
        console.log('Map assigned to window.map');
        console.log('GPS Marker assigned to window.gpsMarker');
    </script>
    <style>.gps-marker {{ background: transparent; border: none; }} .deploy-marker {{ background: transparent; border: none; }}</style>
    """
    
    # Stats for dashboard
    start_time = data['start_time']
    mission_time = str(timedelta(seconds=int(time.time() - start_time)))
    ultrasonic_distance = get_ultrasonic_distance()
    stats = {
        'mission_time': mission_time,
        'deploy_count': data['deploy_count'],
        'total': sess.total_patches,
        'blobs': len(cluster_info),
        'ultrasonic_distance': ultrasonic_distance
    }
    return render_template('dashboard.html', map_html=map_html, stats=stats, cluster_info=cluster_info, gps_logs=gps_logs, deployments=deployments, sid=sid)

@app.route('/history')
def history():
    sessions_query = ReefSession.query.order_by(ReefSession.start_time.desc()).all()
    sessions = []
    for s in sessions_query:
        if s.status == 'completed':
            duration_secs = (s.end_time - s.start_time).total_seconds()
            duration = f"{duration_secs / 3600:.1f}h"
            status = 'Completed'
        else:
            duration = 'Ongoing' if s.status == 'ongoing' else 'Paused'
            status = 'Ongoing' if s.status == 'ongoing' else 'Paused'
        sessions.append({
            'id': s.id,
            'start_time': s.start_time.strftime('%Y-%m-%d %H:%M'),
            'duration': duration,
            'deploy_count': s.deploy_count,
            'total_distance': f"{s.total_distance:.2f} km" if s.total_distance else "0 km",
            'status': status
        })
    return render_template('history.html', sessions=sessions)

@app.route('/help')
def help_page():
    return render_template('help.html')

@app.route('/api/update_gps', methods=['POST'])
def update_gps():
    try:
        if 'session_id' not in session:
            return jsonify({'error': 'No session'}), 401
        
        data = request.json
        if not data:
            return jsonify({'error': 'No JSON data'}), 400
        
        lat, lon = data['lat'], data['lon']
        session_id = session['session_id']
        sess_data = clusters_data.get(session_id)
        if not sess_data:
            return jsonify({'error': 'Session data not found'}), 404
        
        # Calc distance to nearest blob
        min_dist = float('inf')
        for blob in sess_data['blobs']:
            dist = haversine_dist(lat, lon, blob['lat'], blob['lon'])
            min_dist = min(min_dist, dist)
        
        in_deploy_zone = min_dist <= sess_data.get('eps', 50.0)
        
        # Update distance travelled
        if sess_data['prev_pos']:
            prev_lat, prev_lon = sess_data['prev_pos']
            dist_delta = haversine_dist(lat, lon, prev_lat, prev_lon) / 1000  # km
            sess_data['total_distance'] += dist_delta
        sess_data['prev_pos'] = (lat, lon)
        
        # Placeholder deploy (ultrasonic later) - but since button removed, this won't trigger unless from elsewhere
        if in_deploy_zone and data.get('deployed'):
            sess_data['deploy_count'] += 1
        
        speed = data.get('speed', 0)  # From GPS
        
        return jsonify({
            'status': 'DEPLOY' if in_deploy_zone else "DON'T DEPLOY",
            'color': 'green' if in_deploy_zone else 'red',
            'min_dist': min_dist,
            'total_distance': sess_data['total_distance'],
            'deploy_count': sess_data['deploy_count'],
            'speed': speed,
            'depth': 20.0  # Placeholder avg
        })
    except Exception as e:
        print(f'API Error: {e}')  # Log to console
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/mission_time')
def mission_time():
    try:
        session_id = session.get('session_id')
        if not session_id or session_id not in clusters_data:
            return jsonify({'time': '00:00:00'})
        start_time = clusters_data[session_id]['start_time']
        mission_time = str(timedelta(seconds=int(time.time() - start_time)))
        return jsonify({'time': mission_time})
    except Exception as e:
        print(f'Time API Error: {e}')  # Log to console
        return jsonify({'time': '00:00:00'})

# Global error handler for JSON APIs
@app.errorhandler(404)
@app.errorhandler(500)
def handle_error(error):
    response = jsonify({'error': str(error)})
    response.status_code = error.code if hasattr(error, 'code') else 500
    return response

# --- GPS setup ---
SERIAL_PORT = '/dev/ttyUSB0'  # Update if needed
BAUD_RATE = 115200
TIMEOUT = 1
MIN_QUAL = 2  # >=2: DGPS/RTK

gps_lat = None
gps_lon = None
current_speed = 0.0
latest_gps = None

def gps_monitor():
    global gps_lat, gps_lon, current_speed, latest_gps, CURRENT_IN_ZONE, CURRENT_SESSION_ID
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)
    except Exception as e:
        print(f"GPS serial error: {e}")
        return
    print("Starting GPS monitor thread...")
    while True:
        try:
            raw_line = ser.readline()
            if raw_line:
                try:
                    line = raw_line.decode('ascii', errors='replace').strip()
                    if line.startswith('$GPVTG'):
                        msg = pynmea2.parse(line)
                        if hasattr(msg, 'spd_over_grnd_kmh') and msg.spd_over_grnd_kmh:
                            current_speed = float(msg.spd_over_grnd_kmh)
                    elif line.startswith('$GNGGA') or line.startswith('$GGA'):
                        msg = pynmea2.parse(line)
                        if isinstance(msg, pynmea2.GGA):
                            qual = int(msg.gps_qual)
                            if qual >= MIN_QUAL and msg.latitude and msg.longitude:
                                gps_lat = msg.latitude
                                gps_lon = msg.longitude
                                hdop = float(msg.horizontal_dil) if msg.horizontal_dil else None
                                sats = int(msg.num_sats) if msg.num_sats else 0
                                emit_data = {
                                    'lat': gps_lat,
                                    'lon': gps_lon,
                                    'qual': qual,
                                    'sats': sats,
                                    'hdop': hdop,
                                    'speed': current_speed
                                }
                                if CURRENT_SESSION_ID and CURRENT_SESSION_ID in clusters_data:
                                    sess_data = clusters_data[CURRENT_SESSION_ID]
                                    # Calc distance to nearest blob
                                    min_dist = float('inf')
                                    for blob in sess_data['blobs']:
                                        dist = haversine_dist(gps_lat, gps_lon, blob['lat'], blob['lon'])
                                        min_dist = min(min_dist, dist)
                                    
                                    in_deploy_zone = min_dist <= sess_data.get('eps', 50.0)
                                    
                                    # Update distance travelled
                                    if sess_data['prev_pos']:
                                        prev_lat, prev_lon = sess_data['prev_pos']
                                        dist_delta = haversine_dist(gps_lat, gps_lon, prev_lat, prev_lon) / 1000  # km
                                        sess_data['total_distance'] += dist_delta
                                    sess_data['prev_pos'] = (gps_lat, gps_lon)
                                    
                                    CURRENT_IN_ZONE = in_deploy_zone
                                    
                                    # Log to DB
                                    try:
                                        log = GPSLog(
                                            session_id=int(CURRENT_SESSION_ID),
                                            timestamp=datetime.now(),
                                            lat=float(gps_lat),
                                            lon=float(gps_lon),
                                            speed=current_speed,
                                            depth=20.0,
                                            qual=qual,
                                            sats=sats,
                                            hdop=hdop
                                        )
                                        db.session.add(log)
                                        db_sess = db.session.get(ReefSession, int(CURRENT_SESSION_ID))
                                        db_sess.total_distance = sess_data['total_distance']
                                        db.session.commit()
                                    except Exception as e:
                                        print(f"GPS log error: {e}")
                                    
                                    emit_data.update({
                                        'status': 'DEPLOY' if in_deploy_zone else "DON'T DEPLOY",
                                        'color': 'green' if in_deploy_zone else 'red',
                                        'min_dist': min_dist,
                                        'total_distance': sess_data['total_distance'],
                                        'deploy_count': sess_data['deploy_count'],
                                        'speed': current_speed,
                                        'depth': 20.0
                                    })
                                else:
                                    emit_data.update({
                                        'status': 'NO GPS SESSION',
                                        'color': 'gray',
                                        'min_dist': None,
                                        'total_distance': 0,
                                        'deploy_count': 0,
                                        'speed': current_speed,
                                        'depth': 0
                                    })
                                socketio.emit('gps_position_update', emit_data)
                except Exception as e:
                    print(f"GPS parse error: {e}")
            time.sleep(0.1)
        except Exception as e:
            print(f"GPS read error: {e}")
            time.sleep(1)

# Start GPS monitor thread automatically
import threading
threading.Thread(target=gps_monitor, daemon=True).start()

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        # Migrate: add df_json if missing
        from sqlalchemy import inspect
        inspector = inspect(db.engine)
        if 'reef_session' in inspector.get_table_names():
            columns = [c['name'] for c in inspector.get_columns('reef_session')]
            if 'df_json' not in columns:
                with db.engine.connect() as conn:
                    conn.execute(db.text("ALTER TABLE reef_session ADD COLUMN df_json TEXT"))
                    conn.commit()
                print("Added df_json column to reef_session table.")
            if 'status' not in columns:
                with db.engine.connect() as conn:
                    conn.execute(db.text("ALTER TABLE reef_session ADD COLUMN status VARCHAR(20) DEFAULT 'ongoing'"))
                    conn.execute(db.text("UPDATE reef_session SET status = 'completed' WHERE end_time IS NOT NULL"))
                    conn.execute(db.text("UPDATE reef_session SET status = 'ongoing' WHERE status IS NULL"))
                    conn.commit()
                print("Added status column to reef_session table.")
    try:
        # Set log_output=True to show the server address (and other startup info)
        socketio.run(app, debug=True, log_output=True)
    finally:
        try:
            GPIO.cleanup()
            print("GPIO cleaned up on shutdown.")
        except:
            pass