from flask import Flask, render_template, request, jsonify, session, redirect
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
from api_key import API_KEY
import io
import time
import json
from datetime import datetime, timedelta
from flask_session import Session  # pip install flask-session
from folium import Element
from folium.plugins import MarkerCluster, LocateControl

app = Flask(__name__)
app.secret_key = 'reefscan_secret'  # Change for prod
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global for demo (use DB later)
clusters_data = {}  # {session_id: {'blobs': [...], 'start_time': ..., 'deploy_count': 0, ...}}

def haversine_dist(lat1, lon1, lat2, lon2):
    R = 6371000
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/new', methods=['GET', 'POST'])
def new_session():
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
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
            
            # Clustering (only on decision 2)
            deploy_df = df[df['patch_decision'] == 2].copy()
            if len(deploy_df) > 0:
                coords = deploy_df[['patch_lon', 'patch_lat']].values
                db = DBSCAN(eps=eps / 6371000, min_samples=min_samples, metric='haversine').fit(np.radians(coords))
                deploy_df['cluster'] = db.labels_
                cluster_sizes = deploy_df['cluster'].value_counts()
                valid_clusters = [c for c in cluster_sizes[cluster_sizes >= min_cluster_size].index if c != -1]
                
                # Prep blobs as list of [centre_lat, centre_lon, radius=eps] for JS checks
                blobs = []
                for cid in valid_clusters:
                    cluster_pts = deploy_df[deploy_df['cluster'] == cid][['patch_lat', 'patch_lon']].values
                    centre = cluster_pts.mean(axis=0)
                    blobs.append({'lat': centre[0], 'lon': centre[1], 'radius': eps})  # For distance calc
                
                # Start session
                session_id = str(time.time())
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
                    'hide_no_deploy': hide_no_deploy
                }
                session['session_id'] = session_id
                
                return redirect('/dashboard')
            else:
                return render_template('new.html', error='No deploy (2) points found!')
        except Exception as e:
            return render_template('new.html', error=f'Error: {str(e)}')
    
    return render_template('new.html')

@app.route('/dashboard')
def dashboard():
    if 'session_id' not in session:
        return redirect('/new')
    
    session_id = session['session_id']
    data = clusters_data.get(session_id, {})
    if not data:
        return redirect('/new')
    
    # Generate map (optimized for speed)
    df = data['df']
    if data.get('hide_no_deploy', True):
        df = df[df['patch_decision'] != 0].copy()  # Filter out no-deploy points
    centre_lat = df['centre_lat'].mean() if 'centre_lat' in df else df['center_lat'].mean()
    centre_lon = df['centre_lon'].mean() if 'centre_lon' in df else df['center_lon'].mean()
    
    eps = data.get('eps', 50.0)
    min_samples = data.get('min_samples', 2)
    min_cluster_size = data.get('min_cluster_size', 2)
    tiles = f'https://api.maptiler.com/maps/satellite/{{z}}/{{x}}/{{y}}.png?key={API_KEY}'
    m = folium.Map(
        location=[centre_lat, centre_lon], 
        zoom_start=15,  # Lower zoom for faster initial load
        tiles=tiles, 
        attr='© MapTiler', 
        max_zoom=22,
        prefer_canvas=True  # Render to canvas for speed with many markers
    )
    
    # Add custom style to override Folium's #map with .folium-map for proper sizing
    m.get_root().header.add_child(Element('<style>.folium-map { position: absolute; top: 0; bottom: 0; right: 0; left: 0; height: 100%; width: 100%; }</style>'))
    
    # Show all points directly (no clustering for visibility)
    points_layer = folium.FeatureGroup(name='Points', show=True)
    
    def get_colour(decision):
        if decision == 0: return 'red'
        if decision == 1: return 'yellow'
        if decision == 2: return 'green'
        return 'blue'
    
    for _, row in df.iterrows():
        marker = folium.CircleMarker(
            [row['patch_lat'], row['patch_lon']], radius=3,
            popup=f"Patch {row['patch_id']}<br>Decision: {row['patch_decision']}<br>Depth: {row['ping_depth']}",
            color=get_colour(row['patch_decision']), fill=True, fillOpacity=0.7
        )
        marker.add_to(points_layer)
    
    m.add_child(points_layer)
    
    # Blobs (unchanged)
    deploy_df = data['df'][data['df']['patch_decision'] == 2].copy()  # Use original df for blobs
    coords = deploy_df[['patch_lon', 'patch_lat']].values
    db = DBSCAN(eps=eps / 6371000, min_samples=min_samples, metric='haversine').fit(np.radians(coords))
    deploy_df['cluster'] = db.labels_
    cluster_sizes = deploy_df['cluster'].value_counts()
    valid_clusters = [c for c in cluster_sizes[cluster_sizes >= min_cluster_size].index if c != -1]
    
    clusters_layer = folium.FeatureGroup(name='Clusters (Blobs)', show=True)
    for cid in valid_clusters:
        cluster_pts = deploy_df[deploy_df['cluster'] == cid][['patch_lat', 'patch_lon']].values
        if len(cluster_pts) >= 3:
            hull_input = cluster_pts[:, [1, 0]]
            hull = ConvexHull(hull_input)
            vertices = cluster_pts[hull.vertices]
            hull_pts = [Point(v[1], v[0]) for v in vertices]
            hull_poly = unary_union(hull_pts).convex_hull
            gdf = gpd.GeoDataFrame({'geometry': [hull_poly]}, crs='EPSG:4326')
            folium.GeoJson(
                gdf,
                style_function=lambda x: {'fillColor': 'green', 'color': 'darkgreen', 'weight': 3, 'fillOpacity': 0.4},
                popup=f"Cluster {cid}<br>Size: {len(cluster_pts)} points"
            ).add_to(clusters_layer)
        else:
            centre = cluster_pts.mean(axis=0)
            folium.Circle([centre[0], centre[1]], radius=eps / 50, color='darkgreen', weight=3, fillColor='green', fillOpacity=0.4).add_to(clusters_layer)
    
    m.add_child(clusters_layer)
    folium.LayerControl().add_to(m)
    
    # Add LocateControl for reliable auto geolocation
    locate = LocateControl(
        auto_start=True,  # Start on map load
        strings={'title': 'Show me where I am', 'popup': 'You are within {distance} {unit} from this point'},
        locate_options={
            'watch': True,  # Use watchPosition for continuous updates
            'setView': 'always',  # Auto-pan/zoom on each update
            'enableHighAccuracy': True,
            'timeout': 10000,
            'maximumAge': 0
        },
        keepCurrentZoomLevel=False,
        returnToPrevBounds=False,
        showPopup=True
    )
    locate.add_to(m)
    
    # Setup JS to assign the dynamic map var to window.map
    setup_code = """
    (function() {
        var checkExist = setInterval(function() {
            var mapKeys = Object.keys(window).filter(function(k) { return k.indexOf('map_') === 0; });
            if (mapKeys.length > 0) {
                window.map = window[mapKeys[0]];
                clearInterval(checkExist);
                console.log('Map instance assigned to window.map');
            }
        }, 100);
    })();
    """
    m.get_root().script.add_child(Element(setup_code))
    
    # Custom JS for proximity checks, UI updates, API calls (hook into locationfound event) - now uses window.map
    blobs_json = json.dumps(data['blobs'])
    eps_str = str(eps)
    js_code = f"""
    let blobs = {blobs_json};
    let eps = {eps_str};
    let prevPos = null;
    let totalDistance = 0;
    
    function haversineDist(lat1, lon1, lat2, lon2) {{
        const R = 6371;
        const dlat = (lat2 - lat1) * Math.PI / 180;
        const dlon = (lon2 - lon1) * Math.PI / 180;
        const a = Math.sin(dlat/2) * Math.sin(dlat/2) + Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) * Math.sin(dlon/2) * Math.sin(dlon/2);
        const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
        return R * c * 1000; // meters
    }}
    
    // Wait a bit more for window.map to be set, then attach listeners
    setTimeout(function() {{
        if (window.map) {{
            // Listen to locationfound event from LocateControl (fires on each update)
            window.map.on('locationfound', function(e) {{
                console.log('Location found:', e.latlng.lat, e.latlng.lng);  // Debug log
                const lat = e.latlng.lat;
                const lon = e.latlng.lng;
                const parentDoc = window.parent.document;
                parentDoc.getElementById('gps-pos').textContent = `${{lat.toFixed(6)}}, ${{lon.toFixed(6)}}`;
                
                // Check nearest blob
                let minDist = Infinity;
                blobs.forEach(blob => {{
                    const dist = haversineDist(lat, lon, blob.lat, blob.lon);
                    if (dist < minDist) minDist = dist;
                }});
                
                const inZone = minDist <= eps;
                const statusBox = parentDoc.getElementById('status-box');
                const statusText = parentDoc.getElementById('status-text');
                statusBox.className = `p-4 rounded shadow ${{inZone ? 'status-deploy' : 'status-no-deploy'}}`;
                statusText.textContent = inZone ? 'DEPLOY' : "DON'T DEPLOY";
                
                // Distance
                if (prevPos) {{
                    const delta = haversineDist(lat, lon, prevPos.lat, prevPos.lon) / 1000;
                    totalDistance += delta;
                }}
                prevPos = {{ lat, lon }};
                
                // Speed (placeholder; plugin doesn't provide, use last known or 0)
                parentDoc.getElementById('speed').textContent = '0.0 km/h';  // Update if speed available
                
                // Send to backend
                fetch('/api/update_gps', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{ lat, lon, speed: 0, deployed: false }})
                }}).then(r => {{
                    if (!r.ok) {{
                        throw new Error(`HTTP ${{r.status}}: ${{r.statusText}}`);
                    }}
                    return r.json();
                }}).then(data => {{
                    parentDoc.getElementById('deploy-count').textContent = data.deploy_count;
                    parentDoc.getElementById('depth').textContent = `${{data.depth}} m`;
                    parentDoc.getElementById('distance').textContent = `${{data.total_distance.toFixed(2)}} km`;
                }}).catch(err => {{
                    console.error('API Error:', err);
                    // Fallback to local distance
                    parentDoc.getElementById('distance').textContent = `${{totalDistance.toFixed(2)}} km`;
                }});
                
                // Optional: Add custom trail (plugin has its own marker)
                if (!window.gpsTrail) {{
                    window.gpsTrail = L.polyline([], {{color: '#3388ff', weight: 4, opacity: 0.8}}).addTo(window.map);
                }}
                window.gpsTrail.addLatLng([lat, lon]);
            }});
            
            // Error handler for geolocation (e.g., permission denied)
            window.map.on('locationerror', function(e) {{
                console.error('Location error:', e.message);
                window.parent.document.getElementById('gps-pos').textContent = 'Location Error: ' + e.message;
            }});
            
            console.log('Custom event listeners attached to map');
        }} else {{
            console.error('window.map not found!');
        }}
    }}, 1000);  // Increased delay to ensure setup_code has run and DOM is ready
    
    // Mission time update (moved outside timeout for immediate start)
    setInterval(() => {{
        fetch('/api/mission_time').then(r => {{
            if (!r.ok) {{
                throw new Error(`HTTP ${{r.status}}: ${{r.statusText}}`);
            }}
            return r.json();
        }}).then(data => {{
            window.parent.document.getElementById('mission-time').textContent = data.time;
        }}).catch(err => {{
            console.error('Time API Error:', err);
        }});
    }}, 1000);
    """
    
    m.get_root().script.add_child(Element(js_code))
    
    map_html = m._repr_html_()
    
    # Stats for dashboard
    start_time = data['start_time']
    mission_time = str(timedelta(seconds=int(time.time() - start_time)))
    stats = {
        'mission_time': mission_time,
        'deploy_count': data['deploy_count'],
        'total': len(df),
        'blobs': len(data['blobs'])
    }
    
    return render_template('dashboard.html', map_html=map_html, stats=stats)

@app.route('/end_session')
def end_session():
    if 'session_id' in session:
        session_id = session['session_id']
        data = clusters_data.pop(session_id, None)
        session.pop('session_id')
        if data:
            final_stats = {
                'mission_time': str(timedelta(seconds=int(time.time() - data['start_time']))),
                'deploy_count': data['deploy_count'],
                'total_distance': f"{data['total_distance']:.2f} km",
                'total_patches': len(data['df']),
                'blobs': len(data['blobs'])
            }
            return render_template('summary.html', stats=final_stats)
    return redirect('/')

@app.route('/history')
def history():
    # Placeholder: List past sessions
    return render_template('history.html', sessions=list(clusters_data.keys()))  # Demo keys

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

if __name__ == '__main__':
    app.run(debug=True)