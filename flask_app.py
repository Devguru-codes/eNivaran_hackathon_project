import os
import tempfile
import base64
import sqlite3
import io
import datetime
import json
from flask import Flask, request, render_template, jsonify, send_from_directory, session, redirect, url_for, flash
from flask.json.provider import DefaultJSONProvider
import logging
from flask.logging import default_handler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('eNivaran')
logger.addHandler(default_handler)

def handle_json_error(e):
    """Handle JSON serialization errors"""
    logger.error(f"JSON Serialization Error: {str(e)}")
    return jsonify({
        'error': 'Internal server error occurred while processing the request.',
        'details': str(e) if app.debug else None
    }), 500

def handle_value_error(e):
    """Handle data type conversion errors"""
    logger.error(f"Value Error: {str(e)}")
    return jsonify({
        'error': 'Invalid data format in request.',
        'details': str(e) if app.debug else None
    }), 400

def handle_key_error(e):
    """Handle missing key errors"""
    logger.error(f"Key Error: {str(e)}")
    return jsonify({
        'error': 'Required data missing from request.',
        'details': str(e) if app.debug else None
    }), 400

def handle_sqlite_error(e):
    """Handle database errors"""
    logger.error(f"Database Error: {str(e)}")
    return jsonify({
        'error': 'Database operation failed.',
        'details': str(e) if app.debug else None
    }), 500

class CustomJSONEncoder(DefaultJSONProvider):
    def __init__(self, app):
        super().__init__(app)
        self.options = {
            'ensure_ascii': False,
            'sort_keys': False,
            'compact': True
        }

    def default(self, obj):
        try:
            if isinstance(obj, datetime.datetime):
                return obj.isoformat()
            if isinstance(obj, sqlite3.Row):
                return dict(obj)
            if isinstance(obj, bytes):
                return base64.b64encode(obj).decode('utf-8')
            return super().default(obj)
        except Exception as e:
            print(f"JSON encoding error: {e}")
            return None
        
    def dumps(self, obj, **kwargs):
        def convert(o):
            if isinstance(o, datetime.datetime):
                return o.isoformat()
            elif isinstance(o, sqlite3.Row):
                return dict(o)
            elif isinstance(o, bytes):
                return base64.b64encode(o).decode('utf-8')
            elif isinstance(o, dict):
                return {k: convert(v) for k, v in o.items()}
            elif isinstance(o, (list, tuple)):
                return [convert(v) for v in o]
            return o

        try:
            return json.dumps(convert(obj), ensure_ascii=False, **kwargs)
        except Exception as e:
            print(f"JSON dumps error: {e}")
            return json.dumps(None)

    def loads(self, s, **kwargs):
        try:
            return json.loads(s, **kwargs)
        except Exception as e:
            print(f"JSON loads error: {e}")
            return None

from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderServiceError

# Import the pothole detection function from the existing file
from pothole_detection import run_pothole_detection
from duplication_detection_code import get_duplicate_detector

app = Flask(__name__)
app.config['SECRET_KEY'] = 'dev-secret-key-replace-later'

# Add Jinja2 filter for base64 encoding
def b64encode_filter(data):
    """Jinja2 filter to base64 encode binary data."""
    if data is None:
        return None
    return base64.b64encode(data).decode('utf-8')

app.jinja_env.filters['b64encode'] = b64encode_filter

# Configure JSON provider
app.json = CustomJSONEncoder(app)

# --- Application Configuration ---
BASE_DIR = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config.update(
    UPLOAD_FOLDER=UPLOAD_FOLDER,
    SECRET_KEY='dev-secret-key-replace-later',  # Move this to environment variable in production
    MAX_CONTENT_LENGTH=16 * 1024 * 1024  # Limit file size to 16MB
)

# Register error handlers
app.register_error_handler(json.JSONDecodeError, handle_json_error)
app.register_error_handler(ValueError, handle_value_error)
app.register_error_handler(KeyError, handle_key_error)
app.register_error_handler(sqlite3.Error, handle_sqlite_error)

# Add generic error handlers
@app.errorhandler(404)
def not_found_error(error):
    if request.is_json:
        return jsonify({'error': 'Resource not found'}), 404
    flash('The requested page was not found.', 'error')
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f'Server Error: {error}')
    if request.is_json:
        return jsonify({'error': 'An internal server error occurred'}), 500
    flash('An unexpected error has occurred.', 'error')
    return render_template('index.html'), 500

@app.errorhandler(Exception)
def unhandled_exception(e):
    app.logger.error(f'Unhandled Exception: {str(e)}')
    if request.is_json:
        return jsonify({
            'error': 'An unexpected error occurred',
            'details': str(e) if app.debug else None
        }), 500
    flash('An unexpected error has occurred.', 'error')
    return render_template('index.html'), 500

# Configure file logging
if not app.debug:
    import logging.handlers
    file_handler = logging.handlers.RotatingFileHandler(
        'enivaran.log',
        maxBytes=1024 * 1024,  # 1 MB
        backupCount=10
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('eNivaran startup')

# --- Database Configuration ---
APP_DB = os.path.join(BASE_DIR, 'enivaran.db')

def dict_factory(cursor, row):
    """Convert SQLite Row to dictionary with proper datetime handling"""
    d = {}
    for idx, col in enumerate(cursor.description):
        value = row[idx]
        if isinstance(value, str) and col[0] in ['submitted_at', 'detected_at', 'created_at', 'last_updated']:
            try:
                # Try parsing with microseconds
                value = datetime.datetime.strptime(value, '%Y-%m-%d %H:%M:%S.%f')
            except ValueError:
                try:
                    # Try parsing without microseconds
                    value = datetime.datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    value = datetime.datetime.now()
        d[col[0]] = value
    return d

def get_coordinates_from_address(street, city, state, zipcode):
    geolocator = Nominatim(user_agent="eNivaran-app")
    address = f"{street}, {city}, {state}, {zipcode}, India"
    try:
        location = geolocator.geocode(address)
        return (location.latitude, location.longitude) if location else (None, None)
    except GeocoderServiceError:
        return None, None

from functools import wraps

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/tools')
@login_required
def tools():
    return render_template('tools.html')

# --- Database Setup ---
APP_DB = os.path.join(os.path.dirname(__file__), 'enivaran.db')

def init_database():
    with sqlite3.connect(APP_DB) as conn:
        conn.execute('PRAGMA foreign_keys = ON')  # Enable foreign key support
        c = conn.cursor()
        
        # Create users table first (for foreign key relationships)
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                full_name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )''')
            
        # Create complaints table with foreign key to users
        c.execute('''
            CREATE TABLE IF NOT EXISTS complaints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT,
                location_lat REAL,
                location_lon REAL,
                issue_type TEXT,
                image BLOB,
                image_filename TEXT,
                submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_duplicate INTEGER DEFAULT 0,
                original_report_id INTEGER,
                user_id INTEGER,
                status TEXT DEFAULT 'Submitted',
                upvotes INTEGER DEFAULT 0,
                remarks TEXT DEFAULT 'Complaint sent for supervision.',
                FOREIGN KEY (user_id) REFERENCES users (id),
                FOREIGN KEY (original_report_id) REFERENCES complaints (id)
            )''')
            
        # Create pothole detections table
        c.execute('''
            CREATE TABLE IF NOT EXISTS pothole_detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                input_image BLOB,
                input_filename TEXT,
                detection_result TEXT,
                annotated_image BLOB,
                detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_id INTEGER,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )''')
            
        # Create pothole stats table
        c.execute('''
            CREATE TABLE IF NOT EXISTS pothole_stats (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                total_potholes INTEGER DEFAULT 0,
                high_priority_count INTEGER DEFAULT 0,
                medium_priority_count INTEGER DEFAULT 0,
                low_priority_count INTEGER DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )''')
            
        # Initialize pothole stats if empty
        if c.execute('SELECT COUNT(*) FROM pothole_stats').fetchone()[0] == 0:
            c.execute('INSERT INTO pothole_stats (id) VALUES (1)')
            
        # Create indexes for better performance
        c.execute('CREATE INDEX IF NOT EXISTS idx_complaints_user_id ON complaints(user_id)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_complaints_submitted_at ON complaints(submitted_at)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)')
        
        conn.commit()

# --- Initialize Application ---
def init_app():
    # Initialize the database
    init_database()
    
    # Create required directories
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Enable foreign key support for all connections
    @app.before_request
    def enable_foreign_keys():
        if request.endpoint != 'static_files':  # Skip for static files
            conn = sqlite3.connect(APP_DB)
            conn.execute('PRAGMA foreign_keys = ON')
            conn.close()

# Initialize the application
init_app()


@app.route('/detect_pothole', methods=['POST'])
@login_required
def detect_pothole():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    result_json, annotated_image_bytes = run_pothole_detection(file_path)
    os.remove(file_path)

    if result_json is None:
        return jsonify({'error': 'Detection failed'}), 500

    annotated_image_b64 = base64.b64encode(annotated_image_bytes).decode('utf-8')
    return jsonify({'result': result_json, 'annotated_image_b64': annotated_image_b64})

@app.route('/static/<path:filename>')
def static_files(filename):
    # Added to serve the illustration image
    return send_from_directory('static', filename)

# --- Authentication Routes ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        login_type = request.form.get('loginType', 'user')
        
        if login_type == 'admin':
            if username == 'admin001' and password == 'admin$001':
                session['user_id'] = 'admin'
                session['username'] = 'admin001'
                session['is_admin'] = True
                flash('Welcome back, Administrator!', 'success')
                return redirect(url_for('admin_dashboard'))
            else:
                flash('Invalid admin credentials.', 'error')
                return render_template('login.html')
        else:
            with sqlite3.connect(APP_DB) as conn:
                conn.row_factory = dict_factory
                user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
                
            if user and check_password_hash(user['password_hash'], password):
                session['user_id'] = user['id']
                session['username'] = user['username']
                session['is_admin'] = False
                flash(f'Welcome back, {user["full_name"]}!', 'success')
                return redirect(url_for('index'))
            else:
                flash('Invalid username or password.', 'error')
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if 'user_id' in session:
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form['username']
        full_name = request.form['full_name']
        password = request.form['password']
        if not all([username, full_name, password]):
            flash('All fields are required.', 'error')
            return redirect(url_for('signup'))
        with sqlite3.connect(APP_DB) as conn:
            if conn.execute('SELECT id FROM users WHERE username = ?', (username,)).fetchone():
                flash('Username already exists.', 'error')
                return redirect(url_for('signup'))
            password_hash = generate_password_hash(password)
            conn.execute('INSERT INTO users (username, full_name, password_hash) VALUES (?, ?, ?)',
                         (username, full_name, password_hash))
            conn.commit()
        flash('Signup successful! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/logout')
def logout():
    if 'user_id' not in session:
        return redirect(url_for('login'))
        
    # Store username temporarily for the message
    username = session.get('username', 'User')
    is_admin = session.get('is_admin', False)
    
    # Clear all session data
    session.clear()
    
    # Flash appropriate goodbye message
    if is_admin:
        flash('Administrator logged out successfully.', 'success')
    else:
        flash(f'Goodbye, {username}! You have been logged out successfully.', 'success')
    
    # Redirect to login page
    return redirect(url_for('login'))

# --- Admin Routes ---
@app.route('/admin')
@login_required
def admin_dashboard():
    if not session.get('is_admin'):
        return redirect(url_for('index'))
    with sqlite3.connect(APP_DB) as conn:
        conn.row_factory = dict_factory
        complaints_raw = conn.execute('''
            SELECT 
                c.id, 
                c.text, 
                CAST(c.location_lat AS FLOAT) as location_lat,
                CAST(c.location_lon AS FLOAT) as location_lon,
                c.issue_type,
                c.image,
                c.submitted_at,
                c.status,
                c.upvotes,
                c.remarks,
                c.is_duplicate,
                c.original_report_id,
                u.username,
                u.full_name as reporter_name,
                c.user_id
            FROM complaints c
            LEFT JOIN users u ON c.user_id = u.id
            ORDER BY c.submitted_at DESC
        ''').fetchall()
        
        # Process and validate each complaint
        processed_complaints = []
        for complaint in complaints_raw:
            try:
                # Create a clean dictionary with basic complaint info
                comp_dict = {
                    'id': int(complaint['id']),
                    'text': str(complaint['text'] or ''),
                    'location_lat': float(complaint['location_lat'] or 0),
                    'location_lon': float(complaint['location_lon'] or 0),
                    'issue_type': str(complaint['issue_type'] or ''),
                    'status': str(complaint['status'] or 'Submitted'),
                    'upvotes': int(complaint['upvotes'] or 0),
                    'remarks': str(complaint['remarks'] or ''),
                    'username': str(complaint['username'] or ''),
                    'reporter_name': str(complaint['reporter_name'] or ''),
                    'is_duplicate': bool(complaint.get('is_duplicate')),
                    'original_report_id': int(complaint['original_report_id']) if complaint.get('original_report_id') else None,
                    'user_id': int(complaint['user_id'])
                }

                # Handle datetime
                if isinstance(complaint['submitted_at'], str):
                    try:
                        submitted_at = datetime.datetime.strptime(complaint['submitted_at'], '%Y-%m-%d %H:%M:%S.%f')
                    except ValueError:
                        try:
                            submitted_at = datetime.datetime.strptime(complaint['submitted_at'], '%Y-%m-%d %H:%M:%S')
                        except ValueError:
                            submitted_at = datetime.datetime.now()
                else:
                    submitted_at = complaint['submitted_at'] or datetime.datetime.now()

                comp_dict['submitted_at'] = submitted_at

                # Handle image data
                if complaint.get('image'):
                    try:
                        comp_dict['image'] = base64.b64encode(complaint['image']).decode('utf-8')
                    except:
                        comp_dict['image'] = None
                else:
                    comp_dict['image'] = None

                processed_complaints.append(comp_dict)
            except Exception as e:
                app.logger.error(f"Error processing complaint {complaint.get('id', 'unknown')}: {str(e)}")
                continue
        
    return render_template('admin_dashboard.html', complaints=processed_complaints)

@app.route('/update_complaint_status/<int:complaint_id>', methods=['POST'])
@login_required
def update_complaint_status(complaint_id):
    if not session.get('is_admin'):
        return redirect(url_for('index'))
    status = request.form.get('status')
    remarks = request.form.get('remarks')
    if not status or not remarks:
        flash('Status and remarks are required.', 'error')
        return redirect(url_for('admin_dashboard'))
    with sqlite3.connect(APP_DB) as conn:
        conn.execute('PRAGMA foreign_keys = ON')
        conn.execute('UPDATE complaints SET status = ?, remarks = ? WHERE id = ?', 
                    (status, remarks, complaint_id))
        conn.commit()
    flash('Complaint status updated successfully.', 'success')
    return redirect(url_for('admin_dashboard'))

# --- Public & User Complaint Routes ---
@app.route('/pothole_stats')
def pothole_stats():
    """Return pothole statistics"""
    with sqlite3.connect(APP_DB) as conn:
        conn.row_factory = dict_factory
        result = conn.execute('''
            SELECT 
                CAST(total_potholes AS INTEGER) as total_potholes,
                CAST(high_priority_count AS INTEGER) as high_priority_count,
                CAST(medium_priority_count AS INTEGER) as medium_priority_count,
                CAST(low_priority_count AS INTEGER) as low_priority_count,
                last_updated
            FROM pothole_stats 
            WHERE id = 1
        ''').fetchone()

        if result:
            try:
                processed = {
                    'total_potholes': int(result['total_potholes'] or 0),
                    'high_priority_count': int(result['high_priority_count'] or 0),
                    'medium_priority_count': int(result['medium_priority_count'] or 0),
                    'low_priority_count': int(result['low_priority_count'] or 0),
                    'last_updated': result['last_updated'].isoformat() if result['last_updated'] else datetime.datetime.now().isoformat()
                }
                return jsonify(processed)
            except Exception as e:
                app.logger.error(f"Error processing pothole stats: {str(e)}")
    
    # Return default values if no stats or error
    return jsonify({
        'total_potholes': 0,
        'high_priority_count': 0,
        'medium_priority_count': 0,
        'low_priority_count': 0,
        'last_updated': datetime.datetime.now().isoformat()
    })

@app.route('/complaints')
@login_required
def view_complaints():
    sort_by = request.args.get('sort', 'time_desc')
    order_clause = "ORDER BY submitted_at DESC"
    if sort_by == 'upvotes_desc':
        order_clause = "ORDER BY upvotes DESC, submitted_at DESC"
    elif sort_by == 'time_asc':
        order_clause = "ORDER BY submitted_at ASC"
    
    with sqlite3.connect(APP_DB) as conn:
        conn.row_factory = dict_factory
        complaints_raw = conn.execute(f'''
            SELECT 
                c.id, 
                c.text, 
                CAST(c.location_lat AS FLOAT) as location_lat,
                CAST(c.location_lon AS FLOAT) as location_lon,
                c.issue_type,
                c.image,
                c.submitted_at,
                c.status,
                c.upvotes,
                c.remarks,
                c.is_duplicate,
                c.original_report_id,
                u.username 
            FROM complaints c
            LEFT JOIN users u ON c.user_id = u.id
            WHERE c.is_duplicate = 0 OR c.is_duplicate IS NULL
            {order_clause}
        ''').fetchall()
        
        # Process and validate each complaint
        processed_complaints = []
        for complaint in complaints_raw:
            try:
                # Create a clean dictionary with basic complaint info
                comp_dict = {
                    'id': int(complaint['id']),
                    'text': str(complaint['text'] or ''),
                    'location_lat': float(complaint['location_lat'] or 0),
                    'location_lon': float(complaint['location_lon'] or 0),
                    'issue_type': str(complaint['issue_type'] or ''),
                    'status': str(complaint['status'] or 'Submitted'),
                    'upvotes': int(complaint['upvotes'] or 0),
                    'remarks': str(complaint['remarks'] or ''),
                    'username': str(complaint['username'] or ''),
                    'is_duplicate': bool(complaint.get('is_duplicate')),
                    'original_report_id': int(complaint['original_report_id']) if complaint.get('original_report_id') else None
                }

                # Handle datetime
                if isinstance(complaint['submitted_at'], str):
                    try:
                        submitted_at = datetime.datetime.strptime(complaint['submitted_at'], '%Y-%m-%d %H:%M:%S.%f')
                    except ValueError:
                        try:
                            submitted_at = datetime.datetime.strptime(complaint['submitted_at'], '%Y-%m-%d %H:%M:%S')
                        except ValueError:
                            submitted_at = datetime.datetime.now()
                else:
                    submitted_at = complaint['submitted_at'] or datetime.datetime.now()

                comp_dict['submitted_at'] = submitted_at

                # Handle image data
                if complaint.get('image'):
                    try:
                        comp_dict['image'] = base64.b64encode(complaint['image']).decode('utf-8')
                    except:
                        comp_dict['image'] = None
                else:
                    comp_dict['image'] = None

                processed_complaints.append(comp_dict)
            except Exception as e:
                app.logger.error(f"Error processing complaint {complaint.get('id', 'unknown')}: {str(e)}")
                continue
    
    return render_template('complaints.html', complaints=processed_complaints, sort_by=sort_by)

@app.route('/upvote_complaint/<int:complaint_id>', methods=['POST'])
@login_required
def upvote_complaint(complaint_id):
    if session.get('is_admin'):
        return jsonify({'error': 'Admins cannot upvote.'}), 403
    with sqlite3.connect(APP_DB) as conn:
        conn.execute('PRAGMA foreign_keys = ON')
        conn.row_factory = dict_factory
        # Update upvote count
        conn.execute('UPDATE complaints SET upvotes = upvotes + 1 WHERE id = ?', (complaint_id,))
        conn.commit()
        
        # Get updated count
        result = conn.execute('SELECT upvotes FROM complaints WHERE id = ?', (complaint_id,)).fetchone()
        if result:
            new_count = result['upvotes']
            return jsonify({'success': True, 'new_count': new_count})
    return jsonify({'error': 'Complaint not found.'}), 404

# --- NEW: My Complaints Route ---
@app.route('/my_complaints')
@login_required
def my_complaints():
    if session.get('is_admin'):
        flash("Admin users can view all complaints via the admin dashboard.", "info")
        return redirect(url_for('admin_dashboard'))
    
    user_id = session['user_id']
    with sqlite3.connect(APP_DB) as conn:
        conn.row_factory = dict_factory
        complaints_raw = conn.execute('''
            SELECT 
                c.id, 
                c.text, 
                c.issue_type,
                c.image,
                c.submitted_at,
                c.status,
                c.upvotes,
                c.remarks,
                c.is_duplicate,
                c.original_report_id,
                u.username 
            FROM complaints c
            LEFT JOIN users u ON c.user_id = u.id
            WHERE c.user_id = ? 
            ORDER BY c.submitted_at DESC
        ''', (user_id,)).fetchall()
        
        # Process and validate each complaint
        processed_complaints = []
        for complaint in complaints_raw:
            try:
                # Create a clean dictionary with basic complaint info
                comp_dict = {
                    'id': int(complaint['id']),
                    'text': str(complaint['text'] or ''),
                    'issue_type': str(complaint['issue_type'] or ''),
                    'status': str(complaint['status'] or 'Submitted'),
                    'upvotes': int(complaint['upvotes'] or 0),
                    'remarks': str(complaint['remarks'] or ''),
                    'username': str(complaint['username'] or ''),
                    'is_duplicate': bool(complaint['is_duplicate']),
                    'original_report_id': int(complaint['original_report_id']) if complaint['original_report_id'] else None
                }

                # Handle image data
                if complaint.get('image'):
                    try:
                        comp_dict['image'] = base64.b64encode(complaint['image']).decode('utf-8')
                    except:
                        comp_dict['image'] = None
                else:
                    comp_dict['image'] = None

                # Handle datetime
                if isinstance(complaint['submitted_at'], str):
                    try:
                        submitted_at = datetime.datetime.strptime(complaint['submitted_at'], '%Y-%m-%d %H:%M:%S.%f')
                    except ValueError:
                        try:
                            submitted_at = datetime.datetime.strptime(complaint['submitted_at'], '%Y-%m-%d %H:%M:%S')
                        except ValueError:
                            submitted_at = datetime.datetime.now()
                else:
                    submitted_at = complaint['submitted_at'] or datetime.datetime.now()

                comp_dict['submitted_at'] = submitted_at
                processed_complaints.append(comp_dict)
            except Exception as e:
                app.logger.error(f"Error processing complaint {complaint.get('id', 'unknown')}: {str(e)}")
                continue
        
    return render_template('my_complaints.html', 
                         complaints=processed_complaints, 
                         now=datetime.datetime.now())

# --- Complaint Submission ---
@app.route('/raise_complaint', methods=['POST'])
@login_required
def raise_complaint():
    user_id = session['user_id']
    if session.get('is_admin'):
         return jsonify({'error': 'Admin users cannot raise complaints.'}), 403

    form = request.form
    if not all([form.get(k) for k in ['text', 'issue_type', 'street', 'city', 'state', 'zipcode']]) or 'image' not in request.files:
        return jsonify({'error': 'All fields and an image are required.'}), 400

    lat, lon = get_coordinates_from_address(form['street'], form['city'], form['state'], form['zipcode'])
    if not lat:
        return jsonify({'error': 'Could not find coordinates for the address.'}), 400

    image_bytes = request.files['image'].read()
    
    # Placeholder for duplication logic
    is_duplicate, original_id = False, None
    # Here you would call your duplication detection logic
    # is_duplicate, original_id, confidence = detector.find_duplicates(...)

    with sqlite3.connect(APP_DB) as conn:
        conn.execute('PRAGMA foreign_keys = ON')
        conn.execute('''
            INSERT INTO complaints (
                text, location_lat, location_lon, issue_type,
                image, image_filename, user_id,
                is_duplicate, original_report_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            form['text'], lat, lon, form['issue_type'],
            image_bytes, secure_filename(request.files['image'].filename),
            user_id, is_duplicate, original_id
        ))
        conn.commit()

    return jsonify({'message': 'Complaint registered successfully.'}), 200


# Debug utility routes
@app.route('/debug/reset_complaints', methods=['POST'])
def debug_reset_complaints():
    """Debug route to reset all complaints. Only works in debug mode."""
    if not app.debug:
        return jsonify({'error': 'This route is only available in debug mode'}), 403
    
    try:
        with sqlite3.connect(APP_DB) as conn:
            conn.execute('PRAGMA foreign_keys = OFF')  # Temporarily disable foreign keys
            conn.execute('DELETE FROM complaints')  # Clear all complaints
            conn.execute('DELETE FROM sqlite_sequence WHERE name="complaints"')  # Reset auto-increment
            conn.execute('UPDATE pothole_stats SET total_potholes = 0, high_priority_count = 0, medium_priority_count = 0, low_priority_count = 0')  # Reset stats
            conn.commit()
            flash('All complaints have been cleared successfully.', 'success')
            return jsonify({'message': 'All complaints cleared successfully'}), 200
    except Exception as e:
        app.logger.error(f'Error resetting complaints: {str(e)}')
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
