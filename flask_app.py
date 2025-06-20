import os
import tempfile
import base64
import sqlite3
import io
import datetime
from flask import Flask, request, render_template, jsonify, send_from_directory, session, redirect, url_for, flash
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderServiceError

# Import the pothole detection function from the existing file
from pothole_detection import run_pothole_detection

from duplication_detection_code import get_duplicate_detector

# (Duplication detection will be added after basic flow is working)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'dev-secret-key-replace-later' # Replace with a strong, random secret key for production

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def get_coordinates_from_address(street, city, state, zipcode):
    """
    Uses geopy to get (latitude, longitude) from address fields.
    Country is always set to India.
    Returns (lat, lon) tuple or (None, None) if not found.
    """
    geolocator = Nominatim(user_agent="pothole-complaint-app")
    address = f"{street}, {city}, {state}, {zipcode}, India"
    try:
        location = geolocator.geocode(address)
        if location:
            return location.latitude, location.longitude
        else:
            return None, None
    except GeocoderServiceError:
        return None, None

# --- Protected Routes ---
# Decorator for login required
from functools import wraps

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'error')
            return redirect(url_for('login'))
        # Admins should generally use the admin dashboard, redirect them there from index/tools
        if session.get('is_admin'):
             # Allow admin access if needed, or redirect
             # return redirect(url_for('admin_dashboard')) # Option to redirect admin away
             pass # Currently allowing admin access to user pages too
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

# --- Database Setups ---

# Pothole DB setup
POTHOLE_DB = os.path.join(os.path.dirname(__file__), 'pothole_data.db')

def init_pothole_db():
    conn = sqlite3.connect(POTHOLE_DB)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS pothole_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input_image BLOB,
            input_filename TEXT,
            detection_result TEXT,
            annotated_image BLOB,
            detected_at TIMESTAMP
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS pothole_stats (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            total_potholes INTEGER,
            high_priority_count INTEGER,
            medium_priority_count INTEGER,
            low_priority_count INTEGER,
            last_updated TIMESTAMP
        )
    ''')
    # Ensure a single row exists for stats
    c.execute('SELECT COUNT(*) FROM pothole_stats')
    if c.fetchone()[0] == 0:
        c.execute('''
            INSERT INTO pothole_stats (id, total_potholes, high_priority_count, medium_priority_count, low_priority_count, last_updated)
            VALUES (1, 0, 0, 0, 0, ?)
        ''', (datetime.datetime.now(),))
    conn.commit()
    conn.close()

init_pothole_db()

@app.route('/detect_pothole', methods=['POST'])
# @login_required # Uncomment this line if pothole detection should require login
def detect_pothole():
    # if 'user_id' not in session: # Manual check if not using decorator
    #     return jsonify({'error': 'Authentication required'}), 401
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Run pothole detection
    result_json, annotated_image_bytes = run_pothole_detection(file_path)

    if result_json is None or annotated_image_bytes is None:
        return jsonify({'error': 'Detection failed'}), 500

    # Store in pothole_images table
    conn = sqlite3.connect(POTHOLE_DB)
    c = conn.cursor()
    c.execute('''
        INSERT INTO pothole_images (input_image, input_filename, detection_result, annotated_image, detected_at)
        VALUES (?, ?, ?, ?, ?)
    ''', (
        file.read() if hasattr(file, 'stream') else open(file_path, 'rb').read(),
        filename,
        str(result_json),
        annotated_image_bytes,
        datetime.datetime.now()
    ))

    # Update stats
    # Extract priority info from result_json
    total = result_json.get('total_potholes', 0)
    high = 0
    medium = 0
    low = 0
    # Robustly count priorities
    priorities = result_json.get('individual_priorities')
    if (
        isinstance(priorities, list)
        and len(priorities) == total
        and all(p in ('high', 'medium', 'low') for p in priorities)
    ):
        for p in priorities:
            if p == 'high':
                high += 1
            elif p == 'medium':
                medium += 1
            elif p == 'low':
                low += 1
    elif 'road_priority' in result_json and total > 0:
        # Fallback: if only road_priority is available, assign all to that priority
        if result_json['road_priority'] == 'high':
            high = total
        elif result_json['road_priority'] == 'medium':
            medium = total
        elif result_json['road_priority'] == 'low':
            low = total
    # If priorities are missing or malformed, only update total, not priority counts

    # Update stats row
    c.execute('SELECT total_potholes, high_priority_count, medium_priority_count, low_priority_count FROM pothole_stats WHERE id=1')
    stats = c.fetchone()
    if stats:
        new_total = stats[0] + total
        new_high = stats[1] + high
        new_medium = stats[2] + medium
        new_low = stats[3] + low
        c.execute('''
            UPDATE pothole_stats
            SET total_potholes=?, high_priority_count=?, medium_priority_count=?, low_priority_count=?, last_updated=?
            WHERE id=1
        ''', (new_total, new_high, new_medium, new_low, datetime.datetime.now()))
    conn.commit()
    conn.close()

    # Encode annotated image as base64 for frontend display
    annotated_image_b64 = base64.b64encode(annotated_image_bytes).decode('utf-8')

    # Clean up uploaded file
    try:
        os.remove(file_path)
    except Exception:
        pass

    return jsonify({
        'result': result_json,
        'annotated_image_b64': annotated_image_b64
    })

import sqlite3
import io
import datetime
from duplication_detection_code import get_duplicate_detector

# Serve static files (CSS, JS)
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

# Complaints DB setup
COMPLAINTS_DB = os.path.join(os.path.dirname(__file__), 'complaints.db')

def init_complaints_db():
    conn = None
    try:
        conn = sqlite3.connect(COMPLAINTS_DB)
        c = conn.cursor()

        # Create table if it doesn't exist (keeping original columns)
        c.execute('''
            CREATE TABLE IF NOT EXISTS complaints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT,
                location_lat REAL,
                location_lon REAL,
                issue_type TEXT,
                image BLOB,
                image_filename TEXT,
                submitted_at TIMESTAMP,
                is_duplicate INTEGER,
                original_report_id INTEGER
            )
        ''')

        # Check and add new columns if they don't exist
        c.execute("PRAGMA table_info(complaints)")
        columns = [column[1] for column in c.fetchall()]

        if 'user_id' not in columns:
            # Add user_id column (linking to users table)
            # Note: Adding FK constraints via ALTER is tricky in SQLite,
            # handle relationship logic in the application for now.
            c.execute("ALTER TABLE complaints ADD COLUMN user_id INTEGER")
            print("Added 'user_id' column to complaints table.")

        if 'status' not in columns:
            # Add status column with default value
            c.execute("ALTER TABLE complaints ADD COLUMN status TEXT DEFAULT 'Submitted'")
            print("Added 'status' column to complaints table.")

        if 'upvotes' not in columns:
            # Add upvotes column with default value
            c.execute("ALTER TABLE complaints ADD COLUMN upvotes INTEGER DEFAULT 0")
            print("Added 'upvotes' column to complaints table.")

        if 'remarks' not in columns:
            # Add remarks column with default value
            c.execute("ALTER TABLE complaints ADD COLUMN remarks TEXT DEFAULT 'Complaint sent for supervision.'")
            print("Added 'remarks' column to complaints table.")

        conn.commit()
        print("Complaints DB initialized/updated successfully.")

    except sqlite3.Error as e:
        print(f"Database error during complaints DB initialization: {e}")
    finally:
        if conn:
            conn.close()

init_complaints_db()

# Users DB setup
USERS_DB = os.path.join(os.path.dirname(__file__), 'users.db')

def init_users_db():
    conn = sqlite3.connect(USERS_DB)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            full_name TEXT NOT NULL,
            password_hash TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_users_db()

# --- Authentication Routes ---

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check for hardcoded admin credentials first
        if username == 'admin001' and password == 'admin$001':
            session['user_id'] = 'admin'
            session['username'] = 'admin001'
            session['is_admin'] = True
            flash('Admin login successful!', 'success')
            return redirect(url_for('admin_dashboard')) # Redirect admin to admin dashboard

        # Check regular users in the database
        conn = sqlite3.connect(USERS_DB)
        conn.row_factory = sqlite3.Row # Return rows as dictionary-like objects
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = c.fetchone()
        conn.close()

        if user and check_password_hash(user['password_hash'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['is_admin'] = False
            flash('Login successful!', 'success')
            return redirect(url_for('index')) # Redirect regular users to index
        else:
            flash('Invalid username or password.', 'error')
            return redirect(url_for('login')) # Redirect back to login page on failure

    # For GET request, just render the login page
    # If already logged in, redirect away from login page
    if 'user_id' in session:
        if session.get('is_admin'):
            return redirect(url_for('admin_dashboard'))
        else:
            return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        full_name = request.form['full_name']
        password = request.form['password']

        if not username or not full_name or not password:
             flash('All fields are required.', 'error')
             return redirect(url_for('signup'))

        conn = None  # Initialize connection variable
        try:
            conn = sqlite3.connect(USERS_DB)
            c = conn.cursor()

            # Check if username already exists
            c.execute('SELECT id FROM users WHERE username = ?', (username,))
            existing_user = c.fetchone()

            if existing_user:
                flash('Username already exists. Please choose another.', 'error')
                # No need to close conn here, finally block will handle it
                return redirect(url_for('signup'))

            # Hash password and insert new user
            password_hash = generate_password_hash(password)
            c.execute('INSERT INTO users (username, full_name, password_hash) VALUES (?, ?, ?)',
                      (username, full_name, password_hash))
            conn.commit()

            flash('Signup successful! Please log in.', 'success')
            # No need to close conn here, finally block will handle it
            return redirect(url_for('login'))

        except sqlite3.Error as e:
            # Log the specific error to the console for debugging
            print(f"Database error during signup: {e}") # Using print for simplicity in dev environment
            flash('A database error occurred during signup. Please try again.', 'error')
            # Ensure redirect happens even if error occurs
            return redirect(url_for('signup'))

        finally:
            # Ensure the connection is closed whether successful or not
            if conn:
                conn.close()

    # For GET request, render signup page
    # If already logged in, redirect away
    if 'user_id' in session:
        if session.get('is_admin'):
            return redirect(url_for('admin_dashboard'))
        else:
            return redirect(url_for('index'))
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    session.pop('is_admin', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

# --- Admin Dashboard Route ---

@app.route('/admin')
def admin_dashboard():
    # Protect this route - only admins allowed
    if not session.get('is_admin'):
        flash('You do not have permission to access this page.', 'error')
        return redirect(url_for('login'))

    conn = sqlite3.connect(COMPLAINTS_DB)
    conn.row_factory = sqlite3.Row # Return rows as dictionary-like objects
    c = conn.cursor()
    # Fetch non-duplicate complaints, ordered by submission time, including new fields
    c.execute('''
        SELECT id, user_id, text, location_lat, location_lon, issue_type, submitted_at, status, upvotes, remarks, image
        FROM complaints
        WHERE is_duplicate = 0 OR is_duplicate IS NULL
        ORDER BY submitted_at DESC
    ''')
    complaints = c.fetchall()
    conn.close()

    # Convert timestamp strings to datetime objects if needed (or handle in template)
    # Assuming they are stored as TIMESTAMP which sqlite3 might return as strings
    processed_complaints = []
    for complaint in complaints:
        comp_dict = dict(complaint) # Convert row object to dict
        # Attempt to parse timestamp if it's a string
        if isinstance(comp_dict['submitted_at'], str):
             try:
                 # Adjust format string if necessary based on how it's stored
                 comp_dict['submitted_at'] = datetime.datetime.strptime(comp_dict['submitted_at'], '%Y-%m-%d %H:%M:%S.%f')
             except ValueError:
                 # Handle cases where parsing might fail or format is different
                 try:
                     comp_dict['submitted_at'] = datetime.datetime.strptime(comp_dict['submitted_at'], '%Y-%m-%d %H:%M:%S')
                 except ValueError:
                     # Fallback if parsing fails
                     pass # Keep original string or handle error
        # Convert image blob to base64 string if it exists
        if 'image' in comp_dict and comp_dict['image'] is not None:
            comp_dict['image'] = "data:image/jpeg;base64," + base64.b64encode(comp_dict['image']).decode('utf-8')
        processed_complaints.append(comp_dict)


    # Pass datetime module to template context
    return render_template('admin_dashboard.html', complaints=processed_complaints, datetime=datetime)

# --- Admin Action Route ---

@app.route('/update_complaint_status/<int:complaint_id>', methods=['POST'])
@login_required # Ensure user is logged in
def update_complaint_status(complaint_id):
    # Ensure only admin can perform this action
    if not session.get('is_admin'):
        flash('Unauthorized action.', 'error')
        return redirect(url_for('login')) # Or redirect to index?

    new_status = request.form.get('status')
    new_remarks = request.form.get('remarks')

    # Basic validation
    allowed_statuses = ['Submitted', 'Approved', 'Rejected', 'On Hold']
    if not new_status or new_status not in allowed_statuses:
        flash('Invalid status selected.', 'error')
        return redirect(url_for('admin_dashboard'))
    if not new_remarks: # Remarks are required
        flash('Remarks are required to update status.', 'error')
        return redirect(url_for('admin_dashboard'))

    conn = None
    try:
        conn = sqlite3.connect(COMPLAINTS_DB)
        c = conn.cursor()
        c.execute('''
            UPDATE complaints
            SET status = ?, remarks = ?
            WHERE id = ?
        ''', (new_status, new_remarks, complaint_id))
        conn.commit()

        if c.rowcount == 0:
            flash(f'Complaint ID {complaint_id} not found.', 'error')
        else:
            flash(f'Complaint ID {complaint_id} status updated successfully.', 'success')

    except sqlite3.Error as e:
        print(f"Database error updating complaint status: {e}")
        flash('A database error occurred while updating status.', 'error')
    finally:
        if conn:
            conn.close()

    return redirect(url_for('admin_dashboard'))

# --- Public Complaints View ---
# In your Flask app setup
# In your Flask app
@app.template_test('isdatetime')
def is_datetime(obj):
    return isinstance(obj, datetime.datetime)
@app.route('/complaints')
@login_required
def view_complaints():
    # Get query parameters for search, sort, filter (implement later)
    search_id = request.args.get('search_id')
    sort_by = request.args.get('sort', 'time_desc') # Default sort: newest first
    # filter_status = request.args.get('filter_status')
    # filter_location = request.args.get('filter_location')
    # filter_time = request.args.get('filter_time')

    conn = None
    complaints = []
    try:
        conn = sqlite3.connect(COMPLAINTS_DB)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        # Base query
        query = '''
            SELECT id, user_id, text, location_lat, location_lon, issue_type, submitted_at, status, upvotes, remarks, image
            FROM complaints
            WHERE (is_duplicate = 0 OR is_duplicate IS NULL)
        '''
        params = []

        # TODO: Add filtering logic based on query parameters
        if search_id:
            query += " AND id = ?"
            try:
                params.append(int(search_id))
            except ValueError:
                flash('Invalid Complaint ID for search.', 'error')
                # Handle invalid ID search - maybe show all or show none? Show all for now.
                query = query.replace(" AND id = ?", "") # Remove the condition

        # TODO: Add sorting logic
        if sort_by == 'upvotes_desc':
            query += " ORDER BY upvotes DESC, submitted_at DESC"
        elif sort_by == 'time_asc':
             query += " ORDER BY submitted_at ASC"
        else: # Default: time_desc
            query += " ORDER BY submitted_at DESC"


        c.execute(query, params)
        raw_complaints = c.fetchall()

        # Process timestamps like in admin view
        for complaint in raw_complaints:
            comp_dict = dict(complaint)
            if isinstance(comp_dict['submitted_at'], str):
                 try:
                     comp_dict['submitted_at'] = datetime.datetime.strptime(comp_dict['submitted_at'], '%Y-%m-%d %H:%M:%S.%f')
                 except ValueError:
                     try:
                         comp_dict['submitted_at'] = datetime.datetime.strptime(comp_dict['submitted_at'], '%Y-%m-%d %H:%M:%S')
                     except ValueError:
                         pass
            # Convert image blob to base64 string if it exists
            if 'image' in comp_dict and comp_dict['image'] is not None:
                comp_dict['image'] = "data:image/jpeg;base64," + base64.b64encode(comp_dict['image']).decode('utf-8')
            complaints.append(comp_dict)


    except sqlite3.Error as e:
        print(f"Database error fetching complaints: {e}")
        flash('Could not retrieve complaints due to a database error.', 'error')
    finally:
        if conn:
            conn.close()

    # Pass sorting/filtering parameters and datetime module back to template
    return render_template('complaints.html', complaints=complaints, sort_by=sort_by, search_id=search_id, datetime=datetime)

@app.route('/upvote_complaint/<int:complaint_id>', methods=['POST'])
@login_required
def upvote_complaint(complaint_id):
    # Admin cannot upvote
    if session.get('is_admin'):
        return jsonify({'error': 'Admin users cannot upvote complaints.'}), 403

    conn = None
    try:
        conn = sqlite3.connect(COMPLAINTS_DB)
        c = conn.cursor()

        # Increment the upvote count
        # Using COALESCE to handle potential NULL values if the default wasn't set properly
        c.execute('''
            UPDATE complaints
            SET upvotes = COALESCE(upvotes, 0) + 1
            WHERE id = ? AND (is_duplicate = 0 OR is_duplicate IS NULL)
        ''', (complaint_id,))

        # Check if the update was successful and get the new count
        if c.rowcount > 0:
            c.execute('SELECT upvotes FROM complaints WHERE id = ?', (complaint_id,))
            result = c.fetchone()
            new_count = result[0] if result else 0
            conn.commit() # Commit only if update was successful
            return jsonify({'success': True, 'new_count': new_count})
        else:
            # Complaint not found or was a duplicate
            conn.rollback() # Rollback if no rows affected
            return jsonify({'error': 'Complaint not found or cannot be upvoted.'}), 404

    except sqlite3.Error as e:
        print(f"Database error during upvote: {e}")
        if conn:
            conn.rollback() # Rollback on error
        return jsonify({'error': 'A database error occurred.'}), 500
    finally:
        if conn:
            conn.close()

# --- User's Complaints View ---

@app.route('/my_complaints')
@login_required
def my_complaints():
    user_id = session['user_id']
    # Admins don't have "my complaints"
    if session.get('is_admin'):
        flash("Admin users view all complaints via the admin dashboard.", "info")
        return redirect(url_for('admin_dashboard'))

    conn = None
    complaints = []
    try:
        conn = sqlite3.connect(COMPLAINTS_DB)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        # Fetch complaints submitted by the current user
        c.execute('''
            SELECT id, text, location_lat, location_lon, issue_type, submitted_at, status, upvotes, remarks, image, is_duplicate, original_report_id
            FROM complaints
            WHERE user_id = ?
            ORDER BY submitted_at DESC
        ''', (user_id,))
        raw_complaints = c.fetchall()

        # Process timestamps
        for complaint in raw_complaints:
            comp_dict = dict(complaint)
            if isinstance(comp_dict['submitted_at'], str):
                 try:
                     comp_dict['submitted_at'] = datetime.datetime.strptime(comp_dict['submitted_at'], '%Y-%m-%d %H:%M:%S.%f')
                 except ValueError:
                     try:
                         comp_dict['submitted_at'] = datetime.datetime.strptime(comp_dict['submitted_at'], '%Y-%m-%d %H:%M:%S')
                     except ValueError:
                         pass
            # Convert image blob to base64 string if it exists
            if 'image' in comp_dict and comp_dict['image'] is not None:
                comp_dict['image'] = "data:image/jpeg;base64," + base64.b64encode(comp_dict['image']).decode('utf-8')
            complaints.append(comp_dict)

    except sqlite3.Error as e:
        print(f"Database error fetching user complaints: {e}")
        flash('Could not retrieve your complaints due to a database error.', 'error')
    finally:
        if conn:
            conn.close()

    # Pass datetime module to template context
    return render_template('my_complaints.html', complaints=complaints, datetime=datetime)


# --- Complaint and Pothole Routes ---

@app.route('/top_complaints', methods=['GET'])
def get_top_complaints():
    conn = None
    try:
        conn = sqlite3.connect(COMPLAINTS_DB)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        # Get top 3 non-duplicate complaints by upvotes
        query = '''
            SELECT id, text, location_lat, location_lon, issue_type, submitted_at, status, upvotes
            FROM complaints
            WHERE (is_duplicate = 0 OR is_duplicate IS NULL)
            ORDER BY upvotes DESC, submitted_at DESC
            LIMIT 3
        '''
        c.execute(query)
        complaints = [dict(row) for row in c.fetchall()]
        
        # Convert datetime objects to string format and image to base64
        for complaint in complaints:
            if isinstance(complaint['submitted_at'], str):
                try:
                    dt = datetime.datetime.strptime(complaint['submitted_at'], '%Y-%m-%d %H:%M:%S.%f')
                    complaint['submitted_at'] = dt.strftime('%Y-%m-%d')
                except ValueError:
                    try:
                        dt = datetime.datetime.strptime(complaint['submitted_at'], '%Y-%m-%d %H:%M:%S')
                        complaint['submitted_at'] = dt.strftime('%Y-%m-%d')
                    except ValueError:
                        pass
            
            # Convert image blob to base64 if it exists
            if 'image' in complaint and complaint['image'] is not None:
                complaint['image'] = "data:image/jpeg;base64," + base64.b64encode(complaint['image']).decode('utf-8')
        
        return jsonify(complaints)
    except sqlite3.Error as e:
        print(f"Database error fetching top complaints: {e}")
        return jsonify([])
    finally:
        if conn:
            conn.close()

@app.route('/raise_complaint', methods=['POST'])
@login_required # Now require login to raise a complaint
def raise_complaint():
    # user_id is guaranteed to be in session due to @login_required
    user_id = session['user_id']
    # Admin cannot raise complaints
    if session.get('is_admin'):
         return jsonify({'error': 'Admin users cannot raise complaints.'}), 403

    # Get form fields
    text = request.form.get('text')
    issue_type = request.form.get('issue_type')
    street = request.form.get('street')
    city = request.form.get('city')
    state = request.form.get('state')
    zipcode = request.form.get('zipcode')
    image_file = request.files.get('image')

    if not all([text, issue_type, street, city, state, zipcode, image_file]):
        return jsonify({'error': 'All fields are required.'}), 400

    # Get coordinates from address
    lat, lon = get_coordinates_from_address(street, city, state, zipcode)
    if lat is None or lon is None:
        return jsonify({'error': 'Could not geocode the provided address.'}), 400

    # Read image bytes
    image_bytes = image_file.read()
    image_filename = secure_filename(image_file.filename)

    # Prepare report dict for duplication detection
    report = {
        'text': text,
        'location': (lat, lon),
        'issue_type': issue_type,
        'image_bytes': image_bytes
    }

    # Load all existing complaints for duplication detection
    conn = sqlite3.connect(COMPLAINTS_DB)
    c = conn.cursor()
    c.execute('SELECT id, text, location_lat, location_lon, issue_type, image FROM complaints WHERE is_duplicate=0')
    rows = c.fetchall()
    detector = get_duplicate_detector()
    for row in rows:
        db_report = {
            'id': row[0],
            'text': row[1],
            'location': (row[2], row[3]),
            'issue_type': row[4],
            'image_bytes': row[5]
        }
        detector.add_report(db_report)

    # Check for duplicates
    is_duplicate, similar_reports, confidence = detector.find_duplicates(report)

    # Prepare data for insertion (including user_id)
    complaint_data = (
        text, lat, lon, issue_type, image_bytes, image_filename,
        datetime.datetime.now(), user_id # Add user_id here
    )

    if is_duplicate:
        original_id = similar_reports[0] if similar_reports else None
        # Store as duplicate for record-keeping, including user_id
        # Note: Defaults for status, upvotes, remarks are handled by DB schema
        c.execute('''
            INSERT INTO complaints (text, location_lat, location_lon, issue_type, image, image_filename, submitted_at, user_id, is_duplicate, original_report_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', complaint_data + (1, original_id)) # Add is_duplicate and original_id
        conn.commit()
        conn.close()
        return jsonify({'message': f'Duplicate complaint detected. Similar to complaint ID {original_id}.'}), 200

    # Not duplicate, store as new complaint, including user_id
    # Note: Defaults for status, upvotes, remarks are handled by DB schema
    c.execute('''
        INSERT INTO complaints (text, location_lat, location_lon, issue_type, image, image_filename, submitted_at, user_id, is_duplicate, original_report_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', complaint_data + (0, None)) # Add is_duplicate (0) and original_id (None)
    conn.commit()
    conn.close()
    return jsonify({'message': 'Complaint registered successfully.'}), 200

@app.route('/pothole_stats', methods=['GET'])
def pothole_stats():
    conn = sqlite3.connect(POTHOLE_DB)
    c = conn.cursor()
    c.execute('SELECT total_potholes, high_priority_count, medium_priority_count, low_priority_count, last_updated FROM pothole_stats WHERE id=1')
    row = c.fetchone()
    conn.close()
    if row:
        return jsonify({
            'total_potholes': row[0],
            'high_priority_count': row[1],
            'medium_priority_count': row[2],
            'low_priority_count': row[3],
            'last_updated': row[4]
        })
    else:
        return jsonify({'error': 'Stats not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)
