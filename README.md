# Smart Pothole Detection & Civic Management


eNivaran is an ML-powered road safety and civic issue reporting system designed to help create safer, smarter cities through community participation and cutting-edge technology. Users can report civic issues like potholes, view existing complaints, and track their status, while administrators can manage and update these reports. The system utilizes an AI model to detect potholes from images and includes functionality to prevent duplicate complaints.

## Core Functionalities

*   **AI Pothole Detection:** Upload an image to the 'Tools' section to automatically detect potholes using a pre-trained ONNX model (`pothole_detector_v1.onnx`). The system provides detection results and statistics.
*   **Civic Issue Reporting:** Users can submit detailed complaints including text description, issue type, location (address automatically converted to coordinates using Geopy), and an accompanying image.
*   **Duplicate Complaint Detection:** Before saving a new complaint, the system checks for similarities (text, location, image) with existing non-duplicate reports to minimize redundancy.
*   **User Authentication:** Secure signup and login system for regular users and a separate administrative login.
*   **Complaint Management:**
    *   **Public View (`/complaints`):** View all non-duplicate complaints, sortable by time or upvotes. Includes basic search by ID. Users can upvote complaints.
    *   **User View (`/my_complaints`):** Logged-in users can view the status and details of complaints they have submitted.
    *   **Admin Dashboard (`/admin`):** Admins can view all non-duplicate complaints, update their status (e.g., Approved, Rejected, On Hold), and add remarks.
*   **Pothole Statistics:** Tracks the total number of potholes detected and categorizes them by severity (high, medium, low).

## Technology Stack

* **Backend:**  
  - Python  
  - Flask

* **Frontend:**  
  - HTML  
  - CSS  
  - JavaScript
  - React

* **Database:**  
  - SQLite (used for Users, Complaints, and Pothole Detection data)

* **Machine Learning & Data Processing:**  
  - **ONNX Runtime:** For running the `pothole_detector_v1.onnx` model  
  - **NumPy:** Numerical computations and array operations  
  - **Pillow (PIL):** Image processing and image byte decoding  
  - **PyTorch & Torchvision:** For image feature extraction (using ResNet50)  
  - **Scikit-learn:**  
    - Clustering with KMeans  
    - Cosine similarity computations  
    - Feature extraction using TF-IDF vectorizer  
    - Data scaling via StandardScaler  
  - **Sentence Transformers:** For text embeddings using Sentence-BERT  
  - **XGBoost:** For training a duplicate detection classifier

* **Computer Vision & Detection:**  
  - **OpenCV (cv2):** Image processing and video frame analysis  
  - **Ultralytics YOLO:** For pothole detection with an explicit model version (compatible with Python 3.11)  
  - **Matplotlib:** Visualization (for annotating images and plotting road priority distribution)

* **Utilities & Others:**  
  - **Werkzeug:** For password hashing and file handling  
  - **Geopy:** For geocoding (address to coordinates conversion) and distance calculations  
  - **Python Logging:** (using `logging`) For tracking events, errors, and system messages  
  - **Time:** For performance measurement during model inference  
  - **Collections:** (`defaultdict`, `deque`) For data organization in clustering and processing tasks

## Setup and Running Instructions

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/RohanChoudhary15/Esoc-X-Hack_DevBytes
    cd Esoc-X-Hack_DevBytes
    ```

2.  **Create and Activate a Virtual Environment (Recommended):**
    ```bash
    # For Linux/macOS
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    Make sure you have `pip` installed. Run the following command in the project's root directory:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Ensure `requirements.txt` includes Flask, Werkzeug, Geopy, onnxruntime, Pillow/OpenCV, sentence-transformers, and any other necessary libraries.*

4.  **Run the Flask Application:**
    ```bash
    flask run
    ```
    Alternatively, you can run:
    ```bash
    python flask_app.py
    ```
    The application will start, typically on `http://127.0.0.1:5000/`.

5.  **Database Initialization:**
    The application automatically initializes the SQLite databases (`users.db`, `complaints.db`, `pothole_data.db`) and creates the necessary tables if they don't exist when you first run `flask_app.py`.

6.  **Accessing the Application:**
    *   Open your web browser and navigate to `http://127.0.0.1:5000/`.
    *   You will be redirected to the login page (`/login`).
    *   **Sign up** for a new user account or use the **Admin** credentials:
        *   Username: `admin001`
        *   Password: `admin$001`
    *   Regular users will be redirected to the homepage (`/`) after login.
    *   The admin user will be redirected to the admin dashboard (`/admin`) after login.

## How It Works

1.  **User Interaction:** Users sign up or log in. They can navigate through different sections: Home, Complaints (view all), My Complaints (view own), Tools (pothole detection).
2.  **Pothole Detection (Tools):** Upload an image. The backend runs the ONNX model via `pothole_detection.py`, saves results and stats to `pothole_data.db`, and displays the annotated image and detection details.
3.  **Raising a Complaint:** Users fill out the complaint form on the main page (or a dedicated page). The address is geocoded via Geopy. The complaint text and image are processed by `duplication_detection_code.py` to check against existing reports in `complaints.db`.
4.  **Saving Complaints:** If not a duplicate, the complaint (including text, location, image blob, user ID, timestamp) is saved to `complaints.db` with `is_duplicate=0`. If it is a duplicate, it's saved with `is_duplicate=1` and a reference to the original report ID.
5.  **Viewing Complaints:** Users can view filtered lists of complaints (all, own) fetched from `complaints.db`. Images are loaded as base64 strings. Users can upvote complaints, updating the count in the database.
6.  **Admin Management:** The admin logs in and accesses the `/admin` dashboard to view complaints and update their status and remarks directly in the `complaints.db`.

---

**Team:** DevBytes
**Hackathon:** ESOC-X
**Problem Statement:** Code Brown
