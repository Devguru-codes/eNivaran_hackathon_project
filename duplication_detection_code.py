import numpy as np
import json
import os
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import geopy.distance
from collections import defaultdict
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def decode_image_bytes(image_bytes):
    """
    Helper to decode image bytes (from SQLite BLOB) to a numpy array (for PIL or OpenCV).
    Args:
        image_bytes: Raw image bytes (e.g., from SQLite BLOB)
    Returns:
        image_array: Decoded numpy array (RGB, as used by PIL)
    """
    import numpy as np
    from PIL import Image
    import io
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        return image
    except Exception as e:
        print(f"Error decoding image bytes: {e}")
        # Return a default white image if decoding fails
        return Image.new('RGB', (224, 224), color='white')

class CivicIssueDuplicateDetector:
    def __init__(self, n_clusters=None, location_threshold=0.1, text_similarity_threshold=0.8):
        """
        Initialize the duplicate detection model using unsupervised clustering

        Args:
            n_clusters: Number of clusters for K-means. Should be set to the number of unique complaints with the same location area, problem type (e.g., pothole, manhole cover removed, etc.), and time of reporting. (default: None - will be determined dynamically)
            location_threshold: Max distance in km to consider location similar (default: 0.1 km = 100m)
            text_similarity_threshold: Threshold for text similarity (default: 0.8)
        """
        # Initialize image feature extractor (ResNet50) with error handling
        try:
            self.image_model = models.resnet50(weights='DEFAULT')
            self.image_model.eval()
            # Remove the classification layer
            self.image_model = torch.nn.Sequential(*(list(self.image_model.children())[:-1]))
            self.image_model_available = True
            print("ResNet50 model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load ResNet50 model: {e}")
            print("Image similarity will use basic features")
            self.image_model = None
            self.image_model_available = False
        
        # Image preprocessing
        self.image_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Text embedding model with error handling
        try:
            self.text_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
            self.text_model_available = True
            print("SentenceTransformer model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load SentenceTransformer: {e}")
            print("Falling back to TF-IDF for text similarity")
            self.text_model = None
            self.text_model_available = False
        
        # TF-IDF vectorizer as alternative text representation
        try:
            self.tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
            self.tfidf_fitted = False
        except Exception as e:
            print(f"Warning: Could not initialize TF-IDF: {e}")
            self.tfidf = None
        
        # Clustering parameters
        self.n_clusters = n_clusters
        self.location_threshold = location_threshold
        self.text_similarity_threshold = text_similarity_threshold
        
        # Storage for processed data and clusters
        self.image_features_db = []
        self.location_db = []
        self.text_embeddings_db = []
        self.text_raw_db = []  # Store raw text for TF-IDF fallback
        self.issue_types_db = []
        self.reports_db = []
        
        # Cluster models
        self.image_kmeans = None
        self.text_kmeans = None
        
        # Cluster assignments
        self.image_clusters = []
        self.location_clusters = defaultdict(list)  # Will store indices by location grid
        self.issue_type_clusters = defaultdict(list)  # Will store indices by issue type
        
        # XGBoost model and scaler
        self.xgb_model = None
        self.scaler = None
        self.has_enough_data_for_xgboost = False
        
    def extract_image_features(self, image_input):
        """
        Extract image features using ResNet50 or fallback to basic features.
        image_input can be a file path, PIL Image, or image bytes (from SQLite BLOB).
        """
        try:
            # Handle different input types
            if isinstance(image_input, str) and os.path.exists(image_input):
                image = Image.open(image_input).convert('RGB')
            elif isinstance(image_input, Image.Image):
                image = image_input
            elif isinstance(image_input, bytes):
                image = decode_image_bytes(image_input)
            elif image_input is None:
                # Return zero features if no image provided
                return np.zeros(2048) if self.image_model_available else np.zeros(100)
            else:
                # If it's a numpy array, convert to PIL Image
                try:
                    if isinstance(image_input, np.ndarray):
                        image = Image.fromarray(image_input.astype('uint8'))
                    else:
                        return np.zeros(2048) if self.image_model_available else np.zeros(100)
                except Exception:
                    return np.zeros(2048) if self.image_model_available else np.zeros(100)
            
            if self.image_model_available and self.image_model is not None:
                # Use ResNet50 features
                image_tensor = self.image_transform(image).unsqueeze(0)
                with torch.no_grad():
                    features = self.image_model(image_tensor)
                return features.squeeze().numpy()
            else:
                # Fallback to basic image features (histogram)
                image_array = np.array(image.resize((64, 64)))
                # Simple color histogram as features
                hist_r = np.histogram(image_array[:,:,0], bins=10, range=(0,255))[0]
                hist_g = np.histogram(image_array[:,:,1], bins=10, range=(0,255))[0]
                hist_b = np.histogram(image_array[:,:,2], bins=10, range=(0,255))[0]
                # Add some basic statistics
                mean_rgb = np.mean(image_array, axis=(0,1))
                std_rgb = np.std(image_array, axis=(0,1))
                features = np.concatenate([hist_r, hist_g, hist_b, mean_rgb, std_rgb])
                # Pad to 100 features
                if len(features) < 100:
                    features = np.pad(features, (0, 100 - len(features)), 'constant')
                return features[:100]
                
        except Exception as e:
            print(f"Error extracting image features: {e}")
            # Return appropriate zero vector based on model availability
            return np.zeros(2048) if self.image_model_available else np.zeros(100)
    
    def extract_text_features(self, text):
        """Extract text embeddings using Sentence-BERT or TF-IDF fallback"""
        if not text or not isinstance(text, str):
            text = ""
            
        try:
            if self.text_model_available and self.text_model is not None:
                # Use SentenceTransformer
                return self.text_model.encode(text)
            else:
                # Fallback to TF-IDF
                if self.tfidf is not None:
                    if not self.tfidf_fitted and len(self.text_raw_db) > 0:
                        # Fit TF-IDF on existing texts
                        all_texts = self.text_raw_db + [text]
                        self.tfidf.fit(all_texts)
                        self.tfidf_fitted = True
                        return self.tfidf.transform([text]).toarray()[0]
                    elif self.tfidf_fitted:
                        return self.tfidf.transform([text]).toarray()[0]
                    else:
                        # Return basic text features (length, word count, etc.)
                        features = [
                            len(text),
                            len(text.split()),
                            text.count('.'),
                            text.count('!'),
                            text.count('?'),
                            len(set(text.lower().split()))  # unique words
                        ]
                        return np.array(features + [0] * (384 - len(features)))  # Pad to 384 like sentence-bert
                else:
                    # Very basic fallback
                    return np.array([len(text), len(text.split())] + [0] * 382)
        except Exception as e:
            print(f"Error extracting text features: {e}")
            return np.zeros(384)  # Default sentence-bert size
    
    def location_to_grid(self, location):
        """Convert location to grid cell for clustering"""
        try:
            # Using a simple grid approach for location clustering
            # Each grid cell is approximately location_threshold x location_threshold km
            lat, lon = location
            lat_grid = int(lat / self.location_threshold)
            lon_grid = int(lon / self.location_threshold)
            return (lat_grid, lon_grid)
        except Exception as e:
            print(f"Error converting location to grid: {e}")
            return (0, 0)  # Default grid position
    
    def add_report(self, report):
        """
        Add a new report to the database.
        report: Dictionary with at least 'text', 'location', 'issue_type', and either 'image_path', 'image_bytes', or 'image_array'.
        """
        try:
            # Validate required fields
            required_fields = ['text', 'location', 'issue_type']
            for field in required_fields:
                if field not in report:
                    raise ValueError(f"Missing required field: {field}")
            
            # Extract features
            image_input = report.get('image_bytes') or report.get('image_array') or report.get('image_path')
            image_features = self.extract_image_features(image_input)
            text_embedding = self.extract_text_features(report['text'])
            location = report['location']
            issue_type = report['issue_type']

            # Store features and report
            index = len(self.reports_db)
            self.image_features_db.append(image_features)
            self.text_embeddings_db.append(text_embedding)
            self.text_raw_db.append(report['text'])  # Store raw text for TF-IDF
            self.location_db.append(location)
            self.issue_types_db.append(issue_type)
            self.reports_db.append(report)

            # Add to location grid
            location_grid = self.location_to_grid(location)
            self.location_clusters[location_grid].append(index)

            # Add to issue type clusters
            self.issue_type_clusters[issue_type].append(index)

            # Check if we have enough data to train XGBoost
            self.check_and_train_xgboost()

            # Return the added index
            return index
            
        except Exception as e:
            print(f"Error adding report: {e}")
            return None
    
    def build_clusters(self):
        """Build clusters from all added reports"""
        try:
            # Determine number of clusters - even with small datasets
            if self.n_clusters is None:
                # Use at least 2 clusters, but not more than half the data points
                self.n_clusters = max(2, min(int(len(self.reports_db) / 2), 50))
            
            # Proceed with clustering even with small datasets
            if len(self.reports_db) >= 2:  # Need at least 2 reports to cluster
                n_clusters_actual = min(self.n_clusters, len(self.reports_db))
                self.image_kmeans = KMeans(
                    n_clusters=n_clusters_actual, 
                    random_state=42,
                    n_init=10
                )
                self.image_clusters = self.image_kmeans.fit_predict(np.array(self.image_features_db))
        except Exception as e:
            print(f"Error building clusters: {e}")
            self.image_clusters = [0] * len(self.reports_db)  # Default all to cluster 0
    
    def check_and_train_xgboost(self):
        """Check if we have enough data to train XGBoost and train if possible"""
        try:
            # Check if we have enough reports of the same type in similar locations
            issue_type_counts = {}
            for issue_type, indices in self.issue_type_clusters.items():
                if len(indices) >= 5:  # We need at least 5 reports of the same type
                    issue_type_counts[issue_type] = len(indices)
            
            # If we have enough data, train XGBoost
            if issue_type_counts and not self.has_enough_data_for_xgboost:
                self.train_xgboost_model()
                self.has_enough_data_for_xgboost = True
        except Exception as e:
            print(f"Error checking/training XGBoost: {e}")
            self.has_enough_data_for_xgboost = False
    
    def train_xgboost_model(self):
        """Train XGBoost model using pseudo-labels from current similarity metrics"""
        try:
            # Create feature vectors for each report pair
            X = []
            y = []  # Pseudo-labels based on current similarity metrics
            
            # Compare each report with every other report
            for i in range(len(self.reports_db)):
                for j in range(i+1, len(self.reports_db)):
                    # Skip if different issue types
                    if self.issue_types_db[i] != self.issue_types_db[j]:
                        continue
                        
                    # Extract features for this pair
                    text_sim = cosine_similarity([self.text_embeddings_db[i]], [self.text_embeddings_db[j]])[0][0]
                    image_sim = cosine_similarity([self.image_features_db[i]], [self.image_features_db[j]])[0][0]
                    
                    # Calculate location similarity
                    loc1 = self.location_db[i]
                    loc2 = self.location_db[j]
                    dist = geopy.distance.distance(loc1, loc2).kilometers
                    loc_sim = 1.0 - min(1.0, dist/self.location_threshold)
                    
                    # Create feature vector for this pair
                    features = [text_sim, image_sim, loc_sim, 
                               int(self.issue_types_db[i] == self.issue_types_db[j])]
                    X.append(features)
                    
                    # Create pseudo-label using improved similarity formula
                    current_sim = 0.4 * text_sim + 0.2 * image_sim + 0.4 * loc_sim
                    is_duplicate = 1 if current_sim >= 0.65 else 0
                    y.append(is_duplicate)
            
            # Train XGBoost model if we have enough pairs
            if len(X) > 5:
                # Normalize features
                self.scaler = StandardScaler()
                X_scaled = self.scaler.fit_transform(X)
                
                # Train XGBoost model
                self.xgb_model = xgb.XGBClassifier(
                    n_estimators=50,
                    max_depth=3,
                    learning_rate=0.1,
                    objective='binary:logistic',
                    random_state=42
                )
                self.xgb_model.fit(X_scaled, y)
                print("XGBoost model trained successfully")
        except Exception as e:
            print(f"Error training XGBoost model: {e}")
            self.xgb_model = None
            self.scaler = None
    
    def find_duplicates(self, new_report):
        """
        Find if a new report is a duplicate of any existing report

        Args:
            new_report: Dictionary with 'text', 'image_path', 'image_bytes', 'image_array', 'location', 'issue_type'

        Returns:
            is_duplicate: Boolean indicating if this is a duplicate
            similar_reports: List of report objects (not indices) of similar reports
            confidence: Confidence score of duplicate detection
        """
        try:
            # Validate required fields
            required_fields = ['text', 'location', 'issue_type']
            for field in required_fields:
                if field not in new_report:
                    print(f"Missing required field in new report: {field}")
                    return False, [], 0.0
            
            # Extract features from new report
            image_input = new_report.get('image_bytes') or new_report.get('image_array') or new_report.get('image_path')
            new_image_features = self.extract_image_features(image_input)
            new_text_embedding = self.extract_text_features(new_report['text'])
            new_location = new_report['location']
            new_issue_type = new_report['issue_type']

            # Storage for results
            similarities = []

            # Check each report in the database
            for idx, report in enumerate(self.reports_db):
                try:
                    # Check issue type match first
                    if report['issue_type'] != new_issue_type:
                        continue

                    # Check location proximity
                    dist = geopy.distance.distance(new_location, self.location_db[idx]).kilometers
                    if dist > self.location_threshold:
                        continue

                    # Text similarity
                    text_sim = cosine_similarity([new_text_embedding], [self.text_embeddings_db[idx]])[0][0]

                    # Image similarity
                    image_sim = cosine_similarity([new_image_features], [self.image_features_db[idx]])[0][0]

                    # Handle NaN values in similarities
                    if np.isnan(text_sim):
                        text_sim = 0.0
                    if np.isnan(image_sim):
                        image_sim = 0.0

                    # Calculate location similarity
                    location_sim = 1.0 - min(1.0, dist / self.location_threshold)

                    # Add debug information to see the scores
                    print(f"DEBUG: Comparing with Report ID {report.get('id', idx)}: "
                          f"Text Sim={text_sim:.2f}, Image Sim={image_sim:.2f}, Loc Sim={location_sim:.2f}")

                    # Use XGBoost model if available, trained, and enough data
                    if (
                        self.xgb_model is not None
                        and self.has_enough_data_for_xgboost
                        and len(self.reports_db) > 10  # Reduced threshold for XGBoost usage
                        and self.scaler is not None
                    ):
                        # Create feature vector
                        features = [[
                            text_sim,
                            image_sim,
                            location_sim,
                            int(new_issue_type == self.issue_types_db[idx])
                        ]]

                        # Scale features
                        features_scaled = self.scaler.transform(features)

                        # Get XGBoost prediction probability
                        prob = self.xgb_model.predict_proba(features_scaled)[0][1]  # Probability of being duplicate

                        if prob >= 0.5:  # Threshold for XGBoost confidence
                            similarities.append((report, prob))
                    else:
                        # --- CORRECTED AND RE-TUNED SIMILARITY LOGIC ---
                        
                        # Rule 1: Lowered strict override for very similar images/locations.
                        if image_sim > 0.85 and location_sim > 0.9:
                            similarities.append((report, 0.95))  # Assign a high confidence score
                            continue  # Move to next report

                        # Rule 2: Re-balanced formula giving more weight to text and location.
                        # Text (0.4) + Location (0.4) + Image (0.2)
                        overall_sim = (0.4 * text_sim) + (0.2 * image_sim) + (0.4 * location_sim)

                        # Rule 3: Lowered the overall threshold to a more reasonable value.
                        if overall_sim >= 0.65:
                            similarities.append((report, overall_sim))
                                
                except Exception as e:
                    print(f"Error comparing with report {idx}: {e}")
                    continue

            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)

            if similarities:
                return True, [report for report, _ in similarities], similarities[0][1]
            else:
                return False, [], 0.0
                
        except Exception as e:
            print(f"Error finding duplicates: {e}")
            return False, [], 0.0
    
    def process_json_input(self, json_data):
        """
        Process JSON input to determine if a report is a duplicate
        
        Args:
            json_data: String containing JSON data or dictionary
            
        Returns:
            Dictionary with duplicate status and original report ID
        """
        try:
            if isinstance(json_data, str):
                data = json.loads(json_data)
            else:
                data = json_data
            
            is_duplicate, similar_reports, confidence = self.find_duplicates(data)
            
            # Format the response with only required information
            response = {
                "is_duplicate": 1 if is_duplicate else 0
            }
            
            # Include only the original report ID, not the index
            if similar_reports and is_duplicate:
                # Map report objects to report IDs
                original_ids = []
                for report in similar_reports:
                    if 'id' in report:
                        original_ids.append(report['id'])
                
                if original_ids:
                    response["original_report_id"] = original_ids[0]  # Return only the most similar report ID
            
            return response
            
        except Exception as e:
            print(f"Error processing JSON input: {e}")
            return {"is_duplicate": 0, "error": str(e)}
    
    def rebuild_clusters_if_needed(self, force=False):
        """Rebuild clusters if database has grown significantly"""
        try:
            # Simple heuristic: rebuild if database has grown by 20%
            if force or (self.image_kmeans is not None and 
                        len(self.reports_db) > 1.2 * len(self.image_clusters)):
                self.build_clusters()
        except Exception as e:
            print(f"Error rebuilding clusters: {e}")
    
    def load_reports_from_json(self, json_file):
        """Load reports from a JSON file"""
        try:
            with open(json_file, 'r') as f:
                reports = json.load(f)
            
            for report in reports:
                self.add_report(report)
            
            # Build clusters after loading
            if len(reports) >= 2:
                self.build_clusters()
                
            print(f"Loaded {len(reports)} reports from {json_file}")
            
        except Exception as e:
            print(f"Error loading reports from JSON: {e}")

def get_duplicate_detector(**kwargs):
    """
    Flask-ready helper to get a CivicIssueDuplicateDetector instance.
    Pass kwargs to customize (e.g., n_clusters, location_threshold, text_similarity_threshold).
    n_clusters should be set to the number of unique complaints with the same location area, problem type (e.g., pothole, manhole cover removed, etc.), and time of reporting.
    """
    try:
        detector = CivicIssueDuplicateDetector(**kwargs)
        print("Duplicate detector initialized successfully")
        return detector
    except Exception as e:
        print(f"Error initializing duplicate detector: {e}")
        return None

# Flask Integration Example
def create_flask_app():
    """Create and configure Flask app with duplicate detection"""
    from flask import Flask, request, jsonify
    
    app = Flask(__name__)
    
    # Initialize detector as global variable
    app.detector = get_duplicate_detector(
        location_threshold=0.1,  # 100 meters
        text_similarity_threshold=0.65  # Lowered threshold
    )
    
    @app.route('/add_report', methods=['POST'])
    def add_report():
        """Add a new report to the detector"""
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No JSON data provided"}), 400
            
            index = app.detector.add_report(data)
            if index is not None:
                return jsonify({"success": True, "index": index}), 200
            else:
                return jsonify({"error": "Failed to add report"}), 500
                
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/check_duplicate', methods=['POST'])
    def check_duplicate():
        """Check if a report is a duplicate"""
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No JSON data provided"}), 400
            
            result = app.detector.process_json_input(data)
            return jsonify(result), 200
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/find_duplicates', methods=['POST'])
    def find_duplicates():
        """Find duplicates for a given report (detailed response)"""
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No JSON data provided"}), 400
            
            is_duplicate, similar_reports, confidence = app.detector.find_duplicates(data)
            
            response = {
                "is_duplicate": is_duplicate,
                "confidence": float(confidence),
                "similar_reports": [
                    {
                        "id": report.get('id', 'unknown'),
                        "text": report.get('text', '')[:100] + "..." if len(report.get('text', '')) > 100 else report.get('text', ''),
                        "issue_type": report.get('issue_type', ''),
                        "location": report.get('location', [])
                    }
                    for report in similar_reports[:5]  # Limit to top 5 similar reports
                ]
            }
            
            return jsonify(response), 200
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/load_reports', methods=['POST'])
    def load_reports():
        """Load reports from uploaded JSON file"""
        try:
            data = request.get_json()
            if not data or 'reports' not in data:
                return jsonify({"error": "No reports data provided"}), 400
            
            reports = data['reports']
            count = 0
            for report in reports:
                if app.detector.add_report(report) is not None:
                    count += 1
            
            # Build clusters after loading
            if count >= 2:
                app.detector.build_clusters()
            
            return jsonify({"success": True, "loaded_reports": count}), 200
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/stats', methods=['GET'])
    def get_stats():
        """Get detector statistics"""
        try:
            stats = {
                "total_reports": len(app.detector.reports_db),
                "issue_types": dict(app.detector.issue_type_clusters),
                "xgboost_trained": app.detector.has_enough_data_for_xgboost,
                "clusters_built": app.detector.image_kmeans is not None
            }
            return jsonify(stats), 200
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    return app

if __name__ == "__main__":
    print("This module is Flask-ready. Use get_duplicate_detector() to create a detector instance.")
    print("add_report expects a dict with 'text', 'location', 'issue_type', and either 'image_path', 'image_bytes', or 'image_array'.")
    
    # Test basic functionality
    try:
        detector = get_duplicate_detector()
        if detector:
            print("Basic initialization test passed")
            
            # Create Flask app
            app = create_flask_app()
            print("Flask app created successfully")
            print("Run with: app.run(debug=True)")
            
        else:
            print("Basic initialization test failed")
    except Exception as e:
        print(f"Basic test failed: {e}")
