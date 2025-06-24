import os
import cv2
import numpy as np
import logging
import time
import json
import argparse
from collections import Counter

# --- Configuration ---
# Ensure uploads directory exists
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configure logging to file and console
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("pothole_detector.log"),
                              logging.StreamHandler()])
logger = logging.getLogger(__name__)


# --- Core Model and Logic Functions ---

def load_model(model_path):
    """
    Loads the YOLO ONNX model and wraps it for consistent inference.
    """
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info(f"Loading model from {model_path}")
        import onnxruntime as ort

        providers = ['CPUExecutionProvider']
        session = ort.InferenceSession(model_path, providers=providers)

        class ONNXWrapper:
            def __init__(self, session):
                self.session = session

            def __call__(self, img, conf=0.25, imgsz=640):
                h_orig, w_orig = img.shape[:2]
                img_resized = cv2.resize(img, (imgsz, imgsz))
                img_normalized = img_resized.transpose(2, 0, 1).astype('float32') / 255.0
                img_batch = np.expand_dims(img_normalized, axis=0)

                outputs = self.session.run(None, {'images': img_batch})
                
                output_data = outputs[0][0].T
                
                valid_detections = output_data[output_data[:, 4] > conf]
                if len(valid_detections) == 0:
                    return [type('obj', (object,), {'boxes': type('obj', (object,), {'xyxy': [], 'conf': []})()})()]
                
                box_coords = valid_detections[:, :4]
                scores = valid_detections[:, 4]

                x1 = box_coords[:, 0] - box_coords[:, 2] / 2
                y1 = box_coords[:, 1] - box_coords[:, 3] / 2
                x2 = box_coords[:, 0] + box_coords[:, 2] / 2
                y2 = box_coords[:, 1] + box_coords[:, 3] / 2
                boxes_for_nms = np.stack([x1, y1, x2, y2], axis=1)

                indices = cv2.dnn.NMSBoxes(boxes_for_nms.tolist(), scores.tolist(), conf, 0.45)
                if len(indices) == 0:
                    return [type('obj', (object,), {'boxes': type('obj', (object,), {'xyxy': [], 'conf': []})()})()]

                indices = indices.flatten()
                final_boxes_normalized = boxes_for_nms[indices]
                final_scores = scores[indices]

                w_scale, h_scale = w_orig / imgsz, h_orig / imgsz
                final_boxes_scaled = []
                for box in final_boxes_normalized:
                    final_boxes_scaled.append([
                        int(box[0] * w_scale), int(box[1] * h_scale),
                        int(box[2] * w_scale), int(box[3] * h_scale)
                    ])
                
                class MockBoxes:
                    def __init__(self, boxes, confs):
                        self.xyxy = np.array(boxes)
                        self.conf = np.array(confs)

                class MockResult:
                    def __init__(self, boxes, confs):
                        self.boxes = MockBoxes(boxes, confs)
                
                return [MockResult(final_boxes_scaled, final_scores)]

        model = ONNXWrapper(session)
        logger.info("ONNX model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to load ONNX model: {e}", exc_info=True)
        raise

def estimate_pothole_depth(image, contour):
    """
    Estimates pothole depth score (0-1) based on shadow analysis.
    """
    try:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(gray_image)
        cv2.drawContours(mask, [contour], 0, 255, -1)
        
        pixel_values = gray_image[mask == 255]
        if pixel_values.size == 0: return 0.0
        
        darkness_score = 1 - (np.mean(pixel_values) / 255.0)
        contrast_score = min(np.std(pixel_values) / 50.0, 1.0) if pixel_values.size > 1 else 0.0
        return max(0.0, min(1.0, (0.7 * darkness_score) + (0.3 * contrast_score)))
    except Exception as e:
        logger.warning(f"Could not estimate depth: {e}")
        return 0.0

def get_individual_pothole_priority(area_ratio, depth_score):
    """
    Determines an individual pothole's priority ('High', 'Medium', 'Low').
    """
    combined_score = (0.6 * area_ratio * 100) + (0.4 * depth_score)
    if combined_score > 0.6 or (area_ratio > 0.01 and depth_score > 0.6):
        return 'High', (0, 0, 255)  # Red
    elif combined_score > 0.3 or (area_ratio > 0.005 and depth_score > 0.4):
        return 'Medium', (0, 165, 255)  # Orange
    else:
        return 'Low', (0, 255, 0)  # Green

def determine_road_priority(potholes_list, proximity_threshold, image_shape):
    """
    Determines the overall road priority based on all detected potholes.
    """
    if not potholes_list:
        return 'Low', (0, 255, 0), []
    
    high_count = sum(1 for p in potholes_list if p['priority'] == 'High')
    medium_count = sum(1 for p in potholes_list if p['priority'] == 'Medium')
    
    clusters, processed = [], set()
    for i, p1 in enumerate(potholes_list):
        if i in processed: continue
        cluster, q = [i], [i]
        processed.add(i)
        while q:
            curr = q.pop(0)
            for j, p2 in enumerate(potholes_list):
                if j not in processed and np.linalg.norm(np.array(p1['position']) - np.array(p2['position'])) < proximity_threshold:
                    processed.add(j)
                    cluster.append(j)
                    q.append(j)
        clusters.append(cluster)
    
    total_area_ratio = sum(p['area_ratio'] for p in potholes_list)
    
    if (high_count >= 2 or (high_count >= 1 and medium_count >= 2) or
        total_area_ratio > 0.05 or len([c for c in clusters if len(c) >= 3]) > 0):
        return 'High', (0, 0, 255), clusters
    elif (high_count >= 1 or medium_count >= 2 or total_area_ratio > 0.02 or 
          len([c for c in clusters if len(c) >= 2]) > 0):
        return 'Medium', (0, 165, 255), clusters
    else:
        return 'Low', (0, 255, 0), clusters


# --- Main Assessment Function for Images ---

def assess_road_image(image_path, model, conf_threshold=0.25, proximity_threshold=150):
    """
    Assesses a single image, returning a JSON report and an annotated image.
    """
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
    else: # Assumes image_path is a numpy array
        image = image_path

    annotated_image = image.copy()
    h, w = image.shape[:2]
    image_area = h * w
    
    results = model(image, conf=conf_threshold)
    detections = results[0].boxes
    
    potholes_list = []
    if len(detections.xyxy) > 0:
        for i in range(len(detections.xyxy)):
            x1, y1, x2, y2 = map(int, detections.xyxy[i])
            confidence = float(detections.conf[i])
            
            contour = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
            area_ratio = cv2.contourArea(contour) / image_area
            depth_score = estimate_pothole_depth(image, contour)
            priority, color = get_individual_pothole_priority(area_ratio, depth_score)
            
            potholes_list.append({
                'id': i, 'position': ((x1 + x2) // 2, (y1 + y2) // 2),
                'bbox': [x1, y1, x2, y2], 'area_ratio': area_ratio,
                'depth_score': depth_score, 'priority': priority, 'confidence': confidence
            })
            
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_image, f"{priority} ({confidence:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    road_priority, road_color, clusters = determine_road_priority(potholes_list, proximity_threshold, (h, w))
    
    for i, cluster in enumerate(clusters):
        if len(cluster) > 1:
            points = np.array([potholes_list[idx]['position'] for idx in cluster]).astype(np.int32)
            hull = cv2.convexHull(points.reshape(-1, 1, 2))
            cv2.polylines(annotated_image, [hull], True, (255, 0, 255), 2)
    
    cv2.putText(annotated_image, f"Road Priority: {road_priority}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, road_color, 3)
    
    priority_counts = Counter(p['priority'] for p in potholes_list)
    source_name = os.path.basename(image_path) if isinstance(image_path, str) else "image_from_memory"

    assessment_data = {
        "source": source_name, "road_priority": road_priority,
        "total_potholes": len(potholes_list),
        "priority_distribution": dict(priority_counts),
        "cluster_count": len([c for c in clusters if len(c) > 1]),
        "potholes": [{k: p[k] for k in ('id', 'bbox', 'priority', 'depth_score', 'confidence')} for p in potholes_list]
    }
    
    return json.dumps(assessment_data, indent=2), annotated_image


# --- Main Execution Block ---

# --- Flask Integration Functions ---

def run_pothole_detection(image_path):
    """
    Flask-ready entry point for pothole detection from file path.
    Returns:
        result_json: JSON-serializable dict with detection info
        annotated_image_bytes: Annotated image as bytes (for SQLite BLOB)
    """
    try:
        model = load_model("pothole_detector_v1.onnx")
        json_output, annotated_image = assess_road_image(image_path, model)
        
        # Convert JSON string back to dict and add image path
        result_dict = json.loads(json_output)
        result_dict['image_path'] = image_path
        
        # Convert annotated image to bytes
        success, img_encoded = cv2.imencode('.jpg', annotated_image)
        if not success:
            return None, None
            
        return result_dict, img_encoded.tobytes()
        
    except Exception as e:
        logger.error(f"Error in run_pothole_detection: {e}", exc_info=True)
        return None, None

def run_pothole_detection_from_bytes(image_bytes):
    """
    Flask-ready entry point for pothole detection from image bytes.
    Returns:
        result_json: JSON-serializable dict with detection info
        annotated_image_bytes: Annotated image as bytes (for SQLite BLOB)
    """
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None, None
            
        # Run detection
        model = load_model("pothole_detector_v1.onnx")
        json_output, annotated_image = assess_road_image(img, model)
        
        # Convert JSON string back to dict
        result_dict = json.loads(json_output)
        
        # Convert annotated image to bytes
        success, img_encoded = cv2.imencode('.jpg', annotated_image)
        if not success:
            return None, None
            
        return result_dict, img_encoded.tobytes()
        
    except Exception as e:
        logger.error(f"Error in run_pothole_detection_from_bytes: {e}", exc_info=True)
        return None, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pothole Detection and Road Priority Assessment for Images.")
    parser.add_argument("--image", type=str, help="Path to a single image for analysis.")
    parser.add_argument("--model", type=str, default="pothole_detector_v1.onnx", help="Path to the ONNX model file.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for detection.")

    args = parser.parse_args()

    try:
        model = load_model(args.model)
        
        if args.image:
            if not os.path.exists(args.image):
                logger.error(f"Image file not found: {args.image}")
            else:
                logger.info(f"--- Processing Image: {args.image} ---")
                json_output, annotated_image = assess_road_image(args.image, model, conf_threshold=args.conf)
                
                output_path = f"{os.path.splitext(args.image)[0]}_assessed.jpg"
                cv2.imwrite(output_path, annotated_image)
                
                print("\n--- Assessment Report ---")
                print(json_output)
                logger.info(f"Annotated image saved to: {output_path}")
        
        else:
            parser.print_help()
            logger.warning("No input file specified. Use the --image argument.")

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
