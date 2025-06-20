from collections import defaultdict, deque
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import torch
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("pothole_detector.log"),
                              logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Load the model with explicit version specification compatible with Python 3.11
def load_model(model_path):
    """
    Loads the YOLO model with error handling.
    
    Args:
        model_path: Path to the ONNX model
        
    Returns:
        model: Loaded YOLO model
    """
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        logger.info(f"Loading model from {model_path}")
        model = YOLO(model_path)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def estimate_pothole_depth(image, binary_mask, contour):
    """
    Estimates the depth of a pothole based on shadow analysis in the pothole region.
    
    Args:
        image: Input image (BGR format)
        binary_mask: Binary mask of the pothole
        contour: Contour of the pothole
        
    Returns:
        depth_score: Estimated depth score (0-1)
    """
    try:
        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Create mask from contour for precise region analysis
        mask = np.zeros_like(gray_image)
        cv2.drawContours(mask, [contour], 0, 255, -1)
        
        # Extract only the pothole region using the mask
        pothole_region = cv2.bitwise_and(gray_image, gray_image, mask=mask)
        
        # Get pixel values excluding zeros (background)
        pixel_values = pothole_region[pothole_region > 0]
        
        if len(pixel_values) == 0:
            return 0.0  # No valid pixels
        
        # Calculate statistics of the pothole region
        mean_value = np.mean(pixel_values)
        min_value = np.min(pixel_values)
        
        # Calculate the depth score based on darkness and contrast
        # Darker regions indicate deeper potholes
        # Normalize to 0-1 range where 1 is deepest
        darkness_score = 1 - (mean_value / 255.0)
        
        # Calculate contrast within the pothole (higher contrast often means deeper)
        if len(pixel_values) > 1:
            std_dev = np.std(pixel_values)
            contrast_score = min(std_dev / 50.0, 1.0)  # Normalize, cap at 1.0
        else:
            contrast_score = 0.0
        
        # Combined score with more weight on darkness
        depth_score = (0.7 * darkness_score) + (0.3 * contrast_score)
        
        # Ensure it's in 0-1 range
        depth_score = max(0.0, min(1.0, depth_score))
        
        return depth_score
    except Exception as e:
        logger.error(f"Error estimating pothole depth: {e}")
        return 0.0  # Return safe default value


def get_individual_pothole_priority(area_ratio, depth_score):
    """
    Determines the priority of an individual pothole based on size and depth.
    
    Args:
        area_ratio: Ratio of pothole area to image area
        depth_score: Estimated depth score (0-1)
        
    Returns:
        priority: String priority level ('High', 'Medium', or 'Low')
        color: BGR color tuple for visualization
    """
    try:
        # Calculate combined score weighted by area and depth
        # Area is more important for overall road damage assessment
        combined_score = (0.6 * area_ratio * 100) + (0.4 * depth_score)
        
        # Determine priority based on combined score
        if combined_score > 0.4 or (area_ratio > 0.01 and depth_score > 0.6):
            priority = 'High'
            color = (0, 0, 255)  # Red (BGR)
        elif combined_score > 0.2 or (area_ratio > 0.005 and depth_score > 0.4):
            priority = 'Medium'
            color = (0, 165, 255)  # Orange (BGR)
        else:
            priority = 'Low'
            color = (0, 255, 0)  # Green (BGR)
        
        return priority, color
    except Exception as e:
        logger.error(f"Error determining pothole priority: {e}")
        return 'Low', (0, 255, 0)  # Return safe default values


def determine_road_priority(potholes_list, proximity_threshold, image_shape):
    """
    Determines the overall road priority based on pothole count, proximity, and severity.
    
    Args:
        potholes_list: List of detected potholes with priority information
        proximity_threshold: Maximum distance to consider potholes as clustered
        image_shape: Shape of the input image
        
    Returns:
        road_priority: Overall road priority ('High', 'Medium', or 'Low')
        road_color: BGR color tuple for visualization
        clusters: List of clusters, where each cluster is a list of pothole indices
    """
    try:
        # If no potholes, return low priority
        if not potholes_list:
            return 'Low', (0, 255, 0), []
        
        # Count high and medium priority potholes
        high_priority_count = sum(1 for p in potholes_list if p['priority'] == 'High')
        medium_priority_count = sum(1 for p in potholes_list if p['priority'] == 'Medium')
        
        # Find clusters of potholes (potholes in proximity indicate concentrated damage)
        clusters = []
        processed = set()
        
        for i, pothole1 in enumerate(potholes_list):
            if i in processed:
                continue
            
            cluster = [i]
            processed.add(i)
            
            for j, pothole2 in enumerate(potholes_list):
                if j in processed or i == j:
                    continue
                
                # Calculate distance between potholes
                pos1 = pothole1['position']
                pos2 = pothole2['position']
                distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                
                # If close enough, add to cluster
                if distance < proximity_threshold:
                    cluster.append(j)
                    processed.add(j)
            
            clusters.append(cluster)
        
        # Calculate total damaged area (as percentage of road)
        total_area_ratio = sum(p['area_ratio'] for p in potholes_list)
        
        # Calculate area of largest cluster as a percentage of image area
        largest_cluster_area = 0
        if clusters:
            for cluster in clusters:
                if len(cluster) > 1:  # Only consider clusters with multiple potholes
                    cluster_points = np.array([potholes_list[idx]['position'] for idx in cluster])
                    hull = cv2.convexHull(np.array(cluster_points).reshape(-1, 1, 2).astype(np.int32))
                    cluster_area = cv2.contourArea(hull) / (image_shape[0] * image_shape[1])
                    largest_cluster_area = max(largest_cluster_area, cluster_area)
        
        # Determine road priority using multiple factors
        if (high_priority_count >= 2 or 
            (high_priority_count >= 1 and medium_priority_count >= 2) or
            total_area_ratio > 0.05 or
            largest_cluster_area > 0.03 or
            len([c for c in clusters if len(c) >= 3]) >= 1):  # Cluster with 3+ potholes
            road_priority = 'High'
            road_color = (0, 0, 255)  # Red (BGR)
        elif (high_priority_count >= 1 or 
            medium_priority_count >= 2 or
            total_area_ratio > 0.02 or
            largest_cluster_area > 0.015 or
            len([c for c in clusters if len(c) >= 2]) >= 1):  # Cluster with 2+ potholes
            road_priority = 'Medium'
            road_color = (0, 165, 255)  # Orange (BGR)
        else:
            road_priority = 'Low'
            road_color = (0, 255, 0)  # Green (BGR)
        
        return road_priority, road_color, clusters
    except Exception as e:
        logger.error(f"Error determining road priority: {e}")
        return 'Low', (0, 255, 0), []  # Return safe default values

def assess_road_priority(image_input, conf_threshold=0.25, proximity_threshold=150, model=None):
    """
    Assesses the road priority for a given image based on pothole detection.

    Args:
        image_input: Path to the image (str) or numpy array (OpenCV BGR)
        conf_threshold: Confidence threshold for pothole detection
        proximity_threshold: Distance threshold for clustering potholes
        model: YOLO model for pothole detection

    Returns:
        annotated_image: Image annotated with pothole detections and priority
        road_info: Dictionary containing road priority information
    """
    try:
        # Accept either file path or numpy array
        if isinstance(image_input, str):
            logger.info(f"Processing image: {image_input}")
            image = cv2.imread(image_input)
            if image is None:
                raise ValueError(f"Unable to read image: {image_input}")
        else:
            logger.info(f"Processing image from array input")
            image = image_input
            if image is None or not isinstance(image, np.ndarray):
                raise ValueError("Invalid image array input")

        original_image = image.copy()

        # Run inference with YOLO model
        if model is None:
            raise ValueError("Model not provided")

        # Run model with updated API call compatible with newer ultralytics versions
        start_time = time.time()
        results = model.predict(image, conf=conf_threshold)[0]
        inference_time = time.time() - start_time
        logger.info(f"Inference completed in {inference_time:.2f} seconds")

        potholes_list = []
        image_area = image.shape[0] * image.shape[1]

        # Process each detection
        for i, detection in enumerate(results.boxes):
            if len(detection.xyxy) == 0:
                continue
                
            box = detection.xyxy.cpu().numpy()[0]
            conf = detection.conf.cpu().numpy()[0]
            
            if conf < conf_threshold:
                continue
                
            x1, y1, x2, y2 = map(int, box)
            
            # Create binary mask for this pothole
            binary_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            cv2.rectangle(binary_mask, (x1, y1), (x2, y2), 255, -1)
            
            # Find contours in the binary mask
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
                
            main_contour = max(contours, key=cv2.contourArea)
            
            # Calculate area ratio (pothole area / image area)
            pothole_area = cv2.contourArea(main_contour)
            area_ratio = pothole_area / image_area
            
            # Estimate depth
            depth_score = estimate_pothole_depth(original_image, binary_mask, main_contour)
            
            # Get priority and color
            priority, color = get_individual_pothole_priority(area_ratio, depth_score)
            
            # Calculate centroid
            M = cv2.moments(main_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            # Add pothole to list
            potholes_list.append({
                'position': (cx, cy),
                'area_ratio': area_ratio,
                'depth_score': depth_score,
                'priority': priority,
                'color': color,
                'box': (x1, y1, x2, y2),
                'contour': main_contour,
                'confidence': float(conf)
            })
            
        logger.info(f"Detected {len(potholes_list)} potholes")
        
        # Determine road priority
        road_priority, road_color, clusters = determine_road_priority(
            potholes_list, proximity_threshold, image.shape)
        
        # Annotate image
        annotated_image = original_image.copy()
        
        # Draw all pothole detections
        for pothole in potholes_list:
            x1, y1, x2, y2 = pothole['box']
            color = pothole['color']
            priority = pothole['priority']
            conf = pothole['confidence']
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw contour
            cv2.drawContours(annotated_image, [pothole['contour']], 0, color, 2)
            
            # Add priority label with confidence
            cv2.putText(annotated_image, f"{priority} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw clusters
        for i, cluster in enumerate(clusters):
            if len(cluster) > 1:  # Only draw clusters with multiple potholes
                # Get all points in the cluster
                cluster_points = np.array([potholes_list[idx]['position'] for idx in cluster])
                
                # Draw convex hull around the cluster
                hull = cv2.convexHull(np.array(cluster_points).reshape(-1, 1, 2).astype(np.int32))
                cv2.polylines(annotated_image, [hull], True, (255, 255, 0), 2)
                
                # Label the cluster
                cx = int(np.mean(cluster_points[:, 0]))
                cy = int(np.mean(cluster_points[:, 1]))
                cv2.putText(annotated_image, f"Cluster {i+1}", (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Add road priority text at the top of the image
        cv2.putText(annotated_image, f"Road Priority: {road_priority}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, road_color, 3)
        
        # Count potholes by priority
        priority_counts = {
            'High': sum(1 for p in potholes_list if p['priority'] == 'High'),
            'Medium': sum(1 for p in potholes_list if p['priority'] == 'Medium'),
            'Low': sum(1 for p in potholes_list if p['priority'] == 'Low')
        }
        
        # Prepare road info
        road_info = {
            'road_priority': road_priority,
            'total_potholes': len(potholes_list),
            'pothole_clusters': len(clusters),
            'individual_priorities': priority_counts,
            'clusters': clusters,
            'inference_time': inference_time
        }
        
        logger.info(f"Road priority assessment completed: {road_priority}")
        
        return annotated_image, road_info
    
    except Exception as e:
        logger.error(f"Error assessing road priority: {e}", exc_info=True)
        return None, None

def process_video_for_road_priority(video_path, conf_threshold=0.25, proximity_threshold=150, model=None, 
                                   skip_frames=0, output_path=None):
    """
    Processes a video to assess road priority frame by frame.
    
    Args:
        video_path: Path to the video
        conf_threshold: Confidence threshold for pothole detection
        proximity_threshold: Distance threshold for clustering potholes
        model: YOLO model for pothole detection
        skip_frames: Number of frames to skip between processing (0 = process all frames)
        output_path: Optional custom output path
        
    Returns:
        output_path: Path to the annotated video
        road_summary: Summary of road priorities across the video
    """
    try:
        if model is None:
            raise ValueError("Model not provided")
        
        logger.info(f"Processing video: {video_path}")
        
        # Open the video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video: {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create output path
        if output_path is None:
            base_name = os.path.basename(video_path)
            base_name_no_ext = os.path.splitext(base_name)[0]
            output_path = f"{base_name_no_ext}_analyzed.mp4"
        
        # Initialize video writer with correct codec for Python 3.11
        # Use h264 codec if available, fall back to mp4v
        try:
            fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Try using H.264 codec first (better quality)
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if not out.isOpened():
                raise Exception("avc1 codec failed")
        except Exception:
            try:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Fall back to mp4v
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            except Exception:
                fourcc = 0  # Fall back to default codec if all else fails
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Initialize tracking variables
        road_priorities = []
        frame_count = 0
        processed_frames = 0
        
        # Process each frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames if specified
            if skip_frames > 0 and frame_count % (skip_frames + 1) != 1:
                # Still write the original frame
                out.write(frame)
                continue
            
            processed_frames += 1
            logger.info(f"Processing frame {frame_count}/{total_frames} (processed {processed_frames})")
            
            try:
                start_time = time.time()
                
                # Process the frame with the updated API
                results = model.predict(frame, conf=conf_threshold)[0]
                
                # Create a copy of the frame
                original_frame = frame.copy()
                
                potholes_list = []
                image_area = frame.shape[0] * frame.shape[1]
                
                # Process each detection with updated box extraction
                for detection in results.boxes:
                    if len(detection.xyxy) == 0:
                        continue
                        
                    box = detection.xyxy.cpu().numpy()[0]
                    conf = detection.conf.cpu().numpy()[0]
                    
                    if conf < conf_threshold:
                        continue
                        
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Create binary mask for this pothole
                    binary_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    cv2.rectangle(binary_mask, (x1, y1), (x2, y2), 255, -1)
                    
                    # Find contours in the binary mask
                    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if not contours:
                        continue
                        
                    main_contour = max(contours, key=cv2.contourArea)
                    
                    # Calculate area ratio (pothole area / image area)
                    pothole_area = cv2.contourArea(main_contour)
                    area_ratio = pothole_area / image_area
                    
                    # Estimate depth
                    depth_score = estimate_pothole_depth(original_frame, binary_mask, main_contour)
                    
                    # Get priority and color
                    priority, color = get_individual_pothole_priority(area_ratio, depth_score)
                    
                    # Calculate centroid
                    M = cv2.moments(main_contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                    else:
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    # Add pothole to list
                    potholes_list.append({
                        'position': (cx, cy),
                        'area_ratio': area_ratio,
                        'depth_score': depth_score,
                        'priority': priority,
                        'color': color,
                        'box': (x1, y1, x2, y2),
                        'contour': main_contour,
                        'confidence': float(conf)
                    })
                
                # Determine road priority
                road_priority, road_color, clusters = determine_road_priority(
                    potholes_list, proximity_threshold, frame.shape)
                
                # Record road priority
                road_priorities.append(road_priority)
                
                # Annotate frame
                annotated_frame = original_frame.copy()
                
                # Draw all pothole detections
                for pothole in potholes_list:
                    x1, y1, x2, y2 = pothole['box']
                    color = pothole['color']
                    priority = pothole['priority']
                    conf = pothole['confidence']
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw contour
                    cv2.drawContours(annotated_frame, [pothole['contour']], 0, color, 2)
                    
                    # Add priority label with confidence
                    cv2.putText(annotated_frame, f"{priority} ({conf:.2f})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw clusters
                for i, cluster in enumerate(clusters):
                    if len(cluster) > 1:  # Only draw clusters with multiple potholes
                        # Get all points in the cluster
                        cluster_points = np.array([potholes_list[idx]['position'] for idx in cluster])
                        
                        # Draw convex hull around the cluster with explicit int32 dtype
                        hull = cv2.convexHull(cluster_points.reshape(-1, 1, 2).astype(np.int32))
                        cv2.polylines(annotated_frame, [hull], True, (255, 255, 0), 2)
                        
                        # Label the cluster
                        cx = int(np.mean(cluster_points[:, 0]))
                        cy = int(np.mean(cluster_points[:, 1]))
                        cv2.putText(annotated_frame, f"Cluster {i+1}", (cx, cy),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Add road priority text at the top of the frame
                cv2.putText(annotated_frame, f"Road Priority: {road_priority}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, road_color, 3)
                
                # Add processing info at the bottom of the frame
                process_time = time.time() - start_time
                cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames} | Processing time: {process_time:.2f}s", 
                            (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Write frame to output video
                out.write(annotated_frame)
                
                logger.info(f"Frame {frame_count} processed in {process_time:.2f}s")
                    
            except Exception as e:
                logger.error(f"Error processing frame {frame_count}: {e}", exc_info=True)
                # Write original frame if error occurs
                out.write(frame)
        
        # Release resources
        cap.release()
        out.release()
        
        # Calculate summary statistics
        priority_distribution = {
            'High': road_priorities.count('High'),
            'Medium': road_priorities.count('Medium'),
            'Low': road_priorities.count('Low')
        }
        
        # Find most common road priority
        most_common_priority = max(priority_distribution, key=priority_distribution.get) if road_priorities else 'Unknown'
        
        # Calculate high priority percentage
        high_priority_percentage = (priority_distribution['High'] / processed_frames) * 100 if processed_frames > 0 else 0
        
        # Prepare road summary
        road_summary = {
            'total_frames': frame_count,
            'processed_frames': processed_frames,
            'priority_distribution': priority_distribution,
            'most_common_road_priority': most_common_priority,
            'high_priority_percentage': high_priority_percentage
        }
        
        logger.info(f"Video processing completed: {output_path}")
        return output_path, road_summary
    
    except Exception as e:
        logger.error(f"Error processing video: {e}", exc_info=True)
        if 'cap' in locals() and cap is not None:
            cap.release()
        if 'out' in locals() and out is not None:
            out.release()
        return None, None

def process_road_image_example(image_path, model_path="pothole_detector_v1.onnx"):
    """
    Example function to process a single road image.
    
    Args:
        image_path: Path to the input image
        model_path: Path to the ONNX model
        
    Returns:
        annotated_image: Image with pothole detections and priority information
        road_info: Dictionary containing road priority information
    """
    try:
        # Load the model
        model = load_model(model_path)
        
        # Process the image to assess road priority
        annotated_image, road_info = assess_road_priority(
            image_path, 
            conf_threshold=0.25,
            proximity_threshold=150,  # Adjust based on your image scale
            model=model
        )
        
        if annotated_image is None or road_info is None:
            logger.error("Failed to process the image")
            return None, None
            
        # Save the output image
        output_path = os.path.splitext(image_path)[0] + "_analyzed.jpg"
        cv2.imwrite(output_path, annotated_image)
        logger.info(f"Saved analyzed image to {output_path}")
        
        # Display the results
        plt.figure(figsize=(12, 8))
        
        # Convert BGR to RGB for matplotlib
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        plt.imshow(annotated_image_rgb)
        plt.title(f"Road Assessment: {road_info['road_priority']} Priority\n"
                f"Total Potholes: {road_info['total_potholes']}, Clusters: {road_info['pothole_clusters']}")
        plt.axis('off')
        plt.savefig('road_assessment_result.png')
        plt.show()
        
        print("\nRoad Priority Assessment Summary:")
        print(f"Road Priority: {road_info['road_priority']}")
        print(f"Total Potholes: {road_info['total_potholes']}")
        print(f"Pothole Clusters: {road_info['pothole_clusters']}")
        print(f"Processing Time: {road_info['inference_time']:.2f} seconds")
        print("\nIndividual Pothole Priorities:")
        print(f"  High: {road_info['individual_priorities']['High']}")
        print(f"  Medium: {road_info['individual_priorities']['Medium']}")
        print(f"  Low: {road_info['individual_priorities']['Low']}")
        
        return annotated_image, road_info
    
    except Exception as e:
        logger.error(f"Error processing image: {e}", exc_info=True)
        return None, None

def process_road_video_example(video_path, model_path="pothole_detector_v1.onnx", skip_frames=0):
    """
    Example function to process a road video.
    
    Args:
        video_path: Path to the input video
        model_path: Path to the ONNX model
        skip_frames: Number of frames to skip between processing (0 = process all frames)
        
    Returns:
        output_path: Path to the output video
        road_summary: Dictionary containing road priority information
    """
    try:
        # Load the model
        model = load_model(model_path)
        
        # Process the video
        start_time = time.time()
        output_path, road_summary = process_video_for_road_priority(
            video_path, 
            conf_threshold=0.25,
            proximity_threshold=150,  # Adjust based on your video scale
            model=model,
            skip_frames=skip_frames
        )
        total_time = time.time() - start_time
        
        if output_path is None or road_summary is None:
            logger.error("Failed to process the video")
            return None, None
        
        print("\nVideo Processing Complete!")
        print(f"Processed video saved to: {output_path}")
        print(f"Total processing time: {total_time:.2f} seconds")
        print("\nRoad Assessment Summary:")
        print(f"Total frames processed: {road_summary['total_frames']}")
        print(f"Frames analyzed: {road_summary['processed_frames']}")
        
        print("\nRoad Priority Distribution:")
        for priority, count in road_summary['priority_distribution'].items():
            percentage = (count / road_summary['processed_frames']) * 100 if road_summary['processed_frames'] > 0 else 0
            print(f"  {priority}: {count} frames ({percentage:.1f}%)")
        
        print(f"\nMost Common Road Priority: {road_summary['most_common_road_priority']}")
        print(f"High Priority Percentage: {road_summary['high_priority_percentage']:.1f}%")
        
        # Create a simple visualization of the road priority distribution
        priorities = ['High', 'Medium', 'Low']
        counts = [road_summary['priority_distribution'][p] for p in priorities]
        colors = ['red', 'orange', 'green']
        
        plt.figure(figsize=(10, 6))
        plt.bar(priorities, counts, color=colors)
        plt.title('Road Priority Distribution')
        plt.xlabel('Priority Level')
        plt.ylabel('Number of Frames')
        for i, count in enumerate(counts):
            plt.text(i, count + 0.5, str(count), ha='center')
        
        # Save the chart
        chart_path = os.path.splitext(video_path)[0] + "_priority_chart.png"
        plt.savefig(chart_path)
        plt.close()
        
        logger.info(f"Generated priority distribution chart: {chart_path}")
        print(f"\nPriority distribution chart saved to: {chart_path}")
        
        return output_path, road_summary
    
    except Exception as e:
        logger.error(f"Error in process_road_video_example: {e}", exc_info=True)
        return None, None


import json

def decode_image_bytes(image_bytes):
    """
    Helper to decode image bytes (from SQLite BLOB) to a numpy array (OpenCV format).
    Args:
        image_bytes: Raw image bytes (e.g., from SQLite BLOB)
    Returns:
        image_array: Decoded numpy array (BGR, as used by OpenCV)
    """
    import numpy as np
    import cv2
    image_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return img

def run_pothole_detection(image_path):
    """
    Flask-ready entry point for pothole detection from file path.
    Uses the hardcoded model_path, processes the given image, and returns results.
    Args:
        image_path: Path to the input image.
    Returns:
        result_json: JSON-serializable dict with detection info
        annotated_image_bytes: Annotated image as bytes (for SQLite BLOB)
    """
    model_path = "pothole_detector_v1.onnx"
    if not os.path.exists(image_path):
        logger.error(f"Input image not found: {image_path}")
        return None, None
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return None, None
    model = load_model(model_path)
    annotated_image, road_info = assess_road_priority(
        image_path,
        conf_threshold=0.25,
        proximity_threshold=150,
        model=model
    )
    if annotated_image is None or road_info is None:
        return None, None
    # Encode annotated image as bytes (for SQLite BLOB)
    success, img_bytes = cv2.imencode('.jpg', annotated_image)
    if not success:
        return None, None
    result_json = {
        "road_priority": road_info.get("road_priority"),
        "total_potholes": road_info.get("total_potholes"),
        "pothole_clusters": road_info.get("pothole_clusters"),
        "individual_priorities": road_info.get("individual_priorities"),
        "clusters": road_info.get("clusters"),
        "inference_time": road_info.get("inference_time"),
    }
    return result_json, img_bytes.tobytes()

def run_pothole_detection_from_bytes(image_bytes):
    """
    Flask-ready entry point for pothole detection from image bytes (e.g., from SQLite BLOB).
    Args:
        image_bytes: Raw image bytes (e.g., from SQLite BLOB)
    Returns:
        result_json: JSON-serializable dict with detection info
        annotated_image_bytes: Annotated image as bytes (for SQLite BLOB)
    """
    model_path = "pothole_detector_v1.onnx"
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return None, None
    image_array = decode_image_bytes(image_bytes)
    if image_array is None:
        logger.error("Failed to decode image bytes")
        return None, None
    model = load_model(model_path)
    annotated_image, road_info = assess_road_priority(
        image_array,
        conf_threshold=0.25,
        proximity_threshold=150,
        model=model
    )
    if annotated_image is None or road_info is None:
        return None, None
    success, img_bytes = cv2.imencode('.jpg', annotated_image)
    if not success:
        return None, None
    result_json = {
        "road_priority": road_info.get("road_priority"),
        "total_potholes": road_info.get("total_potholes"),
        "pothole_clusters": road_info.get("pothole_clusters"),
        "individual_priorities": road_info.get("individual_priorities"),
        "clusters": road_info.get("clusters"),
        "inference_time": road_info.get("inference_time"),
    }
    return result_json, img_bytes.tobytes()

if __name__ == "__main__":
    # Example usage for manual/script testing (no hardcoded image path)
    print("This module is Flask-ready. Use run_pothole_detection(image_path) to process an image.")
