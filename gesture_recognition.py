import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import json
import math

class GestureRecognizer:
    def __init__(self):
        # Initialize OpenCV-based detection
        self.hand_cascade = None
        try:
            # Try to load hand cascade (may not be available)
            # Note: haarcascade_hand.xml is not included in standard OpenCV distribution
            # Fallback to face detection for demo purposes
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.hand_cascade = cv2.CascadeClassifier(cascade_path)
            if self.hand_cascade.empty():
                print(f"Warning: Cascade classifier at {cascade_path} is empty")
                self.hand_cascade = None
            else:
                print(f"Successfully loaded cascade classifier from {cascade_path}")
        except Exception as e:
            print(f"Error loading cascade classifier: {e}")
            self.hand_cascade = None
        
        # Gesture templates for simple recognition
        self.gesture_templates = {
            'open_hand': {'min_area': 5000, 'aspect_ratio_range': (0.7, 1.3)},
            'help_gesture': {'min_area': 5500, 'aspect_ratio_range': (0.8, 1.2)},
            'fist': {'min_area': 2000, 'aspect_ratio_range': (0.8, 1.2)},
            'pointing': {'min_area': 1500, 'aspect_ratio_range': (0.4, 0.8)}
        }
    
    def decode_base64_image(self, base64_string):
        """Decode base64 image string to OpenCV format"""
        try:
            # Validate input
            if not isinstance(base64_string, str):
                raise ValueError("Base64 image must be a string")
                
            if not base64_string or base64_string.strip() == "":
                raise ValueError("Base64 image string is empty")
            
            # Remove data URL prefix if present
            if 'data:image' in base64_string:
                try:
                    base64_string = base64_string.split(',')[1]
                except IndexError:
                    raise ValueError("Invalid data URL format")
            
            # Decode base64
            try:
                image_data = base64.b64decode(base64_string)
                if len(image_data) == 0:
                    raise ValueError("Decoded image data is empty")
            except Exception as e:
                raise ValueError(f"Base64 decoding failed: {str(e)}")
            
            # Convert to PIL Image
            try:
                pil_image = Image.open(BytesIO(image_data))
            except Exception as e:
                raise ValueError(f"Failed to open image data: {str(e)}")
            
            # Convert to OpenCV format
            opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # Validate image dimensions
            if opencv_image.shape[0] <= 0 or opencv_image.shape[1] <= 0:
                raise ValueError("Invalid image dimensions")
                
            return opencv_image
            
        except Exception as e:
            print(f"Failed to decode image: {str(e)}")
            raise ValueError(f"Failed to decode image: {str(e)}")
    
    def detect_hands(self, image):
        """Detect hand regions in the image"""
        try:
            # Validate input image
            if image is None or image.size == 0:
                print("Error: Empty or invalid image provided to detect_hands")
                return []
                
            # Check image dimensions
            if len(image.shape) < 2:
                print(f"Error: Invalid image dimensions {image.shape}")
                return []
                
            # Convert to grayscale
            try:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            except Exception as e:
                print(f"Error converting image to grayscale: {e}")
                return []
            
            # Detect using cascade classifier
            detections = []
            if self.hand_cascade is not None:
                try:
                    detections = self.hand_cascade.detectMultiScale(
                        gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
                    )
                    print(f"Detected {len(detections)} potential hand regions")
                except Exception as e:
                    print(f"Error during cascade detection: {e}")
                    return []
            else:
                print("Warning: No cascade classifier available for detection")
            
            return detections
            
        except Exception as e:
            print(f"Hand detection error: {e}")
            return []
    
    def analyze_contours(self, image):
        """Analyze contours for gesture recognition"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Threshold the image
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by area
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Minimum area threshold
                    valid_contours.append(contour)
            
            return valid_contours
            
        except Exception as e:
            print(f"Contour analysis error: {e}")
            return []
    
    def count_fingers(self, image, contours):
        """Count fingers in the hand using contour analysis"""
        try:
            if not contours:
                return 0
                
            # Find the largest contour (assuming it's the hand)
            max_contour = max(contours, key=cv2.contourArea)
            
            # Get convex hull and defects
            hull = cv2.convexHull(max_contour, returnPoints=False)
            if len(hull) < 4:  # Not enough points for defects
                return 0
                
            defects = cv2.convexityDefects(max_contour, hull)
            if defects is None:
                return 0
                
            # Count fingers based on convexity defects
            finger_count = 0
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(max_contour[s][0])
                end = tuple(max_contour[e][0])
                far = tuple(max_contour[f][0])
                
                # Calculate angle between fingers
                a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                
                # Apply cosine law
                angle = math.acos((b**2 + c**2 - a**2) / (2*b*c)) * 57.295
                
                # If angle is less than 90 degrees, it's likely a finger
                if angle <= 90:
                    finger_count += 1
            
            # Add 1 for the thumb (usually not counted in defects)
            return finger_count + 1
            
        except Exception as e:
            print(f"Finger counting error: {e}")
            return 0
    
    def classify_gesture_simple(self, roi, detection_box):
        """Simple gesture classification based on region properties"""
        try:
            x, y, w, h = detection_box
            area = w * h
            aspect_ratio = w / h if h > 0 else 1.0
            
            # Analyze contours in the ROI
            contours = self.analyze_contours(roi)
            
            # Count fingers using contour analysis (simplified approach)
            finger_count = self.count_fingers(roi, contours)
            
            # Simple classification logic
            if area > 8000:
                # Five-finger outward palm (help gesture)
                if 0.8 <= aspect_ratio <= 1.2 and finger_count >= 5:
                    return {
                        'name': 'Help Gesture',
                        'confidence': 0.9,
                        'area': area,
                        'aspect_ratio': aspect_ratio,
                        'finger_count': finger_count,
                        'message': 'help gesture detected',
                        'emergency': True
                    }
                # Regular open hand
                elif 0.8 <= aspect_ratio <= 1.2:
                    return {
                        'name': 'Open Hand',
                        'confidence': 0.7,
                        'area': area,
                        'aspect_ratio': aspect_ratio,
                        'finger_count': finger_count
                    }
                elif aspect_ratio < 0.8:
                    return {
                        'name': 'Pointing',
                        'confidence': 0.6,
                        'area': area,
                        'aspect_ratio': aspect_ratio,
                        'finger_count': finger_count
                    }
            elif 3000 <= area <= 8000:
                return {
                    'name': 'Fist',
                    'confidence': 0.6,
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'finger_count': finger_count
                }
            
            return {
                'name': 'Unknown',
                'confidence': 0.3,
                'area': area,
                'aspect_ratio': aspect_ratio,
                'finger_count': finger_count
            }
            
        except Exception as e:
            return {
                'name': 'Error',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def recognize_gesture(self, image):
        """Main gesture recognition function"""
        try:
            # Detect hand regions
            detections = self.detect_hands(image)
            
            gestures = []
            
            # Process each detection
            for (x, y, w, h) in detections:
                # Extract region of interest
                roi = image[y:y+h, x:x+w]
                
                # Classify gesture
                gesture = self.classify_gesture_simple(roi, (x, y, w, h))
                gesture['bbox'] = {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)}
                gestures.append(gesture)
            
            return {
                'gestures': gestures,
                'hand_count': len(gestures),
                'confidence': max([g.get('confidence', 0) for g in gestures]) if gestures else 0.0,
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'error': f'Gesture recognition failed: {str(e)}',
                'gestures': [],
                'hand_count': 0,
                'confidence': 0.0,
                'status': 'error'
            }
    
    def detect_emergency_gestures(self, image):
        """Detect emergency gestures"""
        try:
            result = self.recognize_gesture(image)
            
            emergency_gestures = []
            
            for gesture in result.get('gestures', []):
                gesture_name = gesture.get('name', '').lower()
                
                # Define emergency gesture patterns
                if 'help' in gesture_name and gesture.get('confidence', 0) > 0.5:
                    emergency_gestures.append({
                        'type': 'Help Signal',
                        'urgency': 'high',
                        'description': 'Help gesture detected - five-finger outward palm',
                        'message': 'help gesture detected'
                    })
                elif 'fist' in gesture_name and gesture.get('confidence', 0) > 0.5:
                    emergency_gestures.append({
                        'type': 'Help Signal',
                        'urgency': 'medium',
                        'description': 'Closed fist detected - possible distress signal'
                    })
                elif 'pointing' in gesture_name:
                    emergency_gestures.append({
                        'type': 'Direction Signal',
                        'urgency': 'low',
                        'description': 'Pointing gesture detected'
                    })
            
            return {
                'emergency_detected': len(emergency_gestures) > 0,
                'emergency_gestures': emergency_gestures,
                'total_gestures': result.get('hand_count', 0)
            }
            
        except Exception as e:
            return {
                'emergency_detected': False,
                'emergency_gestures': [],
                'error': str(e)
            }
    
    def process_image(self, image_data):
        """Process image data and return gesture analysis"""
        try:
            # Validate input
            if image_data is None:
                raise ValueError("No image data provided")
                
            # Decode image
            if isinstance(image_data, str):
                if not image_data or image_data.strip() == "":
                    raise ValueError("Empty image data string provided")
                try:
                    image = self.decode_base64_image(image_data)
                except Exception as e:
                    raise ValueError(f"Failed to decode base64 image: {str(e)}")
            else:
                # If not string, assume it's already an image array
                if not hasattr(image_data, 'shape') or len(image_data.shape) < 2:
                    raise ValueError("Invalid image data format")
                image = image_data
            
            # Validate decoded image
            if image is None or image.size == 0 or len(image.shape) < 2:
                raise ValueError("Invalid image after decoding")
                
            # Recognize gestures
            gesture_result = self.recognize_gesture(image)
            
            # Detect emergency gestures
            emergency_result = self.detect_emergency_gestures(image)
            
            return {
                'gesture_analysis': gesture_result,
                'emergency_analysis': emergency_result,
                'image_processed': True,
                'timestamp': str(np.datetime64('now'))
            }
            
        except ValueError as e:
            print(f"Validation error in process_image: {str(e)}")
            return {
                'error': f'Image validation failed: {str(e)}',
                'gesture_analysis': {'gestures': [], 'hand_count': 0, 'confidence': 0.0},
                'emergency_analysis': {'emergency_detected': False, 'emergency_gestures': []},
                'image_processed': False
            }
        except Exception as e:
            print(f"Unexpected error in process_image: {str(e)}")
            return {
                'error': f'Image processing failed: {str(e)}',
                'gesture_analysis': {'gestures': [], 'hand_count': 0, 'confidence': 0.0},
                'emergency_analysis': {'emergency_detected': False, 'emergency_gestures': []},
                'image_processed': False
            }
    
    def get_gesture_description(self, gesture_name):
        """Get description for a gesture"""
        descriptions = {
            'open hand': 'An open palm gesture, often used for greeting or showing openness',
            'help gesture': 'Five-finger outward palm, indicating a request for help or assistance',
            'fist': 'A closed fist, can indicate determination or potential distress',
            'pointing': 'A pointing gesture, used to indicate direction or draw attention',
            'unknown': 'Gesture not recognized or unclear',
            'error': 'Error occurred during gesture recognition'
        }
        
        return descriptions.get(gesture_name.lower(), 'No description available')

# Global instance
gesture_recognizer = GestureRecognizer()

def recognize_gesture_from_image(image_data):
    """Global function for gesture recognition"""
    return gesture_recognizer.process_image(image_data)

def get_gesture_info(gesture_name):
    """Global function to get gesture information"""
    return gesture_recognizer.get_gesture_description(gesture_name)
