import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import json

class GestureRecognizer:
    def __init__(self):
        # Initialize OpenCV-based detection
        self.hand_cascade = None
        try:
            # Try to load hand cascade (may not be available)
            self.hand_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_hand.xml')
        except:
            # Fallback to face detection for demo purposes
            self.hand_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Gesture templates for simple recognition
        self.gesture_templates = {
            'open_hand': {'min_area': 5000, 'aspect_ratio_range': (0.7, 1.3)},
            'fist': {'min_area': 2000, 'aspect_ratio_range': (0.8, 1.2)},
            'pointing': {'min_area': 1500, 'aspect_ratio_range': (0.4, 0.8)}
        }
    
    def decode_base64_image(self, base64_string):
        """Decode base64 image string to OpenCV format"""
        try:
            # Remove data URL prefix if present
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            # Decode base64
            image_data = base64.b64decode(base64_string)
            
            # Convert to PIL Image
            pil_image = Image.open(BytesIO(image_data))
            
            # Convert to OpenCV format
            opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            return opencv_image
            
        except Exception as e:
            raise ValueError(f"Failed to decode image: {str(e)}")
    
    def detect_hands(self, image):
        """Detect hand regions in the image"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect using cascade classifier
            detections = []
            if self.hand_cascade is not None:
                detections = self.hand_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
                )
            
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
    
    def classify_gesture_simple(self, roi, detection_box):
        """Simple gesture classification based on region properties"""
        try:
            x, y, w, h = detection_box
            area = w * h
            aspect_ratio = w / h if h > 0 else 1.0
            
            # Analyze contours in the ROI
            contours = self.analyze_contours(roi)
            
            # Simple classification logic
            if area > 8000:
                if 0.8 <= aspect_ratio <= 1.2:
                    return {
                        'name': 'Open Hand',
                        'confidence': 0.7,
                        'area': area,
                        'aspect_ratio': aspect_ratio
                    }
                elif aspect_ratio < 0.8:
                    return {
                        'name': 'Pointing',
                        'confidence': 0.6,
                        'area': area,
                        'aspect_ratio': aspect_ratio
                    }
            elif 3000 <= area <= 8000:
                return {
                    'name': 'Fist',
                    'confidence': 0.6,
                    'area': area,
                    'aspect_ratio': aspect_ratio
                }
            
            return {
                'name': 'Unknown',
                'confidence': 0.3,
                'area': area,
                'aspect_ratio': aspect_ratio
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
                if 'fist' in gesture_name and gesture.get('confidence', 0) > 0.5:
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
            # Decode image
            if isinstance(image_data, str):
                image = self.decode_base64_image(image_data)
            else:
                image = image_data
            
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
            
        except Exception as e:
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