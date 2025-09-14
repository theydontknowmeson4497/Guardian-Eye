import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import json
from datetime import datetime
import os
from sklearn.cluster import KMeans
from collections import Counter
import math

class SituationAnalyzer:
    def __init__(self):
        self.emergency_keywords = [
            'help', 'emergency', 'fire', 'accident', 'injured', 'danger',
            'threat', 'violence', 'robbery', 'attack', 'medical', 'urgent'
        ]
        
        self.safety_zones = {
            'safe': ['home', 'office', 'school', 'hospital', 'police_station'],
            'moderate': ['street', 'park', 'mall', 'restaurant'],
            'risky': ['isolated_area', 'dark_alley', 'abandoned_building']
        }
        
        self.time_risk_factors = {
            'night': 0.8,  # Higher risk at night
            'evening': 0.6,
            'morning': 0.3,
            'afternoon': 0.2
        }
    
    def decode_base64_image(self, base64_string):
        """Decode base64 image string to OpenCV format"""
        try:
            if 'data:image' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            image_bytes = base64.b64decode(base64_string)
            pil_image = Image.open(BytesIO(image_bytes))
            opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            return opencv_image
        except Exception as e:
            print(f"Error decoding image: {e}")
            return None
    
    def analyze_lighting_conditions(self, image):
        """Analyze lighting conditions in the image"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate average brightness
            avg_brightness = np.mean(gray)
            
            # Calculate brightness distribution
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            
            # Determine lighting condition
            if avg_brightness < 50:
                condition = "very_dark"
                risk_level = 0.9
            elif avg_brightness < 100:
                condition = "dark"
                risk_level = 0.7
            elif avg_brightness < 150:
                condition = "moderate"
                risk_level = 0.3
            else:
                condition = "bright"
                risk_level = 0.1
            
            return {
                "condition": condition,
                "brightness_score": float(avg_brightness),
                "risk_level": risk_level,
                "visibility": "good" if avg_brightness > 100 else "poor"
            }
        except Exception as e:
            return {"error": f"Lighting analysis failed: {str(e)}"}
    
    def detect_crowd_density(self, image):
        """Estimate crowd density using simple computer vision techniques"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Use Haar cascade for person detection (simplified)
            # In a real implementation, you'd use a proper person detection model
            
            # For now, we'll use edge detection as a proxy for activity/movement
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # Estimate crowd density based on edge density
            if edge_density > 0.15:
                density = "high"
                people_estimate = "many (>10)"
                safety_factor = 0.6  # Crowds can be safer but also riskier
            elif edge_density > 0.08:
                density = "medium"
                people_estimate = "moderate (5-10)"
                safety_factor = 0.4
            elif edge_density > 0.03:
                density = "low"
                people_estimate = "few (1-5)"
                safety_factor = 0.3
            else:
                density = "empty"
                people_estimate = "none or very few"
                safety_factor = 0.8  # Being alone can be risky
            
            return {
                "density": density,
                "estimated_people": people_estimate,
                "edge_density": float(edge_density),
                "safety_factor": safety_factor
            }
        except Exception as e:
            return {"error": f"Crowd analysis failed: {str(e)}"}
    
    def analyze_environment_safety(self, image):
        """Analyze environmental factors for safety assessment"""
        try:
            # Color analysis for environment type
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Analyze dominant colors
            pixels = hsv.reshape(-1, 3)
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            colors = kmeans.cluster_centers_
            labels = kmeans.labels_
            color_counts = Counter(labels)
            
            # Determine environment type based on colors
            environment_type = "unknown"
            safety_score = 0.5
            
            # Green dominance suggests outdoor/park
            green_ratio = np.sum((colors[:, 0] >= 35) & (colors[:, 0] <= 85)) / len(colors)
            if green_ratio > 0.3:
                environment_type = "outdoor/nature"
                safety_score = 0.4
            
            # Gray/concrete dominance suggests urban
            gray_pixels = np.sum((hsv[:,:,1] < 50) & (hsv[:,:,2] > 50))
            total_pixels = hsv.shape[0] * hsv.shape[1]
            gray_ratio = gray_pixels / total_pixels
            
            if gray_ratio > 0.4:
                environment_type = "urban/concrete"
                safety_score = 0.6
            
            # Detect potential hazards (very basic)
            hazards = []
            
            # Check for fire/red dominance
            red_pixels = np.sum((hsv[:,:,0] < 10) | (hsv[:,:,0] > 170))
            if (red_pixels / total_pixels) > 0.3:
                hazards.append("potential_fire_or_emergency_lights")
                safety_score += 0.3
            
            return {
                "environment_type": environment_type,
                "safety_score": min(safety_score, 1.0),
                "dominant_colors": colors.tolist(),
                "potential_hazards": hazards,
                "color_analysis": {
                    "green_ratio": float(green_ratio),
                    "gray_ratio": float(gray_ratio)
                }
            }
        except Exception as e:
            return {"error": f"Environment analysis failed: {str(e)}"}
    
    def get_time_based_risk(self):
        """Calculate risk based on current time"""
        current_hour = datetime.now().hour
        
        if 22 <= current_hour or current_hour <= 5:  # Night time
            return self.time_risk_factors['night']
        elif 18 <= current_hour <= 21:  # Evening
            return self.time_risk_factors['evening']
        elif 6 <= current_hour <= 11:  # Morning
            return self.time_risk_factors['morning']
        else:  # Afternoon
            return self.time_risk_factors['afternoon']
    
    def calculate_overall_risk(self, lighting, crowd, environment, gesture_data=None):
        """Calculate overall risk assessment"""
        try:
            # Base risk factors
            lighting_risk = lighting.get('risk_level', 0.5)
            crowd_risk = crowd.get('safety_factor', 0.5)
            env_risk = 1 - environment.get('safety_score', 0.5)
            time_risk = self.get_time_based_risk()
            
            # Emergency gesture detection
            emergency_risk = 0.0
            if gesture_data and 'emergency_signals' in gesture_data:
                if gesture_data['emergency_signals']:
                    emergency_risk = 0.9  # High risk if emergency gestures detected
            
            # Weighted average
            weights = {
                'lighting': 0.25,
                'crowd': 0.20,
                'environment': 0.20,
                'time': 0.15,
                'emergency': 0.20
            }
            
            overall_risk = (
                lighting_risk * weights['lighting'] +
                crowd_risk * weights['crowd'] +
                env_risk * weights['environment'] +
                time_risk * weights['time'] +
                emergency_risk * weights['emergency']
            )
            
            # Determine risk level
            if overall_risk >= 0.7:
                risk_level = "high"
                recommendation = "Immediate attention required. Consider seeking help or moving to a safer location."
            elif overall_risk >= 0.5:
                risk_level = "medium"
                recommendation = "Exercise caution. Stay alert and consider safety measures."
            elif overall_risk >= 0.3:
                risk_level = "low"
                recommendation = "Generally safe. Maintain normal awareness."
            else:
                risk_level = "very_low"
                recommendation = "Safe environment. Continue normal activities."
            
            return {
                "overall_risk_score": float(overall_risk),
                "risk_level": risk_level,
                "recommendation": recommendation,
                "risk_factors": {
                    "lighting": float(lighting_risk),
                    "crowd": float(crowd_risk),
                    "environment": float(env_risk),
                    "time": float(time_risk),
                    "emergency_gestures": float(emergency_risk)
                }
            }
        except Exception as e:
            return {"error": f"Risk calculation failed: {str(e)}"}
    
    def analyze_situation(self, image_data, gesture_data=None, location_data=None):
        """Main function to analyze the overall situation"""
        try:
            # Decode image
            if isinstance(image_data, str):
                image = self.decode_base64_image(image_data)
            else:
                image = image_data
            
            if image is None:
                return {"error": "Failed to decode image"}
            
            # Perform various analyses
            lighting_analysis = self.analyze_lighting_conditions(image)
            crowd_analysis = self.detect_crowd_density(image)
            environment_analysis = self.analyze_environment_safety(image)
            
            # Calculate overall risk
            risk_assessment = self.calculate_overall_risk(
                lighting_analysis, crowd_analysis, environment_analysis, gesture_data
            )
            
            # Compile final response
            response = {
                "timestamp": datetime.now().isoformat(),
                "lighting_conditions": lighting_analysis,
                "crowd_density": crowd_analysis,
                "environment_safety": environment_analysis,
                "risk_assessment": risk_assessment,
                "emergency_detected": False
            }
            
            # Check for emergency conditions
            if (risk_assessment.get('risk_level') == 'high' or 
                (gesture_data and gesture_data.get('emergency_signals'))):
                response["emergency_detected"] = True
                response["emergency_details"] = {
                    "type": "situation_analysis",
                    "urgency": "high",
                    "suggested_actions": [
                        "Contact emergency services if needed",
                        "Move to a safer location",
                        "Alert trusted contacts",
                        "Stay calm and assess options"
                    ]
                }
            
            return response
            
        except Exception as e:
            return {"error": f"Situation analysis failed: {str(e)}"}
    
    def get_safety_tips(self, risk_level):
        """Get safety tips based on risk level"""
        tips = {
            "high": [
                "Stay alert and aware of your surroundings",
                "Keep emergency contacts readily available",
                "Consider moving to a well-lit, populated area",
                "Trust your instincts if something feels wrong",
                "Have an exit strategy planned"
            ],
            "medium": [
                "Maintain situational awareness",
                "Keep your phone charged and accessible",
                "Stay in well-lit areas when possible",
                "Let someone know your location"
            ],
            "low": [
                "Continue normal safety practices",
                "Stay aware of your environment",
                "Keep emergency contacts updated"
            ],
            "very_low": [
                "Enjoy your activities safely",
                "Maintain basic safety awareness"
            ]
        }
        
        return tips.get(risk_level, tips["medium"])

# Initialize global analyzer instance
situation_analyzer = SituationAnalyzer()

def analyze_situation_from_data(image_data, gesture_data=None, location_data=None):
    """Main function to analyze situation from provided data"""
    return situation_analyzer.analyze_situation(image_data, gesture_data, location_data)

def get_safety_recommendations(risk_level):
    """Get safety recommendations based on risk level"""
    return situation_analyzer.get_safety_tips(risk_level)