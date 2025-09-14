from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
from datetime import datetime
import logging
from werkzeug.exceptions import BadRequest

# Import our custom modules
from gesture_recognition import recognize_gesture_from_image, get_gesture_info
from situation_analysis import analyze_situation_from_data, get_safety_recommendations

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['JSON_SORT_KEYS'] = False

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum size is 16MB."}), 413

@app.errorhandler(400)
def bad_request(e):
    return jsonify({"error": "Bad request. Please check your input data."}), 400

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error. Please try again later."}), 500

@app.route('/', methods=['GET'])
def home():
    """Serve the frontend HTML file"""
    try:
        frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'frontend', 'index.html')
        with open(frontend_path, 'r', encoding='utf-8') as f:
            return f.read(), 200, {'Content-Type': 'text/html'}
    except FileNotFoundError:
        return jsonify({
            "message": "Smart India Hackathon - Safety & Gesture Recognition API",
            "status": "running",
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
            "endpoints": {
                "gesture_recognition": "/api/recognize-gesture",
                "situation_analysis": "/api/analyze-situation",
                "combined_analysis": "/api/analyze-complete",
                "safety_tips": "/api/safety-tips",
                "health": "/health"
            }
        })

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/recognize-gesture', methods=['POST'])
def recognize_gesture():
    """Endpoint for gesture recognition from image"""
    try:
        # Validate request
        if not request.json:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Get image data
        image_data = request.json.get('image')
        if not image_data:
            return jsonify({"error": "No image data provided"}), 400
            
        # Validate image data format
        if isinstance(image_data, str):
            if not image_data.strip():
                return jsonify({"error": "Empty image data provided"}), 400
        else:
            return jsonify({"error": "Invalid image data format"}), 400
        
        # Process gesture recognition
        logger.info("Processing gesture recognition request")
        result = recognize_gesture_from_image(image_data)
        
        if "error" in result:
            logger.error(f"Gesture recognition error: {result['error']}")
            return jsonify(result), 400
        
        # Add metadata
        result["timestamp"] = datetime.now().isoformat()
        result["processing_time"] = "<1s"  # Placeholder
        
        # Add gesture descriptions
        if "gestures" in result:
            for gesture in result["gestures"]:
                gesture["description"] = get_gesture_info(gesture["gesture"])
        
        logger.info(f"Gesture recognition completed. Found {result.get('hands_detected', 0)} hands")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Gesture recognition endpoint error: {str(e)}")
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

@app.route('/api/analyze-situation', methods=['POST'])
def analyze_situation():
    """Endpoint for situation analysis from image"""
    try:
        # Validate request
        if not request.json:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Get image data
        image_data = request.json.get('image')
        if not image_data:
            return jsonify({"error": "No image data provided"}), 400
            
        # Validate image data format
        if isinstance(image_data, str):
            if not image_data.strip():
                return jsonify({"error": "Empty image data provided"}), 400
        else:
            return jsonify({"error": "Invalid image data format"}), 400
        
        # Optional: Get additional context data
        location_data = request.json.get('location')
        gesture_data = request.json.get('gesture_context')
        
        # Process situation analysis
        logger.info("Processing situation analysis request")
        result = analyze_situation_from_data(image_data, gesture_data, location_data)
        
        if "error" in result:
            logger.error(f"Situation analysis error: {result['error']}")
            return jsonify(result), 400
        
        # Add safety recommendations
        if "risk_assessment" in result and "risk_level" in result["risk_assessment"]:
            risk_level = result["risk_assessment"]["risk_level"]
            result["safety_tips"] = get_safety_recommendations(risk_level)
        
        logger.info(f"Situation analysis completed. Risk level: {result.get('risk_assessment', {}).get('risk_level', 'unknown')}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Situation analysis endpoint error: {str(e)}")
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

@app.route('/api/analyze-complete', methods=['POST'])
def analyze_complete():
    """Endpoint for combined gesture recognition and situation analysis"""
    try:
        # Validate request
        if not request.json:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Get image data
        image_data = request.json.get('image')
        if not image_data:
            return jsonify({"error": "No image data provided"}), 400
            
        # Validate image data format
        if isinstance(image_data, str):
            if not image_data.strip():
                return jsonify({"error": "Empty image data provided"}), 400
        else:
            return jsonify({"error": "Invalid image data format"}), 400
        
        # Optional: Get additional context data
        location_data = request.json.get('location')
        
        logger.info("Processing complete analysis request")
        
        # Step 1: Gesture recognition
        gesture_result = recognize_gesture_from_image(image_data)
        
        if "error" in gesture_result:
            logger.error(f"Gesture recognition error in complete analysis: {gesture_result['error']}")
            return jsonify({"error": f"Gesture recognition failed: {gesture_result['error']}"}), 400
        
        # Step 2: Situation analysis with gesture context
        situation_result = analyze_situation_from_data(image_data, gesture_result, location_data)
        
        if "error" in situation_result:
            logger.error(f"Situation analysis error in complete analysis: {situation_result['error']}")
            return jsonify({"error": f"Situation analysis failed: {situation_result['error']}"}), 400
        
        # Combine results
        combined_result = {
            "timestamp": datetime.now().isoformat(),
            "analysis_type": "complete",
            "gesture_recognition": gesture_result,
            "situation_analysis": situation_result,
            "overall_assessment": {
                "emergency_detected": False,
                "risk_level": situation_result.get("risk_assessment", {}).get("risk_level", "unknown"),
                "immediate_action_required": False
            }
        }
        
        # Check for emergency conditions
        emergency_gestures = gesture_result.get("emergency_signals", [])
        high_risk = situation_result.get("risk_assessment", {}).get("risk_level") == "high"
        situation_emergency = situation_result.get("emergency_detected", False)
        
        # Check for help gesture specifically
        help_gesture_detected = False
        if "gestures" in gesture_result:
            for gesture in gesture_result.get("gestures", []):
                if gesture.get("gesture") == "help" or gesture.get("name") == "Help Gesture" or gesture.get("message") == "help gesture detected" or gesture.get("emergency", False):
                    help_gesture_detected = True
                    emergency_gestures.append("Help gesture detected")
                    
                    # Add emergency analysis to the response if not already present
                    if "emergency_analysis" not in combined_result:
                        combined_result["emergency_analysis"] = {
                            "emergency_detected": True,
                            "emergency_gestures": []
                        }
                    
                    combined_result["emergency_analysis"]["emergency_gestures"].append({
                        "type": gesture.get("name", "Help Gesture"),
                        "message": "help gesture detected",
                        "confidence": gesture.get("confidence", 0.9)
                    })
                    break
        
        if emergency_gestures or high_risk or situation_emergency or help_gesture_detected:
            combined_result["overall_assessment"]["emergency_detected"] = True
            combined_result["overall_assessment"]["immediate_action_required"] = True
            
            # Compile emergency response
            emergency_actions = []
            if emergency_gestures:
                emergency_actions.extend(["Emergency gesture detected", "Contact emergency services"])
            if help_gesture_detected:
                emergency_actions.extend(["Help gesture detected", "Immediate assistance required"])
            if high_risk:
                emergency_actions.extend(["High risk situation identified", "Move to safer location"])
            
            combined_result["emergency_response"] = {
                "urgency": "high",
                "recommended_actions": emergency_actions,
                "emergency_contacts": {
                    "police": "100",
                    "medical": "108",
                    "fire": "101",
                    "women_helpline": "1091"
                }
            }
        
        # Add safety recommendations
        risk_level = situation_result.get("risk_assessment", {}).get("risk_level", "medium")
        combined_result["safety_recommendations"] = get_safety_recommendations(risk_level)
        
        logger.info(f"Complete analysis finished. Emergency: {combined_result['overall_assessment']['emergency_detected']}")
        return jsonify(combined_result)
        
    except Exception as e:
        logger.error(f"Complete analysis endpoint error: {str(e)}")
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

@app.route('/api/safety-tips', methods=['GET'])
def safety_tips():
    """Endpoint to get safety tips based on risk level"""
    try:
        risk_level = request.args.get('risk_level', 'medium')
        
        if risk_level not in ['very_low', 'low', 'medium', 'high']:
            return jsonify({"error": "Invalid risk level. Use: very_low, low, medium, high"}), 400
        
        tips = get_safety_recommendations(risk_level)
        
        return jsonify({
            "risk_level": risk_level,
            "safety_tips": tips,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Safety tips endpoint error: {str(e)}")
        return jsonify({"error": f"Failed to get safety tips: {str(e)}"}), 500

@app.route('/api/emergency', methods=['POST'])
def emergency_alert():
    """Endpoint for emergency alerts"""
    try:
        if not request.json:
            return jsonify({"error": "No JSON data provided"}), 400
        
        alert_type = request.json.get('type', 'general')
        location = request.json.get('location')
        message = request.json.get('message', 'Emergency alert triggered')
        
        # Log emergency alert
        logger.warning(f"EMERGENCY ALERT: Type: {alert_type}, Location: {location}, Message: {message}")
        
        # In a real implementation, this would trigger actual emergency services
        response = {
            "alert_id": f"ALERT_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "status": "received",
            "timestamp": datetime.now().isoformat(),
            "type": alert_type,
            "message": "Emergency alert has been logged and processed",
            "next_steps": [
                "Alert has been logged in the system",
                "If this is a real emergency, contact local emergency services",
                "Emergency contacts: Police (100), Medical (108), Fire (101)"
            ]
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Emergency alert endpoint error: {str(e)}")
        return jsonify({"error": f"Failed to process emergency alert: {str(e)}"}), 500

@app.route('/api/gesture-info/<gesture_name>', methods=['GET'])
def gesture_info(gesture_name):
    """Get information about a specific gesture"""
    try:
        description = get_gesture_info(gesture_name)
        
        return jsonify({
            "gesture": gesture_name,
            "description": description,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Gesture info endpoint error: {str(e)}")
        return jsonify({"error": f"Failed to get gesture info: {str(e)}"}), 500

if __name__ == '__main__':
    # Get port from environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))
    
    # Get debug mode from environment variable
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting Smart India Hackathon API server on port {port}")
    logger.info(f"Debug mode: {debug_mode}")
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug_mode,
        threaded=True
    )
