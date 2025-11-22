from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

app = Flask(__name__)
CORS(app)

# Feature encoding maps
ENCODING_MAPS = {
    'internet_quality': {'excellent': 4, 'good': 3, 'average': 2, 'poor': 1},
    'device_availability': {'always': 4, 'mostly': 3, 'sometimes': 2, 'rarely': 1},
    'family_support': {'strong': 4, 'moderate': 3, 'limited': 2, 'none': 1},
    'participation': {'high': 4, 'medium': 3, 'low': 2, 'none': 1}
}

# Training data - realistic student performance dataset
TRAINING_DATA = np.array([
    # [marks, attendance, backlogs, assignment%, internet, device, family, stress, study_hrs, participation, RESULT]
    [420, 95, 0, 90, 4, 4, 4, 3, 6, 4, 1],
    [380, 88, 1, 85, 3, 4, 3, 4, 5, 3, 1],
    [250, 78, 2, 70, 3, 3, 3, 5, 4, 3, 1],
    [220, 76, 2, 65, 2, 3, 2, 6, 3, 2, 1],
    [180, 70, 3, 50, 2, 2, 2, 7, 2, 2, 0],
    [150, 65, 4, 40, 1, 2, 1, 8, 2, 1, 0],
    [350, 85, 1, 80, 3, 3, 3, 4, 5, 3, 1],
    [280, 80, 2, 75, 3, 3, 3, 5, 4, 3, 1],
    [190, 72, 3, 55, 2, 2, 2, 7, 3, 2, 0],
    [410, 92, 0, 88, 4, 4, 4, 2, 6, 4, 1],
    [240, 77, 2, 68, 2, 3, 2, 6, 3, 2, 1],
    [170, 68, 4, 45, 1, 2, 1, 8, 2, 1, 0],
    [320, 82, 1, 78, 3, 3, 3, 4, 5, 3, 1],
    [260, 79, 2, 72, 3, 3, 2, 5, 4, 3, 1],
    [200, 74, 3, 60, 2, 2, 2, 6, 3, 2, 0],
    [440, 96, 0, 92, 4, 4, 4, 2, 7, 4, 1],
    [210, 75, 2, 65, 2, 3, 2, 6, 3, 2, 1],
    [160, 66, 4, 42, 1, 2, 1, 8, 2, 1, 0],
    [370, 87, 1, 83, 3, 4, 3, 3, 5, 3, 1],
    [290, 81, 2, 76, 3, 3, 3, 5, 4, 3, 1],
    [230, 78, 2, 70, 3, 3, 3, 5, 4, 2, 1],
    [195, 73, 3, 58, 2, 2, 2, 7, 3, 2, 0],
    [340, 84, 1, 79, 3, 3, 3, 4, 5, 3, 1],
    [270, 80, 2, 74, 3, 3, 3, 5, 4, 3, 1],
    [185, 71, 3, 52, 2, 2, 2, 7, 2, 2, 0],
])

# Initialize model and scaler globally
model = None
scaler = None

def train_model():
    """Train the Random Forest model"""
    global model, scaler
    
    # Separate features and labels
    X = TRAINING_DATA[:, :-1]
    y = TRAINING_DATA[:, -1]
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Random Forest Classifier
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        min_samples_split=2,
        min_samples_leaf=1
    )
    model.fit(X_scaled, y)
    
    # Calculate training accuracy
    accuracy = model.score(X_scaled, y)
    print(f"Model trained with accuracy: {accuracy * 100:.2f}%")
    
    return model, scaler

def encode_categorical_features(data):
    """Convert categorical features to numerical values"""
    encoded = data.copy()
    
    for feature, mapping in ENCODING_MAPS.items():
        if feature in encoded:
            encoded[feature] = mapping.get(encoded[feature], 2)  # Default to 2 (average)
    
    return encoded

def get_feature_vector(data):
    """Extract feature vector in correct order"""
    return [
        float(data['total_marks']),
        float(data['attendance']),
        int(data['backlogs']),
        float(data['assignment_submission']),
        int(data['internet_quality']),
        int(data['device_availability']),
        int(data['family_support']),
        float(data['stress_level']),
        float(data['study_hours']),
        int(data['participation'])
    ]

def calculate_risk_factors(data, features):
    """Analyze individual risk factors"""
    risk_factors = []
    
    # Critical factors
    if data['total_marks'] < 200:
        risk_factors.append({
            'factor': 'Academic Performance',
            'severity': 'high',
            'message': 'Total marks below passing threshold (40%)'
        })
    
    if data['attendance'] < 75:
        risk_factors.append({
            'factor': 'Attendance',
            'severity': 'critical',
            'message': 'Attendance below 75% - Detention risk'
        })
    
    if data['backlogs'] > 2:
        risk_factors.append({
            'factor': 'Backlogs',
            'severity': 'high',
            'message': f'{data["backlogs"]} backlogs affecting graduation eligibility'
        })
    
    # Secondary factors
    if data['assignment_submission'] < 60:
        risk_factors.append({
            'factor': 'Assignments',
            'severity': 'medium',
            'message': 'Low assignment submission rate'
        })
    
    if data['stress_level'] > 7:
        risk_factors.append({
            'factor': 'Mental Health',
            'severity': 'medium',
            'message': 'High stress levels detected'
        })
    
    if data['study_hours'] < 3:
        risk_factors.append({
            'factor': 'Study Habits',
            'severity': 'medium',
            'message': 'Insufficient daily study hours'
        })
    
    if data['internet_quality'] == 1 or data['device_availability'] == 1:
        risk_factors.append({
            'factor': 'Digital Access',
            'severity': 'low',
            'message': 'Limited internet or device access'
        })
    
    if data['family_support'] <= 2:
        risk_factors.append({
            'factor': 'Support System',
            'severity': 'low',
            'message': 'Limited family support'
        })
    
    return risk_factors

def generate_recommendations(data, prediction_result, risk_factors):
    """Generate AI-powered personalized recommendations"""
    recommendations = []
    
    # Critical recommendations based on risk factors
    for risk in risk_factors:
        if risk['severity'] == 'critical':
            recommendations.append({
                'priority': 'urgent',
                'category': risk['factor'],
                'action': f"⚠️ IMMEDIATE ACTION: {risk['message']}"
            })
    
    # Academic recommendations
    if data['total_marks'] < 200:
        recommendations.append({
            'priority': 'high',
            'category': 'Academic',
            'action': 'Schedule extra tutoring sessions for weak subjects'
        })
        recommendations.append({
            'priority': 'high',
            'category': 'Academic',
            'action': 'Join study groups with high-performing students'
        })
    elif data['total_marks'] < 300:
        recommendations.append({
            'priority': 'medium',
            'category': 'Academic',
            'action': 'Focus on improving internal assessment marks'
        })
    
    # Attendance recommendations
    if data['attendance'] < 75:
        recommendations.append({
            'priority': 'urgent',
            'category': 'Attendance',
            'action': 'Meet with HOD to discuss attendance improvement plan'
        })
    elif data['attendance'] < 85:
        recommendations.append({
            'priority': 'medium',
            'category': 'Attendance',
            'action': 'Maintain regular attendance to build buffer above 75%'
        })
    
    # Backlog recommendations
    if data['backlogs'] > 2:
        recommendations.append({
            'priority': 'urgent',
            'category': 'Backlogs',
            'action': 'Enroll in backlog clearing sessions this semester'
        })
        recommendations.append({
            'priority': 'high',
            'category': 'Backlogs',
            'action': 'Dedicate 2 hours daily specifically for backlog subjects'
        })
    elif data['backlogs'] > 0:
        recommendations.append({
            'priority': 'medium',
            'category': 'Backlogs',
            'action': 'Clear pending backlogs to reduce academic burden'
        })
    
    # Study habits
    if data['study_hours'] < 3:
        recommendations.append({
            'priority': 'high',
            'category': 'Study Habits',
            'action': 'Gradually increase study hours to 5-6 hours daily'
        })
    
    # Mental health
    if data['stress_level'] > 7:
        recommendations.append({
            'priority': 'high',
            'category': 'Mental Health',
            'action': 'Contact HIT counseling center for stress management support'
        })
        recommendations.append({
            'priority': 'medium',
            'category': 'Mental Health',
            'action': 'Practice daily meditation and maintain regular sleep schedule'
        })
    
    # Assignment submission
    if data['assignment_submission'] < 60:
        recommendations.append({
            'priority': 'medium',
            'category': 'Assignments',
            'action': 'Set reminders for assignment deadlines and submit early'
        })
    
    # Digital access
    if data['internet_quality'] <= 2 or data['device_availability'] <= 2:
        recommendations.append({
            'priority': 'medium',
            'category': 'Resources',
            'action': 'Utilize college computer labs and library for online resources'
        })
    
    # Overall recommendations
    if prediction_result == 1:
        recommendations.append({
            'priority': 'low',
            'category': 'Encouragement',
            'action': '✅ Excellent progress! Maintain current performance levels'
        })
    else:
        recommendations.append({
            'priority': 'urgent',
            'category': 'Intervention',
            'action': 'Schedule immediate meeting with faculty mentor and academic advisor'
        })
    
    return recommendations

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        data = request.json
        
        # Encode categorical features
        encoded_data = encode_categorical_features(data)
        
        # Extract feature vector
        features = get_feature_vector(encoded_data)
        features_array = np.array([features])
        
        # Standardize features
        features_scaled = scaler.transform(features_array)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        prediction_proba = model.predict_proba(features_scaled)[0]
        
        # Calculate success probability
        success_probability = prediction_proba[1] * 100
        
        # Determine risk level
        if success_probability >= 75:
            risk_level = 'Low Risk'
        elif success_probability >= 50:
            risk_level = 'Low-Medium Risk'
        elif success_probability >= 35:
            risk_level = 'Medium Risk'
        else:
            risk_level = 'High Risk'
        
        # Calculate academic index
        academic_index = int((
            (encoded_data['total_marks'] / 500 * 100) +
            encoded_data['attendance'] +
            encoded_data['assignment_submission']
        ) / 3)
        
        # Calculate risk score
        risk_score = int(100 - success_probability)
        
        # Get feature importances
        feature_names = [
            'Total Marks', 'Attendance', 'Backlogs', 'Assignments',
            'Internet', 'Device', 'Family Support', 'Stress', 'Study Hours', 'Participation'
        ]
        importances = model.feature_importances_
        
        # Calculate risk factors
        risk_factors = calculate_risk_factors(encoded_data, features)
        
        # Generate recommendations
        recommendations = generate_recommendations(encoded_data, int(prediction), risk_factors)
        
        # Prepare response
        response = {
            'prediction': {
                'will_pass': bool(prediction),
                'success_probability': round(success_probability, 2),
                'risk_score': risk_score,
                'risk_level': risk_level,
                'academic_index': academic_index
            },
            'risk_factors': risk_factors,
            'recommendations': recommendations,
            'feature_importance': [
                {'feature': name, 'importance': round(imp * 100, 2)}
                for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:5]
            ],
            'student_info': {
                'name': data.get('student_name', 'Unknown'),
                'roll_number': data.get('roll_number', 'N/A')
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        'model_type': 'Random Forest Classifier',
        'n_estimators': 100,
        'training_samples': len(TRAINING_DATA),
        'features': 10,
        'accuracy': round(model.score(scaler.transform(TRAINING_DATA[:, :-1]), TRAINING_DATA[:, -1]) * 100, 2)
    })

if __name__ == '__main__':
    print("Training AI model...")
    train_model()
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)