// API Configuration
const API_URL = 'http://localhost:5000';

// Update range slider displays
document.getElementById('assignmentSubmission').addEventListener('input', function(e) {
    document.getElementById('assignmentValue').textContent = e.target.value + '%';
});

document.getElementById('stressLevel').addEventListener('input', function(e) {
    document.getElementById('stressValue').textContent = e.target.value + '/10';
});

document.getElementById('studyHours').addEventListener('input', function(e) {
    document.getElementById('studyValue').textContent = e.target.value + ' hrs';
});

// Form submission handler
document.getElementById('predictionForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    // Show loading
    document.getElementById('loading').classList.add('show');
    document.getElementById('results').classList.remove('show');

    // Collect form data
    const formData = {
        student_name: document.getElementById('studentName').value,
        roll_number: document.getElementById('rollNumber').value,
        total_marks: parseFloat(document.getElementById('totalMarks').value),
        attendance: parseFloat(document.getElementById('attendance').value),
        backlogs: parseInt(document.getElementById('backlogs').value),
        assignment_submission: parseFloat(document.getElementById('assignmentSubmission').value),
        internet_quality: document.getElementById('internetQuality').value,
        device_availability: document.getElementById('deviceAvailability').value,
        family_support: document.getElementById('familySupport').value,
        stress_level: parseFloat(document.getElementById('stressLevel').value),
        study_hours: parseFloat(document.getElementById('studyHours').value),
        participation: document.getElementById('participation').value
    };

    try {
        // Call Flask API
        const response = await fetch(`${API_URL}/api/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        });

        if (!response.ok) {
            throw new Error('Prediction failed');
        }

        const result = await response.json();
        displayResults(result);
        
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to connect to AI server. Please ensure Flask server is running on port 5000.');
    }

    // Hide loading
    document.getElementById('loading').classList.remove('show');
});

function displayResults(result) {
    const resultsDiv = document.getElementById('results');
    const statusBadge = document.getElementById('statusBadge');
    const riskLevel = document.getElementById('riskLevel');
    const studentInfo = document.getElementById('studentInfo');

    // Update status
    if (result.prediction.will_pass) {
        statusBadge.textContent = '✓ PASS PREDICTED';
        statusBadge.className = 'status-badge status-pass';
    } else {
        statusBadge.textContent = '✗ AT RISK - INTERVENTION NEEDED';
        statusBadge.className = 'status-badge status-fail';
    }

    riskLevel.textContent = result.prediction.risk_level;
    studentInfo.textContent = `${result.student_info.name} (${result.student_info.roll_number})`;

    // Update metrics
    document.getElementById('successProb').textContent = result.prediction.success_probability + '%';
    document.getElementById('riskScore').textContent = result.prediction.risk_score;
    document.getElementById('academicIndex').textContent = result.prediction.academic_index;

    // Display risk factors
    const riskFactorsSection = document.getElementById('riskFactorsSection');
    const riskFactorsList = document.getElementById('riskFactorsList');
    
    if (result.risk_factors && result.risk_factors.length > 0) {
        riskFactorsSection.style.display = 'block';
        riskFactorsList.innerHTML = '';
        
        result.risk_factors.forEach(risk => {
            const riskItem = document.createElement('div');
            riskItem.className = 'risk-factor-item';
            riskItem.innerHTML = `
                <span class="risk-badge risk-${risk.severity}">${risk.severity}</span>
                <div>
                    <strong>${risk.factor}:</strong> ${risk.message}
                </div>
            `;
            riskFactorsList.appendChild(riskItem);
        });
    } else {
        riskFactorsSection.style.display = 'none';
    }

    // Display recommendations
    const recommendationList = document.getElementById('recommendationList');
    recommendationList.innerHTML = '';
    
    result.recommendations.forEach(rec => {
        const recItem = document.createElement('div');
        recItem.className = 'recommendation-item';
        recItem.innerHTML = `
            <span class="recommendation-priority priority-${rec.priority}">${rec.priority}</span>
            <strong>${rec.category}:</strong> ${rec.action}
        `;
        recommendationList.appendChild(recItem);
    });

    // Display feature importance
    const featureImportanceList = document.getElementById('featureImportanceList');
    featureImportanceList.innerHTML = '';
    
    result.feature_importance.forEach(feature => {
        const featureItem = document.createElement('div');
        featureItem.className = 'feature-item';
        featureItem.innerHTML = `
            <div>
                <strong>${feature.feature}</strong>
                <div class="feature-bar" style="width: ${feature.importance}%; margin-top: 5px;"></div>
            </div>
            <span>${feature.importance}%</span>
        `;
        featureImportanceList.appendChild(featureItem);
    });

    // Show results with animation
    resultsDiv.classList.add('show');
    resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function resetForm() {
    document.getElementById('predictionForm').reset();
    document.getElementById('assignmentValue').textContent = '75%';
    document.getElementById('stressValue').textContent = '5/10';
    document.getElementById('studyValue').textContent = '4 hrs';
    document.getElementById('results').classList.remove('show');
}

// Test API connection on page load
window.addEventListener('load', async function() {
    try {
        const response = await fetch(`${API_URL}/api/model-info`);
        if (response.ok) {
            const info = await response.json();
            console.log('AI Model Info:', info);
            console.log(`✓ Connected to AI server - Model Accuracy: ${info.accuracy}%`);
        }
    } catch (error) {
        console.warn('⚠ Flask server not running. Please start the server with: python app.py');
    }
});