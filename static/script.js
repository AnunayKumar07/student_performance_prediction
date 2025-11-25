/*
    UNIVERSAL API URL (Works on Codespaces + Vercel + Local)
    --------------------------------------------------------
    - On Vercel: window.location.origin = https://yourapp.vercel.app
    - On Codespaces: window.location.origin = https://your-codespace-url.github.dev
    - On Localhost: window.location.origin = http://localhost:5000
*/
const API_URL = window.location.origin;


// Update slider displays
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

    // Prepare form data
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
        // API request to backend
        const response = await fetch(`${API_URL}/api/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData)
        });

        if (!response.ok) {
            throw new Error("Prediction error");
        }

        const result = await response.json();
        displayResults(result);

    } catch (err) {
        console.error(err);

        // Message works for both Vercel + Codespaces
        alert("⚠ Unable to connect to AI server.\n\nIf running locally or in Codespaces:\n➡ Start backend using: python app.py\nIf on Vercel:\n➡ Deployment may still be building.\nCheck Network tab for API errors.");
    }

    // Hide loader
    document.getElementById('loading').classList.remove('show');
});


// Display results in UI
function displayResults(result) {
    const resultsDiv = document.getElementById('results');
    const statusBadge = document.getElementById('statusBadge');
    const riskLevel = document.getElementById('riskLevel');
    const studentInfo = document.getElementById('studentInfo');

    // PASS/FAIL
    if (result.prediction.will_pass) {
        statusBadge.textContent = '✓ PASS PREDICTED';
        statusBadge.className = 'status-badge status-pass';
    } else {
        statusBadge.textContent = '✗ AT RISK - INTERVENTION NEEDED';
        statusBadge.className = 'status-badge status-fail';
    }

    riskLevel.textContent = result.prediction.risk_level;
    studentInfo.textContent = `${result.student_info.name} (${result.student_info.roll_number})`;

    // Metrics
    document.getElementById('successProb').textContent = result.prediction.success_probability + '%';
    document.getElementById('riskScore').textContent = result.prediction.risk_score;
    document.getElementById('academicIndex').textContent = result.prediction.academic_index;

    // Risk Factors
    const rfSection = document.getElementById('riskFactorsSection');
    const rfList = document.getElementById('riskFactorsList');
    rfList.innerHTML = '';

    if (result.risk_factors?.length > 0) {
        rfSection.style.display = 'block';
        result.risk_factors.forEach(risk => {
            rfList.innerHTML += `
                <div class="risk-factor-item">
                    <span class="risk-badge risk-${risk.severity}">${risk.severity}</span>
                    <div><strong>${risk.factor}:</strong> ${risk.message}</div>
                </div>`;
        });
    } else {
        rfSection.style.display = 'none';
    }

    // Recommendations
    const recommendationList = document.getElementById('recommendationList');
    recommendationList.innerHTML = '';
    result.recommendations.forEach(rec => {
        recommendationList.innerHTML += `
            <div class="recommendation-item">
                <span class="recommendation-priority priority-${rec.priority}">${rec.priority}</span>
                <strong>${rec.category}:</strong> ${rec.action}
            </div>`;
    });

    // Feature Importance
    const featureList = document.getElementById('featureImportanceList');
    featureList.innerHTML = '';
    result.feature_importance.forEach(f => {
        featureList.innerHTML += `
            <div class="feature-item">
                <div>
                    <strong>${f.feature}</strong>
                    <div class="feature-bar" style="width:${f.importance}%"></div>
                </div>
                <span>${f.importance}%</span>
            </div>`;
    });

    resultsDiv.classList.add('show');
    resultsDiv.scrollIntoView({ behavior: 'smooth' });
}


// Reset form
function resetForm() {
    document.getElementById('predictionForm').reset();
    document.getElementById('assignmentValue').textContent = '75%';
    document.getElementById('stressValue').textContent = '5/10';
    document.getElementById('studyValue').textContent = '4 hrs';
    document.getElementById('results').classList.remove('show');
}


// Test connection on page load
window.addEventListener('load', async () => {
    try {
        const res = await fetch(`${API_URL}/api/model-info`);
        if (res.ok) {
            const info = await res.json();
            console.log("✓ Connected to AI backend");
            console.log("Model Info:", info);
        }
    } catch (err) {
        console.warn("⚠ Backend not reachable yet.");
    }
});
