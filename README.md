# student_performance_prediction

Student Performance Prediction AI project using Flask, Random Forest, and Supabase.

## Installation & Setup

Prerequisites:

- Python 3.8 or higher
- pip
- A Supabase project

Step 1: Install dependencies

```powershell
pip install -r requirements.txt
```

Step 2: Create the Supabase table

1. Open your Supabase project dashboard.
2. Go to SQL Editor.
3. Open `supabase_schema.sql` from this project.
4. Paste the SQL into Supabase SQL Editor.
5. Click Run.

Step 3: Configure environment variables

Copy `.env.example` to `.env`:

```powershell
Copy-Item .env.example .env -Force
```

Fill in the values from Supabase:

```env
SUPABASE_URL=https://your-project-ref.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
SUPABASE_TABLE=student_predictions
```

Use the service role key only on the Flask backend. Never put it in frontend JavaScript.

Step 4: Run the server

```powershell
python app.py
```

Step 5: Open the website

Navigate to:

```text
http://localhost:5000
```

## Supabase Data

Each prediction submitted from the website is stored in the `student_predictions` table by default. The saved row includes student details, original form input, encoded model features, prediction output, risk factors, recommendations, feature importance, model metadata, and `created_at`.

The website also exposes:

```text
GET /api/predictions?limit=10
```

This endpoint feeds the recent prediction history panel.

## Link

https://student-performance-prediction-theta.vercel.app/
