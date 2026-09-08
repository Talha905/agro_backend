# Deploying AgroSaathi Backend to Render

This FastAPI backend handles:
1. **Plant Disease Detection** (`/predict-disease`) via TFLite model.
2. **Crop Recommendation** (`/recommend-crop`) via Scikit-Learn ML model.
3. **AI Growth Plan Generation** (`/generate-growth-plan`) via Gemini API.

---

## Step-by-Step Deployment Instructions

### 1. Push Code to GitHub
Ensure all files in the repository (including `backend/`) are committed and pushed to your GitHub repository:
```bash
git add backend/
git commit -m "Configure backend for Render deployment"
git push origin develop
```

---

### 2. Create a Free Web Service on Render
1. Go to [dashboard.render.com](https://dashboard.render.com) and log in.
2. Click **New +** -> **Web Service**.
3. Select **Build and deploy from a Git repository** and connect your GitHub account.
4. Choose your repository: `Talha905/agro_FYP`.
5. Fill in the deployment details:
   * **Name**: `agrosaathi-backend`
   * **Root Directory**: `backend`
   * **Environment**: `Python 3`
   * **Region**: Choose closest to India (e.g., *Singapore*)
   * **Branch**: `develop`
   * **Build Command**: `pip install -r requirements.txt`
   * **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`

---

### 3. Add Environment Variables
Under the **Environment Variables** section on Render, add:
* **Key**: `GEMINI_API_KEY`
* **Value**: *(Your Google Gemini API Key)*

---

### 4. Deploy & Get Your Live Public URL
1. Click **Create Web Service**.
2. Render will build and launch your container.
3. Once deployed, Render gives you a live HTTPS URL, e.g.:
   `https://agrosaathi-backend.onrender.com`
4. Test it by opening `https://agrosaathi-backend.onrender.com/health` in your browser. You should see:
   ```json
   {
     "status": "healthy",
     "service": "AgroSaathi ML Backend",
     "disease_model_loaded": true,
     "recommend_model_loaded": true
   }
   ```

---

### 5. Update Flutter App Base URL
Once your Render URL is live, update `baseUrl` in your Flutter app:
* [api_service.dart](file:///c:/Users/thele/OneDrive/Desktop/FPY3/agrosaathi/lib/services/api_service.dart#L6)
* [ai_growth_plan_service.dart](file:///c:/Users/thele/OneDrive/Desktop/FPY3/agrosaathi/lib/services/ai_growth_plan_service.dart#L15)

And rebuild your APK:
```bash
flutter build apk --release
```
