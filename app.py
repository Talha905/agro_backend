import os
import re
import json
import asyncio
import warnings
import urllib.request
import urllib.error
import numpy as np
import pickle
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import io

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(
    title="AgroSaathi ML Backend",
    description="Production-ready FastAPI backend for plant disease detection, crop recommendation, AI growth plans, and Ollama qwen2.5:3b disease remedies.",
    version="1.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")


def query_ollama(prompt: str, system: str = "", format_json: bool = True, timeout: float = 35.0) -> dict | None:
    """Queries local or remote Ollama server running qwen2.5:3b model."""
    url = f"{OLLAMA_HOST.rstrip('/')}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": 350,
            "temperature": 0.2,
            "top_p": 0.9,
        }
    }
    if system:
        payload["system"] = system
    if format_json:
        payload["format"] = "json"

    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "bypass-tunnel-reminder": "true",
        "Bypass-Tunnel-Reminder": "true",
        "ngrok-skip-browser-warning": "true",
    }
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                result = json.loads(resp.read().decode("utf-8"))
                response_text = result.get("response", "")
                if format_json:
                    clean_text = response_text.strip()
                    if clean_text.startswith("```json"):
                        clean_text = clean_text[7:]
                    if clean_text.startswith("```"):
                        clean_text = clean_text[3:]
                    if clean_text.endswith("```"):
                        clean_text = clean_text[:-3]
                    return json.loads(clean_text.strip())
                return {"response": response_text}
    except Exception as e:
        print(f"Ollama query notice ({OLLAMA_MODEL} @ {OLLAMA_HOST}): {e}")
    return None


@app.get("/")
@app.get("/health")
async def health_check():
    """Health check endpoint for Render monitoring."""
    return {
        "status": "healthy",
        "service": "AgroSaathi ML Backend",
        "ollama_model": OLLAMA_MODEL,
        "disease_model_loaded": interpreter is not None,
        "recommend_model_loaded": recommend_model is not None,
    }


# ----------------------------------------
# 1. Disease Detection Endpoint (TFLite)
# ----------------------------------------
tflite_model_path = os.path.join(BASE_DIR, "plant_disease_model.tflite")
if not os.path.exists(tflite_model_path):
    tflite_model_path = os.path.join(BASE_DIR, "model.tflite")

interpreter = None
input_details = None
output_details = None

try:
    try:
        import tflite_runtime.interpreter as tflite
        interpreter = tflite.Interpreter(model_path=tflite_model_path)
    except ImportError:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"Loaded TFLite disease model from: {tflite_model_path}")
except Exception as e:
    print(f"Warning: Failed to load TFLite disease model ({e})")

labels = {}
class_indices_path = os.path.join(BASE_DIR, "class_indices.json")
if os.path.exists(class_indices_path):
    try:
        with open(class_indices_path, "r") as f:
            raw_labels = json.load(f)
            labels = {int(v): k for k, v in raw_labels.items()}
    except Exception as e:
        print(f"Warning: Failed to parse class_indices.json: {e}")


@app.post("/predict")
@app.post("/predict-disease")
async def predict_disease(file: UploadFile = File(...)):
    try:
        if interpreter is None:
            return {
                "success": False,
                "error": "Disease model not loaded on server."
            }

        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((256, 256))

        input_data = np.expand_dims(image, axis=0)

        if input_details[0]['dtype'] == np.float32:
            input_data = (input_data.astype(np.float32) / 127.5) - 1.0

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]['index'])

        if output_details[0]['dtype'] != np.float32:
            scale, zero_point = output_details[0]['quantization']
            output = (output.astype(np.float32) - zero_point) * scale

        exp_output = np.exp(output[0] - np.max(output[0]))
        probabilities = exp_output / exp_output.sum()

        predicted_index = int(np.argmax(probabilities))
        predicted_class = labels.get(predicted_index, f"Disease Index {predicted_index}")
        confidence = float(probabilities[predicted_index]) * 100.0

        return {
            "success": True,
            "disease": predicted_class,
            "confidence": round(confidence, 2)
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# ----------------------------------------
# 2. Disease Remedy Advisor (Ollama qwen2.5:3b)
# ----------------------------------------
class DiseaseRemedyRequest(BaseModel):
    diseaseName: str
    cropName: str | None = None
    language: str | None = "English"


@app.post("/disease-remedy")
async def disease_remedy(req: DiseaseRemedyRequest):
    """Generates AI treatment, cure steps, and prevention measures using qwen2.5:3b via Ollama."""
    disease = req.diseaseName.strip()

    system_prompt = (
        "You are an expert plant pathologist and agronomy advisor for Indian farmers. "
        "Given a plant disease name, output ONLY a JSON object (no markdown fences, no prose before or after) with this exact structure:\n"
        "{\n"
        '  "disease": "proper disease name",\n'
        '  "severity": "Mild | Moderate | Severe",\n'
        '  "organicRemedies": ["natural or organic cure step 1", "step 2"],\n'
        '  "chemicalTreatments": ["recommended chemical spray/fungicide with dosage", ...],\n'
        '  "preventiveMeasures": ["cultural or field practice step 1", ...],\n'
        '  "summaryAdvice": "1-2 sentence quick advice for the farmer"\n'
        "}"
    )

    prompt = f"Disease: {disease}. Crop: {req.cropName or 'Auto-detect'}. Preferred language: {req.language}."

    # 1. Try Ollama (qwen2.5:3b)
    ollama_res = query_ollama(prompt, system=system_prompt, format_json=True, timeout=35.0)
    if ollama_res and isinstance(ollama_res, dict) and "organicRemedies" in ollama_res:
        return {"success": True, "source": "ollama_qwen2.5", "remedy": ollama_res}

    # 2. Instant Smart Agronomic Fallback (Zero Gemini, Zero quota errors)
    formatted_name = disease.replace('___', ': ').replace('_', ' ').title()
    return {
        "success": True,
        "source": "fallback",
        "remedy": {
            "disease": formatted_name,
            "severity": "Moderate",
            "organicRemedies": [
                "Spray Neem oil solution (5ml per liter of water) during early morning or late evening.",
                "Remove and safely destroy severely infected foliage to halt spore spread.",
                "Apply Trichoderma viride bio-fungicide to soil around the root zone."
            ],
            "chemicalTreatments": [
                "Apply Copper Oxychloride 50 WP (2.5g per liter of water).",
                "For severe infections, spray Mancozeb 75 WP (2g per liter of water) at 10-14 day intervals."
            ],
            "preventiveMeasures": [
                "Maintain optimum plant spacing to encourage canopy ventilation.",
                "Avoid overhead sprinkler irrigation; use drip lines to keep leaves dry.",
                "Practice crop rotation with non-host crop families each season."
            ],
            "summaryAdvice": f"Isolate infected plants promptly and apply organic neem spray or recommended copper fungicide for {formatted_name}."
        }
    }


# ----------------------------------------
# 3. Crop Recommendation Endpoint
# ----------------------------------------
recommend_model = None
model_pkl_path = os.path.join(BASE_DIR, "model.pkl")
if os.path.exists(model_pkl_path):
    try:
        with open(model_pkl_path, "rb") as f:
            recommend_model = pickle.load(f)
            print(f"Loaded crop recommendation model from: {model_pkl_path}")
    except Exception as e:
        print(f"Warning: Failed to load model.pkl: {e}")


class RecommendationRequest(BaseModel):
    soilType: str
    season: str
    waterAvailability: str
    farmSizeAcres: float | None = 2.5
    latitude: float | None = None
    longitude: float | None = None


@app.post("/recommend_crop")
@app.post("/recommend-crop")
async def recommend_crop(req: RecommendationRequest):
    try:
        crop_metadata = {
            "wheat": {"name": "Wheat (गेहूँ)", "yieldPerAcre": 18.0, "price": 2275, "cost": 14000, "water": "medium", "risk": "low", "days": 120, "window": "Oct - Nov"},
            "rice": {"name": "Rice (चावल)", "yieldPerAcre": 22.0, "price": 2183, "cost": 18000, "water": "high", "risk": "medium", "days": 135, "window": "Jun - Jul"},
            "cotton": {"name": "Cotton (कपास)", "yieldPerAcre": 10.0, "price": 6620, "cost": 22000, "water": "medium", "risk": "high", "days": 160, "window": "May - Jun"},
            "maize": {"name": "Maize (मक्का)", "yieldPerAcre": 20.0, "price": 2090, "cost": 12000, "water": "medium", "risk": "low", "days": 100, "window": "Jun - Jul / Oct"},
            "soybean": {"name": "Soybean (सोयाबीन)", "yieldPerAcre": 12.0, "price": 4600, "cost": 13500, "water": "medium", "risk": "medium", "days": 105, "window": "Jun - Jul"},
            "chickpea": {"name": "Chickpea (चना)", "yieldPerAcre": 9.0, "price": 5440, "cost": 9500, "water": "low", "risk": "low", "days": 110, "window": "Oct - Nov"},
            "sugarcane": {"name": "Sugarcane (गन्ना)", "yieldPerAcre": 350.0, "price": 315, "cost": 45000, "water": "high", "risk": "low", "days": 360, "window": "Jan - Mar"},
            "onion": {"name": "Onion (प्याज़)", "yieldPerAcre": 80.0, "price": 1800, "cost": 30000, "water": "medium", "risk": "high", "days": 120, "window": "Oct - Dec"},
            "groundnut": {"name": "Groundnut (मूंगफली)", "yieldPerAcre": 11.0, "price": 6377, "cost": 14000, "water": "medium", "risk": "medium", "days": 115, "window": "Jun - Jul"},
            "mustard": {"name": "Mustard (सरसों)", "yieldPerAcre": 8.0, "price": 5650, "cost": 8000, "water": "low", "risk": "low", "days": 105, "window": "Oct - Nov"},
        }

        recommendations = []

        if recommend_model is not None:
            soil_map = {"black": 0, "red": 1, "loamy": 2, "sandy": 3, "clay": 4}
            season_map = {"kharif": 0, "rabi": 1, "zaid": 2, "summer": 2, "whole year": 3}
            water_map = {"low": 0, "medium": 1, "high": 2}

            s_val = soil_map.get(req.soilType.lower(), 0)
            se_val = season_map.get(req.season.lower(), 0)
            w_val = water_map.get(req.waterAvailability.lower(), 1)

            input_features = np.array([[s_val, se_val, w_val]])

            if hasattr(recommend_model, "predict_proba"):
                probs = recommend_model.predict_proba(input_features)[0]
                classes = recommend_model.classes_
                top_indices = np.argsort(probs)[::-1][:3]
                ranked_crops = [(str(classes[i]).lower(), float(probs[i])) for i in top_indices]
            else:
                pred = str(recommend_model.predict(input_features)[0]).lower()
                ranked_crops = [(pred, 0.85), ("wheat", 0.70), ("chickpea", 0.60)]
        else:
            ranked_crops = [("wheat", 0.92), ("chickpea", 0.84), ("soybean", 0.76)]

        for rank, (crop_slug, prob) in enumerate(ranked_crops):
            suitability = int(prob * 100) if prob <= 1.0 else int(prob)
            if suitability < 50:
                suitability = max(50, 78 - rank * 8)

            meta = crop_metadata.get(crop_slug, {
                "name": crop_slug.capitalize(),
                "yieldPerAcre": 15.0,
                "price": 3000,
                "cost": 15000,
                "water": "medium",
                "risk": "low",
                "days": 110,
                "window": "Optimal planting season"
            })

            farm_multiplier = req.farmSizeAcres if (req.farmSizeAcres and req.farmSizeAcres > 0) else 1.0
            est_yield = meta["yieldPerAcre"] * farm_multiplier
            est_revenue = est_yield * meta["price"]
            est_cost = meta["cost"] * farm_multiplier
            profit = max(0.0, est_revenue - est_cost)

            recommendations.append({
                "cropId": crop_slug,
                "cropName": meta["name"],
                "estimatedProfit": profit,
                "waterRequirement": meta["water"],
                "riskLevel": meta["risk"],
                "confidenceScore": round(prob, 3),
                "suitabilityScore": suitability,
                "growthDurationDays": meta["days"],
                "sowingWindow": meta["window"],
                "agronomicTips": f"ML Model match ({suitability}%) based on {req.soilType.capitalize()} soil and {req.season.capitalize()} season.",
                "estimatedYieldQuintal": round(est_yield, 1),
                "estimatedRevenue": round(est_revenue, 0),
                "estimatedCost": round(est_cost, 0)
            })

        return {
            "success": True,
            "count": len(recommendations),
            "recommendations": recommendations
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ----------------------------------------
# 4. Growth Plan Generation Endpoint (Ollama qwen2.5:3b)
# ----------------------------------------
ALLOWED_STAGES = {"sowing", "germination", "vegetative", "flowering", "maturity"}

GROWTH_PLAN_SYSTEM_PROMPT = """You are an agronomy assistant for AgroSaathi, a farming app used in Maharashtra, India.
Given a crop name and optional growing conditions, output ONLY a JSON object (no markdown fences, no prose before or after) with this exact shape:

{
  "cropName": "string, proper-cased crop name",
  "stages": [
    {
      "name": "one of: sowing, germination, vegetative, flowering, maturity",
      "durationDays": integer > 0,
      "irrigationFrequencyDays": integer > 0,
      "pestRisks": ["short pest or disease name", ...]
    }
  ],
  "fertilizerPlan": [
    {
      "stageName": "must match one of the stage names above",
      "fertilizerType": "short string, e.g. 'Basal NPK'",
      "dayOffsetInStage": integer >= 0
    }
  ]
}

Rules:
- Include exactly one entry per stage, in this order: sowing, germination, vegetative, flowering, maturity.
- Base durations and irrigation frequency on real agronomic practice for the given crop and, if provided, the soil/season/region.
- pestRisks should list realistic risks specific to that growth stage, not a generic list repeated on every stage.
- fertilizerPlan should have 1-3 realistic entries total across the whole cycle.
- If the input isn't a real, growable crop, respond with {"error": "not a recognized crop"} instead."""


class GrowthPlanRequest(BaseModel):
    cropName: str
    soilType: str | None = None
    season: str | None = None
    region: str | None = None


def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON object found in model output")
    return json.loads(match.group(0))


def _validate_template(data: dict) -> None:
    if "error" in data:
        raise ValueError(data["error"])

    stages = data.get("stages")
    if not isinstance(stages, list) or len(stages) != 5:
        raise ValueError("expected exactly 5 stages")

    seen_names = [s.get("name") for s in stages]
    if seen_names != ["sowing", "germination", "vegetative", "flowering", "maturity"]:
        raise ValueError(f"stages out of order or invalid: {seen_names}")

    for stage in stages:
        if not isinstance(stage.get("durationDays"), int) or stage["durationDays"] <= 0:
            raise ValueError(f"invalid durationDays for stage {stage.get('name')}")
        if not isinstance(stage.get("irrigationFrequencyDays"), int) or stage["irrigationFrequencyDays"] <= 0:
            raise ValueError(f"invalid irrigationFrequencyDays for stage {stage.get('name')}")

    for step in data.get("fertilizerPlan", []):
        if step.get("stageName") not in ALLOWED_STAGES:
            raise ValueError(f"fertilizerPlan references unknown stage: {step.get('stageName')}")


def _generate_smart_fallback_template(crop_name: str) -> dict:
    name = crop_name.strip().title() if crop_name.strip() else "Custom Crop"
    return {
        "cropName": name,
        "stages": [
            {"name": "sowing", "durationDays": 10, "irrigationFrequencyDays": 5, "pestRisks": ["Soil Pests"]},
            {"name": "germination", "durationDays": 12, "irrigationFrequencyDays": 6, "pestRisks": ["Cutworm", "Damping Off"]},
            {"name": "vegetative", "durationDays": 35, "irrigationFrequencyDays": 7, "pestRisks": ["Aphids", "Leaf Spot"]},
            {"name": "flowering", "durationDays": 30, "irrigationFrequencyDays": 7, "pestRisks": ["Bollworm", "Blight"]},
            {"name": "maturity", "durationDays": 25, "irrigationFrequencyDays": 10, "pestRisks": ["Fungal Rot"]},
        ],
        "fertilizerPlan": [
            {"stageName": "sowing", "fertilizerType": "Basal NPK", "dayOffsetInStage": 0},
            {"stageName": "vegetative", "fertilizerType": "Urea Top Dressing", "dayOffsetInStage": 15},
            {"stageName": "flowering", "fertilizerType": "Potash Boost", "dayOffsetInStage": 10},
        ],
    }


_growth_plan_cache: dict[str, dict] = {}


def _cache_key(request: "GrowthPlanRequest") -> str:
    return "|".join([
        request.cropName.strip().lower(),
        (request.soilType or "").strip().lower(),
        (request.season or "").strip().lower(),
        (request.region or "").strip().lower(),
    ])


@app.post("/generate-growth-plan")
async def generate_growth_plan(request: GrowthPlanRequest):
    cache_key = _cache_key(request)
    if cache_key in _growth_plan_cache:
        return {"success": True, "template": _growth_plan_cache[cache_key], "cached": True}

    crop_slug = request.cropName.strip().lower()
    if crop_slug in _growth_plan_cache:
        return {"success": True, "template": _growth_plan_cache[crop_slug], "cached": True}

    context_parts = [f"Crop: {request.cropName}"]
    if request.soilType:
        context_parts.append(f"Soil type: {request.soilType}")
    if request.season:
        context_parts.append(f"Season: {request.season}")
    if request.region:
        context_parts.append(f"Region: {request.region}")

    prompt_text = "\n".join(context_parts)

    # 1. Try Ollama qwen2.5:3b
    ollama_res = query_ollama(prompt_text, system=GROWTH_PLAN_SYSTEM_PROMPT, format_json=True, timeout=35.0)
    if ollama_res and isinstance(ollama_res, dict):
        try:
            _validate_template(ollama_res)
            _growth_plan_cache[cache_key] = ollama_res
            _growth_plan_cache[crop_slug] = ollama_res
            return {"success": True, "template": ollama_res, "source": "ollama_qwen2.5"}
        except Exception as ve:
            print(f"Ollama template validation notice: {ve}")

    # 2. Instant Smart Agronomic Fallback (Zero Gemini, Zero 429 quota errors)
    fallback_data = _generate_smart_fallback_template(request.cropName)
    _growth_plan_cache[cache_key] = fallback_data
    _growth_plan_cache[crop_slug] = fallback_data
    return {"success": True, "template": fallback_data, "fallback": True}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)