import os
import re
import json
import asyncio
import warnings
import urllib.request
import urllib.error
import numpy as np
import pickle
import time
from datetime import datetime, timezone
from collections import deque
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from PIL import Image, ImageOps
import io

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(
    title="AgroSaathi ML Backend",
    description="Production-ready FastAPI backend for plant disease detection, crop recommendation, AI growth plans, and Ollama qwen2.5:3b disease remedies with live UI Dashboard.",
    version="1.3.0"
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

# In-memory Request Log Ring Buffer (Max 100 logs)
MAX_LOGS = 100
REQUEST_LOGS = deque(maxlen=MAX_LOGS)
LOG_COUNTER = 0


def record_log(log_data: dict):
    global LOG_COUNTER
    LOG_COUNTER += 1
    log_data["id"] = LOG_COUNTER
    REQUEST_LOGS.appendleft(log_data)


def query_ollama_detailed(prompt: str, system: str = "", format_json: bool = True, timeout: float = 35.0) -> tuple[dict | None, float, str, str | None]:
    """Queries local or remote Ollama server running qwen2.5:3b model and returns detailed diagnostics."""
    start_time = time.time()
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
            elapsed = time.time() - start_time
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
                    return json.loads(clean_text.strip()), elapsed, "success", None
                return {"response": response_text}, elapsed, "success", None
            else:
                return None, elapsed, "error", f"HTTP Status {resp.status}"
    except urllib.error.URLError as e:
        elapsed = time.time() - start_time
        err_msg = str(e.reason) if hasattr(e, "reason") else str(e)
        if "timed out" in err_msg.lower() or "timeout" in err_msg.lower():
            return None, elapsed, "timeout", f"Read operation timed out after {round(elapsed, 1)}s"
        return None, elapsed, "error", err_msg
    except Exception as e:
        elapsed = time.time() - start_time
        err_str = str(e)
        if "timed out" in err_str.lower() or "timeout" in err_str.lower():
            return None, elapsed, "timeout", f"Read operation timed out after {round(elapsed, 1)}s"
        return None, elapsed, "error", err_str


def query_ollama(prompt: str, system: str = "", format_json: bool = True, timeout: float = 35.0) -> dict | None:
    res, _, _, _ = query_ollama_detailed(prompt, system=system, format_json=format_json, timeout=timeout)
    return res


@app.get("/")
@app.get("/health")
async def health_check():
    """Health check endpoint for Render monitoring."""
    return {
        "status": "healthy",
        "service": "AgroSaathi ML Backend",
        "ollama_model": OLLAMA_MODEL,
        "ollama_host": OLLAMA_HOST,
        "disease_model_loaded": interpreter is not None,
        "recommend_model_loaded": recommend_model is not None,
        "dashboard_url": "/dashboard",
    }


# ----------------------------------------
# 1. Disease Detection Endpoint (TFLite)
# ----------------------------------------
tflite_model_path = os.path.join(BASE_DIR, "plant_disease_model_quantized.tflite")
if not os.path.exists(tflite_model_path):
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
async def predict_disease(request: Request, file: UploadFile = File(...)):
    start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"
    filename = file.filename or "uploaded_image.jpg"
    try:
        if interpreter is None:
            res_body = {"success": False, "error": "Disease model not loaded on server."}
            dur = round((time.time() - start_time) * 1000, 1)
            record_log({
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                "endpoint": "/predict-disease",
                "method": "POST",
                "client_ip": client_ip,
                "request_body": {"filename": filename, "content_type": file.content_type},
                "ollama_attempt": {"status": "skipped", "duration_sec": 0, "host": OLLAMA_HOST, "model": OLLAMA_MODEL},
                "source": "tflite_error",
                "response_status": 500,
                "response_sent": True,
                "response_body": res_body,
                "total_duration_ms": dur
            })
            return res_body

        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image = ImageOps.exif_transpose(image).convert("RGB")
        target_h = input_details[0]['shape'][1] if len(input_details[0]['shape']) > 2 else 256
        target_w = input_details[0]['shape'][2] if len(input_details[0]['shape']) > 2 else 256
        image = image.resize((target_w, target_h))

        input_data = np.expand_dims(image, axis=0)

        # Note: Model contains built-in Rescaling(1./255) layer, so input must be raw float32 [0..255]
        if input_details[0]['dtype'] == np.float32:
            input_data = input_data.astype(np.float32)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]['index'])

        if output_details[0]['dtype'] != np.float32:
            scale, zero_point = output_details[0]['quantization']
            output = (output.astype(np.float32) - zero_point) * scale

        raw_output = output[0]
        # Check if the TFLite model output already includes Softmax probabilities
        if abs(float(np.sum(raw_output)) - 1.0) < 0.05:
            probabilities = raw_output
        else:
            exp_output = np.exp(raw_output - np.max(raw_output))
            probabilities = exp_output / exp_output.sum()

        predicted_index = int(np.argmax(probabilities))
        predicted_class = labels.get(predicted_index, f"Disease Index {predicted_index}")
        confidence = float(probabilities[predicted_index]) * 100.0

        res_body = {
            "success": True,
            "disease": predicted_class,
            "confidence": round(confidence, 2)
        }
        dur = round((time.time() - start_time) * 1000, 1)
        record_log({
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "endpoint": "/predict-disease",
            "method": "POST",
            "client_ip": client_ip,
            "request_body": {"filename": filename, "image_size_bytes": len(contents)},
            "ollama_attempt": {"status": "skipped", "duration_sec": 0, "host": OLLAMA_HOST, "model": OLLAMA_MODEL},
            "source": "tflite_model",
            "response_status": 200,
            "response_sent": True,
            "response_body": res_body,
            "total_duration_ms": dur
        })
        return res_body

    except Exception as e:
        res_body = {"success": False, "error": str(e)}
        dur = round((time.time() - start_time) * 1000, 1)
        record_log({
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "endpoint": "/predict-disease",
            "method": "POST",
            "client_ip": client_ip,
            "request_body": {"filename": filename},
            "ollama_attempt": {"status": "skipped", "duration_sec": 0, "host": OLLAMA_HOST, "model": OLLAMA_MODEL},
            "source": "tflite_error",
            "response_status": 500,
            "response_sent": True,
            "response_body": res_body,
            "total_duration_ms": dur
        })
        return res_body


# ----------------------------------------
# 2. Disease Remedy Advisor (Ollama qwen2.5:3b)
# ----------------------------------------
class DiseaseRemedyRequest(BaseModel):
    diseaseName: str
    cropName: str | None = None
    language: str | None = "English"


@app.post("/disease-remedy")
async def disease_remedy(req: DiseaseRemedyRequest, request: Request):
    """Generates AI treatment, cure steps, and prevention measures using qwen2.5:3b via Ollama."""
    start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"
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
    ollama_res, dur_sec, ollama_status, ollama_err = query_ollama_detailed(prompt, system=system_prompt, format_json=True, timeout=35.0)

    if ollama_res and isinstance(ollama_res, dict) and "organicRemedies" in ollama_res:
        resp_data = {"success": True, "source": "ollama_qwen2.5", "remedy": ollama_res}
        source_used = "ollama_qwen2.5"
    else:
        formatted_name = disease.replace('___', ': ').replace('_', ' ').title()
        resp_data = {
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
        source_used = "fallback"

    total_dur_ms = round((time.time() - start_time) * 1000, 1)

    record_log({
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "endpoint": "/disease-remedy",
        "method": "POST",
        "client_ip": client_ip,
        "request_body": req.dict(),
        "ollama_attempt": {
            "host": OLLAMA_HOST,
            "model": OLLAMA_MODEL,
            "duration_sec": round(dur_sec, 2),
            "status": ollama_status,
            "error": ollama_err,
        },
        "source": source_used,
        "response_status": 200,
        "response_sent": True,
        "response_body": resp_data,
        "total_duration_ms": total_dur_ms,
    })

    return resp_data


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
async def recommend_crop(req: RecommendationRequest, request: Request):
    start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"
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

        res_body = {
            "success": True,
            "count": len(recommendations),
            "recommendations": recommendations
        }
        total_dur_ms = round((time.time() - start_time) * 1000, 1)
        record_log({
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "endpoint": "/recommend-crop",
            "method": "POST",
            "client_ip": client_ip,
            "request_body": req.dict(),
            "ollama_attempt": {"status": "skipped", "duration_sec": 0, "host": OLLAMA_HOST, "model": OLLAMA_MODEL},
            "source": "pickle_model" if recommend_model is not None else "rule_engine",
            "response_status": 200,
            "response_sent": True,
            "response_body": res_body,
            "total_duration_ms": total_dur_ms,
        })
        return res_body
    except Exception as e:
        res_body = {"success": False, "error": str(e)}
        total_dur_ms = round((time.time() - start_time) * 1000, 1)
        record_log({
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "endpoint": "/recommend-crop",
            "method": "POST",
            "client_ip": client_ip,
            "request_body": req.dict(),
            "ollama_attempt": {"status": "skipped", "duration_sec": 0, "host": OLLAMA_HOST, "model": OLLAMA_MODEL},
            "source": "error",
            "response_status": 500,
            "response_sent": True,
            "response_body": res_body,
            "total_duration_ms": total_dur_ms,
        })
        return res_body


# ----------------------------------------
# 4. Growth Plan Generation Endpoint (Ollama qwen2.5:3b)
# ----------------------------------------
ALLOWED_STAGES = {"sowing", "germination", "vegetative", "flowering", "maturity"}

GROWTH_PLAN_SYSTEM_PROMPT = (
    "You are an agronomy expert for Indian crops. Output ONLY valid JSON with 5 stages (sowing, germination, vegetative, flowering, maturity) in this exact structure:\n"
    "{\n"
    '  "cropName": "Crop Name",\n'
    '  "stages": [\n'
    '    {"name": "sowing", "durationDays": 10, "irrigationFrequencyDays": 5, "pestRisks": ["Soil Pests"]},\n'
    '    {"name": "germination", "durationDays": 12, "irrigationFrequencyDays": 6, "pestRisks": ["Cutworm"]},\n'
    '    {"name": "vegetative", "durationDays": 35, "irrigationFrequencyDays": 7, "pestRisks": ["Aphids"]},\n'
    '    {"name": "flowering", "durationDays": 30, "irrigationFrequencyDays": 7, "pestRisks": ["Bollworm"]},\n'
    '    {"name": "maturity", "durationDays": 25, "irrigationFrequencyDays": 10, "pestRisks": ["Fungal Rot"]}\n'
    '  ],\n'
    '  "fertilizerPlan": [\n'
    '    {"stageName": "sowing", "fertilizerType": "Basal NPK", "dayOffsetInStage": 0},\n'
    '    {"stageName": "vegetative", "fertilizerType": "Urea", "dayOffsetInStage": 15}\n'
    '  ]\n'
    "}"
)


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


def _normalize_and_validate_template(data: dict, default_crop_name: str) -> dict:
    if not isinstance(data, dict) or "error" in data:
        raise ValueError(data.get("error", "Invalid data format") if isinstance(data, dict) else "Not a dict")

    crop_name = data.get("cropName") or data.get("crop_name") or default_crop_name
    raw_stages = data.get("stages") or data.get("growth_stages") or []

    if not isinstance(raw_stages, list) or len(raw_stages) < 3:
        raise ValueError("Insufficient stages returned")

    stage_order = ["sowing", "germination", "vegetative", "flowering", "maturity"]
    normalized_stages = []

    for i, s in enumerate(raw_stages):
        if not isinstance(s, dict):
            continue
        raw_name = str(s.get("name") or s.get("stage") or s.get("stage_name") or "").strip().lower()
        if not raw_name:
            raw_name = stage_order[i] if i < len(stage_order) else f"stage_{i+1}"
        
        matched_name = raw_name
        for target in stage_order:
            if target in raw_name:
                matched_name = target
                break

        dur = s.get("durationDays") or s.get("duration_days") or s.get("duration") or 14
        try:
            dur = int(dur)
        except (ValueError, TypeError):
            dur = 14

        irrig = s.get("irrigationFrequencyDays") or s.get("irrigation_frequency_days") or s.get("irrigation") or 7
        try:
            irrig = int(irrig)
        except (ValueError, TypeError):
            irrig = 7

        pests = s.get("pestRisks") or s.get("pest_risks") or s.get("pests") or []
        if not isinstance(pests, list):
            pests = [str(pests)]

        normalized_stages.append({
            "name": matched_name,
            "durationDays": max(1, dur),
            "irrigationFrequencyDays": max(1, irrig),
            "pestRisks": [str(p) for p in pests if p],
        })

    existing_names = {s["name"] for s in normalized_stages}
    for req_stage in stage_order:
        if req_stage not in existing_names:
            normalized_stages.append({
                "name": req_stage,
                "durationDays": 15,
                "irrigationFrequencyDays": 7,
                "pestRisks": ["General Pests"],
            })

    normalized_stages.sort(key=lambda s: stage_order.index(s["name"]) if s["name"] in stage_order else 99)
    normalized_stages = normalized_stages[:5]

    raw_fert = data.get("fertilizerPlan") or data.get("fertilizer_plan") or []
    normalized_fert = []
    if isinstance(raw_fert, list):
        for f in raw_fert:
            if isinstance(f, dict):
                stg = str(f.get("stageName") or f.get("stage_name") or f.get("stage") or "sowing").strip().lower()
                matched_stg = "sowing"
                for target in stage_order:
                    if target in stg:
                        matched_stg = target
                        break
                ftype = str(f.get("fertilizerType") or f.get("fertilizer_type") or f.get("fertilizer") or "NPK Blend")
                offset = f.get("dayOffsetInStage") or f.get("day_offset_in_stage") or f.get("day_offset") or 0
                try:
                    offset = int(offset)
                except (ValueError, TypeError):
                    offset = 0
                normalized_fert.append({
                    "stageName": matched_stg,
                    "fertilizerType": ftype,
                    "dayOffsetInStage": max(0, offset),
                })

    if not normalized_fert:
        normalized_fert = [
            {"stageName": "sowing", "fertilizerType": "Basal NPK", "dayOffsetInStage": 0},
            {"stageName": "vegetative", "fertilizerType": "Urea Top Dressing", "dayOffsetInStage": 15},
        ]

    return {
        "cropName": str(crop_name).title(),
        "stages": normalized_stages,
        "fertilizerPlan": normalized_fert,
    }


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
async def generate_growth_plan(request: GrowthPlanRequest, req_obj: Request):
    start_time = time.time()
    client_ip = req_obj.client.host if req_obj.client else "unknown"
    cache_key = _cache_key(request)
    crop_slug = request.cropName.strip().lower()

    if cache_key in _growth_plan_cache or crop_slug in _growth_plan_cache:
        cached_tpl = _growth_plan_cache.get(cache_key) or _growth_plan_cache.get(crop_slug)
        res_body = {"success": True, "template": cached_tpl, "cached": True}
        total_dur_ms = round((time.time() - start_time) * 1000, 1)
        record_log({
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "endpoint": "/generate-growth-plan",
            "method": "POST",
            "client_ip": client_ip,
            "request_body": request.dict(),
            "ollama_attempt": {"status": "skipped_cache_hit", "duration_sec": 0, "host": OLLAMA_HOST, "model": OLLAMA_MODEL},
            "source": "in_memory_cache",
            "response_status": 200,
            "response_sent": True,
            "response_body": res_body,
            "total_duration_ms": total_dur_ms,
        })
        return res_body

    context_parts = [f"Crop: {request.cropName}"]
    if request.soilType:
        context_parts.append(f"Soil type: {request.soilType}")
    if request.season:
        context_parts.append(f"Season: {request.season}")
    if request.region:
        context_parts.append(f"Region: {request.region}")

    prompt_text = "\n".join(context_parts)

    # 1. Try Ollama qwen2.5:3b (with 60s timeout)
    ollama_res, dur_sec, ollama_status, ollama_err = query_ollama_detailed(prompt_text, system=GROWTH_PLAN_SYSTEM_PROMPT, format_json=True, timeout=60.0)

    if ollama_res and isinstance(ollama_res, dict):
        try:
            cleaned_template = _normalize_and_validate_template(ollama_res, request.cropName)
            _growth_plan_cache[cache_key] = cleaned_template
            _growth_plan_cache[crop_slug] = cleaned_template
            res_body = {"success": True, "template": cleaned_template, "source": "ollama_qwen2.5"}
            total_dur_ms = round((time.time() - start_time) * 1000, 1)
            record_log({
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                "endpoint": "/generate-growth-plan",
                "method": "POST",
                "client_ip": client_ip,
                "request_body": request.dict(),
                "ollama_attempt": {
                    "host": OLLAMA_HOST,
                    "model": OLLAMA_MODEL,
                    "duration_sec": round(dur_sec, 2),
                    "status": ollama_status,
                    "error": ollama_err,
                },
                "source": "ollama_qwen2.5",
                "response_status": 200,
                "response_sent": True,
                "response_body": res_body,
                "total_duration_ms": total_dur_ms,
            })
            return res_body
        except Exception as ve:
            print(f"Ollama template normalization notice: {ve}")
            if not ollama_err:
                ollama_err = f"Template normalization error: {ve}"

    # 2. Instant Smart Agronomic Fallback (Zero Gemini, Zero 429 quota errors)
    fallback_data = _generate_smart_fallback_template(request.cropName)
    _growth_plan_cache[cache_key] = fallback_data
    _growth_plan_cache[crop_slug] = fallback_data
    res_body = {"success": True, "template": fallback_data, "fallback": True}
    total_dur_ms = round((time.time() - start_time) * 1000, 1)
    record_log({
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "endpoint": "/generate-growth-plan",
        "method": "POST",
        "client_ip": client_ip,
        "request_body": request.dict(),
        "ollama_attempt": {
            "host": OLLAMA_HOST,
            "model": OLLAMA_MODEL,
            "duration_sec": round(dur_sec, 2),
            "status": ollama_status,
            "error": ollama_err,
        },
        "source": "fallback",
        "response_status": 200,
        "response_sent": True,
        "response_body": res_body,
        "total_duration_ms": total_dur_ms,
    })
    return res_body


# ----------------------------------------
# 5. Live Dashboard & Logging API
# ----------------------------------------
@app.get("/api/request-logs")
async def get_request_logs():
    logs_list = list(REQUEST_LOGS)
    total_count = len(logs_list)
    ollama_successes = sum(1 for l in logs_list if l.get("source") == "ollama_qwen2.5")
    fallbacks = sum(1 for l in logs_list if l.get("source") == "fallback")

    return {
        "status": "success",
        "summary": {
            "total_requests": total_count,
            "ollama_successes": ollama_successes,
            "fallbacks": fallbacks,
            "ollama_host": OLLAMA_HOST,
            "ollama_model": OLLAMA_MODEL,
        },
        "logs": logs_list
    }


@app.post("/api/clear-logs")
async def clear_request_logs():
    REQUEST_LOGS.clear()
    return {"status": "success", "message": "Request logs cleared."}


@app.get("/api/ollama-status")
async def check_ollama_status():
    start_t = time.time()
    url = f"{OLLAMA_HOST.rstrip('/')}/api/tags"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "bypass-tunnel-reminder": "true",
        "ngrok-skip-browser-warning": "true",
    }
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            elapsed = round((time.time() - start_t) * 1000, 1)
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name") for m in data.get("models", [])]
                return {
                    "online": True,
                    "status": "Connected",
                    "host": OLLAMA_HOST,
                    "target_model": OLLAMA_MODEL,
                    "latency_ms": elapsed,
                    "available_models": models
                }
    except Exception as e:
        elapsed = round((time.time() - start_t) * 1000, 1)
        return {
            "online": False,
            "status": "Offline / Unreachable",
            "host": OLLAMA_HOST,
            "target_model": OLLAMA_MODEL,
            "latency_ms": elapsed,
            "error": str(e)
        }


@app.get("/dashboard", response_class=HTMLResponse)
@app.get("/logs-ui", response_class=HTMLResponse)
async def render_dashboard():
    """Serves an interactive single-page dashboard to monitor backend requests and Ollama performance."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AgroSaathi Backend Dashboard & Request Monitor</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
  <style>
    body { background-color: #0f172a; color: #f8fafc; font-family: system-ui, -apple-system, sans-serif; }
    .badge-ollama { background-color: #1e1b4b; color: #818cf8; border: 1px solid #4338ca; }
    .badge-fallback { background-color: #451a03; color: #fbbf24; border: 1px solid #92400e; }
    .badge-tflite { background-color: #312e81; color: #c084fc; border: 1px solid #6b21a8; }
    .badge-pickle { background-color: #064e3b; color: #34d399; border: 1px solid #047857; }
    pre { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }
  </style>
</head>
<body class="p-4 md:p-6 min-h-screen flex flex-col">
  <!-- Header -->
  <header class="flex flex-col md:flex-row justify-between items-start md:items-center pb-4 mb-6 border-b border-slate-800 gap-4">
    <div>
      <div class="flex items-center gap-3">
        <h1 class="text-2xl font-bold text-emerald-400 flex items-center gap-2">
          <i class="fa-solid fa-seedling text-emerald-500"></i> AgroSaathi AI Dashboard
        </h1>
        <span id="live-badge" class="flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-950 text-emerald-400 border border-emerald-800">
          <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span> LIVE MONITORING
        </span>
      </div>
      <p class="text-xs text-slate-400 mt-1">Real-time trace analyzer for incoming mobile requests, Ollama qwen2.5 execution, and JSON responses.</p>
    </div>

    <!-- Actions Bar -->
    <div class="flex flex-wrap items-center gap-3">
      <div class="flex items-center bg-slate-800 rounded-lg p-1 text-xs border border-slate-700">
        <label class="px-2 text-slate-400">Refresh:</label>
        <select id="refresh-interval" class="bg-slate-900 text-slate-200 border-none rounded px-2 py-1 focus:ring-1 focus:ring-emerald-500 outline-none">
          <option value="2000" selected>2s</option>
          <option value="5000">5s</option>
          <option value="10000">10s</option>
          <option value="0">Paused</option>
        </select>
      </div>

      <button onclick="fetchLogs()" class="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-slate-200 rounded-lg border border-slate-700 transition flex items-center gap-1.5">
        <i class="fa-solid fa-rotate-right"></i> Refresh Now
      </button>

      <button onclick="clearLogs()" class="px-3 py-1.5 bg-red-950/60 hover:bg-red-900/80 text-xs font-semibold text-red-300 border border-red-800/80 rounded-lg transition flex items-center gap-1.5">
        <i class="fa-solid fa-trash-can"></i> Clear Logs
      </button>
    </div>
  </header>

  <!-- Metric Cards -->
  <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
    <div class="bg-slate-800/80 border border-slate-700/80 rounded-xl p-4 flex flex-col justify-between shadow-lg">
      <div class="flex items-center justify-between text-slate-400 text-xs font-semibold mb-2">
        <span>TOTAL REQUESTS CAPTURED</span>
        <i class="fa-solid fa-server text-emerald-400 text-sm"></i>
      </div>
      <div class="text-3xl font-extrabold text-white" id="card-total-requests">0</div>
      <div class="text-[11px] text-slate-400 mt-2">Captured in current session (Max 100)</div>
    </div>

    <div class="bg-slate-800/80 border border-slate-700/80 rounded-xl p-4 flex flex-col justify-between shadow-lg">
      <div class="flex items-center justify-between text-slate-400 text-xs font-semibold mb-2">
        <span>OLLAMA QWEN2.5 RESPONSES</span>
        <i class="fa-solid fa-brain text-indigo-400 text-sm"></i>
      </div>
      <div class="text-3xl font-extrabold text-indigo-400" id="card-ollama-successes">0</div>
      <div class="text-[11px] text-slate-400 mt-2" id="card-ollama-percent">0% of total AI queries</div>
    </div>

    <div class="bg-slate-800/80 border border-slate-700/80 rounded-xl p-4 flex flex-col justify-between shadow-lg">
      <div class="flex items-center justify-between text-slate-400 text-xs font-semibold mb-2">
        <span>SMART FALLBACKS USED</span>
        <i class="fa-solid fa-shield-halved text-amber-400 text-sm"></i>
      </div>
      <div class="text-3xl font-extrabold text-amber-400" id="card-fallbacks">0</div>
      <div class="text-[11px] text-slate-400 mt-2">Triggered on timeout or tunnel offline</div>
    </div>

    <div class="bg-slate-800/80 border border-slate-700/80 rounded-xl p-4 flex flex-col justify-between shadow-lg">
      <div class="flex items-center justify-between text-slate-400 text-xs font-semibold mb-2">
        <span>OLLAMA TUNNEL CONNECTIVITY</span>
        <button onclick="checkTunnelStatus()" class="hover:text-emerald-400 transition"><i class="fa-solid fa-arrows-rotate"></i></button>
      </div>
      <div class="flex items-center gap-2 mt-1">
        <span id="tunnel-status-dot" class="w-3 h-3 rounded-full bg-slate-600"></span>
        <span id="tunnel-status-text" class="text-lg font-bold text-slate-300">Checking...</span>
      </div>
      <div class="text-[11px] text-slate-400 mt-2 truncate" id="tunnel-host-info" title="">Host: Loading...</div>
    </div>
  </div>

  <!-- Interactive Live Quick Tests -->
  <div class="bg-slate-800/40 border border-slate-700/60 rounded-xl p-4 mb-6">
    <div class="text-xs font-bold text-slate-300 mb-3 flex items-center gap-2">
      <i class="fa-solid fa-flask text-emerald-400"></i> LIVE TEST SIMULATOR (Trigger API requests directly from backend)
    </div>
    <div class="flex flex-wrap gap-3">
      <button onclick="sendTestRemedy('Potato Late Blight')" class="px-3 py-1.5 bg-indigo-900/60 hover:bg-indigo-800 text-xs text-indigo-200 border border-indigo-700 rounded-lg transition">
        <i class="fa-solid fa-bug mr-1"></i> Test Disease Remedy (Potato Blight)
      </button>
      <button onclick="sendTestGrowthPlan('Rice')" class="px-3 py-1.5 bg-emerald-900/60 hover:bg-emerald-800 text-xs text-emerald-200 border border-emerald-700 rounded-lg transition">
        <i class="fa-solid fa-wheat-awn mr-1"></i> Test Growth Plan (Rice)
      </button>
      <button onclick="sendTestCropRec()" class="px-3 py-1.5 bg-teal-900/60 hover:bg-teal-800 text-xs text-teal-200 border border-teal-700 rounded-lg transition">
        <i class="fa-solid fa-chart-line mr-1"></i> Test Crop Recommendation
      </button>
    </div>
  </div>

  <!-- Filters & Search Toolbar -->
  <div class="flex flex-col sm:flex-row items-center justify-between gap-4 mb-4">
    <div class="relative w-full sm:w-80">
      <i class="fa-solid fa-magnifying-glass absolute left-3 top-2.5 text-slate-500 text-xs"></i>
      <input type="text" id="search-input" onkeyup="filterLogs()" placeholder="Search crop, disease, or payload..." class="w-full bg-slate-800 text-slate-200 text-xs rounded-lg pl-8 pr-3 py-2 border border-slate-700 focus:outline-none focus:border-emerald-500">
    </div>

    <div class="flex items-center gap-3 w-full sm:w-auto justify-end text-xs">
      <select id="filter-endpoint" onchange="filterLogs()" class="bg-slate-800 text-slate-200 border border-slate-700 rounded-lg px-3 py-1.5 focus:outline-none focus:border-emerald-500">
        <option value="ALL">All Endpoints</option>
        <option value="/disease-remedy">/disease-remedy</option>
        <option value="/generate-growth-plan">/generate-growth-plan</option>
        <option value="/predict-disease">/predict-disease</option>
        <option value="/recommend-crop">/recommend-crop</option>
      </select>

      <select id="filter-source" onchange="filterLogs()" class="bg-slate-800 text-slate-200 border border-slate-700 rounded-lg px-3 py-1.5 focus:outline-none focus:border-emerald-500">
        <option value="ALL">All Sources</option>
        <option value="ollama_qwen2.5">Ollama (qwen2.5:3b)</option>
        <option value="fallback">Smart Fallback</option>
        <option value="tflite_model">TFLite Model</option>
        <option value="pickle_model">Pickle Model</option>
      </select>
    </div>
  </div>

  <!-- Logs Table -->
  <div class="bg-slate-800/60 border border-slate-700/80 rounded-xl overflow-hidden shadow-xl flex-grow flex flex-col">
    <div class="overflow-x-auto">
      <table class="w-full text-left text-xs border-collapse">
        <thead class="bg-slate-900/90 text-slate-400 font-semibold border-b border-slate-700">
          <tr>
            <th class="p-3"># / TIME (UTC)</th>
            <th class="p-3">ENDPOINT</th>
            <th class="p-3">INPUT REQUEST</th>
            <th class="p-3">OLLAMA EXECUTION</th>
            <th class="p-3">SOURCE USED</th>
            <th class="p-3">LATENCY</th>
            <th class="p-3">RESPONSE SENT</th>
          </tr>
        </thead>
        <tbody id="logs-tbody" class="divide-y divide-slate-800 text-slate-300">
          <tr>
            <td colspan="7" class="p-8 text-center text-slate-500">
              <i class="fa-solid fa-spinner fa-spin text-lg mb-2"></i><br>Loading live request logs...
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- JSON Modal Viewer -->
  <div id="json-modal" class="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4 hidden z-50">
    <div class="bg-slate-900 border border-slate-700 rounded-xl max-w-3xl w-full max-h-[85vh] flex flex-col shadow-2xl">
      <div class="p-4 border-b border-slate-800 flex justify-between items-center">
        <h3 id="modal-title" class="font-bold text-emerald-400 text-sm flex items-center gap-2">
          <i class="fa-solid fa-code"></i> Request Details
        </h3>
        <button onclick="closeModal()" class="text-slate-400 hover:text-white"><i class="fa-solid fa-xmark text-lg"></i></button>
      </div>
      <div class="p-4 overflow-y-auto flex-grow">
        <pre id="modal-json" class="bg-slate-950 p-4 rounded-lg text-emerald-300 text-xs overflow-x-auto leading-relaxed border border-slate-800"></pre>
      </div>
      <div class="p-3 border-t border-slate-800 flex justify-end gap-2 bg-slate-900/50">
        <button onclick="copyModalJson()" class="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs rounded-lg border border-slate-700 transition flex items-center gap-1">
          <i class="fa-solid fa-copy"></i> Copy JSON
        </button>
        <button onclick="closeModal()" class="px-3 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold rounded-lg transition">
          Close
        </button>
      </div>
    </div>
  </div>

  <script>
    let rawLogs = [];
    let timer = null;

    async function fetchLogs() {
      try {
        const res = await fetch('/api/request-logs');
        if (!res.ok) return;
        const data = await res.json();
        
        rawLogs = data.logs || [];
        updateSummaryCards(data.summary);
        renderLogsTable();
      } catch (err) {
        console.error("Failed to fetch logs:", err);
      }
    }

    async function checkTunnelStatus() {
      const dot = document.getElementById('tunnel-status-dot');
      const text = document.getElementById('tunnel-status-text');
      const hostInfo = document.getElementById('tunnel-host-info');

      text.innerText = "Pinging...";
      dot.className = "w-3 h-3 rounded-full bg-amber-400 animate-ping";

      try {
        const res = await fetch('/api/ollama-status');
        const data = await res.json();
        hostInfo.innerText = `Host: ${data.host}`;
        hostInfo.title = data.host;

        if (data.online) {
          dot.className = "w-3 h-3 rounded-full bg-emerald-400";
          text.className = "text-lg font-bold text-emerald-400";
          text.innerText = `Connected (${data.latency_ms}ms)`;
        } else {
          dot.className = "w-3 h-3 rounded-full bg-red-500";
          text.className = "text-lg font-bold text-red-400";
          text.innerText = "Offline / Timeout";
        }
      } catch (e) {
        dot.className = "w-3 h-3 rounded-full bg-red-500";
        text.className = "text-lg font-bold text-red-400";
        text.innerText = "Error Checking";
      }
    }

    function updateSummaryCards(summary) {
      if (!summary) return;
      document.getElementById('card-total-requests').innerText = summary.total_requests || 0;
      document.getElementById('card-ollama-successes').innerText = summary.ollama_successes || 0;
      document.getElementById('card-fallbacks').innerText = summary.fallbacks || 0;

      const totalAI = (summary.ollama_successes || 0) + (summary.fallbacks || 0);
      const pct = totalAI > 0 ? Math.round((summary.ollama_successes / totalAI) * 100) : 0;
      document.getElementById('card-ollama-percent').innerText = `${pct}% of total AI queries`;
    }

    function renderLogsTable() {
      const tbody = document.getElementById('logs-tbody');
      const searchVal = document.getElementById('search-input').value.toLowerCase();
      const endpointVal = document.getElementById('filter-endpoint').value;
      const sourceVal = document.getElementById('filter-source').value;

      const filtered = rawLogs.filter(log => {
        if (endpointVal !== "ALL" && log.endpoint !== endpointVal) return false;
        if (sourceVal !== "ALL" && log.source !== sourceVal) return false;
        if (searchVal) {
          const str = JSON.stringify(log).toLowerCase();
          if (!str.includes(searchVal)) return false;
        }
        return true;
      });

      if (filtered.length === 0) {
        tbody.innerHTML = `
          <tr>
            <td colspan="7" class="p-8 text-center text-slate-500">
              No matching request logs found.
            </td>
          </tr>`;
        return;
      }

      let html = '';
      filtered.forEach(log => {
        // Source Badge
        let sourceBadge = '';
        if (log.source === 'ollama_qwen2.5') {
          sourceBadge = `<span class="badge-ollama px-2 py-0.5 rounded text-[10px] font-bold">OLLAMA (qwen2.5)</span>`;
        } else if (log.source === 'fallback') {
          sourceBadge = `<span class="badge-fallback px-2 py-0.5 rounded text-[10px] font-bold">SMART FALLBACK</span>`;
        } else if (log.source === 'tflite_model') {
          sourceBadge = `<span class="badge-tflite px-2 py-0.5 rounded text-[10px] font-bold">TFLite CNN</span>`;
        } else if (log.source === 'pickle_model') {
          sourceBadge = `<span class="badge-pickle px-2 py-0.5 rounded text-[10px] font-bold">RandomForest ML</span>`;
        } else {
          sourceBadge = `<span class="bg-slate-800 text-slate-300 border border-slate-700 px-2 py-0.5 rounded text-[10px] font-bold">${log.source}</span>`;
        }

        // Ollama Attempt Badge
        const o = log.ollama_attempt || {};
        let ollamaBadge = '';
        if (o.status === 'success') {
          ollamaBadge = `<span class="text-emerald-400 font-semibold"><i class="fa-solid fa-check mr-1"></i>Success (${o.duration_sec}s)</span>`;
        } else if (o.status === 'timeout') {
          ollamaBadge = `<span class="text-red-400 font-semibold" title="${o.error || ''}"><i class="fa-solid fa-clock-rotate-left mr-1"></i>Timeout (${o.duration_sec}s)</span>`;
        } else if (o.status === 'error') {
          ollamaBadge = `<span class="text-red-400 font-semibold" title="${o.error || ''}"><i class="fa-solid fa-triangle-exclamation mr-1"></i>Error (${o.error || ''})</span>`;
        } else {
          ollamaBadge = `<span class="text-slate-500">Skipped</span>`;
        }

        const payloadSummary = escapeHtml(JSON.stringify(log.request_body));

        html += `
          <tr class="hover:bg-slate-800/40 transition border-b border-slate-800/60">
            <td class="p-3 whitespace-nowrap text-slate-400">
              <span class="font-bold text-slate-200">#${log.id}</span><br>
              <span class="text-[10px]">${log.timestamp}</span>
            </td>
            <td class="p-3 whitespace-nowrap">
              <span class="font-bold text-emerald-300">${log.endpoint}</span><br>
              <span class="text-[10px] text-slate-500">${log.method} &bull; ${log.client_ip}</span>
            </td>
            <td class="p-3 max-w-[200px]">
              <div class="truncate text-slate-300 font-mono text-[11px]" title='${payloadSummary}'>${payloadSummary}</div>
              <button onclick="viewDetails(${log.id})" class="text-[10px] text-emerald-400 hover:underline mt-0.5">Inspect Full Payload</button>
            </td>
            <td class="p-3 whitespace-nowrap">
              ${ollamaBadge}
            </td>
            <td class="p-3 whitespace-nowrap">
              ${sourceBadge}
            </td>
            <td class="p-3 whitespace-nowrap font-mono text-slate-300">
              ${log.total_duration_ms} ms
            </td>
            <td class="p-3 max-w-[220px]">
              <div class="flex items-center justify-between">
                <span class="px-1.5 py-0.5 rounded text-[10px] font-bold ${log.response_status === 200 ? 'bg-emerald-950 text-emerald-400 border border-emerald-800' : 'bg-red-950 text-red-400 border border-red-800'}">HTTP ${log.response_status}</span>
                <button onclick="viewDetails(${log.id})" class="px-2 py-1 bg-slate-800 hover:bg-slate-700 text-[10px] rounded text-slate-200 border border-slate-700 transition">View JSON</button>
              </div>
            </td>
          </tr>
        `;
      });

      tbody.innerHTML = html;
    }

    function filterLogs() {
      renderLogsTable();
    }

    function viewDetails(logId) {
      const log = rawLogs.find(l => l.id === logId);
      if (!log) return;

      document.getElementById('modal-title').innerHTML = `<i class="fa-solid fa-code text-emerald-400"></i> Request #${log.id} Details (${log.endpoint})`;
      document.getElementById('modal-json').innerText = JSON.stringify(log, null, 2);
      document.getElementById('json-modal').classList.remove('hidden');
    }

    function closeModal() {
      document.getElementById('json-modal').classList.add('hidden');
    }

    function copyModalJson() {
      const text = document.getElementById('modal-json').innerText;
      navigator.clipboard.writeText(text);
      alert("JSON copied to clipboard!");
    }

    async function clearLogs() {
      if (!confirm("Are you sure you want to clear all request logs?")) return;
      await fetch('/api/clear-logs', { method: 'POST' });
      fetchLogs();
    }

    async function sendTestRemedy(diseaseName) {
      await fetch('/disease-remedy', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ diseaseName: diseaseName, cropName: "Potato", language: "English" })
      });
      fetchLogs();
    }

    async function sendTestGrowthPlan(cropName) {
      await fetch('/generate-growth-plan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cropName: cropName, soilType: "Loamy", season: "Kharif" })
      });
      fetchLogs();
    }

    async function sendTestCropRec() {
      await fetch('/recommend-crop', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ soilType: "black", season: "kharif", waterAvailability: "medium", farmSizeAcres: 3.5 })
      });
      fetchLogs();
    }

    function escapeHtml(str) {
      return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
    }

    function setupAutoRefresh() {
      const select = document.getElementById('refresh-interval');
      const val = parseInt(select.value);
      if (timer) clearInterval(timer);

      if (val > 0) {
        timer = setInterval(fetchLogs, val);
      }
    }

    document.getElementById('refresh-interval').addEventListener('change', setupAutoRefresh);

    // Initial Load
    fetchLogs();
    checkTunnelStatus();
    setupAutoRefresh();
  </script>
</body>
</html>"""
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)