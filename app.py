import os
import re
import numpy as np
import pandas as pd
import joblib

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

from PIL import Image, UnidentifiedImageError
import pytesseract
import pdfplumber
import docx
from bs4 import BeautifulSoup
from striprtf.striprtf import rtf_to_text


# Auto-detect Tesseract on Windows
if os.name == "nt":
    for path in [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            break


# ==================================================
# APP CONFIG
# ==================================================
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MODEL_PATH = os.path.join(BASE_DIR, "models", "fraud_detection_model.pkl")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load ML model
MODEL = joblib.load(MODEL_PATH)
scaler = MODEL["scaler"]
iso = MODEL["isolation_forest"]
corrector = MODEL["corrector"]


# ==================================================
# HELPERS
# ==================================================
def extract_numbers_from_text(text):
    return [float(n) for n in re.findall(r"[-+]?\d*\.\d+|\d+", text)]


def parse_document(filepath, ext):
    raw = ""
    values = []

    try:
        # TXT
        if ext == "txt":
            raw = open(filepath, "r", encoding="utf-8", errors="ignore").read()
            values = extract_numbers_from_text(raw)

        # CSV
        elif ext == "csv":
            try:
                df = pd.read_csv(filepath)
                raw = df.to_string()
                values = df.select_dtypes(include="number").values.flatten().tolist()
            except:
                raw = open(filepath, "r", encoding="utf-8", errors="ignore").read()
                values = extract_numbers_from_text(raw)

        # XLS / XLSX
        elif ext in ["xls", "xlsx"]:
            df = pd.read_excel(filepath)
            raw = df.to_string()
            values = df.select_dtypes(include="number").values.flatten().tolist()

        # PDF
        elif ext == "pdf":
            text = ""
            with pdfplumber.open(filepath) as pdf:
                for page in pdf.pages:
                    text += (page.extract_text() or "") + "\n"
            raw = text
            values = extract_numbers_from_text(text)

        # DOCX
        elif ext == "docx":
            docx_obj = docx.Document(filepath)
            raw = "\n".join([p.text for p in docx_obj.paragraphs])
            values = extract_numbers_from_text(raw)

        # DOC
        elif ext == "doc":
            try:
                import mammoth

                with open(filepath, "rb") as doc_file:
                    result = mammoth.extract_raw_text(doc_file)
                    raw = result.value  # Extracted text
                    values = extract_numbers_from_text(raw)

            except Exception as e:
                print("Mammoth DOC error:", e)

                # Fallback attempt: decode bytes
                try:
                    raw_bytes = open(filepath, "rb").read()
                    decoded = raw_bytes.decode("latin-1", errors="ignore")
                    cleaned = re.sub(r"[\x00-\x1F\x7F-\x9F]", " ", decoded)
                    raw = cleaned
                    values = extract_numbers_from_text(raw)
                except Exception:
                    # OCR fallback
                    try:
                        img = Image.open(filepath)
                        raw = pytesseract.image_to_string(img)
                        values = extract_numbers_from_text(raw)
                    except:
                        raw = ""
                        values = []


        # RTF
        elif ext == "rtf":
            raw = rtf_to_text(open(filepath, encoding="utf-8", errors="ignore").read())
            values = extract_numbers_from_text(raw)

        # HTML
        elif ext in ["html", "htm"]:
            soup = BeautifulSoup(open(filepath, encoding="utf-8", errors="ignore").read(), "lxml")
            raw = soup.get_text(" ")
            values = extract_numbers_from_text(raw)

        # JSON / XML
        elif ext in ["json", "xml"]:
            raw = open(filepath, encoding="utf-8", errors="ignore").read()
            values = extract_numbers_from_text(raw)

        # IMAGES (ALL FORMATS)
        elif ext in ["png", "jpg", "jpeg", "bmp", "tiff", "tif", "gif", "webp", "heic", "heif"]:
            try:
                img = Image.open(filepath)
                if ext in ["heic", "heif"] and img.mode == "RGBA":
                    img = img.convert("RGB")

                raw = pytesseract.image_to_string(img, config="--psm 6")
                values = extract_numbers_from_text(raw)

            except UnidentifiedImageError:
                raw = ""
                values = []

        # SVG fallback
        elif ext == "svg":
            raw = open(filepath).read()
            values = extract_numbers_from_text(raw)

        # DEFAULT fallback
        else:
            raw = open(filepath, encoding="utf-8", errors="ignore").read()
            values = extract_numbers_from_text(raw)

    except Exception as e:
        print("Parse Error:", e)

    return values, raw


# ==================================================
# FEATURE EXTRACTION
# ==================================================
def extract_features(values):
    if not values:
        return dict(count=0, mean=0, std=0, min=0, max=0, sum=0)

    arr = np.array(values)
    return dict(
        count=len(arr),
        mean=float(arr.mean()),
        std=float(arr.std()),
        min=float(arr.min()),
        max=float(arr.max()),
        sum=float(arr.sum()),
    )


# ==================================================
# EXPENSE EXTRACTION
# ==================================================
def extract_expense_categories(raw_text):
    expenses = {}
    pattern = re.compile(r"([A-Za-z\s\-]+?)[:\.\- ]+(\d[\d,\.]+)")
    matches = pattern.findall(raw_text)

    FINANCE_KEYS = [
        "revenue", "income", "sales", "profit", "operating expense",
        "administrative", "cogs", "marketing", "travel", "utilities",
        "rent", "salary", "tax", "insurance", "maintenance", "misc"
    ]

    for label, value in matches:
        lower = label.lower()
        for key in FINANCE_KEYS:
            if key in lower:
                num = float(value.replace(",", ""))
                expenses[key.title()] = expenses.get(key.title(), 0) + num

    return expenses


# ==================================================
# TREND + ANOMALY
# ==================================================
def generate_anomaly_scores(values):
    if not values:
        return []

    arr = np.array(values, dtype=float)
    min_val = arr.min()
    max_val = arr.max()

    norm = (arr - min_val) / (max_val - min_val + 1e-9)

    scores = []
    for i, (score, raw) in enumerate(zip(norm, arr)):
        scores.append({
            "name": f"Point {i+1}",
            "score": float(score),
            "rawValue": float(raw)
        })

    return scores

def generate_trends(values):
    """
    Create real dynamic financial trends based on extracted numbers.
    Works for ANY dataset size.
    """

    values = [float(v) for v in values if isinstance(v, (int, float, np.number))]

    if not values:
        return []

    # CASE 1: very small dataset (1 or 2 numbers)
    if len(values) < 3:
        expanded = (values * 3)[:3]  # repeat to 3 values
        return [
            {"name": f"Point {i+1}", "value": float(v)} 
            for i, v in enumerate(expanded)
        ]

    # CASE 2: moderate dataset (3–10 points) → direct mapping
    if len(values) <= 10:
        return [
            {"name": f"Point {i+1}", "value": float(v)}
            for i, v in enumerate(values)
        ]

    # CASE 3: large dataset → smooth into 5 segments
    segment_count = 5
    chunk_size = len(values) // segment_count
    chunks = [values[i*chunk_size:(i+1)*chunk_size] for i in range(segment_count)]

    return [
        {"name": f"Segment {i+1}", "value": float(np.mean(chunk))}
        for i, chunk in enumerate(chunks)
    ]

    

def extract_key_anomaly_drivers(anomaly_scores, threshold=0.7):
    drivers = []

    for item in anomaly_scores:
        if item["score"] >= threshold:
            drivers.append(
                f"{item['name']} exceeded threshold ({threshold*100}%) with score {item['score']:.2f}"
            )

    if not drivers:
        drivers.append("No significant anomalies detected based on threshold.")

    return drivers

# ==================================================
# API: Analyze Document
# ==================================================
@app.route("/analyze-document", methods=["POST"])
def analyze_document():
    filepath = None

    try:
        file = request.files.get("document")
        if not file:
            return jsonify({"status": "failure", "message": "No file uploaded"})

        filename = secure_filename(file.filename)
        ext = filename.split(".")[-1].lower()

        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # ---------------------------------------------------
        # PARSE DOCUMENT
        # ---------------------------------------------------
        values, raw_text = parse_document(filepath, ext)

        # Always convert values to float safely
        values = [float(v) for v in values if str(v).replace('.', '', 1).isdigit()]

        # ---------------------------------------------------
        # FEATURE EXTRACTION
        # ---------------------------------------------------
        features = extract_features(values)
        df = pd.DataFrame([features])

        # ---------------------------------------------------
        # FRAUD MODEL PREDICTION
        # ---------------------------------------------------
        X_scaled = scaler.transform(df.values)
        iso_pred = (iso.predict(X_scaled) == -1).astype(int).reshape(-1, 1)
        combined = np.hstack([X_scaled, iso_pred])

        fraud_flag = int(corrector.predict(combined)[0])
        fraud_prob = float(corrector.predict_proba(combined)[0][1] * 100)

        # ---------------------------------------------------
        # TREND GENERATION
        # ---------------------------------------------------
        numeric_trends = generate_trends(values)

        # ---------------------------------------------------
        # ANOMALY SCORE GENERATION
        # ---------------------------------------------------
        anomaly_scores = generate_anomaly_scores(values)

        # ---------------------------------------------------
        # KEY ANOMALY DRIVERS
        # ---------------------------------------------------
        key_drivers = extract_key_anomaly_drivers(anomaly_scores, threshold=0.7)

        # ---------------------------------------------------
        # EXPENSE CATEGORY EXTRACTION
        # ---------------------------------------------------
        expenses_map = extract_expense_categories(raw_text)
        expense_arr = [{"name": k, "value": float(v)} for k, v in expenses_map.items()]

        # ---------------------------------------------------
        # FINAL RESPONSE JSON
        # ---------------------------------------------------
        return jsonify({
            "status": "success",
            "fraud_detected": bool(fraud_flag),
            "fraud_probability": fraud_prob,
            "engineeredFeatures": features,

            "trendType": "numeric",
            "numericTrends": numeric_trends,

            "anomalyScoresOverTime": anomaly_scores,
            "keyAnomalyDrivers": key_drivers,

            "expenses": expense_arr,
            "textPreview": raw_text[:600]
        })

    except Exception as e:
        print("Analyze Document Error:", e)
        return jsonify({"status": "error", "message": str(e)})

    finally:
        # Cleanup uploaded file
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except PermissionError:
                pass


# ==================================================
# API: AI Explanation of Trend
# ==================================================
@app.route("/explain-visual", methods=["POST"])
def explain_visual():
    try:
        data = request.json
        trends = data.get("numericTrends", [])

        # No trend data
        if not trends or len(trends) == 0:
            return jsonify({
                "status": "success",
                "explanation": "No trend data available to analyze."
            })

        # Extract values
        values = [float(t.get("value", 0)) for t in trends]
        names = [t.get("name", "") for t in trends]

        start = values[0]
        end = values[-1]
        change = end - start

        # --------------------
        # Determine trend direction
        # --------------------
        if change > 0:
            direction = "📈 Increasing Trend"
        elif change < 0:
            direction = "📉 Decreasing Trend"
        else:
            direction = "➖ Stable / No Movement"

        # --------------------
        # Volatility Measurement
        # --------------------
        diffs = [abs(values[i] - values[i - 1]) for i in range(1, len(values))]
        volatility = sum(diffs) / len(diffs)

        if volatility > (0.3 * max(values)):
            volatility_label = "⚠ High volatility detected"
        elif volatility > (0.15 * max(values)):
            volatility_label = "⚠ Moderate volatility"
        else:
            volatility_label = "✔ Stable movement"

        # --------------------
        # Build explanation
        # --------------------
        explanation = f"""
📊 **Financial Trend Analysis**

**Start Value:** {start:,.2f}  
**End Value:** {end:,.2f}  
**Net Change:** {change:,.2f}

### Trend Direction
{direction}

### Volatility
{volatility_label}

---

### Detailed Observations
"""

        # Add each segment’s value
        for name, val in zip(names, values):
            explanation += f"- **{name}** → {val:,.2f}\n"

        return jsonify({"status": "success", "explanation": explanation})

    except Exception as e:
        print("Explain Visual Error:", e)
        return jsonify({"status": "error", "message": str(e)})



@app.route("/explain-balance-sheet", methods=["POST"])
def explain_balance_sheet():
    try:
        data = request.json

        totalAssets = float(data.get("totalAssets", 0))
        totalLiabilities = float(data.get("totalLiabilities", 0))
        totalEquity = float(data.get("totalEquity", totalAssets - totalLiabilities))

        explanation = f"""
📊 **Balance Sheet Analysis**

**Total Assets:** {totalAssets:,.2f}  
**Total Liabilities:** {totalLiabilities:,.2f}  
**Equity:** {totalEquity:,.2f}

---

### 📘 Formula  
**Equity = Assets - Liabilities**

---

"""

        # Additional reasoning based on values
        if totalEquity > 0:
            explanation += (
                "✔ The company is in a strong financial position with **positive equity**, "
                "indicating assets exceed liabilities."
            )
        elif totalEquity == 0:
            explanation += (
                "⚠ The company has **neutral equity** — assets equal liabilities. "
                "This indicates no retained value."
            )
        else:
            explanation += (
                "🚨 The company is in a **negative equity** state. Liabilities exceed assets, "
                "which is a red flag and may indicate insolvency risk."
            )

        return jsonify({"status": "success", "explanation": explanation})

    except Exception as e:
        print("Balance Sheet Explanation Error:", e)
        return jsonify({"status": "error", "message": str(e)})





from textwrap import dedent

@app.route("/explain-fraud", methods=["POST"])
def explain_fraud():
    try:
        data = request.json

        fraud_prob = float(data.get("fraud_probability", 0))
        features = data.get("engineeredFeatures", {})
        anomaly_scores = data.get("anomalyScoresOverTime", [])
        key_drivers = data.get("keyAnomalyDrivers", [])
        trends = data.get("numericTrends", [])
        expenses = data.get("expenses", [])
        text_preview = data.get("textPreview", "")[:600]

        # Fraud risk level text
        if fraud_prob > 80:
            risk_label = "🚨 CRITICAL — Extremely High Fraud Probability"
        elif fraud_prob > 50:
            risk_label = "⚠ MODERATE — Suspicious Activity Detected"
        else:
            risk_label = "✅ LOW — Minimal Fraud Indicators"

        # Numeric feature extraction
        f_count = features.get("count", 0)
        f_mean = features.get("mean", 0)
        f_std = features.get("std", 0)
        f_min = features.get("min", 0)
        f_max = features.get("max", 0)
        f_sum = features.get("sum", 0)

        # Anomaly summary
        anomaly_points = [a for a in anomaly_scores if a.get("score", 0) >= 0.7]
        anomaly_summary = "\n".join(
            [f"- {a['name']} scored {a['score']:.2f}" for a in anomaly_points]
        ) or "No major anomaly spikes detected."

        # Trend summary
        trend_summary = "\n".join(
            [f"- {t['name']}: {t['value']:.2f}" for t in trends]
        ) if trends else "No trend data found."

        # Expense summary
        expense_summary = "\n".join(
            [f"- {e['name']}: {e['value']:,}" for e in expenses]
        ) if expenses else "No recognizable expense categories extracted."

        # Build final explanation safely
        explanation = dedent("""
        📌 **AI Fraud Probability Report**

        Fraud Probability Score: **{fraud_prob:.2f}%**  
        Risk Evaluation: **{risk_label}**

        ---

        ## 📊 Statistical Feature Summary
        • Total Numerical Entries: **{f_count}**  
        • Mean Value: **{f_mean:.2f}**  
        • Standard Deviation: **{f_std:.2f}**  
        • Minimum Value: **{f_min:.2f}**  
        • Maximum Value: **{f_max:.2f}**  
        • Sum of Values: **{f_sum:.2f}**

        ---

        ## 🚨 Anomaly Detection Overview
        Threshold for anomalies: **70%**

        Detected anomaly points:
        {anomaly_summary}

        Key Drivers:
        {key_drivers}

        ---

        ## 📈 Trend Summary
        {trend_summary}

        ---

        ## 🧾 Expense Breakdown
        {expense_summary}

        ---

        ## 📄 Extracted Text Preview
        {text_preview}

        ---

        ### 🧠 Final Interpretation
        The fraud model evaluates anomaly scoring, statistical variance,
        trend fluctuations, and financial category behavior to determine
        possible manipulation or suspicious patterns.

        """).format(
            fraud_prob=fraud_prob,
            risk_label=risk_label,
            f_count=f_count,
            f_mean=f_mean,
            f_std=f_std,
            f_min=f_min,
            f_max=f_max,
            f_sum=f_sum,
            anomaly_summary=anomaly_summary,
            key_drivers="\n".join(f"- {d}" for d in key_drivers),
            trend_summary=trend_summary,
            expense_summary=expense_summary,
            text_preview=text_preview
        )

        return jsonify({"status": "success", "explanation": explanation})

    except Exception as e:
        print("Explain Fraud Error:", e)
        return jsonify({"status": "error", "message": str(e)})

# ==================================================
# RUN SERVER
# ==================================================
if __name__ == "__main__":
    app.run(debug=True, port=5000, host="0.0.0.0")


