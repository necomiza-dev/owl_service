# main.py
import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# ---------------------------------------------------------------
# Пути к файлам
# ---------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

with open(BASE_DIR / "owl_weights.json", encoding="utf-8") as f:
    CLF = json.load(f)

with open(BASE_DIR / "owl_regression.json", encoding="utf-8") as f:
    REG = json.load(f)

with open(BASE_DIR / "owl_meta.json", encoding="utf-8") as f:
    META = json.load(f)


# ---------------------------------------------------------------
# Функции активации
# ---------------------------------------------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ---------------------------------------------------------------
# Классификатор 4 -> 8 -> 3 (только прямой проход)
# ---------------------------------------------------------------
def classifier_forward(x_norm: np.ndarray) -> np.ndarray:
    hidden = np.array([
        sigmoid(np.dot(x_norm, np.array(n["weights"])) + n["bias"])
        for n in CLF["hidden"]
    ])
    outputs = np.array([
        sigmoid(np.dot(hidden, np.array(n["weights"])) + n["bias"])
        for n in CLF["output"]
    ])
    return outputs


def classify_owl(features: list[float]) -> dict:
    """features: [body_length_cm, wingspan_cm, mass_g, wing_length_cm]"""
    x = np.array(features, dtype=float)
    mn = np.array(CLF["mn"], dtype=float)
    mx = np.array(CLF["mx"], dtype=float)
    x_norm = (x - mn) / (mx - mn)

    probs = classifier_forward(x_norm)
    idx = int(np.argmax(probs))
    return {
        "class_name": CLF["names"][idx],
        "probability": float(probs[idx]),
        "all_probs": {
            CLF["names"][i]: float(probs[i]) for i in range(len(probs))
        },
    }


# ---------------------------------------------------------------
# Регрессия 3 -> 8 -> 1 (линейный выход)
# ---------------------------------------------------------------
def regression_forward(x_norm: np.ndarray) -> float:
    hidden = np.array([
        sigmoid(np.dot(x_norm, np.array(n["weights"])) + n["bias"])
        for n in REG["hidden"]
    ])
    return float(np.dot(hidden, np.array(REG["out_weights"])) + REG["out_bias"])


def predict_mass(features: list[float]) -> float:
    """features: [body_length_cm, wingspan_cm, wing_length_cm]"""
    x = np.array(features, dtype=float)
    x_mn = np.array(REG["x_mn"], dtype=float)
    x_mx = np.array(REG["x_mx"], dtype=float)
    x_norm = (x - x_mn) / (x_mx - x_mn)

    y_norm = regression_forward(x_norm)
    y_real = y_norm * (REG["y_mx"] - REG["y_mn"]) + REG["y_mn"]
    return float(y_real)


# ---------------------------------------------------------------
# FastAPI + шаблоны
# ---------------------------------------------------------------
app = FastAPI(title="Owl Service")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# ---------- Классификация ----------
@app.get("/", response_class=HTMLResponse)
def index_get(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "meta": META["classifier"],
            "result": None,
            "form_values": {},
        },
    )


@app.post("/", response_class=HTMLResponse)
def index_post(
    request: Request,
    body_length_cm: float = Form(...),
    wingspan_cm: float = Form(...),
    mass_g: float = Form(...),
    wing_length_cm: float = Form(...),
):
    features = [body_length_cm, wingspan_cm, mass_g, wing_length_cm]
    result = classify_owl(features)

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "meta": META["classifier"],
            "result": result,
            "form_values": {
                "body_length_cm": body_length_cm,
                "wingspan_cm": wingspan_cm,
                "mass_g": mass_g,
                "wing_length_cm": wing_length_cm,
            },
        },
    )


# ---------- Регрессия ----------
@app.get("/regression", response_class=HTMLResponse)
def regression_get(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="regression.html",
        context={
            "meta": META["regression"],
            "result": None,
            "form_values": {},
        },
    )


@app.post("/regression", response_class=HTMLResponse)
def regression_post(
    request: Request,
    body_length_cm: float = Form(...),
    wingspan_cm: float = Form(...),
    wing_length_cm: float = Form(...),
):
    features = [body_length_cm, wingspan_cm, wing_length_cm]
    mass = predict_mass(features)

    return templates.TemplateResponse(
        request=request,
        name="regression.html",
        context={
            "meta": META["regression"],
            "result": {
                "value": mass,
                "unit": META["regression"]["unit"],
                "target_name": META["regression"]["target_name_ru"],
            },
            "form_values": {
                "body_length_cm": body_length_cm,
                "wingspan_cm": wingspan_cm,
                "wing_length_cm": wing_length_cm,
            },
       },
    )