"""
Archivo: app.py
Propósito: Microservicio API REST en FastAPI que sirve el modelo dual:
           1. Clasificación Supervisada con KNN (k=7 vecinos)
           2. Perfilamiento No Supervisado con K-Means (k=3 clústeres)
Ejecución local: uvicorn app:app --reload
Ejecución Render: uvicorn app:app --host 0.0.0.0 --port $PORT
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

# 1. Inicialización de la aplicación FastAPI
app = FastAPI(
    title="API Dual de Telemedicina — Clasificación (KNN) + Clustering (K-Means)",
    description=(
        "Microservicio productivo de soporte a la decisión clínica. "
        "Evalúa el riesgo agudo de descompensación mediante KNN (k=7) e identifica "
        "el fenotipo fisiológico del paciente mediante K-Means (k=3)."
    ),
    version="2.0.0"
)

RUTA_ARTEFACTO = "artefactos/modelos_salud_dual.joblib"
FEATURES = [
    "edad",
    "frecuencia_cardiaca",
    "spo2",
    "presion_sistolica",
    "glucosa_mg_dl",
    "pasos_diarios"
]


# 2. Función de respaldo automático (si suben a Render sin la carpeta artefactos/)
def construir_artefacto_respaldo(ruta_destino: str) -> dict:
    """Entrena y serializa el paquete KNN + K-Means en frío si no existe en disco."""
    os.makedirs(os.path.dirname(ruta_destino), exist_ok=True)
    rng = np.random.default_rng(42)
    n_samples = 3000
    n_g1, n_g2 = int(n_samples * 0.45), int(n_samples * 0.35)
    n_g3 = n_samples - n_g1 - n_g2

    g1 = pd.DataFrame({
        "edad": rng.integers(25, 50, size=n_g1),
        "frecuencia_cardiaca": rng.normal(68, 7, size=n_g1).clip(50, 95),
        "spo2": rng.normal(98.0, 0.9, size=n_g1).clip(95, 100),
        "presion_sistolica": rng.normal(115, 9, size=n_g1).clip(90, 135),
        "glucosa_mg_dl": rng.normal(90, 10, size=n_g1).clip(70, 115),
        "pasos_diarios": rng.normal(8800, 1600, size=n_g1).clip(4500, 16000)
    })
    g2 = pd.DataFrame({
        "edad": rng.integers(45, 68, size=n_g2),
        "frecuencia_cardiaca": rng.normal(82, 9, size=n_g2).clip(62, 115),
        "spo2": rng.normal(95.2, 1.2, size=n_g2).clip(91, 98),
        "presion_sistolica": rng.normal(138, 11, size=n_g2).clip(115, 168),
        "glucosa_mg_dl": rng.normal(142, 22, size=n_g2).clip(105, 220),
        "pasos_diarios": rng.normal(4100, 1100, size=n_g2).clip(1200, 7000)
    })
    g3 = pd.DataFrame({
        "edad": rng.integers(62, 86, size=n_g3),
        "frecuencia_cardiaca": rng.normal(108, 12, size=n_g3).clip(85, 165),
        "spo2": rng.normal(89.5, 2.2, size=n_g3).clip(80, 94),
        "presion_sistolica": rng.normal(162, 14, size=n_g3).clip(130, 210),
        "glucosa_mg_dl": rng.normal(175, 30, size=n_g3).clip(110, 310),
        "pasos_diarios": rng.normal(1400, 600, size=n_g3).clip(150, 3200)
    })

    df = pd.concat([g1, g2, g3], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
    logit = (
        -10.8 + 0.045 * df["edad"] + 0.060 * df["frecuencia_cardiaca"]
        + 0.480 * (95.0 - df["spo2"]) + 0.030 * df["presion_sistolica"]
        + 0.018 * df["glucosa_mg_dl"] - 0.0003 * df["pasos_diarios"]
        + rng.normal(0, 0.65, size=len(df))
    )
    y = ((1 / (1 + np.exp(-logit))) >= 0.5).astype(int)
    X = df[FEATURES]

    pipe_knn = Pipeline([("escalador", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=7, weights="distance"))])
    pipe_knn.fit(X, y)

    pipe_kmeans = Pipeline([("escalador", StandardScaler()), ("kmeans", KMeans(n_clusters=3, random_state=42, n_init=10))])
    pipe_kmeans.fit(X)

    centroides = pd.DataFrame(
        pipe_kmeans.named_steps["escalador"].inverse_transform(pipe_kmeans.named_steps["kmeans"].cluster_centers_),
        columns=FEATURES
    )
    orden = centroides["frecuencia_cardiaca"].argsort().values
    mapa_clusters = {
        int(orden[0]): "Cluster A — Fenotipo Activo y Saludable",
        int(orden[1]): "Cluster B — Fenotipo Sedentario / Vigilancia Metabólica",
        int(orden[2]): "Cluster C — Fenotipo Crítico Cardiorrespiratorio"
    }

    paquete = {
        "pipeline_knn": pipe_knn,
        "pipeline_kmeans": pipe_kmeans,
        "mapa_clusters": mapa_clusters,
        "features": FEATURES
    }
    joblib.dump(paquete, ruta_destino)
    return paquete


# 3. Carga de modelos en memoria al iniciar el servidor
try:
    if os.path.exists(RUTA_ARTEFACTO):
        artefacto = joblib.load(RUTA_ARTEFACTO)
        print(f"[OK] Artefacto dual cargado desde '{RUTA_ARTEFACTO}'.")
    else:
        print(f"[AVISO] No se encontró '{RUTA_ARTEFACTO}'. Generando artefacto de respaldo...")
        artefacto = construir_artefacto_respaldo(RUTA_ARTEFACTO)
        print("[OK] Artefacto de respaldo generado y cargado exitosamente.")

    modelo_knn = artefacto["pipeline_knn"]
    modelo_kmeans = artefacto["pipeline_kmeans"]
    mapa_clusters = {int(k): v for k, v in artefacto["mapa_clusters"].items()}
except Exception as e:
    modelo_knn, modelo_kmeans, mapa_clusters = None, None, {}
    print(f"[ERROR CRÍTICO] Falló la carga de modelos: {e}")


# 4. Contrato de Entrada con Pydantic (El Guardia Geométrico para KNN y K-Means)
class SignosVitales(BaseModel):
    edad: int = Field(..., ge=18, le=110, description="Edad del paciente en años (18 a 110)", examples=[74])
    frecuencia_cardiaca: float = Field(..., ge=35.0, le=220.0, description="Latidos por minuto (35 a 220)", examples=[114.0])
    spo2: float = Field(..., ge=60.0, le=100.0, description="Saturación de oxígeno SpO2 % (60 a 100)", examples=[88.5])
    presion_sistolica: float = Field(..., ge=60.0, le=260.0, description="Presión arterial sistólica mmHg (60 a 260)", examples=[168.0])
    glucosa_mg_dl: float = Field(..., ge=40.0, le=600.0, description="Glucosa capilar mg/dL (40 a 600)", examples=[188.0])
    pasos_diarios: int = Field(..., ge=0, le=50000, description="Pasos diarios del wearable (0 a 50,000)", examples=[1100])


# 5. Contrato de Salida Dual (KNN + K-Means)
class RespuestaDual(BaseModel):
    riesgo_descompensacion_knn: int
    probabilidad_riesgo_knn: float
    nivel_alerta_knn: str
    cluster_id_kmeans: int
    fenotipo_kmeans: str
    distancia_centroide_kmeans: float


# 6. Función auxiliar de inferencia individual y por lotes
def ejecutar_inferencia_df(df_entrada: pd.DataFrame) -> List[Dict]:
    df_ordenado = df_entrada[FEATURES]

    # Inferencia Supervisada (KNN)
    preds_knn = modelo_knn.predict(df_ordenado)
    probs_knn = modelo_knn.predict_proba(df_ordenado)[:, 1]

    # Inferencia No Supervisada (K-Means)
    clusters_kmeans = modelo_kmeans.predict(df_ordenado)
    distancias_matriz = modelo_kmeans.transform(df_ordenado)

    resultados = []
    for i in range(len(df_ordenado)):
        pred_k = int(preds_knn[i])
        prob_k = float(probs_knn[i])
        c_id = int(clusters_kmeans[i])
        dist_c = float(distancias_matriz[i, c_id])

        resultados.append({
            "riesgo_descompensacion_knn": pred_k,
            "probabilidad_riesgo_knn": round(prob_k, 4),
            "nivel_alerta_knn": "ALERTA ROJA — ALTO RIESGO" if pred_k == 1 else "VERDE — PACIENTE ESTABLE",
            "cluster_id_kmeans": c_id,
            "fenotipo_kmeans": mapa_clusters.get(c_id, f"Cluster {c_id}"),
            "distancia_centroide_kmeans": round(dist_c, 3)
        })
    return resultados


# 7. Endpoints de la API REST
@app.get("/")
def inicio():
    return {
        "servicio": "API Dual Telemedicina MLOps (KNN + K-Means)",
        "estado": "activo",
        "modelos_activos": ["KNeighborsClassifier (k=7)", "KMeans (k=3)"],
        "documentacion_interactiva": "Agrega /docs a esta URL para abrir Swagger UI"
    }


@app.get("/health")
def health_check():
    if modelo_knn is None or modelo_kmeans is None:
        raise HTTPException(status_code=503, detail="Los modelos KNN/K-Means no están cargados en memoria")
    return {
        "status": "healthy",
        "knn_cargado": True,
        "kmeans_cargado": True,
        "clusters_configurados": mapa_clusters
    }


@app.post("/predict", response_model=RespuestaDual)
def predecir_paciente_tiempo_real(paciente: SignosVitales):
    """Evalúa un paciente individual en milisegundos usando KNN (riesgo) y K-Means (fenotipo)."""
    if modelo_knn is None or modelo_kmeans is None:
        raise HTTPException(status_code=503, detail="Modelos no disponibles")

    df_paciente = pd.DataFrame([paciente.model_dump()])
    resultado = ejecutar_inferencia_df(df_paciente)[0]
    return RespuestaDual(**resultado)


@app.post("/predict_batch")
def predecir_lote_pacientes(pacientes: List[SignosVitales]):
    """Evalúa un lote masivo de pacientes en una sola petición HTTP (Inferencia Batch)."""
    if modelo_knn is None or modelo_kmeans is None:
        raise HTTPException(status_code=503, detail="Modelos no disponibles")
    if not pacientes:
        raise HTTPException(status_code=400, detail="La lista de pacientes no puede estar vacía")

    df_lote = pd.DataFrame([p.model_dump() for p in pacientes])
    resultados = ejecutar_inferencia_df(df_lote)

    total_alertas = sum(1 for r in resultados if r["riesgo_descompensacion_knn"] == 1)
    return {
        "total_procesados": len(resultados),
        "total_alertas_rojas_knn": total_alertas,
        "predicciones": resultados
    }