"""
models/vessel_feasibility.py

Serves feasibility (fit / no-fit) and operational risk (low/medium/high)
predictions from the trained RandomForest classifiers in
models/trained/vessel_feasibility_model.joblib and
models/trained/vessel_risk_model.joblib (see
training/train_vessel_feasibility.py).

Public entry point keeps the same shape as the original placeholder so
app.py and cargoai-backend/src/services/mlClient.js do not need to change:

    check_feasibility(vessel_type, loa_m, beam_m, draft_m,
                       cargo_weight_tons, port, port_max_loa_m,
                       port_max_beam_m, port_max_draft_m,
                       congestion_index=None) -> dict

If the trained artifacts are missing, falls back to the original
rule-based size-vs-limit checks + fixed congestion lookup.
"""

from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FEASIBILITY_MODEL_PATH = ROOT / "models" / "trained" / "vessel_feasibility_model.joblib"
RISK_MODEL_PATH = ROOT / "models" / "trained" / "vessel_risk_model.joblib"

# fixed congestion lookup, used only as a fallback / default when the
# caller doesn't supply a live congestion_index for the port
_DEFAULT_CONGESTION = {
    "Rotterdam": 0.25, "Singapore": 0.35, "Los Angeles": 0.45,
    "Jebel Ali": 0.30, "Santos": 0.55, "Mundra": 0.20,
    "Hamburg": 0.30, "Colombo": 0.40,
}


@lru_cache(maxsize=1)
def _load_models():
    try:
        import joblib
        return joblib.load(FEASIBILITY_MODEL_PATH), joblib.load(RISK_MODEL_PATH)
    except FileNotFoundError:
        return None, None


def _fallback_rule_based(
    loa_m, beam_m, draft_m, cargo_weight_tons,
    port, port_max_loa_m, port_max_beam_m, port_max_draft_m,
    congestion_index=None,
) -> dict:
    """Original placeholder behaviour, kept as a safety net."""
    fits = (loa_m <= port_max_loa_m) and (beam_m <= port_max_beam_m) and (draft_m <= port_max_draft_m)
    congestion = congestion_index if congestion_index is not None else _DEFAULT_CONGESTION.get(port, 0.35)
    risk_level = "high" if (not fits or congestion > 0.5) else ("medium" if congestion > 0.3 else "low")
    return {
        "feasible": bool(fits),
        "risk_level": risk_level,
        "risk_probability": None,
        "model": "rule_based_fallback",
        "margins_m": {
            "loa": round(port_max_loa_m - loa_m, 2),
            "beam": round(port_max_beam_m - beam_m, 2),
            "draft": round(port_max_draft_m - draft_m, 2),
        },
    }


def check_feasibility(
    vessel_type: str,
    loa_m: float,
    beam_m: float,
    draft_m: float,
    cargo_weight_tons: float,
    port: str,
    port_max_loa_m: float,
    port_max_beam_m: float,
    port_max_draft_m: float,
    congestion_index: float | None = None,
) -> dict:
    """
    Predict whether a vessel is a workable fit for a port, and the
    operational risk level of that fit.

    Returns:
        {
          "feasible": bool,
          "risk_level": "low" | "medium" | "high",
          "risk_probability": {"low": float, "medium": float, "high": float} | None,
          "model": "random_forest_classifier",
          "margins_m": {"loa": float, "beam": float, "draft": float},
        }
    """
    feasibility_clf, risk_clf = _load_models()
    congestion = congestion_index if congestion_index is not None else _DEFAULT_CONGESTION.get(port, 0.35)

    if feasibility_clf is None or risk_clf is None:
        return _fallback_rule_based(
            loa_m, beam_m, draft_m, cargo_weight_tons,
            port, port_max_loa_m, port_max_beam_m, port_max_draft_m,
            congestion_index=congestion,
        )

    import pandas as pd

    row = pd.DataFrame([{
        "vessel_type": vessel_type,
        "port": port,
        "loa_m": loa_m,
        "beam_m": beam_m,
        "draft_m": draft_m,
        "cargo_weight_tons": cargo_weight_tons,
        "port_max_loa_m": port_max_loa_m,
        "port_max_beam_m": port_max_beam_m,
        "port_max_draft_m": port_max_draft_m,
        "congestion_index": congestion,
        "loa_margin": port_max_loa_m - loa_m,
        "beam_margin": port_max_beam_m - beam_m,
        "draft_margin": port_max_draft_m - draft_m,
    }])

    feasible = bool(feasibility_clf.predict(row)[0])
    risk_level = str(risk_clf.predict(row)[0])
    risk_proba = risk_clf.predict_proba(row)[0]
    risk_probability = {cls: round(float(p), 4) for cls, p in zip(risk_clf.classes_, risk_proba)}

    return {
        "feasible": feasible,
        "risk_level": risk_level,
        "risk_probability": risk_probability,
        "model": "random_forest_classifier",
        "margins_m": {
            "loa": round(port_max_loa_m - loa_m, 2),
            "beam": round(port_max_beam_m - beam_m, 2),
            "draft": round(port_max_draft_m - draft_m, 2),
        },
    }


if __name__ == "__main__":
    import pprint
    pprint.pprint(check_feasibility(
        vessel_type="panamax", loa_m=225, beam_m=32.2, draft_m=12.5,
        cargo_weight_tons=68000, port="Rotterdam",
        port_max_loa_m=400, port_max_beam_m=63, port_max_draft_m=24.0,
    ))
