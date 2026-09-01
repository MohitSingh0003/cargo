import os
import joblib
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "artifacts",
    "feasibility_model.joblib"
)


def load_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Feasibility model not found. "
            "Run training/train_feasibility.py first."
        )

    return joblib.load(MODEL_PATH)


def assess_feasibility(
    cargo_type: str,
    quantity_mt: float,
    vessel_type: str,
    origin: str,
    destination: str
):

    model_data = load_model()

    model = model_data["model"]
    encoder = model_data["encoder"]

    # Demo/default operational values.
    vessel_capacity = {
        "Handysize": 25000,
        "Panamax": 80000,
        "Capesize": 180000
    }.get(vessel_type, 80000)

    port_capacity = 200000

    port_draft_limit = 18

    vessel_draft = {
        "Handysize": 10,
        "Panamax": 13,
        "Capesize": 17
    }.get(vessel_type, 13)

    congestion = 0.30

    input_df = pd.DataFrame([{
        "cargo_type": cargo_type,
        "quantity_mt": quantity_mt,
        "vessel_type": vessel_type,
        "vessel_capacity_mt": vessel_capacity,
        "port_capacity_mt": port_capacity,
        "draft_m": vessel_draft,
        "port_draft_limit_m": port_draft_limit,
        "congestion": congestion
    }])

    categorical_columns = [
        "cargo_type",
        "vessel_type"
    ]

    encoded = encoder.transform(
        input_df[categorical_columns]
    )

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(
            categorical_columns
        )
    )

    numerical_columns = [
        "quantity_mt",
        "vessel_capacity_mt",
        "port_capacity_mt",
        "draft_m",
        "port_draft_limit_m",
        "congestion"
    ]

    X = pd.concat(
        [
            input_df[numerical_columns].reset_index(drop=True),
            encoded_df.reset_index(drop=True)
        ],
        axis=1
    )

    prediction = int(
        model.predict(X)[0]
    )

    probability = float(
        model.predict_proba(X)[0][1]
    )

    reasons = []

    if quantity_mt > vessel_capacity:
        reasons.append(
            "Cargo quantity exceeds vessel capacity."
        )

    if quantity_mt > port_capacity:
        reasons.append(
            "Cargo quantity exceeds port capacity."
        )

    if vessel_draft > port_draft_limit:
        reasons.append(
            "Vessel draft exceeds port draft limit."
        )

    if not reasons:
        reasons.append(
            "Cargo and vessel parameters are within expected limits."
        )

    return {
        "cargoType": cargo_type,
        "vesselType": vessel_type,
        "origin": origin,
        "destination": destination,
        "feasible": bool(prediction),
        "probability": round(probability, 3),
        "risk": (
            "LOW"
            if probability >= 0.75
            else "MEDIUM"
            if probability >= 0.50
            else "HIGH"
        ),
        "reasons": reasons
    }
