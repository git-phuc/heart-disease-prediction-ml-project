from pathlib import Path
import tempfile
import socket
import joblib
import numpy as np
import pandas as pd
import gradio as gr

# =========================================================
# PATHS (app.py nằm trong Inference/)
# =========================================================
ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "saved_models"
DATA_DIR  = ROOT_DIR / "Data"

MODEL_PATHS = {
    "RF (tuned)": MODEL_DIR / "CHD_RF_tuned.joblib",
    "ET (tuned)": MODEL_DIR / "CHD_ET_tuned.joblib",
}

# =========================================================
# LOAD MODELS
# =========================================================
def load_bundles():
    bundles = {}
    for k, p in MODEL_PATHS.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing model file: {p}")
        bundles[k] = joblib.load(p)
    return bundles

BUNDLES = load_bundles()

FEATURES = BUNDLES["RF (tuned)"].get("feature_names", None)
if not FEATURES:
    raise ValueError("❌ Model bundle thiếu 'feature_names'. Hãy re-save model có kèm feature_names.")

# =========================================================
# LABELS (VI) - hiển thị cho người dùng
# =========================================================
FEATURE_LABELS_VI = {
    "age": "Tuổi bệnh nhân",
    "male": "Giới tính",
    "education": "Trình độ học vấn",
    "currentsmoker": "Hiện đang hút thuốc",
    "cigsperday": "Số điếu thuốc mỗi ngày",
    "bpmeds": "Đang dùng thuốc huyết áp",
    "prevalentstroke": "Tiền sử đột quỵ",
    "prevalenthyp": "Tiền sử tăng huyết áp",
    "diabetes": "Tiểu đường",
    "totchol": "Cholesterol toàn phần",
    "sysbp": "Huyết áp tâm thu (SysBP)",
    "diabp": "Huyết áp tâm trương (DiaBP)",
    "bmi": "Chỉ số BMI",
    "heartrate": "Nhịp tim",
    "glucose": "Đường huyết (Glucose)",
}

def norm_key(col: str) -> str:
    return str(col).strip().lower()

def ui_label(col: str) -> str:
    key = norm_key(col)
    base = FEATURE_LABELS_VI.get(key, col)
    return f"{base} [{col}]"

# =========================================================
# EDUCATION NOTE (1/2/3/4)
# (Bạn có thể chỉnh lại mô tả theo dataset của bạn)
# =========================================================
EDU_LABEL_TO_VAL = {
    "1 - Tiểu học": 1,
    "2 - THCS": 2,
    "3 - THPT": 3,
    "4 - Cao đẳng/Đại học+": 4,
}
EDU_VAL_TO_LABEL = {v: k for k, v in EDU_LABEL_TO_VAL.items()}

def parse_education(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return v
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, float):
        return int(v)
    if isinstance(v, str):
        s = v.strip()
        # nếu user chọn label "1 - ..."
        if s in EDU_LABEL_TO_VAL:
            return EDU_LABEL_TO_VAL[s]
        # nếu user gõ "1"...
        try:
            return int(float(s))
        except Exception:
            return v
    return v

def parse_gender_to_male(v):
    """
    Trả về 1 nếu Nam/Male, 0 nếu Nữ/Female.
    """
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return v
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, float):
        return int(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ["nam", "male", "m", "man", "1"]:
            return 1
        if s in ["nữ", "nu", "female", "f", "woman", "0"]:
            return 0
    return v

BINARY_TEXT_MAP = {
    "1": 1, "0": 0,
    "yes": 1, "no": 0,
    "true": 1, "false": 0,
    "có": 1, "co": 1,
    "không": 0, "khong": 0,
}

# =========================================================
# VALIDATION RULES (giới hạn nhập)
# - Single: strict (thiếu là lỗi)
# - Batch: lenient (thiếu -> NaN vẫn predict)
# =========================================================
VALIDATION_RULES = {
    "age":        {"type": "int",   "min": 15,  "max": 100, "step": 1},
    "education":  {"type": "int",   "choices": [1, 2, 3, 4]},
    "cigsperday": {"type": "float", "min": 0,   "max": 80,  "step": 1},
    "totchol":    {"type": "float", "min": 80,  "max": 500, "step": 1},
    "sysbp":      {"type": "float", "min": 70,  "max": 260, "step": 1},
    "diabp":      {"type": "float", "min": 40,  "max": 160, "step": 1},
    "bmi":        {"type": "float", "min": 10,  "max": 70,  "step": 0.1},
    "heartrate":  {"type": "float", "min": 35,  "max": 220, "step": 1},
    "glucose":    {"type": "float", "min": 40,  "max": 500, "step": 1},

    # binary flags
    "male":            {"type": "binary"},
    "currentsmoker":   {"type": "binary"},
    "bpmeds":          {"type": "binary"},
    "prevalentstroke": {"type": "binary"},
    "prevalenthyp":    {"type": "binary"},
    "diabetes":        {"type": "binary"},
}

def _is_missing(v):
    if v is None:
        return True
    if isinstance(v, float) and np.isnan(v):
        return True
    if isinstance(v, str) and v.strip() == "":
        return True
    return False

def validate_row_strict(row: dict) -> list[str]:
    """Single prediction: thiếu là lỗi."""
    errors = []
    for col in FEATURES:
        k = norm_key(col)
        v = row.get(col, None)

        if _is_missing(v):
            errors.append(f"- Thiếu giá trị: {ui_label(col)}")
            continue

        rule = VALIDATION_RULES.get(k)
        if not rule:
            continue

        t = rule["type"]
        if t == "binary":
            try:
                iv = int(float(v))
                if iv not in (0, 1):
                    errors.append(f"- {ui_label(col)} chỉ nhận 0/1")
            except Exception:
                errors.append(f"- {ui_label(col)} không hợp lệ (cần 0/1)")
        elif "choices" in rule:
            try:
                iv = int(float(v))
                if iv not in set(rule["choices"]):
                    errors.append(f"- {ui_label(col)} chỉ nhận {rule['choices']}")
            except Exception:
                errors.append(f"- {ui_label(col)} không hợp lệ")
        else:
            mn, mx = rule.get("min", None), rule.get("max", None)
            try:
                fv = float(v)
                if mn is not None and fv < mn:
                    errors.append(f"- {ui_label(col)} phải ≥ {mn}")
                if mx is not None and fv > mx:
                    errors.append(f"- {ui_label(col)} phải ≤ {mx}")
            except Exception:
                errors.append(f"- {ui_label(col)} không phải số")
    return errors

def validate_row_lenient(row: dict) -> list[str]:
    """Batch: thiếu -> bỏ qua (NaN), không coi là lỗi; chỉ check range nếu có giá trị."""
    errors = []
    for col in FEATURES:
        k = norm_key(col)
        v = row.get(col, None)

        if _is_missing(v):
            continue  # cho phép NaN

        rule = VALIDATION_RULES.get(k)
        if not rule:
            continue

        t = rule["type"]
        if t == "binary":
            try:
                iv = int(float(v))
                if iv not in (0, 1):
                    errors.append(f"- {ui_label(col)} chỉ nhận 0/1")
            except Exception:
                errors.append(f"- {ui_label(col)} không hợp lệ (cần 0/1)")
        elif "choices" in rule:
            try:
                iv = int(float(v))
                if iv not in set(rule["choices"]):
                    errors.append(f"- {ui_label(col)} chỉ nhận {rule['choices']}")
            except Exception:
                errors.append(f"- {ui_label(col)} không hợp lệ")
        else:
            mn, mx = rule.get("min", None), rule.get("max", None)
            try:
                fv = float(v)
                if mn is not None and fv < mn:
                    errors.append(f"- {ui_label(col)} phải ≥ {mn}")
                if mx is not None and fv > mx:
                    errors.append(f"- {ui_label(col)} phải ≤ {mx}")
            except Exception:
                errors.append(f"- {ui_label(col)} không phải số")
    return errors

def coerce_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    - education label -> int
    - gender label -> male 0/1
    - binary text -> 0/1
    - convert numeric if possible
    """
    # education
    for c in df.columns:
        if norm_key(c) == "education":
            df[c] = df[c].apply(parse_education)

    # gender
    for c in df.columns:
        if norm_key(c) == "male":
            df[c] = df[c].apply(parse_gender_to_male)

    # binary text mapping
    binary_cols = [c for c in df.columns if VALIDATION_RULES.get(norm_key(c), {}).get("type") == "binary"]
    for c in binary_cols:
        if df[c].dtype == object:
            def _map_bin(x):
                if _is_missing(x):
                    return x
                s = str(x).strip().lower()
                return BINARY_TEXT_MAP.get(s, x)
            df[c] = df[c].apply(_map_bin)

    # numeric convert
    for c in df.columns:
        if df[c].dtype == object:
            try:
                df[c] = df[c].astype(float)
            except Exception:
                pass

    return df

# =========================================================
# INFER SCHEMA SAMPLE (optional) để set default đẹp hơn cho slider
# =========================================================
def try_load_schema_sample():
    candidates = [
        DATA_DIR / "D11KS.csv",
        DATA_DIR / "D1KS.csv",
        DATA_DIR / "cleaned_merged_heart_dataset.csv",
        DATA_DIR / "raw_merged_heart_dataset.csv",
    ]
    for fp in candidates:
        if fp.exists():
            try:
                df = pd.read_csv(fp)
                return df.head(800)
            except Exception:
                pass
    return None

SCHEMA_RAW = try_load_schema_sample()

def _median_or_mid(col, mn, mx):
    default = (mn + mx) / 2.0
    try:
        if SCHEMA_RAW is not None and col in SCHEMA_RAW.columns:
            s = SCHEMA_RAW[col]
            if pd.api.types.is_numeric_dtype(s):
                default = float(np.nanmedian(s.astype(float)))
    except Exception:
        pass
    return float(min(max(default, mn), mx))

# =========================================================
# COMPONENT BUILDER (UI input) - giới hạn nhập (không âm)
# =========================================================
def make_component(col: str):
    key = norm_key(col)
    rule = VALIDATION_RULES.get(key)

    # Gender: chọn 1 trong 2
    if key == "male":
        return gr.Radio(
            choices=["Nam", "Nữ"],
            value="Nam",
            label=f"{ui_label(col)} (chọn 1 trong 2)"
        )

    # Binary: radio 0/1 kèm note
    if rule and rule.get("type") == "binary":
        return gr.Radio(
            choices=[0, 1],
            value=0,
            label=f"{ui_label(col)} (1=Có, 0=Không)"
        )

    # Education: dropdown có note
    if key == "education":
        return gr.Dropdown(
            choices=list(EDU_LABEL_TO_VAL.keys()),
            value="1 - Tiểu học",
            label=f"{ui_label(col)} (chọn mức)"
        )

    # Numeric with min/max -> Slider
    if rule and rule.get("type") in ("int", "float") and ("min" in rule and "max" in rule):
        mn, mx = float(rule["min"]), float(rule["max"])
        step = float(rule.get("step", 1))
        default = _median_or_mid(col, mn, mx)
        return gr.Slider(
            minimum=mn, maximum=mx, step=step, value=default,
            label=ui_label(col)
        )

    # fallback
    return gr.Textbox(label=ui_label(col), placeholder=f"Nhập {ui_label(col)}")

# =========================================================
# PREDICT (Single + Batch)
# =========================================================
def predict_single(model_name, *vals):
    bundle = BUNDLES[model_name]
    pipe = bundle["pipeline"]
    thr  = float(bundle.get("threshold", 0.5))

    row = {FEATURES[i]: vals[i] for i in range(len(FEATURES))}

    # convert UI values -> model expected
    for col in list(row.keys()):
        if norm_key(col) == "education":
            row[col] = parse_education(row[col])
        if norm_key(col) == "male":
            row[col] = parse_gender_to_male(row[col])

    errors = validate_row_strict(row)
    if errors:
        detail = "Dữ liệu nhập chưa hợp lệ:\n" + "\n".join(errors)
        return "❌ KẾT LUẬN: DỮ LIỆU KHÔNG HỢP LỆ", None, None, round(thr, 6), detail

    df = pd.DataFrame([row])
    df = coerce_df(df)

    proba = float(pipe.predict_proba(df)[:, 1][0])
    risk_pct = proba * 100.0
    pred  = int(proba >= thr)

    decision = "⚠️ KẾT LUẬN: CÓ BỆNH (CHD=1)" if pred == 1 else "✅ KẾT LUẬN: KHÔNG BỆNH (CHD=0)"
    detail = f"Model: {model_name} | Threshold: {thr:.4f} | Risk: {risk_pct:.2f}% (proba={proba:.6f})"
    return decision, round(risk_pct, 2), round(proba, 6), round(thr, 6), detail

def build_feature_matrix_from_csv(df: pd.DataFrame):
    """
    - Match cột theo case-insensitive
    - Nếu thiếu cột -> tạo cột NaN
    - Trả về X (đủ FEATURES), và warnings
    """
    warnings = []
    df_cols = list(df.columns)
    colmap_lower = {str(c).strip().lower(): c for c in df_cols}

    X = pd.DataFrame(index=df.index)

    missing = []
    used_original_cols = set()

    for f in FEATURES:
        fl = str(f).strip().lower()
        if f in df.columns:
            X[f] = df[f]
            used_original_cols.add(f)
        elif fl in colmap_lower:
            orig = colmap_lower[fl]
            X[f] = df[orig]
            used_original_cols.add(orig)
        else:
            X[f] = np.nan
            missing.append(f)

    if missing:
        warnings.append(f"⚠️ CSV thiếu {len(missing)} feature → đã fill NaN và vẫn dự đoán: {missing}")

    # columns in CSV that are not used
    unused = [c for c in df_cols if c not in used_original_cols]
    if unused:
        warnings.append(f"ℹ️ CSV có {len(unused)} cột không dùng: {unused[:30]}" + (" ..." if len(unused) > 30 else ""))

    return X, warnings

def predict_batch(model_name, file_obj):
    bundle = BUNDLES[model_name]
    pipe = bundle["pipeline"]
    thr  = float(bundle.get("threshold", 0.5))

    if file_obj is None:
        return "❌ Bạn chưa upload file CSV.", None, None

    try:
        df = pd.read_csv(file_obj.name)
    except Exception as e:
        return f"❌ Không đọc được CSV: {e}", None, None

    X, warnings = build_feature_matrix_from_csv(df)
    X = coerce_df(X)

    # validation per row (lenient)
    is_valid = []
    val_errs = []
    for i in range(len(X)):
        row = {c: X.iloc[i][c] for c in FEATURES}
        errs = validate_row_lenient(row)
        is_valid.append(len(errs) == 0)
        val_errs.append(" | ".join([e.replace("- ", "") for e in errs]) if errs else "")

    # Predict
    proba = pipe.predict_proba(X)[:, 1]
    pred  = (proba >= thr).astype(int)
    risk_pct = proba * 100.0
    label = np.where(pred == 1, "CHD (1) - High risk", "No CHD (0) - Low risk")

    out = df.copy()
    out["proba_1"] = proba
    out["risk_%"]  = risk_pct
    out["pred"]    = pred
    out["label"]   = label
    out["is_valid"] = is_valid
    out["validation_error"] = val_errs

    # save output
    tmpdir = Path(tempfile.mkdtemp())
    out_path = tmpdir / f"predictions_{model_name.replace(' ', '_').replace('(', '').replace(')', '')}.csv"
    out.to_csv(out_path, index=False)

    msg = f"✅ Done. Threshold={thr:.4f}. Rows={len(out)}. Valid={int(np.sum(is_valid))}, Invalid={int(np.sum(~np.array(is_valid)))}."
    if warnings:
        msg += "\n" + "\n".join(warnings)

    preview = out.head(20)
    return msg, preview, str(out_path)

def reload_models():
    global BUNDLES, FEATURES
    BUNDLES = load_bundles()
    FEATURES = BUNDLES["RF (tuned)"].get("feature_names", FEATURES)
    return "✅ Reloaded model files from disk."

# =========================================================
# UI
# =========================================================
css = """
#app {max-width: 1120px; margin: 0 auto;}
.gradio-container {font-family: ui-sans-serif, system-ui;}
"""

theme = gr.themes.Soft()

with gr.Blocks(theme=theme, css=css, title="CHD 10Y Predictor", elem_id="app") as demo:
    gr.Markdown(
        """
        # 🫀 CHD 10-year Risk Predictor
        - Chọn **model (RF/ET)** → nhập thông số → nhận **KẾT LUẬN + % nguy cơ**
        - **Education**: 1=Tiểu học, 2=THCS, 3=THPT, 4=CĐ/ĐH+
        - **Gender**: chọn Nam/Nữ (tự map sang `male`: Nam=1, Nữ=0)
        - Batch CSV: thiếu feature **không dừng**, tự fill **NaN** và cảnh báo
        """
    )

    with gr.Row():
        model_dd = gr.Dropdown(
            choices=list(MODEL_PATHS.keys()),
            value="RF (tuned)",
            label="Chọn model để inference"
        )
        btn_reload = gr.Button("Reload models", variant="secondary")

    reload_status = gr.Markdown("")
    btn_reload.click(fn=reload_models, inputs=[], outputs=[reload_status])

    with gr.Tabs():
        # ------------------ Single ------------------
        with gr.Tab("Single prediction"):
            gr.Markdown("### Nhập feature (2 cột) — đã giới hạn, không nhập âm/phi lý")

            comp_map = {}
            with gr.Row():
                with gr.Column():
                    for i in range(0, len(FEATURES), 2):
                        f = FEATURES[i]
                        comp_map[f] = make_component(f)
                with gr.Column():
                    for i in range(1, len(FEATURES), 2):
                        f = FEATURES[i]
                        comp_map[f] = make_component(f)

            comps = [comp_map[f] for f in FEATURES]

            with gr.Row():
                btn_pred = gr.Button("Predict", variant="primary")
                btn_clear = gr.Button("Reset", variant="secondary")

            out_decision = gr.Textbox(label="Kết luận")
            out_riskpct  = gr.Number(label="Nguy cơ mắc CHD (%)")
            out_proba    = gr.Number(label="Probability (proba=1)")
            out_thr      = gr.Number(label="Threshold used")
            out_detail   = gr.Textbox(label="Chi tiết / Lỗi nhập liệu", lines=6)

            btn_pred.click(
                fn=predict_single,
                inputs=[model_dd] + comps,
                outputs=[out_decision, out_riskpct, out_proba, out_thr, out_detail]
            )

            btn_clear.click(
                fn=lambda: [None]*len(comps) + ["", None, None, None, ""],
                inputs=[],
                outputs=comps + [out_decision, out_riskpct, out_proba, out_thr, out_detail]
            )

        # ------------------ Batch ------------------
        with gr.Tab("Batch CSV"):
            gr.Markdown(
                """
                ### Upload CSV để predict hàng loạt
                - Nếu thiếu feature: hệ thống **tự thêm cột thiếu = NaN** và vẫn dự đoán (sẽ cảnh báo).
                - Nếu tên cột khác chữ hoa/thường: hệ thống match **case-insensitive**.
                - Output thêm: `proba_1`, `risk_%`, `pred`, `label`, `is_valid`, `validation_error`
                """
            )

            file_in = gr.File(label="Upload CSV", file_types=[".csv"])
            btn_batch = gr.Button("Run batch prediction", variant="primary")

            batch_msg = gr.Textbox(label="Status / Warnings", lines=6)
            batch_preview = gr.Dataframe(label="Preview (first 20 rows)", interactive=False)
            batch_outfile = gr.File(label="Download predictions CSV")

            btn_batch.click(
                fn=predict_batch,
                inputs=[model_dd, file_in],
                outputs=[batch_msg, batch_preview, batch_outfile]
            )

# =========================================================
# LAUNCH - tự tìm port trống
# =========================================================
def find_free_port(host="127.0.0.1"):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((host, 0))
    port = s.getsockname()[1]
    s.close()
    return port

if __name__ == "__main__":
    port = find_free_port()
    print(f"* Running on local URL:  http://127.0.0.1:{port}")
    demo.launch(server_name="127.0.0.1", server_port=port, inbrowser=True)
