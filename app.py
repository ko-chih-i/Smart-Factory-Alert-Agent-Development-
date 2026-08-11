import json
from datetime import datetime, timedelta
import math
import os
import random
import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Page Configuration
st.set_page_config(
    page_title="Pegatron Smart Factory Anomaly Alert Dashboard",
    page_icon="🏭",
    layout="wide",
)

# ── 業界統計參數 ──
MODIFIED_Z_THRESHOLD = 3.5  # Iglewicz & Hoaglin (1993) 穩健離群門檻
TREND_Z_THRESHOLD = 2.0  # 早期預警敏感度：變化速率超出歷史波動 2 個穩健標準差
DEBOUNCE_WINDOW = 5  # 業務規則：連續 5 分鐘才視為有效趨勢

# ── 💡 Ollama Qwen 2.5 整合函式 ──
@st.cache_data(ttl=3600, show_spinner=False)
def get_ollama_qwen_analysis(
    temp: float,
    pressure: float,
    vibration: float,
    sev: str,
    warn_reason_text: str,
    model_name: str = "qwen2.5",
) -> tuple[str, str]:
    """將異常感測數據拋給 Ollama Qwen 2.5 進行動態根因分析與 SOP 指引生成"""
    url = "http://localhost:11434/api/generate"

    prompt = f"""你是一名智慧工廠設備維護專家。監控系統觸發了【{sev}】等級預警，請根據數據進行動態根因推論與 SOP 處置指引。

【當前感測數據】
- 溫度: {temp} °C
- 壓力: {pressure} bar
- 震動: {vibration} g

【系統偵測細節】
- {warn_reason_text}

請嚴格輸出 JSON 格式，包含兩個 key：
1. "root_cause": 潛在故障真因分析（簡潔一句話，30字內）
2. "action_suggestion": 現場工程師應採取的具體 SOP 處置動作50字內

只輸出 JSON，不要加入額外說明：
"""

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "format": "json",
    }

    try:
        response = requests.post(url, json=payload, timeout=8)
        if response.status_code == 200:
            res_json = response.json()
            response_text = res_json.get("response", "{}")
            parsed = json.loads(response_text)
            root_cause = parsed.get("root_cause", "Qwen2.5 推論真因中...")
            action = parsed.get("action_suggestion", "請檢查設備運作狀況。")
            return root_cause, action
        else:
            return (
                f"LLM API 錯誤 (Code {response.status_code})",
                "依標準預防性流程檢修",
            )
    except Exception as e:
        return (
            "Ollama 連線失敗 (請確認已執行 ollama run qwen2.5)",
            "執行基礎巡檢",
        )


@st.cache_data
def generate_sensor_data(
    num_rows=200,
    anomaly_ratio=0.12,
    inject_missing=False,
    missing_rate=0.04,
    seed=42,
):
    random.seed(seed)
    np.random.seed(seed)
    start_time = datetime.strptime("2024-06-03 19:05:00", "%Y-%m-%d %H:%M:%S")
    data = []

    current_temp = 47.5
    current_pressure = 1.025
    current_vibration = 0.030

    anomaly_phase_remaining = 0
    current_anomaly_type = None

    for i in range(num_rows):
        current_time = (start_time + timedelta(minutes=i)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        if anomaly_phase_remaining <= 0:
            if random.random() < (anomaly_ratio * 0.28):
                anomaly_phase_remaining = random.randint(4, 11)
                current_anomaly_type = random.choice([
                    "high_temp",
                    "low_temp",
                    "high_press",
                    "low_press",
                    "high_vib",
                    "compound",
                ])
            else:
                current_anomaly_type = None

        cycle_phase = (i / 18.0) * math.pi * 2.0
        target_normal_temp = 47.5 + 1.2 * math.sin(cycle_phase)
        target_normal_press = 1.025 + 0.01 * math.cos(cycle_phase)
        target_normal_vib = 0.030 + 0.004 * math.sin(cycle_phase * 2.0)

        target_temp = target_normal_temp
        target_press = target_normal_press
        target_vib = target_normal_vib

        if anomaly_phase_remaining > 0 and current_anomaly_type:
            anomaly_phase_remaining -= 1
            if current_anomaly_type == "high_temp":
                target_temp = 53.5 + random.uniform(0.0, 6.5)
            elif current_anomaly_type == "low_temp":
                target_temp = 36.0 + random.uniform(0.0, 5.0)
            elif current_anomaly_type == "high_press":
                target_press = 1.10 + random.uniform(0.0, 0.15)
            elif current_anomaly_type == "low_press":
                target_press = 0.85 + random.uniform(0.0, 0.10)
            elif current_anomaly_type == "high_vib":
                target_vib = 0.08 + random.uniform(0.0, 0.06)
            elif current_anomaly_type == "compound":
                target_temp = 54.0 + random.uniform(0.0, 5.0)
                target_press = 1.12 + random.uniform(0.0, 0.10)
                target_vib = 0.08 + random.uniform(0.0, 0.05)

        current_temp = (
            0.65 * current_temp + 0.35 * target_temp + (random.random() - 0.5) * 0.3
        )
        current_pressure = (
            0.70 * current_pressure
            + 0.30 * target_press
            + (random.random() - 0.5) * 0.014
        )
        current_vibration = (
            0.70 * current_vibration
            + 0.30 * target_vib
            + (random.random() - 0.5) * 0.006
        )

        temp = round(float(current_temp), 1)
        pressure = round(float(current_pressure), 2)
        vibration = round(float(current_vibration), 2)

        if not current_anomaly_type and anomaly_phase_remaining <= 0:
            if temp > 50.0:
                temp = 49.8
            if temp < 45.0:
                temp = 45.2
            if pressure > 1.05:
                pressure = 1.04
            if pressure < 1.00:
                pressure = 1.01
            if vibration > 0.04:
                vibration = 0.038
            if vibration < 0.02:
                vibration = 0.022

        is_physically_abnormal = (
            (temp > 52.0)
            or (temp < 43.0)
            or (pressure > 1.08)
            or (pressure < 0.97)
            or (vibration > 0.07)
        )
        label = "abnormal" if is_physically_abnormal else "normal"

        if inject_missing and random.random() < missing_rate:
            target_col = random.choice(["temp", "pressure", "vibration"])
            if target_col == "temp":
                temp = None
            elif target_col == "pressure":
                pressure = None
            else:
                vibration = None

        data.append({
            "timestamp": current_time,
            "temp": temp,
            "pressure": pressure,
            "vibration": vibration,
            "label": label,
        })

    return pd.DataFrame(data)


def _robust_std(series: pd.Series) -> float:
    med = series.median()
    mad = (series - med).abs().median()
    return float(mad * 1.4826) if mad > 1e-9 else float(series.std() + 1e-9)


def run_anomaly_pipeline(
    df,
    impute_method="線性插值 (Linear Interpolation)",
    enable_llm: bool = False,
):
    clean_df = df.copy().reset_index(drop=True)
    imputed_info = []

    for col in ["temp", "pressure", "vibration"]:
        missing_count = clean_df[col].isnull().sum()
        if missing_count > 0:
            if "線性插值" in impute_method or "Linear" in impute_method:
                clean_df[col] = (
                    clean_df[col].interpolate(method="linear").bfill().ffill()
                )
                method_name = "線性插值 (Linear)"
            elif (
                "前向填補" in impute_method
                or "LOCF" in impute_method
                or "Forward" in impute_method
            ):
                clean_df[col] = clean_df[col].ffill().bfill()
                method_name = "前向填補 (Forward Fill / LOCF)"
            else:
                median_val = clean_df[col].median()
                clean_df[col] = clean_df[col].fillna(median_val)
                method_name = f"中位數 ({median_val:.2f})"

            imputed_info.append(
                f"{col}: 填補 {missing_count} 筆 [{method_name}]"
            )

    clean_df["temp"] = clean_df["temp"].round(1)
    clean_df["pressure"] = clean_df["pressure"].round(2)
    clean_df["vibration"] = clean_df["vibration"].round(2)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(
        clean_df[["temp", "pressure", "vibration"]]
    )
    (
        clean_df["temp_z"],
        clean_df["pressure_z"],
        clean_df["vibration_z"],
    ) = (scaled[:, 0], scaled[:, 1], scaled[:, 2])

    iso_forest = IsolationForest(contamination="auto", random_state=42)
    iso_forest.fit(clean_df[["temp", "pressure", "vibration"]])
    iso_preds = iso_forest.predict(clean_df[["temp", "pressure", "vibration"]])
    iso_scores = iso_forest.decision_function(
        clean_df[["temp", "pressure", "vibration"]]
    )

    median_score = float(np.median(iso_scores))
    mad_score = float(np.median(np.abs(iso_scores - median_score)))
    mad_score = mad_score if mad_score > 1e-9 else 1e-9

    def modified_z_score(score):
        return 0.6745 * (score - median_score) / mad_score

    temp_diff_std = _robust_std(clean_df["temp"].diff().fillna(0))
    press_diff_std = _robust_std(clean_df["pressure"].diff().fillna(0))
    vib_diff_std = _robust_std(clean_df["vibration"].diff().fillna(0))

    (
        reasons_list,
        warning_reasons,
        severities,
        anomaly_scores,
        actions,
        predicted_labels,
    ) = ([], [], [], [], [], [])
    is_temp_anom_list, is_press_anom_list, is_vib_anom_list = [], [], []

    total_rows = len(clean_df)

    for i in range(total_rows):
        row = clean_df.iloc[i]
        reasons = []
        rule_score = 0.0

        if row["temp"] > 52.0:
            reasons.append(f"過熱 ({row['temp']}°C > 52°C)")
            rule_score += 0.45
        elif row["temp"] < 43.0:
            reasons.append(f"過冷 ({row['temp']}°C < 43°C)")
            rule_score += 0.40

        if row["pressure"] > 1.08:
            reasons.append(f"壓力過高 ({row['pressure']} > 1.08 bar)")
            rule_score += 0.40
        elif row["pressure"] < 0.97:
            reasons.append(f"壓力過低 ({row['pressure']} < 0.97 bar)")
            rule_score += 0.35

        if row["vibration"] > 0.07:
            reasons.append(f"劇烈震動 ({row['vibration']} > 0.07 g)")
            rule_score += 0.50

        has_physical_violation = len(reasons) > 0

        mz = modified_z_score(iso_scores[i])
        is_statistical_outlier = (mz > MODIFIED_Z_THRESHOLD) and (
            iso_preds[i] == -1
        )

        start_i = max(0, i - (DEBOUNCE_WINDOW - 1))
        window_len = i - start_i + 1
        is_persistent = window_len >= DEBOUNCE_WINDOW

        temp_trend = row["temp"] - clean_df.iloc[start_i]["temp"]
        press_trend = row["pressure"] - clean_df.iloc[start_i]["pressure"]
        vib_trend = row["vibration"] - clean_df.iloc[start_i]["vibration"]

        temp_trend_z = temp_trend / temp_diff_std
        press_trend_z = press_trend / press_diff_std
        vib_trend_z = vib_trend / vib_diff_std

        has_upward_trend = (
            (temp_trend_z > TREND_Z_THRESHOLD)
            or (press_trend_z > TREND_Z_THRESHOLD)
            or (vib_trend_z > TREND_Z_THRESHOLD)
        )

        is_early_warning = (
            is_statistical_outlier
            and has_upward_trend
            and is_persistent
            and not has_physical_violation
        )

        # 僅在動態串流模式且屬於當前最新單筆資料點 (i == total_rows - 1) 時觸發 LLM 推論
        should_call_llm = enable_llm and (i == total_rows - 1)

        if has_physical_violation:
            pred_label = "abnormal"
            sev = "ALERT"
            final_score = round(min(1.0, max(0.75, 0.40 + rule_score)), 2)
            base_cause = ", ".join(reasons)
            warn_reason = f"物理指標超標 (規格門檻): {base_cause}"

            if should_call_llm:
                root_cause, act = get_ollama_qwen_analysis(
                    round(row["temp"], 1),
                    round(row["pressure"], 2),
                    round(row["vibration"], 2),
                    sev,
                    warn_reason,
                )
                root_cause = f"🤖 [Qwen2.5] {root_cause}"
            else:
                root_cause = base_cause
                act = "依規範派員巡檢修復"

        elif is_early_warning:
            pred_label = "normal"
            sev = "WARNING"
            excess = min(mz - MODIFIED_Z_THRESHOLD, 5.0) / 5.0
            final_score = round(0.50 + excess * 0.24, 2)
            warn_reason = f"Modified Z-score ({mz:.2f} > {MODIFIED_Z_THRESHOLD}) 且呈現上升趨勢"

            if should_call_llm:
                root_cause, act = get_ollama_qwen_analysis(
                    round(row["temp"], 1),
                    round(row["pressure"], 2),
                    round(row["vibration"], 2),
                    sev,
                    warn_reason,
                )
                root_cause = f"🤖 [Qwen2.5 預警] {root_cause}"
            else:
                root_cause = (
                    f"統計離群 (Modified Z-score = {mz:.2f} >"
                    f" {MODIFIED_Z_THRESHOLD})"
                )
                act = "派員進行感測器校正與預防性巡檢"

        else:
            pred_label = "normal"
            sev = "NORMAL"
            final_score = 0.0
            root_cause = "無異常 (Normal)"
            warn_reason = "感測器數值在統計正常範圍內"
            act = "設備運作正常，維持預防性維護"

        temp_viol = (row["temp"] > 52.0) or (row["temp"] < 43.0)
        press_viol = (row["pressure"] > 1.08) or (row["pressure"] < 0.97)
        vib_viol = row["vibration"] > 0.07

        if has_physical_violation:
            is_temp_anom = temp_viol
            is_press_anom = press_viol
            is_vib_anom = vib_viol
        elif is_early_warning:
            is_temp_anom = temp_trend_z > TREND_Z_THRESHOLD
            is_press_anom = press_trend_z > TREND_Z_THRESHOLD
            is_vib_anom = vib_trend_z > TREND_Z_THRESHOLD
            if not (is_temp_anom or is_press_anom or is_vib_anom):
                max_idx = int(
                    np.argmax(
                        [abs(temp_trend_z), abs(press_trend_z), abs(vib_trend_z)]
                    )
                )
                is_temp_anom = max_idx == 0
                is_press_anom = max_idx == 1
                is_vib_anom = max_idx == 2
        else:
            is_temp_anom = False
            is_press_anom = False
            is_vib_anom = False

        is_temp_anom_list.append(is_temp_anom)
        is_press_anom_list.append(is_press_anom)
        is_vib_anom_list.append(is_vib_anom)

        reasons_list.append(root_cause)
        warning_reasons.append(warn_reason)
        severities.append(sev)
        anomaly_scores.append(final_score)
        actions.append(act)
        predicted_labels.append(pred_label)

    clean_df["predicted_label"] = predicted_labels
    clean_df["severity"] = severities
    clean_df["anomaly_score"] = anomaly_scores
    clean_df["warning_reason"] = warning_reasons
    clean_df["root_cause"] = reasons_list
    clean_df["action_suggestion"] = actions
    clean_df["is_temp_anom"] = is_temp_anom_list
    clean_df["is_press_anom"] = is_press_anom_list
    clean_df["is_vib_anom"] = is_vib_anom_list
    if "label" in clean_df.columns:
        clean_df["gt_match"] = (
            clean_df["predicted_label"] == clean_df["label"]
        ).map({True: "✓ 一致 (Match)", False: "✗ 差異 (Diff)"})
    return clean_df, imputed_info


def main():
    st.sidebar.title("和碩 Pegatron 智慧工廠")
    st.sidebar.caption("Assignment 3 — Streamlit + Ollama AI Agent 儀表板")

    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 Ollama AI Agent 設定")
    enable_llm = st.sidebar.toggle(
        "啟用 Qwen 2.5 動態根因診斷",
        value=True,
        help="切換至『動態 1 秒逐筆串流模式』且觸發 WARNING 或 ALERT 時，透過地端 Ollama 呼叫 qwen2.5 進行即時 LLM 推論",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ 1. 設定與生成 CSV 數據")

    num_rows = st.sidebar.slider("感測器筆數 (Rows)", 100, 500, 200, 50)
    anomaly_prob = st.sidebar.slider("異常機率", 0.05, 0.30, 0.12, 0.01)

    if "data_seed" not in st.session_state:
        st.session_state.data_seed = 42

    seed_input = st.sidebar.number_input(
        "隨機種子 (Random Seed)", value=st.session_state.data_seed, step=1
    )
    if seed_input != st.session_state.data_seed:
        st.session_state.data_seed = seed_input
        st.session_state.test_executed = False

    inject_missing = st.sidebar.toggle(
        "可選遺失值處理 (Missing Values Imputation)", value=True
    )

    impute_method = "線性插值 (Linear Interpolation)"
    if inject_missing:
        impute_method = st.sidebar.selectbox(
            "填補演算法 (Imputation Strategy)",
            options=[
                "線性插值 (Linear Interpolation)",
                "前向填補 / LOCF (Forward Fill)",
                "中位數填補 (Median Imputation)",
            ],
            index=0,
        )

    gen_csv_clicked = st.sidebar.button(
        "🎲 生成 / 重新產生 CSV 數據檔", use_container_width=True
    )
    if gen_csv_clicked:
        st.session_state.data_seed = random.randint(1, 10000)
        st.session_state.test_executed = False
        st.session_state.stream_count = 1
        st.rerun()

    generated_df = generate_sensor_data(
        num_rows=num_rows,
        anomaly_ratio=anomaly_prob,
        inject_missing=inject_missing,
        seed=st.session_state.data_seed,
    )

    st.sidebar.download_button(
        label="📥 下載生成的 CSV 數據檔 (sensor_data.csv)",
        data=generated_df.to_csv(index=False).encode("utf-8"),
        file_name="sensor_data.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("📂 2. 手動上傳 CSV 檔案 (可選)")
    uploaded_file = st.sidebar.file_uploader(
        "匯入感測器 CSV 檔案", type=["csv"]
    )

    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        raw_df.to_csv("sensor_data.csv", index=False)
        if "last_uploaded_name" not in st.session_state or st.session_state.last_uploaded_name != uploaded_file.name:
            st.session_state.last_uploaded_name = uploaded_file.name
            st.session_state.test_executed = False
    else:
        raw_df = generated_df
        raw_df.to_csv("sensor_data.csv", index=False)

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚡ 3. 執行模式選擇")
    exec_mode = st.sidebar.radio(
        "請選擇執行模式",
        options=["靜態一次性載入分析", "動態 1 秒逐筆串流模式"],
        index=0,
    )
    stream_active = exec_mode == "動態 1 秒逐筆串流模式"

    total_raw_rows = len(raw_df)

    if stream_active:
        if "stream_count" not in st.session_state:
            st.session_state.stream_count = 1

        col_s1, col_s2 = st.sidebar.columns(2)
        if col_s1.button("🔄 重頭開始", use_container_width=True):
            st.session_state.stream_count = 1
            st.rerun()
        if col_s2.button("⏩ 一次載入全部", use_container_width=True):
            st.session_state.stream_count = total_raw_rows
            st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.subheader("🚀 4. 啟動測試")
    run_test_clicked = st.sidebar.button(
        "🚀 開始執行 AI Agent 測試", type="primary", use_container_width=True
    )

    if "test_executed" not in st.session_state:
        st.session_state.test_executed = False

    if st.sidebar.button("↩️ 重置並返回 CSV 預覽頁面", use_container_width=True):
        st.session_state.test_executed = False
        st.session_state.stream_count = 1
        st.rerun()

    if run_test_clicked:
        st.session_state.test_executed = True
        if stream_active and "stream_count" not in st.session_state:
            st.session_state.stream_count = 1

    # ── 初始準備與 CSV 數據預覽頁面 ──
    if not st.session_state.test_executed:
        st.title("🏭 智慧工廠設備異常警報 AI 儀表板")

        col_act, col_prev = st.columns([1, 2])
        with col_act:
            st.markdown("### 📋 測試準備狀態")
            st.markdown(
                f"- **數據來源**: `sensor_data.csv`\n- **數據總筆數**: {len(raw_df)} 筆"
            )
            if st.button(
                "🚀 開始執行 AI Agent 測試 (Start Test)",
                type="primary",
                key="tmpl_start_btn",
                use_container_width=True,
            ):
                st.session_state.test_executed = True
                if stream_active:
                    st.session_state.stream_count = 1
                st.rerun()

        with col_prev:
            st.markdown("### 📄 產出之 CSV 原始數據預覽 (`sensor_data.csv`)")
            st.dataframe(raw_df.head(15), use_container_width=True)

        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            with st.expander(
                "🧮 1. 怎麼算出 Anomaly Score？（分數算式說明）", expanded=False
            ):
                st.markdown(f"""
                    **Anomaly Score（異常預警指數）推導與計算機制**：

                    **第一層：物理規則判斷（警告 🔴）**
                    依規格門檻判定：temp > 52°C 或 < 43°C、pressure > 1.08 或 < 0.97 bar、
                    vibration > 0.07 g，任一超標即判定為**警告 (ALERT)**，分數固定落在 0.75~1.0。

                    **第二層：統計離群判斷（預警 🟡）**
                    1. Isolation Forest 計算決策分數 `S = decision_function([temp, pressure, vibration])`
                    2. 用穩健統計量 **Median + MAD** (Median Absolute Deviation) 取代 Mean/Std，
                       避免資料中既有異常值污染基線 (masking effect)：

                       `Modified Z = 0.6745 × (S - median(S)) / MAD(S)`

                       門檻採用 **Modified Z > {MODIFIED_Z_THRESHOLD}**
                       (參考 Iglewicz & Hoaglin, 1993《How to Detect and Handle Outliers》建議值)
                    3. 同時要求連續 **{DEBOUNCE_WINDOW} 分鐘**呈現離群狀態，且變化速率超出歷史波動
                       **{TREND_Z_THRESHOLD} 個穩健標準差**，才升級為預警，避免單點雜訊觸發誤報。

                    **第三層：正常（綠 🟢）**
                    以上皆非，數值落於統計與規格正常範圍內。
                    """)

        with col_exp2:
            with st.expander(
                "📁 2. 上傳 CSV 建議格式規範與範例說明", expanded=False
            ):
                st.markdown("""
                    **上傳 CSV 建議欄位名稱與資料型別**：
                    - `timestamp` *(建議)*：時間戳記，例如 `2024-06-03 19:05:00`
                    - `temp` *(必填)*：設備溫度 (°C)，正常約 `45.0 ~ 50.0`
                    - `pressure` *(必填)*：管線壓力 (bar)，正常約 `1.00 ~ 1.05`
                    - `vibration` *(必填)*：三軸震動 (g)，正常約 `0.02 ~ 0.04`
                    - `label` *(選填)*：實際標籤 (`normal` / `abnormal`)，若省略將由 Agent 進行 100% 無監督推論。

                    💡 *側面選單亦可點擊「下載 CSV 數據檔」按鈕取得目前範本。*
                    """)

        return

    # ── 點擊按鈕後的正式分析與串流儀表板區塊 ──
    if stream_active:
        raw_df = raw_df.iloc[: st.session_state.get("stream_count", total_raw_rows)]

    effective_enable_llm = enable_llm and stream_active

    processed_df, imputed_info = run_anomaly_pipeline(
        raw_df, impute_method=impute_method, enable_llm=effective_enable_llm
    )

    latest_row = processed_df.iloc[-1] if len(processed_df) > 0 else None
    current_sev = latest_row["severity"] if latest_row is not None else "NORMAL"
    latest_score = (
        latest_row["anomaly_score"] if latest_row is not None else 0.0
    )
    latest_time = (
        str(latest_row["timestamp"]).split()[-1]
        if (latest_row is not None and "timestamp" in latest_row)
        else "N/A"
    )

    if latest_row is not None and current_sev != "NORMAL":
        latest_cause = (
            latest_row["root_cause"]
            if latest_row["root_cause"]
            else "設備異常"
        )
        latest_action = (
            f"建議: {latest_row['action_suggestion']}"
            if latest_row["action_suggestion"]
            else "建議: 派員巡檢"
        )
    else:
        latest_cause = "運作正常"
        latest_action = "建議: 維持預防性維護"

    sev_display_map = {
        "NORMAL": ("正常", "🟢"),
        "WARNING": ("預警", "🟡"),
        "ALERT": ("警告", "🔴"),
    }
    sev_label, sev_icon = sev_display_map.get(current_sev, ("正常", "🟢"))

    m1, m2 = st.columns(2)
    status_text = "正常" if current_sev == "NORMAL" else "需注意"
    m1.metric(
        "機台當前狀態",
        f"{sev_icon} {sev_label}",
        delta=f"快照 {latest_time} ({status_text})",
        delta_color="normal" if current_sev == "NORMAL" else "inverse",
    )
    m2.metric(
        "即時預警指數",
        f"{latest_score*100:.1f}%",
        delta="Isolation Forest + Qwen 2.5 LLM",
    )

    st.metric(
        "當前異常真因 (Qwen2.5 推論)",
        f"{latest_cause}",
        delta=f"{latest_action}",
        delta_color="normal" if current_sev == "NORMAL" else "inverse",
    )

    with st.expander("📄 點此檢視原始 CSV 感測數據 (`sensor_data.csv`)", expanded=False):
        st.dataframe(raw_df, use_container_width=True)

    st.markdown("---")

    fig_ts = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            "🌡️ 溫度時序分層趨勢 (°C)",
            "🗜️ 壓力時序分層趨勢 (bar)",
            "⚡ 震動時序分層趨勢 (g)",
        ),
    )
    fig_ts.add_trace(
        go.Scatter(
            x=processed_df["timestamp"],
            y=processed_df["temp"],
            mode="lines",
            name="溫度 (°C)",
            line=dict(color="#f97316", width=2),
        ),
        row=1,
        col=1,
    )
    fig_ts.add_hline(
        y=52,
        line_dash="dash",
        line_color="#ef4444",
        annotation_text="上限 52°C",
        row=1,
        col=1,
    )
    fig_ts.add_hline(
        y=43,
        line_dash="dash",
        line_color="#ef4444",
        annotation_text="下限 43°C",
        row=1,
        col=1,
    )

    fig_ts.add_trace(
        go.Scatter(
            x=processed_df["timestamp"],
            y=processed_df["pressure"],
            mode="lines",
            name="壓力 (bar)",
            line=dict(color="#3b82f6", width=2),
        ),
        row=2,
        col=1,
    )
    fig_ts.add_hline(
        y=1.08,
        line_dash="dash",
        line_color="#ef4444",
        annotation_text="上限 1.08 bar",
        row=2,
        col=1,
    )
    fig_ts.add_hline(
        y=0.97,
        line_dash="dash",
        line_color="#ef4444",
        annotation_text="下限 0.97 bar",
        row=2,
        col=1,
    )

    fig_ts.add_trace(
        go.Scatter(
            x=processed_df["timestamp"],
            y=processed_df["vibration"],
            mode="lines",
            name="震動 (g)",
            line=dict(color="#a855f7", width=2),
        ),
        row=3,
        col=1,
    )
    fig_ts.add_hline(
        y=0.07,
        line_dash="dash",
        line_color="#ef4444",
        annotation_text="上限 0.07 g",
        row=3,
        col=1,
    )

    fig_ts.update_layout(
        paper_bgcolor="#0b132b",
        plot_bgcolor="#0b132b",
        font=dict(color="#e2e8f0"),
        height=650,
        hovermode="x unified",
        showlegend=False,
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    st.subheader("🚨 設備異常警報清單與 Agent 推論結果")

    def style_severity_rows(row):
        sev = str(row.get("severity", "")).upper()
        if sev == "ALERT":
            return [
                "color: #ef4444; font-weight: bold; background-color: rgba(153, 27,"
                " 27, 0.25);"
                for _ in row
            ]
        elif sev == "WARNING":
            return [
                "color: #eab308; font-weight: bold; background-color: rgba(161, 98,"
                " 7, 0.20);"
                for _ in row
            ]
        return ["" for _ in row]

    display_df = processed_df.iloc[::-1][
        [
            "timestamp",
            "temp",
            "pressure",
            "vibration",
            "severity",
            "anomaly_score",
            "root_cause",
            "action_suggestion",
        ]
    ]
    st.dataframe(
        display_df.style.apply(style_severity_rows, axis=1),
        use_container_width=True,
    )

    if stream_active and st.session_state.get("stream_count", 0) < total_raw_rows:
        time.sleep(1)
        st.session_state.stream_count += 1
        st.rerun()


if __name__ == "__main__":
    main()
