def run_anomaly_pipeline(
    df,
    impute_method="線性插值 (Linear Interpolation)",
    enable_llm: bool = True,
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

  total_len = len(clean_df)

  for i in range(total_len):
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

    # 💡 核心優化：只對「最新一筆 (i == total_len - 1)」呼叫 LLM，避免歷史數據迴圈卡死
    is_latest_row = i == (total_len - 1)

    if has_physical_violation:
      pred_label = "abnormal"
      sev = "ALERT"
      final_score = round(min(1.0, max(0.75, 0.40 + rule_score)), 2)
      base_cause = ", ".join(reasons)
      warn_reason = f"物理指標超標 (規格門檻): {base_cause}"

      if enable_llm and is_latest_row:
        root_cause, act = get_ollama_gemma_analysis(
            round(row["temp"], 1),
            round(row["pressure"], 2),
            round(row["vibration"], 2),
            sev,
            warn_reason,
        )
        root_cause = f"🤖 [Gemma2:2B] {root_cause}"
      else:
        root_cause = base_cause
        act = "依規範派員巡檢修復"

    elif is_early_warning:
      pred_label = "normal"
      sev = "WARNING"
      excess = min(mz - MODIFIED_Z_THRESHOLD, 5.0) / 5.0
      final_score = round(0.50 + excess * 0.24, 2)
      warn_reason = f"Modified Z-score ({mz:.2f} > {MODIFIED_Z_THRESHOLD}) 且呈現上升趨勢"

      if enable_llm and is_latest_row:
        root_cause, act = get_ollama_gemma_analysis(
            round(row["temp"], 1),
            round(row["pressure"], 2),
            round(row["vibration"], 2),
            sev,
            warn_reason,
        )
        root_cause = f"🤖 [Gemma2:2B 預警] {root_cause}"
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
