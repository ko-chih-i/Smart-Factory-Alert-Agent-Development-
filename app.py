<!-- ... existing code ... -->
# ── 💡 Ollama Qwen 2.5 整合函式 ──
@st.cache_data(ttl=3600, show_spinner=False)
def get_ollama_qwen_analysis(
    temp: float,
    pressure: float,
    vibration: float,
    sev: str,
    warn_reason_text: str,
    model_name: str = "qwen2.5",
    base_url: str = "http://localhost:11434",
) -> tuple[str, str]:
    """將異常感測數據拋給 Ollama Qwen 2.5 進行動態根因分析與 SOP 指引生成"""
    url = f"{base_url.rstrip('/')}/api/generate"

    prompt = f"""你是一名智慧工廠設備維護專家。監控系統觸發了【{sev}】等級預警，請根據數據進行動態根因推論與 SOP 處置指引。
<!-- ... existing code ... -->
def main():
    st.sidebar.title("和碩 Pegatron 智慧工廠")
    st.sidebar.caption("Assignment 3 — Streamlit + Ollama AI Agent 儀表板")

    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 Ollama AI Agent 設定")
    
    ollama_url = st.sidebar.text_input(
        "Ollama Server 網址",
        value=st.secrets.get("OLLAMA_URL", "http://localhost:11434"),
        help="本機測試用 localhost；若部署至 Streamlit Cloud，請輸入 ngrok 提供的外網網址"
    )

    enable_llm = st.sidebar.toggle(
        "啟用 Qwen 2.5 動態根因診斷",
        value=True,
        help="切換至『動態 1 秒逐筆串流模式』且觸發 WARNING 或 ALERT 時，透過地端 Ollama 呼叫 qwen2.5 進行即時 LLM 推論",
    )
<!-- ... existing code ... -->
```

---

### 步驟 2：使用 ngrok 把本機 Ollama 穿透到外網

1. **下載並安裝 ngrok**（免費）：
   至 [ngrok.com](https://ngrok.com/) 下載安裝，或用指令（macOS/Linux: `brew install ngrok`，Windows 下載 `.exe`）。
2. **開啟電腦的 Ollama**：
   確保本機已執行 `ollama run qwen2.5`。
3. **允許 Ollama 跨域請求（CORS，重要！）**：
   在終端機啟動 Ollama 前設定環境變數：
   * **Windows (PowerShell)**:
     ```powershell
     $env:OLLAMA_ORIGINS="*"
     ollama serve
     ```
   * **Mac / Linux**:
     ```bash
     OLLAMA_ORIGINS="*" ollama serve
     ```
4. **開啟 ngrok 穿透**：
   在終端機輸入：
   ```bash
   ngrok http 11434
   ```
5. **複製生成的網址**：
   ngrok 會提供一個公網 URL，例如：
   `https://a1b2-34-56-78-90.ngrok-free.app`

---

### 步驟 3：在 Streamlit Cloud 上設定

1. 將更新後的程式碼推送到 GitHub，並部署至 `share.streamlit.io`。
2. 開啟雲端儀表板後，在側邊欄的 **「Ollama Server 網址」** 欄位貼上剛剛 ngrok 產生的 `https://xxxx.ngrok-free.app` 網址。
3. （可選）也可在 Streamlit Cloud 專案的 **App Settings -> Secrets** 設定預設值：
   ```toml
   OLLAMA_URL = "https://xxxx.ngrok-free.app"
   ```

這樣雲端運行的 Streamlit 就能順利連回你電腦上執行的 Qwen 2.5 了！
