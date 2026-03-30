import streamlit as st
import os
import yt_dlp
from faster_whisper import WhisperModel

st.set_page_config(page_title="AI 语音转文字工具", page_icon="📝")

@st.cache_resource
def load_model():
    return WhisperModel("base", device="cpu", compute_type="int8")

model = load_model()

# --- 核心转录逻辑 ---
def transcribe_audio(file_path):
    segments, info = model.transcribe(file_path, beam_size=5)
    full_text = ""
    for segment in segments:
        ts = f"[{int(segment.start)//60:02d}:{int(segment.start)%60:02d}]"
        line = f"{ts} {segment.text}\n"
        full_text += line
        st.write(line)
    return full_content

st.title("🎙️ 多功能转录助手")
tab1, tab2 = st.tabs(["🔗 YouTube 链接", "📂 上传本地文件"])

# --- Tab 1: YouTube 链接 ---
with tab1:
    url = st.text_input("输入链接:")
    if st.button("开始解析链接"):
        # 使用方案 A 的下载代码
        with st.spinner("正在尝试下载..."):
            path, err = download_audio_stealth(url)
            if err:
                st.error("YouTube 暂时封锁了服务器 IP，请尝试【上传本地文件】模式。")
            else:
                transcribe_audio(path)
                os.remove(path)

# --- Tab 2: 本地上传 ---
with tab2:
    uploaded_file = st.file_uploader("选择 mp3/wav 文件", type=["mp3", "wav", "m4a"])
    if uploaded_file is not None:
        if st.button("开始转录上传文件"):
            with open("temp_upload.mp3", "wb") as f:
                f.write(uploaded_file.getbuffer())
            transcribe_audio("temp_upload.mp3")
            os.remove("temp_upload.mp3")
