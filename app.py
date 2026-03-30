import streamlit as st
import yt_dlp
import os
from faster_whisper import WhisperModel

# 配置页面
st.set_page_config(page_title="YouTube Whisper 转录器", page_icon="📝")

st.title("🎥 YouTube 视频自动转录")
st.markdown("输入 YouTube 链接，利用 OpenAI Whisper 模型自动生成字幕。")

# 1. 初始化模型 (建议使用 tiny 或 base 以适应 Streamlit 内存)
@st.cache_resource
def load_model():
    # device="cpu" 是因为 Streamlit Cloud 免费版没有 GPU
    # compute_type="int8" 可以进一步降低内存占用
    return WhisperModel("tiny", device="cpu", compute_type="int8")

model = load_model()

# 2. 定义下载音频的函数
def download_audio(url):
    ydl_opts = {
        'format': 'm4a/bestaudio/best',
        'outtmpl': 'temp_audio.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return "temp_audio.mp3"

# 3. 界面交互
url = st.text_input("在此粘贴 YouTube 链接:", placeholder="https://www.youtube.com/watch?v=...")

if st.button("开始转录"):
    if url:
        try:
            with st.spinner("正在提取音频..."):
                audio_file = download_audio(url)
            
            with st.spinner("Whisper 正在努力转录中 (这可能需要几分钟)..."):
                segments, info = model.transcribe(audio_file, beam_size=5)
                
                st.success(f"检测到语言: {info.language} (置信度: {info.language_probability:.2f})")
                
                full_text = ""
                transcript_container = st.empty() # 创建一个动态更新区域
                
                for segment in segments:
                    segment_text = f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}\n"
                    full_text += segment_text
                    # 实时的在界面上输出（可选）
                    # st.write(segment_text) 
                
                st.text_area("转录结果:", full_text, height=300)
                
                # 提供下载按钮
                st.download_button(
                    label="下载转录文本",
                    data=full_text,
                    file_name="transcript.txt",
                    mime="text/plain"
                )
                
                # 清理临时文件
                if os.path.exists(audio_file):
                    os.remove(audio_file)
                    
        except Exception as e:
            st.error(f"发生错误: {str(e)}")
    else:
        st.warning("请输入有效的链接。")
