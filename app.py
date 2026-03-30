import streamlit as st
import os
from pytubefix import YouTube
from pytubefix.cli import on_progress
from faster_whisper import WhisperModel

# --- 页面配置 ---
st.set_page_config(page_title="AI 视频转录助手", page_icon="🎙️")

st.title("🎙️ YouTube 视频转录工具")
st.markdown("使用 OpenAI Whisper 模型，直接从 YouTube 链接提取文本。")

# --- 1. 加载模型 (缓存以节省内存) ---
@st.cache_resource
def load_whisper_model():
    # 使用 'base' 模型平衡速度与准确度
    # compute_type="int8" 极大减少内存占用，防止 Streamlit 403/崩溃
    return WhisperModel("base", device="cpu", compute_type="int8")

model = load_whisper_model()

# --- 2. 核心功能：下载与转录 ---
def process_video(url):
    try:
        # 下载音频
        with st.status("正在从 YouTube 获取音频...", expanded=True) as status:
            yt = YouTube(url, on_progress_callback=on_progress)
            st.info(f"正在处理: **{yt.title}**")
            
            # 仅提取音频流以节省带宽
            audio_stream = yt.streams.get_audio_only()
            audio_path = audio_stream.download(filename="temp_audio.mp3")
            status.update(label="音频下载完成！开始转录...", state="running")

            # 执行转录
            segments, info = model.transcribe(audio_path, beam_size=5)
            
            full_text = []
            progress_bar = st.progress(0)
            
            # 迭代转录结果
            for segment in segments:
                timestamp = f"[{int(segment.start)//60:02d}:{int(segment.start)%60:02d}]"
                line = f"{timestamp} {segment.text}"
                full_text.append(line)
                # 实时显示在界面上
                st.write(line)

            status.update(label="转录成功！", state="complete")
            return "\n".join(full_text), audio_path

    except Exception as e:
        st.error(f"处理失败: {str(e)}")
        return None, None

# --- 3. 网页交互界面 ---
url_input = st.text_input("请输入 YouTube 视频链接:", placeholder="https://www.youtube.com/watch?v=...")

if st.button("🚀 开始任务"):
    if url_input:
        transcript, file_to_clean = process_video(url_input)
        
        if transcript:
            st.divider()
            st.subheader("最终转录文本")
            st.text_area("您可以复制以下内容:", transcript, height=400)
            
            # 提供下载
            st.download_button(
                label="📥 下载转录结果 (.txt)",
                data=transcript,
                file_name="transcript.txt",
                mime="text/plain"
            )
            
            # 清理服务器临时文件
            if os.path.exists(file_to_clean):
                os.remove(file_to_clean)
    else:
        st.warning("请先输入链接！")

st.sidebar.info("💡 提示：对于长视频，转录可能需要几分钟，请耐心等待。")
