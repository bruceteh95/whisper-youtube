import streamlit as st
import os
from pytubefix import YouTube
from faster_whisper import WhisperModel

st.set_page_config(page_title="OAuth Whisper 转录器", page_icon="🔑")

# --- 1. 加载模型 ---
@st.cache_resource
def load_model():
    return WhisperModel("base", device="cpu", compute_type="int8")

model = load_model()

# --- 2. 带有 OAuth 的下载函数 ---
def download_with_oauth(url):
    try:
        # 使用 use_oauth=True 开启授权模式
        # allow_oauth_cache=True 会尝试寻找本地 token，避免重复授权
        yt = YouTube(
            url, 
            use_oauth=True, 
            allow_oauth_cache=True
        )
        
        # 如果需要授权，pytubefix 会在后台等待。
        # 注意：在 Streamlit 云端，我们需要在日志或通过捕获输出来引导用户
        st.info(f"正在准备下载: {yt.title}")
        
        audio = yt.streams.get_audio_only()
        path = audio.download(filename="temp_audio.mp3")
        return path, None
    except Exception as e:
        return None, str(e)

# --- 3. 界面逻辑 ---
st.title("🛡️ YouTube 稳定转录器 (OAuth 版)")
st.write("如果这是你第一次运行，请查看下方说明完成 Google 授权。")

url_input = st.text_input("YouTube 链接:")

if st.button("开始转录"):
    if url_input:
        with st.status("正在处理中...") as status:
            # 执行下载
            audio_path, error = download_with_oauth(url_input)
            
            if error:
                st.error(f"下载失败: {error}")
                st.info("💡 如果提示需要授权，请检查 Streamlit 的侧边栏日志或控制台输出。")
            else:
                status.update(label="音频下载成功！正在 AI 转录...", state="running")
                
                # 开始转录
                segments, info = model.transcribe(audio_path, beam_size=5)
                
                result_text = ""
                for segment in segments:
                    line = f"[{int(segment.start)//60:02d}:{int(segment.start)%60:02d}] {segment.text}\n"
                    result_text += line
                    st.write(line)
                
                st.success("转录完成！")
                st.download_button("下载文本", result_text, file_name="output.txt")
                
                # 清理
                if os.path.exists(audio_path):
                    os.remove(audio_path)
    else:
        st.warning("请输入链接")

# --- 4. 关键点：如何在 Streamlit Cloud 完成授权 ---
with st.sidebar:
    st.header("🔑 首次运行授权指南")
    st.markdown("""
    1. 点击 **'开始转录'** 按钮。
    2. 观察 Streamlit 右下角的 **'Manage App' -> 'Logs'** (日志窗口)。
    3. 你会看到一行文字：  
       `Please open https://www.google.com/device and input code XXXX-XXXX`
    4. 在你的浏览器打开该链接，输入代码，选择你的 Google 账号授权。
    5. 授权完成后，程序会自动继续运行。
    """)
