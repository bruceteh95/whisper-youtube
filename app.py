import streamlit as st
import requests
import os
import time
from faster_whisper import WhisperModel

st.set_page_config(page_title="Whisper 转录 (v10 API)", page_icon="⚡")

@st.cache_resource
def load_model():
    # Streamlit Cloud 内存有限，强制使用 base + int8
    return WhisperModel("base", device="cpu", compute_type="int8")

model = load_model()

def download_audio_v10(youtube_url):
    """使用最新的 Cobalt v10 接口"""
    # 使用官方推荐的公共实例或备用实例
    api_url = "https://api.cobalt.tools/" 
    
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    
    # v10 的新参数格式
    payload = {
        "url": youtube_url,
        "videoQuality": "720",
        "audioFormat": "mp3",
        "filenameStyle": "basic",
        "downloadMode": "audio", # 明确指定下载模式为音频
        "youtubeVideoCodec": "h264"
    }

    try:
        # 第一步：向 API 发送处理请求
        response = requests.post(api_url, json=payload, headers=headers)
        result = response.json()
        
        # v10 返回的状态通常是 'tunnel' (直接下载流) 或 'redirect'
        if result.get("status") in ["tunnel", "redirect", "success"]:
            download_url = result.get("url")
            
            # 第二步：从返回的 URL 下载实际文件
            audio_response = requests.get(download_url, stream=True)
            if audio_response.status_code == 200:
                with open("temp_audio.mp3", "wb") as f:
                    for chunk in audio_response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return "temp_audio.mp3", None
            else:
                return None, "无法从提取链接获取文件数据"
        else:
            return None, f"API 错误反馈: {result.get('text', '无法解析视频')}"
            
    except Exception as e:
        return None, f"连接 Cobalt 失败: {str(e)}"

# --- UI 界面 ---
st.title("⚡ 智能 YouTube 转录助手")
st.caption("基于 Cobalt v10 API & Faster-Whisper")

url = st.text_input("输入 YouTube URL:", placeholder="https://www.youtube.com/watch?v=...")

if st.button("开始自动转录"):
    if url:
        with st.status("正在通过 API 提取音频...", expanded=True) as status:
            file_path, err = download_audio_v10(url)
            
            if err:
                st.error(f"提取失败: {err}")
                st.info("💡 如果接口繁忙，请稍后再试，或检查链接是否有效。")
            else:
                status.update(label="音频提取成功！AI 正在识别文本...", state="running")
                
                # 开始 Whisper 转录
                segments, info = model.transcribe(file_path, beam_size=5)
                
                full_content = ""
                for segment in segments:
                    # 格式化时间戳 [分:秒]
                    ts = f"[{int(segment.start)//60:02d}:{int(segment.start)%60:02d}]"
                    line = f"{ts} {segment.text}\n"
                    full_content += line
                    st.write(line)
                
                status.update(label="转录大功告成！", state="complete")
                
                st.divider()
                st.download_button("📥 点击下载转录文本", full_content, file_name="transcript.txt")
                
                # 清理文件
                if os.path.exists(file_path):
                    os.remove(file_path)
    else:
        st.warning("请输入链接")
