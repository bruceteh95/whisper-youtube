import streamlit as st
import requests
import os
from faster_whisper import WhisperModel

st.set_page_config(page_title="全自动 Whisper 转录", page_icon="🚀")

@st.cache_resource
def load_model():
    return WhisperModel("base", device="cpu", compute_type="int8")

model = load_model()

def download_audio_via_api(youtube_url):
    """使用公共 Cobalt 实例获取音频下载地址"""
    # 这是一个开源的媒体提取 API
    api_url = "https://api.cobalt.tools/api/json"
    
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
    
    payload = {
        "url": youtube_url,
        "isAudioOnly": True,
        "audioFormat": "mp3",
        "vQuality": "720", # 虽是音频但需传参
    }

    try:
        response = requests.post(api_url, json=payload, headers=headers)
        data = response.json()
        
        if data.get("status") == "stream":
            download_url = data.get("url")
            # 真正从流地址下载文件到本地
            audio_data = requests.get(download_url).content
            with open("temp_audio.mp3", "wb") as f:
                f.write(audio_data)
            return "temp_audio.mp3", None
        else:
            return None, f"API 报错: {data.get('text', '未知错误')}"
    except Exception as e:
        return None, f"请求失败: {str(e)}"

st.title("🚀 极速 YouTube 转录 (无需授权)")

url = st.text_input("在此输入 YouTube 链接:")

if st.button("开始提取"):
    if url:
        with st.status("正在绕过限制提取音频...") as status:
            file_path, err = download_audio_via_api(url)
            
            if err:
                st.error(err)
            else:
                status.update(label="音频获取成功！开始 AI 识别...", state="running")
                segments, info = model.transcribe(file_path, beam_size=5)
                
                final_text = ""
                for segment in segments:
                    line = f"[{int(segment.start)//60:02d}:{int(segment.start)%60:02d}] {segment.text}\n"
                    final_text += line
                    st.write(line)
                
                st.success("转录完成！")
                st.download_button("保存结果", final_text)
                os.remove(file_path)
    else:
        st.warning("请输入链接")
