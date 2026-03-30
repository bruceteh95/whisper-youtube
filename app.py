import streamlit as st
import streamlit.components.v1 as components
import base64

st.set_page_config(layout="wide", page_title="Whisper MP3 Timeline")

st.title("🎵 MP3 上传识别 (带时间轴)")
st.markdown("""
上传您的 MP3 文件，浏览器将直接在本地进行识别并生成带时间轴的文本。
* **隐私**: 音频文件不会离开您的浏览器。
* **性能**: 建议使用 Chrome/Edge 以获得 WebGPU 加速。
""")

# 1. Streamlit 文件上传器
uploaded_file = st.file_uploader("选择一个 MP3 文件", type=["mp3", "wav", "m4a"])

audio_base64 = ""
if uploaded_file is not None:
    # 将文件读入并转为 Base64，以便传递给前端 JS
    audio_bytes = uploaded_file.read()
    audio_base64 = base64.b64encode(audio_bytes).decode()
    st.audio(audio_bytes, format='audio/mp3')
    st.success("文件已准备就绪，请点击下方按钮开始识别。")

# 2. 前端 HTML/JS 逻辑
html_code = f"""
<div id="container" style="font-family: sans-serif; color: #31333F; padding: 20px; border: 1px solid #ddd; border-radius: 10px; background: #fff;">
    <div id="status" style="margin-bottom: 15px; font-weight: bold; color: #FF4B4B;">状态: 等待文件上传...</div>
    
    <div id="progress-container" style="display: none; width: 100%; background: #eee; border-radius: 5px; margin-bottom: 15px;">
        <div id="progress-bar" style="width: 0%; height: 10px; background: #4CAF50; border-radius: 5px; transition: width 0.3s;"></div>
        <small id="progress-text">正在加载模型...</small>
    </div>

    <button id="run-btn" {"disabled" if not audio_base64 else ""} style="padding: 10px 20px; background: #008CBA; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; opacity: {'0.5' if not audio_base64 else '1'};">
        开始本地识别 (生成时间轴)
    </button>

    <div style="margin-top: 20px;">
        <strong>识别结果 (SRT 格式):</strong>
        <div id="output" style="white-space: pre-wrap; background: #262730; color: #f0f2f6; padding: 15px; border-radius: 5px; min-height: 150px; margin-top: 10px; font-family: monospace; font-size: 14px; overflow-y: auto; max-height: 400px;">等待识别开始...</div>
    </div>
</div>

<script type="module">
    import {{ pipeline, env }} from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2';

    env.allowLocalModels = false;
    env.useBrowserCache = true;

    const runBtn = document.getElementById('run-btn');
    const status = document.getElementById('status');
    const output = document.getElementById('output');
    const progContainer = document.getElementById('progress-container');
    const progBar = document.getElementById('progress-bar');
    const progText = document.getElementById('progress-text');

    // 格式化时间函数 [秒 -> HH:MM:SS,mmm]
    function formatTime(s) {{
        const ms = Math.floor((s % 1) * 1000);
        const secs = Math.floor(s % 60);
        const mins = Math.floor((s / 60) % 60);
        const hrs = Math.floor(s / 3600);
        return `${{hrs.toString().padStart(2, '0')}}:${{mins.toString().padStart(2, '0')}}:${{secs.toString().padStart(2, '0')}},${{ms.toString().padStart(3, '0')}}`;
    }}

    runBtn.onclick = async () => {{
        const audioData = "{audio_base64}";
        if (!audioData) return;

        try {{
            runBtn.disabled = true;
            status.innerText = "状态: 初始化模型中...";
            progContainer.style.display = 'block';

            // 1. 加载模型
            const transcriber = await pipeline('automatic-speech-recognition', 'Xenova/whisper-tiny', {{
                progress_callback: (data) => {{
                    if (data.status === 'progress') {{
                        progBar.style.width = data.progress + '%';
                        progText.innerText = `模型下载中: ${{Math.round(data.progress)}}%`;
                    }}
                }}
            }});

            status.innerText = "状态: 正在转换音频并识别...";
            
            // 2. 将 Base64 转为 Blob URL 以供识别
            const audioUrl = `data:audio/mp3;base64,${{audioData}}`;

            // 3. 执行识别并开启 return_timestamps
            const result = await transcriber(audioUrl, {{
                chunk_length_s: 30,
                stride_length_s: 5,
                return_timestamps: true, // 开启时间轴输出
                language: 'chinese'      // 默认设为中文，您也可以设为 'auto'
            }});

            // 4. 构建带时间轴的显示字符串 (SRT 风格)
            let timelineText = "";
            result.chunks.forEach((chunk, index) => {{
                const start = formatTime(chunk.timestamp[0]);
                const end = formatTime(chunk.timestamp[1] || chunk.timestamp[0] + 2);
                timelineText += `${{index + 1}}\\n${{start}} --> ${{end}}\\n${{chunk.text.trim()}}\\n\\n`;
            }});

            output.innerText = timelineText || result.text;
            status.innerText = "状态: ✅ 识别完成！";
            progContainer.style.display = 'none';

        }} catch (err) {{
            console.error(err);
            status.innerText = "状态: ❌ 错误";
            output.innerText = "错误详情: " + err.message;
        }} finally {{
            runBtn.disabled = false;
        }}
    }};
</script>
"""

# 3. 渲染组件
components.html(html_code, height=700, scrolling=True)

st.info("💡 提示：如果您的电脑有独立显卡，此过程会非常快。识别结果采用标准的 SRT 字幕格式显示。")
