import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(layout="wide", page_title="Browser-based Whisper")

st.title("🚀 浏览器端 Whisper 转录 (Transformers.js)")
st.markdown("""
这个应用直接在你的**浏览器本地**运行模型，音频数据不会上传到服务器。
* **模型**: `onnx-community/whisper-base`
* **加速**: 自动尝试 WebGPU
""")

# 定义嵌入的 HTML/JS 代码
html_code = """
<div id="container" style="font-family: sans-serif; color: #31333F;">
    <button id="load-btn" style="padding: 10px 20px; background: #FF4B4B; color: white; border: none; border-radius: 5px; cursor: pointer;">
        加载模型并转录测试音频
    </button>
    <div id="status" style="margin-top: 15px; font-weight: bold;">等待操作...</div>
    <pre id="output" style="white-space: pre-wrap; background: #f0f2f6; padding: 15px; border-radius: 5px; margin-top: 10px; min-height: 100px;">转录结果将显示在这里...</pre>
</div>

<script type="module">
    // 引入 Transformers.js
    import { pipeline } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0-alpha.0';

    const loadBtn = document.getElementById('load-btn');
    const status = document.getElementById('status');
    const output = document.getElementById('output');

    loadBtn.onclick = async () => {
        try {
            loadBtn.disabled = true;
            status.innerText = "⏳ 正在加载模型 (约 200MB+)...";
            
            // 初始化 ASR 流水线
            const transcriber = await pipeline('automatic-speech-recognition', 'onnx-community/whisper-base', {
                device: 'webgpu', // 优先使用 WebGPU 加速
            });

            status.innerText = "🎙️ 正在转录测试音频...";
            
            // 测试音频 URL
            const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav';
            
            const result = await transcriber(url, { 
                chunk_length_s: 30, 
                stride_length_s: 5 
            });

            output.innerText = result.text;
            status.innerText = "✅ 转录完成！";
        } catch (err) {
            status.innerText = "❌ 错误: " + err.message;
            console.error(err);
        } finally {
            loadBtn.disabled = false;
        }
    };
</script>
"""

# 在 Streamlit 中渲染组件
components.html(html_code, height=400)

st.info("注意：首次运行需要下载模型，请保持网络畅通。模型会被缓存到浏览器的 IndexedDB 中。")
