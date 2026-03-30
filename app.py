import streamlit as st
import streamlit.components.v1 as components

# 页面配置
st.set_page_config(layout="wide", page_title="Whisper Browser-Side")

st.title("🎙️ 浏览器本地语音转录 (兼容版)")
st.markdown("""
通过 **Transformers.js** 在您的浏览器内运行 Whisper 模型，无需上传音频到服务器。
* **技术栈**: WebGPU/WASM + ONNX Runtime Web
* **注意**: 首次运行需下载约 75MB 的模型权重。
""")

# HTML/JavaScript 代码块
# 使用 @xenova/transformers v2.17.2，这是一个经过广泛测试的稳定版本
html_code = """
<div id="container" style="font-family: sans-serif; color: #31333F; padding: 20px; border: 1px solid #ddd; border-radius: 10px;">
    <div id="status" style="margin-bottom: 15px; font-weight: bold; color: #FF4B4B;">状态: 等待初始化...</div>
    
    <div id="progress-container" style="display: none; width: 100%; background: #eee; border-radius: 5px; margin-bottom: 15px;">
        <div id="progress-bar" style="width: 0%; height: 10px; background: #4CAF50; border-radius: 5px; transition: width 0.3s;"></div>
    </div>

    <button id="run-btn" style="padding: 10px 20px; background: #008CBA; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 16px;">
        加载模型并转录测试音频
    </button>

    <div style="margin-top: 20px;">
        <strong>转录结果:</strong>
        <pre id="output" style="white-space: pre-wrap; background: #f9f9f9; padding: 15px; border-radius: 5px; border: 1px solid #ccc; min-height: 80px; margin-top: 10px; color: #333;">等待结果...</pre>
    </div>
</div>

<script type="module">
    // 1. 导入浏览器专用 ESM 模块
    import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2';

    // 2. 环境配置：禁用 Node.js 相关检查
    env.allowLocalModels = false; 
    env.useBrowserCache = true;

    const runBtn = document.getElementById('run-btn');
    const status = document.getElementById('status');
    const output = document.getElementById('output');
    const progContainer = document.getElementById('progress-container');
    const progBar = document.getElementById('progress-bar');

    runBtn.onclick = async () => {
        try {
            runBtn.disabled = true;
            status.innerText = "状态: 正在下载模型 (首次运行较慢)...";
            progContainer.style.display = 'block';

            // 3. 创建转录流水线 (使用 tiny 模型以保证兼容性)
            const transcriber = await pipeline('automatic-speech-recognition', 'Xenova/whisper-tiny', {
                progress_callback: (data) => {
                    if (data.status === 'progress') {
                        progBar.style.width = data.progress + '%';
                    }
                }
            });

            status.innerText = "状态: 正在识别音频...";
            
            // 4. 执行转录
            const testAudioUrl = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav';
            const result = await transcriber(testAudioUrl, {
                chunk_length_s: 30,
                stride_length_s: 5,
                language: 'english'
            });

            // 5. 显示结果
            output.innerText = result.text;
            status.innerText = "状态: ✅ 转录完成！";
            progContainer.style.display = 'none';

        } catch (err) {
            console.error("Whisper Error:", err);
            status.innerText = "状态: ❌ 错误 (请检查控制台)";
            output.innerText = "错误详情:\\n" + err.message;
        } finally {
            runBtn.disabled = false;
        }
    };
</script>
"""

# 在 Streamlit 中渲染嵌入式 HTML
# 设置足够的高度以显示进度条和文本
components.html(html_code, height=450, scrolling=True)

st.sidebar.header("排错指南")
st.sidebar.info("""
1. **F12 控制台**: 如果点击没反应，请按 F12 查看 Console 报错。
2. **网络问题**: 如果模型下载卡住，可能是无法连接到 Hugging Face。
3. **内存**: `whisper-base` 需要更多内存，若当前 `tiny` 运行成功，可尝试手动修改脚本中的模型名称。
""")
