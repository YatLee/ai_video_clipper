import os
import time
import shutil
import subprocess
import zipfile
import whisper
import json
from openai import OpenAI
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from typing import List

app = FastAPI()

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. 初始化 Whisper 模型 (修正：必须在全局加载) ---
print("正在加载 Whisper 模型...")
whisper_model = whisper.load_model("base") 

# --- 2. 配置大模型 ---
client = OpenAI(
    api_key="******", 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

def real_ai_analysis(video_path):
    # A. 语音转文字
    print(f"AI 正在识别语音: {video_path}")
    result = whisper_model.transcribe(video_path)
    full_text_with_time = ""
    for segment in result['segments']:
        full_text_with_time += f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['text']}\n"
    print(result["text"])
    # B. 调用大模型
    print("AI 正在分析营销逻辑...")
    prompt = f"""
    PROMPT
    """

    response = client.chat.completions.create(
        model="qwen-max", # 修正：请确认模型名称是否为 qwen-max
        messages=[{"role": "user", "content": prompt}],
        # 阿里云某些模式下可能不支持 json_object，如报错可去掉下一行
        response_format={ 'type': 'json_object' } 
    )
    
    content = response.choices[0].message.content
    analysis = json.loads(content)
    
    if isinstance(analysis, dict) and "segments" in analysis:
        return analysis["segments"]
    elif isinstance(analysis, dict) and "clips" in analysis: # 增加一些常见的兼容性
        return analysis["clips"]
    
    return analysis if isinstance(analysis, list) else []
    return ""

def cut_video_clip(input_path, start_sec, end_sec, output_filename):
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    # 稍微增加一点缓冲时间，让声音更完整
    start_sec = max(0, start_sec - 0.3)
    duration = end_sec - start_sec + 0.5
    
    command = [
        'ffmpeg', '-y',
        '-ss', str(start_sec),
        '-t', str(duration),
        '-i', input_path,
        '-c:v', 'libx264',
        '-c:a', 'aac',
        '-strict', 'experimental',
        output_path
    ]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_path

@app.post("/api/upload-and-analyze")
async def upload_and_analyze(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
            
    # 修正：传递真实的本地路径 file_path 而不是 file.filename
    try:
        analysis_result = real_ai_analysis(file_path)
        
        processed_clips = []
        for i, clip in enumerate(analysis_result):
            safe_category = clip["category"].replace("/", "_") # 防止分类名包含非法字符
            clip_name = f"clip_{i}_{safe_category}_{file.filename}"
            
            print(f"正在生成切片: {clip_name}")
            try:
                cut_video_clip(file_path, clip["start_sec"], clip["end_sec"], clip_name)
                clip["clip_filename"] = clip_name
                processed_clips.append(clip)
            except Exception as e:
                print(f"单个切割失败: {e}")

        return {
            "status": "success",
            "original_filename": file.filename,
            "ai_clips": processed_clips
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


# 接口 1: 下载单个片段
@app.get("/api/download/single/{filename}")
async def download_single(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type='video/mp4', filename=filename)
    raise HTTPException(status_code=404, detail="文件不存在")

# 接口 2: 打包下载所有片段
@app.post("/api/download/all")
async def download_all(filenames: List[str]):
    zip_name = f"bundle_{int(time.time())}.zip"
    zip_path = os.path.join(OUTPUT_DIR, zip_name)
    
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for fname in filenames:
            fpath = os.path.join(OUTPUT_DIR, fname)
            if os.path.exists(fpath):
                zipf.write(fpath, arcname=fname)
    
    return FileResponse(zip_path, media_type='application/zip', filename=zip_name)