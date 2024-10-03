import os
import uuid
import requests
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import datetime
import cv2
import base64
from moviepy.editor import VideoFileClip
import asyncio
import aiohttp  # 추가된 임포트
import openai  # 필요한 경우 제거 가능

# Swagger 헤더 설정
SWAGGER_HEADERS = {
    "title": "LINKBRICKS HORIZON-AI Video Frame Analysis API ENGINE",
    "version": "100.100.100",
    "description": (
        "## 비디오 프레임 분석 엔진 \n"
        "- API Swagger \n"
        "- 비디오에서 프레임을 추출하고 Linkbricks Horizon-Ai로 분석 \n"
        "- MP4, MOV, AVI, MKV, WMV, FLV, OGG, WebM \n"
        "- YOUTUBE 지원"
    ),
    "contact": {
        "name": "Linkbricks Horizon AI",
        "url": "https://www.linkbricks.com",
        "email": "contact@linkbricks.com",
        "license_info": {
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT",
        },
    },
}

app = FastAPI(**SWAGGER_HEADERS)

# 인증 키 설정
REQUIRED_AUTH_KEY = "linkbricks-saxoji-benedict-ji-01034726435!@#$%231%$#@%"

# 파일을 저장할 디렉토리 설정
VIDEO_DIR = "video"
if not os.path.exists(VIDEO_DIR):
    os.makedirs(VIDEO_DIR)

# 모델 정의
class VideoFrameAnalysisRequest(BaseModel):
    api_key: str
    auth_key: str
    video_url: str  # 동영상 URL (유튜브 링크 포함)
    seconds_per_frame: int  # 프레임 추출 간격(초)
    downloader_api_key: str  # 유튜브 동영상 다운로드를 위한 API 키

# 유튜브 URL인지 확인하는 함수
def is_youtube_url(url: str) -> bool:
    return "youtube.com" in url or "youtu.be" in url

# 유튜브 URL을 표준 형식으로 변환하는 함수
def normalize_youtube_url(video_url: str) -> str:
    # youtu.be 형식 처리
    if "youtu.be" in video_url:
        video_id = video_url.split('/')[-1].split('?')[0]
        return f"https://www.youtube.com/watch?v={video_id}"
    
    # youtube.com/embed 형식 처리
    if "youtube.com/embed" in video_url:
        video_id = video_url.split('/')[-1].split('?')[0]
        return f"https://www.youtube.com/watch?v={video_id}"
    
    # youtube.com/watch 형식 (이미 표준화된 URL)
    if "youtube.com/watch" in video_url:
        return video_url.split('&')[0]  # 추가 쿼리 매개변수 제거
    
    # 예상치 못한 형식 처리
    raise ValueError("유효하지 않은 유튜브 URL 형식입니다.")

# URL로부터 동영상을 다운로드하는 함수
def download_video(video_url: str, downloader_api_key: str) -> str:
    if is_youtube_url(video_url):
        # 유튜브 동영상 처리
        api_url = "https://zylalabs.com/api/3219/youtube+mp4+video+downloader+api/5880/get+mp4"
        api_headers = {
            'Authorization': f'Bearer {downloader_api_key}'
        }

        response = requests.get(f"{api_url}?id={video_url.split('v=')[-1]}", headers=api_headers)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="API로부터 동영상 정보를 가져오는 데 실패했습니다.")

        data = response.json()

        smallest_resolution = None
        smallest_mp4_url = None

        for format in data.get('formats', []):
            if format.get('mimeType', '').startswith('video/mp4'):
                width = format.get('width')
                height = format.get('height')
                if width and height:
                    if smallest_resolution is None or (width * height) < (smallest_resolution[0] * smallest_resolution[1]):
                        smallest_resolution = (width, height)
                        smallest_mp4_url = format.get('url')

        if smallest_mp4_url:
            video_response = requests.get(smallest_mp4_url, stream=True)
            video_file = os.path.join(VIDEO_DIR, f"{uuid.uuid4()}.mp4")
            with open(video_file, 'wb') as file:
                for chunk in video_response.iter_content(chunk_size=1024):
                    if chunk:
                        file.write(chunk)
        else:
            raise HTTPException(status_code=500, detail="적절한 MP4 파일을 찾을 수 없습니다.")

    else:
        # 일반 웹 동영상 파일 처리
        video_response = requests.get(video_url, stream=True)
        if video_response.status_code != 200:
            raise HTTPException(status_code=500, detail="제공된 URL에서 동영상 파일을 다운로드하는 데 실패했습니다.")
        
        # 원본 확장자로 파일 저장
        video_file_extension = video_url.split('.')[-1]
        video_file = os.path.join(VIDEO_DIR, f"{uuid.uuid4()}.{video_file_extension}")
        
        with open(video_file, 'wb') as file:
            for chunk in video_response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)

    return video_file

# 초를 hh:mm:ss 형식으로 변환하는 함수
def seconds_to_timecode(seconds: int) -> str:
    return str(datetime.timedelta(seconds=seconds))

# 지정된 간격으로 동영상에서 프레임을 추출하는 함수
def extract_frames_from_video(video_file: str, seconds_per_frame: int):
    frames = []
    timecodes = []

    video = cv2.VideoCapture(video_file)
    fps = video.get(cv2.CAP_PROP_FPS)
    if not fps or fps == 0.0:
        fps = 25  # 기본 FPS 설정
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    frames_to_skip = int(fps * seconds_per_frame)
    curr_frame = 0

    while curr_frame < total_frames - 1:
        video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        success, frame = video.read()
        if not success:
            break
        # 프레임을 JPEG로 인코딩하여 base64로 변환
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        frames.append(frame_base64)
        timecode_seconds = curr_frame / fps
        timecodes.append(seconds_to_timecode(int(timecode_seconds)))
        curr_frame += frames_to_skip

    video.release()
    return frames, timecodes

# 여러 이미지를 GPT-4o API로 분석하는 함수
async def analyze_frames_with_gpt4(api_key: str, frames: List[str], timecodes: List[str]) -> List[str]:
    # 메시지 내용 구성
    content_list = []
    content_list.append({"type": "text", "text": "다음은 비디오에서 추출한 이미지들입니다. 각 이미지의 타임코드는 다음과 같습니다. 각 이미지에 대해 무엇이 보이는지 설명해주세요."})

    for i, frame_base64 in enumerate(frames):
        # 타임코드 추가
        content_list.append({"type": "text", "text": f"타임코드: {timecodes[i]}"})
        # 이미지 추가
        content_list.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{frame_base64}",
                "detail": "low"  # 필요에 따라 "high"로 변경 가능
            }
        })

    # 메시지 구성
    messages = [
        {
            "role": "user",
            "content": content_list
        }
    ]

    try:
        # API 요청
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4o",
                    "messages": messages,
                    "max_tokens": 2000
                }
            ) as resp:
                if resp.status != 200:
                    error_detail = await resp.text()
                    raise HTTPException(status_code=resp.status, detail=f"OpenAI API 에러: {error_detail}")
                result = await resp.json()
                description = result['choices'][0]['message']['content']
                # 응답을 분석하여 각 타임코드별 설명을 리스트로 반환
                analyzed_descriptions = description.strip().split('\n')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이미지 분석 중 오류 발생: {str(e)}")

    return analyzed_descriptions

# 전체 분석 결과를 요약하는 함수
async def summarize_descriptions(api_key: str, descriptions: List[str]) -> str:
    messages = [
        {"role": "system", "content": "당신은 비디오 콘텐츠를 요약하는 도우미입니다."},
        {"role": "user", "content": "다음은 각 시간대별 비디오에서 추출된 프레임 이미지 설명입니다. 이를 기반으로 비디오의 전체적인 내용을 요약해주세요:\n" + "\n".join(descriptions)}
    ]

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4o",
                    "messages": messages,
                    "max_tokens": 2000
                }
            ) as resp:
                if resp.status != 200:
                    error_detail = await resp.text()
                    raise HTTPException(status_code=resp.status, detail=f"OpenAI API 에러: {error_detail}")
                result = await resp.json()
                summary_text = result['choices'][0]['message']['content']
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"요약 중 오류 발생: {str(e)}")

    return summary_text

@app.post("/process_video_frames/")
async def process_video_frames(request: VideoFrameAnalysisRequest):
    if request.auth_key != REQUIRED_AUTH_KEY:
        raise HTTPException(status_code=403, detail="유효하지 않은 인증 키입니다.")

    try:
        # 유튜브 URL을 표준화 처리
        if is_youtube_url(request.video_url):
            normalized_video_url = normalize_youtube_url(request.video_url)
        else:
            normalized_video_url = request.video_url

        video_file = download_video(normalized_video_url, request.downloader_api_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"동영상 다운로드 중 오류 발생: {str(e)}")

    try:
        frames_base64, timecodes = extract_frames_from_video(video_file, request.seconds_per_frame)
    except Exception as e:
        # 동영상 파일 삭제
        if os.path.exists(video_file):
            os.remove(video_file)
        raise HTTPException(status_code=500, detail=f"프레임 추출 중 오류 발생: {str(e)}")

    try:
        # 동영상 파일 삭제
        if os.path.exists(video_file):
            os.remove(video_file)

        analyzed_descriptions = await analyze_frames_with_gpt4(request.api_key, frames_base64, timecodes)
        summary_text = await summarize_descriptions(request.api_key, analyzed_descriptions)
    except Exception as e:
        # 필요한 경우 추가로 정리 작업 수행
        raise HTTPException(status_code=500, detail=f"프레임 분석 중 오류 발생: {str(e)}")
    finally:
        # 모든 프레임 데이터 삭제 (메모리에서)
        del frames_base64

    return {"analysis": "\n".join(analyzed_descriptions), "summary": summary_text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
