import os
import uuid
import requests
import json
import base64
import datetime
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from moviepy.editor import VideoFileClip
import asyncio
import aiohttp
import openai
import time

# ----- Playwright 관련 임포트 -----
from playwright.async_api import async_playwright

# ------------------------------------------------
# FastAPI 설정 (Swagger, 인증키 등)
# ------------------------------------------------
SWAGGER_HEADERS = {
    "title": "LINKBRICKS HORIZON-AI Video Frame Analysis API ENGINE",
    "version": "100.100.100",
    "description": (
        "## 비디오 프레임 분석 API 엔진 \n"
        "- API Swagger \n"
        "- 비디오에서 프레임을 추출하고 Linkbricks Horizon-Ai로 분석 \n"
        "- MP4, MOV, AVI, MKV, WMV, FLV, OGG, WebM \n"
        "- YOUTUBE, TIKTOK 지원"
    ),
    "contact": {
        "name": "Linkbricks Horizon AI",
        "url": "https://www.linkbricks.com",
        "email": "contact@linkbricks.com",
        "license_info": {
            "name": "GNU GPL 3.0",
            "url": "https://www.gnu.org/licenses/gpl-3.0.html",
        },
    },
}

app = FastAPI(**SWAGGER_HEADERS)

# 인증 키
REQUIRED_AUTH_KEY = "linkbricks-saxoji-benedict-ji-01034726435!@#$%231%$#@%"

# 파일을 저장할 디렉토리
VIDEO_DIR = "video"
os.makedirs(VIDEO_DIR, exist_ok=True)

# ------------------------------------------------
# Pydantic 모델
# ------------------------------------------------
class VideoFrameAnalysisRequest(BaseModel):
    api_key: str
    auth_key: str
    video_url: str  # (유튜브/TikTok/인스타 등)
    seconds_per_frame: int = None  # 프레임 간격(초) → interval 방식에 사용
    downloader_api_key: str  # 동영상 다운로드 API 키
    extraction_type: str  # "interval" 또는 "keyframe"


# ------------------------------------------------
# URL 판별/정규화
# ------------------------------------------------
def is_youtube_url(url: str) -> bool:
    return "youtube.com" in url or "youtu.be" in url

def is_tiktok_url(url: str) -> bool:
    return "tiktok.com" in url

def is_instagram_url(url: str) -> bool:
    return "instagram.com/reel/" in url or "instagram.com/p/" in url

def normalize_youtube_url(video_url: str) -> str:
    if "youtu.be" in video_url:
        video_id = video_url.split('/')[-1].split('?')[0]
        return f"https://www.youtube.com/watch?v={video_id}"
    if "youtube.com/embed" in video_url:
        video_id = video_url.split('/')[-1].split('?')[0]
        return f"https://www.youtube.com/watch?v={video_id}"
    if "youtube.com/shorts" in video_url:
        video_id = video_url.split('/')[-1].split('?')[0]
        return f"https://www.youtube.com/watch?v={video_id}"
    if "youtube.com/watch" in video_url:
        return video_url.split('&')[0]
    raise ValueError("유효하지 않은 유튜브 URL 형식입니다.")

def normalize_instagram_url(video_url: str) -> str:
    if "/reel/" in video_url:
        video_id = video_url.split("/reel/")[-1].split("/")[0]
        return f"https://www.instagram.com/p/{video_id}/"
    return video_url

# ------------------------------------------------
# Playwright를 이용한 (의사) 스트리밍 다운로드
#  - 실제로는 전체 body를 한번에 받아서 쓰므로, 대용량이면 메모리 부담
# ------------------------------------------------
async def playwright_stream_download(file_url: str, local_file_path: str):
    """
    Playwright의 'route'를 통해 해당 요청을 가로채
    body를 전부 읽은 뒤 chunk 단위로 파일에 기록합니다.
    
    ※ 현재 Python Playwright는 JS의 ReadableStream처럼
      정말 '조각단위'로 바로 쓰는 방식을 공식 제공하지 않습니다.
    ※ body() 전체를 한 번에 받아 메모리에 로드한 뒤 파일에 쓰게 됩니다.
      (대용량에선 메모리 사용량이 커지므로 주의)
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()

        # 1) route 핸들러 등록
        async def handle_route(route):
            # 다운로드할 URL과 정확히 일치할 때만 이 로직 적용
            if route.request.url == file_url:
                resp = await route.fetch()
                if not resp.ok:
                    # 에러 상태면 abort
                    await route.abort()
                    return
                # body() → 전체 바이트 (메모리)
                content = await resp.body()

                # (가짜) chunk 로직: 여기서는 content를 한번에 받고, 파이썬에서 조금씩 쓰는 식
                chunk_size = 1024 * 64  # 64KB씩 쓴다고 가정
                with open(local_file_path, 'wb') as f:
                    for i in range(0, len(content), chunk_size):
                        f.write(content[i:i+chunk_size])

                # route.fulfill() → 응답을 브라우저에도 반환(필요 없으면 생략 가능)
                await route.fulfill(response=resp)
            else:
                # 타겟 URL이 아니면 그냥 통과
                await route.continue_()

        await context.route("**/*", handle_route)

        # 2) 실제 페이지 열기
        page = await context.new_page()
        await page.goto(file_url)  # 동영상 URL 접속 (혹은 어떤 페이지든)

        # 필요하면 약간 대기 (혹은 networkidle 등 대기)
        await page.wait_for_timeout(3000)

        await browser.close()

# ------------------------------------------------
# 동영상 다운로드 함수 (유튜브=Playwright, 틱톡/인스타=기존 로직)
#   - async 로 선언 (FastAPI의 이벤트 루프에 그대로 연결)
# ------------------------------------------------
async def download_video(video_url: str, downloader_api_key: str) -> (str, str):
    """
    :return: (video_file_path, caption_or_None)
    """
    # -------------------------
    # 1) YouTube
    # -------------------------
    if is_youtube_url(video_url):
        # 새 API 호출
        api_url = f"https://zylalabs.com/api/5789/video+downloader+api/7526/download+media?url={video_url}"
        api_headers = {"Authorization": f'Bearer {downloader_api_key}'}

        resp = requests.get(api_url, headers=api_headers)
        if resp.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail=f"동영상 정보 API 실패: {resp.status_code}, {resp.text}"
            )

        data = resp.json()
        video_links = data.get('links', [])
        if not video_links:
            raise HTTPException(status_code=500, detail="동영상 링크가 없습니다.")

        # 최고 해상도 mp4 링크 찾기
        highest_resolution = 0
        highest_mp4_url = None
        for link_info in video_links:
            container = link_info.get('container', '')
            mime_type = link_info.get('mimeType', '')
            if ('mp4' in container) or ('video/mp4' in mime_type):
                width = link_info.get('width', 0)
                height = link_info.get('height', 0)
                resolution = width * height
                if resolution > highest_resolution:
                    highest_resolution = resolution
                    highest_mp4_url = link_info.get('link')

        if not highest_mp4_url:
            raise HTTPException(status_code=500, detail="mp4 다운로드 링크를 찾을 수 없습니다.")

        # Playwright로 (가짜) 스트리밍 다운로드
        video_file = os.path.join(VIDEO_DIR, f"{uuid.uuid4()}.mp4")
        try:
            await playwright_stream_download(highest_mp4_url, video_file)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Playwright 다운로드 오류: {e}")

        return (video_file, None)

    # -------------------------
    # 2) TikTok → 기존 로직
    # -------------------------
    elif is_tiktok_url(video_url):
        api_url = "https://zylalabs.com/api/4640/tiktok+download+connector+api/5719/download+video"
        headers = {"Authorization": f'Bearer {downloader_api_key}'}

        try:
            r = requests.get(f"{api_url}?url={video_url}", headers=headers)
            if r.status_code != 200:
                raise HTTPException(500, "TikTok API 실패")
        except requests.exceptions.RequestException as e:
            raise HTTPException(500, f"TikTok API 요청 중 오류: {e}")

        data = r.json()
        download_url = data.get('download_url')
        if not download_url:
            raise HTTPException(500, "TikTok 다운로드 URL 없음")

        # 기존 requests + stream
        vid_resp = requests.get(download_url, stream=True, timeout=30)
        if vid_resp.status_code != 200:
            raise HTTPException(500, "TikTok 동영상 다운로드 실패")

        video_file = os.path.join(VIDEO_DIR, f"{uuid.uuid4()}.mp4")
        with open(video_file, 'wb') as f:
            for chunk in vid_resp.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        return (video_file, None)

    # -------------------------
    # 3) Instagram → 기존 로직
    # -------------------------
    elif is_instagram_url(video_url):
        norm_url = normalize_instagram_url(video_url)
        api_url = f"https://zylalabs.com/api/1943/instagram+reels+downloader+api/2944/reel+downloader?url={norm_url}"
        headers = {"Authorization": f'Bearer {downloader_api_key}'}

        r = requests.get(api_url, headers=headers)
        if r.status_code != 200:
            raise HTTPException(500, "인스타그램 API 실패")

        data = r.json()
        reel_video_url = data.get("video")
        caption = data.get("caption", "")

        if not reel_video_url:
            raise HTTPException(500, "인스타 mp4 없음")

        # 일반 다운로드
        r2 = requests.get(reel_video_url, stream=True)
        if r2.status_code != 200:
            raise HTTPException(500, "인스타 동영상 다운로드 실패")

        video_file = os.path.join(VIDEO_DIR, f"{uuid.uuid4()}.mp4")
        with open(video_file, 'wb') as f:
            for chunk in r2.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

        return (video_file, caption)

    # -------------------------
    # 4) 일반 웹 동영상
    # -------------------------
    else:
        r = requests.get(video_url, stream=True)
        if r.status_code != 200:
            raise HTTPException(500, "일반 동영상 다운로드 실패")

        ext = video_url.split('.')[-1].lower()
        video_file = os.path.join(VIDEO_DIR, f"{uuid.uuid4()}.{ext}")

        with open(video_file, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

        return (video_file, None)


# ------------------------------------------------
# 비디오 프레임 추출 (interval/keyframe)
# ------------------------------------------------
def seconds_to_timecode(seconds: int) -> str:
    return str(datetime.timedelta(seconds=seconds))

def extract_keyframes_from_video(video_file: str, seconds_per_frame: int, threshold: float = 0.6):
    frames = []
    timecodes = []

    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps == 0.0:
        fps = 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames = int(fps * seconds_per_frame)
    curr_frame = 0

    success, prev_frame = cap.read()
    if not success:
        raise ValueError("동영상을 읽지 못했습니다.")

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    frames.append(base64.b64encode(cv2.imencode('.jpg', prev_frame)[1]).decode('utf-8'))
    timecodes.append(seconds_to_timecode(0))

    while curr_frame < total_frames - 1:
        curr_frame += skip_frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        success, curr_data = cap.read()
        if not success:
            break

        curr_gray = cv2.cvtColor(curr_data, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(prev_gray, curr_gray)
        non_zero_count = np.count_nonzero(frame_diff)
        diff_ratio = non_zero_count / frame_diff.size

        if diff_ratio > threshold:
            frames.append(base64.b64encode(cv2.imencode('.jpg', curr_data)[1]).decode('utf-8'))
            timecodes.append(seconds_to_timecode(int(curr_frame / fps)))
            prev_gray = curr_gray

    cap.release()

    # 최대 250 프레임 제한
    if len(frames) > 250:
        step = len(frames) // 250
        frames = frames[::step][:250]
        timecodes = timecodes[::step][:250]

    return frames, timecodes

def extract_frames_from_video(video_file: str, seconds_per_frame: int):
    frames = []
    timecodes = []

    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps == 0.0:
        fps = 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames = int(fps * seconds_per_frame)
    curr_frame = 0

    while curr_frame < total_frames - 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        success, frame = cap.read()
        if not success:
            break

        _, buffer = cv2.imencode('.jpg', frame)
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        frames.append(frame_b64)
        timecodes.append(seconds_to_timecode(int(curr_frame / fps)))

        curr_frame += skip_frames

    cap.release()

    # 최대 250 프레임 제한
    if len(frames) > 250:
        step = len(frames) // 250
        frames = frames[::step][:250]
        timecodes = timecodes[::step][:250]

    return frames, timecodes

# ------------------------------------------------
# GPT-4o API 분석/요약
# ------------------------------------------------
async def analyze_frames_with_gpt4(api_key: str, frames: List[str], timecodes: List[str]) -> List[str]:
    content_list = [{"type": "text", "text": "다음은 비디오에서 추출한 이미지들입니다. 각 이미지의 타임코드는 다음과 같습니다. 각 이미지에 대해 무엇이 보이는지 설명해주세요."}]
    for i, f_b64 in enumerate(frames):
        content_list.append({"type": "text", "text": f"타임코드: {timecodes[i]}"})
        content_list.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{f_b64}",
                "detail": "high"
            }
        })

    messages = [{"role": "user", "content": content_list}]

    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.post(
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
                    err_text = await resp.text()
                    raise HTTPException(resp.status, f"OpenAI API 에러: {err_text}")
                data = await resp.json()
                description = data['choices'][0]['message']['content']
                return description.strip().split('\n')
    except Exception as e:
        raise HTTPException(500, f"이미지 분석 중 오류: {e}")

async def summarize_descriptions(api_key: str, descriptions: List[str]) -> str:
    messages = [
        {"role": "system", "content": "당신은 비디오 콘텐츠를 요약하는 도우미입니다."},
        {"role": "user", "content": "다음은 각 시간대별 비디오에서 추출된 프레임 이미지 설명입니다. 이를 기반으로 비디오 전체 내용을 요약해주세요:\n" + "\n".join(descriptions)}
    ]

    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.post(
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
                    err_text = await resp.text()
                    raise HTTPException(resp.status, f"OpenAI 요약 에러: {err_text}")
                data = await resp.json()
                return data['choices'][0]['message']['content']
    except Exception as e:
        raise HTTPException(500, f"요약 중 오류: {e}")


# ------------------------------------------------
# 메인 엔드포인트
# ------------------------------------------------
@app.post("/process_video_frames/")
async def process_video_frames(request: VideoFrameAnalysisRequest):
    # 1) 인증키 검사
    if request.auth_key != REQUIRED_AUTH_KEY:
        raise HTTPException(status_code=403, detail="유효하지 않은 인증 키입니다.")

    # 2) URL 정규화 (유튜브/인스타)
    try:
        if is_youtube_url(request.video_url):
            normalized_url = normalize_youtube_url(request.video_url)
        elif is_instagram_url(request.video_url):
            normalized_url = normalize_instagram_url(request.video_url)
        else:
            normalized_url = request.video_url
    except ValueError as e:
        raise HTTPException(400, f"잘못된 URL 형식: {e}")

    # 3) 동영상 다운로드 (Playwright or 기존 로직)
    try:
        video_file, caption = await download_video(normalized_url, request.downloader_api_key)
    except Exception as e:
        raise HTTPException(500, f"동영상 다운로드 오류: {e}")

    # 4) 프레임 추출 (interval or keyframe)
    try:
        if request.extraction_type == "interval":
            if request.seconds_per_frame is None:
                raise HTTPException(400, "interval 방식을 위해 seconds_per_frame이 필요합니다.")
            frames_b64, timecodes = extract_frames_from_video(video_file, request.seconds_per_frame)
        elif request.extraction_type == "keyframe":
            if request.seconds_per_frame is None:
                raise HTTPException(400, "keyframe 방식을 위해 seconds_per_frame이 필요합니다.")
            frames_b64, timecodes = extract_keyframes_from_video(video_file, request.seconds_per_frame)
        else:
            raise HTTPException(400, "extraction_type은 'interval' 또는 'keyframe'이어야 합니다.")
    except Exception as e:
        if os.path.exists(video_file):
            os.remove(video_file)
        raise HTTPException(500, f"프레임 추출 중 오류: {e}")

    # 5) (선택) 동영상 파일 삭제
    if os.path.exists(video_file):
        os.remove(video_file)

    # 6) 프레임 AI 분석 & 요약
    try:
        analyzed_descriptions = await analyze_frames_with_gpt4(request.api_key, frames_b64, timecodes)
        summary_text = await summarize_descriptions(request.api_key, analyzed_descriptions)

        if caption:
            summary_text = f"[caption]: {caption}\n" + summary_text
    except Exception as e:
        raise HTTPException(500, f"프레임 분석/요약 중 오류: {e}")
    finally:
        # 메모리상에서 프레임 제거
        del frames_b64

    return {
        "analysis": "\n".join(analyzed_descriptions),
        "summary": summary_text
    }

# ------------------------------------------------
# 메인 실행 (uvicorn)
# ------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
