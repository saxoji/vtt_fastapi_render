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

# ==== Playwright & Stealth ====
from playwright.async_api import async_playwright
from playwright_stealth import stealth_async

# ------------------------------------------------
# FastAPI 애플리케이션 설정
# ------------------------------------------------
app = FastAPI(
    title="LINKBRICKS HORIZON-AI Video Frame Analysis API ENGINE",
    version="100.100.100",
    description=(
        "## 비디오 프레임 분석 API 엔진 \n"
        "- API Swagger \n"
        "- 비디오에서 프레임을 추출하고 Linkbricks Horizon-Ai로 분석 \n"
        "- MP4, MOV, AVI, MKV, WMV, FLV, OGG, WebM \n"
        "- YOUTUBE, TIKTOK, INSTAGRAM 지원"
    ),
    contact={
        "name": "Linkbricks Horizon AI",
        "url": "https://www.linkbricks.com",
        "email": "contact@linkbricks.com",
        "license_info": {
            "name": "GNU GPL 3.0",
            "url": "https://www.gnu.org/licenses/gpl-3.0.html",
        },
    },
)

# ------------------------------------------------
# 전역 설정
# ------------------------------------------------
REQUIRED_AUTH_KEY = "linkbricks-saxoji-benedict-ji-01034726435!@#$%231%$#@%"
VIDEO_DIR = "video"
os.makedirs(VIDEO_DIR, exist_ok=True)

# ------------------------------------------------
# Pydantic 모델
# ------------------------------------------------
class VideoFrameAnalysisRequest(BaseModel):
    api_key: str
    auth_key: str
    video_url: str          # (유튜브, TikTok, 인스타, 일반URL 등)
    seconds_per_frame: int = None  # 프레임 추출 간격(초) - interval 방식에만
    downloader_api_key: str        # zylalabs API 등 동영상 다운로드 키
    extraction_type: str           # "interval" or "keyframe"

# ------------------------------------------------
# URL 판별 및 정규화
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
        return video_url.split('&')[0]  # 쿼리 파라미터 제거
    raise ValueError("유효하지 않은 유튜브 URL 형식입니다.")

def normalize_instagram_url(video_url: str) -> str:
    if "/reel/" in video_url:
        video_id = video_url.split("/reel/")[-1].split("/")[0]
        return f"https://www.instagram.com/p/{video_id}/"
    return video_url

# ------------------------------------------------
# (2)+(3)번 적용: Playwright + Stealth + '미리' 방문 세션 확보
# ------------------------------------------------
async def playwright_stream_download(
    watch_url: str,
    file_url: str,
    local_file_path: str
):
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--disable-features=IsolateOrigins,site-per-process',
                '--disable-site-isolation-trials'
            ]
        )
        
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/121.0.0.0 Safari/537.36"
            ),
            viewport={'width': 1920, 'height': 1080},
            locale="en-US",
            timezone_id="America/New_York",
            geolocation={"latitude": 40.7128, "longitude": -74.0060},
            permissions=['geolocation'],
            extra_http_headers={
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Fetch-User': '?1',
            }
        )
        
        page = await context.new_page()
        await stealth_async(page)
        
        # Set additional JavaScript properties
        await page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => false,
            });
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-US', 'en'],
            });
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5],
            });
        """)
        
        try:
            # Pre-visit with longer timeout and wait for network idle
            await page.goto(watch_url, wait_until="networkidle", timeout=30000)
            await page.wait_for_timeout(5000)  # Additional wait
            
        except Exception as e:
            print(f"[WARN] Pre-visit({watch_url}) 실패: {e}")
        
        async def handle_route(route):
            if route.request.url == file_url:
                headers = {
                    'Referer': 'https://www.youtube.com/',
                    'Origin': 'https://www.youtube.com',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Connection': 'keep-alive',
                    'Accept-Encoding': 'gzip, deflate, br',
                }
                
                resp = await route.fetch(headers=headers)
                if not resp.ok:
                    print(f"Route response not OK: {resp.status}")
                    await route.abort()
                    return
                
                content = await resp.body()
                chunk_size = 1024 * 64
                with open(local_file_path, 'wb') as f:
                    for i in range(0, len(content), chunk_size):
                        f.write(content[i:i+chunk_size])
                
                await route.fulfill(response=resp)
            else:
                await route.continue_()
        
        await context.route("**/*", handle_route)
        
        # Add cookies (optional - add if you have specific cookies)
        # await context.add_cookies([{"name": "CONSENT", "value": "YES+", "domain": ".youtube.com"}])
        
        await page.goto(file_url, wait_until="load", timeout=30000)
        await page.wait_for_timeout(5000)
        
        await browser.close()

# ------------------------------------------------
# 유튜브/틱톡/인스타/일반 URL에 따른 다운로드 처리
# ------------------------------------------------
async def download_video(video_url: str, downloader_api_key: str):
    """
    :return: (video_file_path, caption_if_any)
    """

    # -------------------------
    # 1) YOUTUBE
    # -------------------------
    if is_youtube_url(video_url):
        # (예시) zylalabs.com API 호출
        api_url = f"https://zylalabs.com/api/5789/video+downloader+api/7526/download+media?url={video_url}"
        headers = {"Authorization": f"Bearer {downloader_api_key}"}

        r = requests.get(api_url, headers=headers)
        if r.status_code != 200:
            raise HTTPException(500, f"유튜브 API 요청 실패: {r.status_code}, {r.text}")

        data = r.json()
        video_links = data.get('links', [])
        if not video_links:
            raise HTTPException(500, "유튜브 링크 정보가 없음")

        # 최고 해상도 mp4 찾기
        highest_resolution = 0
        highest_mp4_url = None
        for link_info in video_links:
            container = link_info.get('container', '')
            mime_type = link_info.get('mimeType', '')
            if ('mp4' in container) or ('video/mp4' in mime_type):
                w = link_info.get('width', 0)
                h = link_info.get('height', 0)
                res = w * h
                if res > highest_resolution:
                    highest_resolution = res
                    highest_mp4_url = link_info.get('link')

        if not highest_mp4_url:
            raise HTTPException(500, "유튜브 mp4 링크 없음")

        # (2)+(3)번 방식: 미리 watch_url 방문 → stealth → 다운로드
        watch_url = video_url  # 유튜브 영상 페이지
        video_file = os.path.join(VIDEO_DIR, f"{uuid.uuid4()}.mp4")

        try:
            await playwright_stream_download(
                watch_url=watch_url,
                file_url=highest_mp4_url,
                local_file_path=video_file
            )
        except Exception as e:
            raise HTTPException(500, f"Playwright 다운로드 오류: {e}")

        return (video_file, None)

    # -------------------------
    # 2) TIKTOK (기존 로직)
    # -------------------------
    elif is_tiktok_url(video_url):
        api_url = "https://zylalabs.com/api/4640/tiktok+download+connector+api/5719/download+video"
        headers = {"Authorization": f"Bearer {downloader_api_key}"}

        try:
            resp = requests.get(f"{api_url}?url={video_url}", headers=headers)
            if resp.status_code != 200:
                raise HTTPException(500, "TikTok API 요청 실패")
        except requests.RequestException as e:
            raise HTTPException(500, f"TikTok API 예외: {e}")

        data = resp.json()
        download_url = data.get("download_url")
        if not download_url:
            raise HTTPException(500, "TikTok 다운로드 링크 없음")

        # 기존 requests + stream
        r2 = requests.get(download_url, stream=True, timeout=30)
        if r2.status_code != 200:
            raise HTTPException(500, "TikTok 동영상 다운로드 실패")

        video_file = os.path.join(VIDEO_DIR, f"{uuid.uuid4()}.mp4")
        with open(video_file, 'wb') as f:
            for chunk in r2.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

        return (video_file, None)

    # -------------------------
    # 3) INSTAGRAM (기존 로직)
    # -------------------------
    elif is_instagram_url(video_url):
        norm_url = normalize_instagram_url(video_url)
        api_url = f"https://zylalabs.com/api/1943/instagram+reels+downloader+api/2944/reel+downloader?url={norm_url}"
        headers = {"Authorization": f"Bearer {downloader_api_key}"}

        resp = requests.get(api_url, headers=headers)
        if resp.status_code != 200:
            raise HTTPException(500, "인스타그램 API 요청 실패")

        data = resp.json()
        reel_video_url = data.get("video")
        caption = data.get("caption", "")
        if not reel_video_url:
            raise HTTPException(500, "인스타 MP4 링크 없음")

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
# 비디오 프레임 추출 로직
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
    skip = int(fps * seconds_per_frame)
    curr_frame = 0

    success, prev_frame = cap.read()
    if not success:
        raise ValueError("동영상을 읽지 못했습니다.")

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    frames.append(base64.b64encode(cv2.imencode('.jpg', prev_frame)[1]).decode('utf-8'))
    timecodes.append(seconds_to_timecode(0))

    while curr_frame < total_frames - 1:
        curr_frame += skip
        cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        success, curr_data = cap.read()
        if not success:
            break

        curr_gray = cv2.cvtColor(curr_data, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, curr_gray)
        nz_count = np.count_nonzero(diff)
        diff_ratio = nz_count / diff.size

        if diff_ratio > threshold:
            frames.append(base64.b64encode(cv2.imencode('.jpg', curr_data)[1]).decode('utf-8'))
            timecodes.append(seconds_to_timecode(int(curr_frame / fps)))
            prev_gray = curr_gray

    cap.release()

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
    skip = int(fps * seconds_per_frame)
    curr_frame = 0

    while curr_frame < total_frames - 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        success, fr = cap.read()
        if not success:
            break

        _, buffer = cv2.imencode('.jpg', fr)
        frames.append(base64.b64encode(buffer).decode('utf-8'))
        timecodes.append(seconds_to_timecode(int(curr_frame / fps)))

        curr_frame += skip

    cap.release()

    if len(frames) > 250:
        step = len(frames) // 250
        frames = frames[::step][:250]
        timecodes = timecodes[::step][:250]

    return frames, timecodes

# ------------------------------------------------
# GPT-4o 분석/요약 로직 (예시)
# ------------------------------------------------
async def analyze_frames_with_gpt4(api_key: str, frames: List[str], timecodes: List[str]) -> List[str]:
    content_list = [
        {"type": "text", "text": "다음은 비디오에서 추출한 이미지들입니다. 각 이미지의 타임코드는 다음과 같습니다. 각 이미지에 대해 무엇이 보이는지 설명해주세요."}
    ]
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
                    err_detail = await resp.text()
                    raise HTTPException(resp.status, f"OpenAI API 에러: {err_detail}")
                data = await resp.json()
                desc = data['choices'][0]['message']['content']
                return desc.strip().split('\n')
    except Exception as e:
        raise HTTPException(500, f"이미지 분석 오류: {e}")

async def summarize_descriptions(api_key: str, descriptions: List[str]) -> str:
    messages = [
        {"role": "system", "content": "당신은 비디오 콘텐츠를 요약하는 도우미입니다."},
        {
            "role": "user",
            "content": "다음은 각 시간대별 비디오에서 추출된 프레임 이미지 설명입니다. 이를 기반으로 비디오의 전체 내용을 요약해주세요:\n" + "\n".join(descriptions)
        }
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
                    err_detail = await resp.text()
                    raise HTTPException(resp.status, f"OpenAI 요약 에러: {err_detail}")
                data = await resp.json()
                return data['choices'][0]['message']['content']
    except Exception as e:
        raise HTTPException(500, f"요약 중 오류: {e}")


# ------------------------------------------------
# 메인 API 엔드포인트
# ------------------------------------------------
@app.post("/process_video_frames/")
async def process_video_frames(req: VideoFrameAnalysisRequest):
    # 1) 인증
    if req.auth_key != REQUIRED_AUTH_KEY:
        raise HTTPException(403, "유효하지 않은 인증 키입니다.")

    # 2) URL 정규화
    try:
        if is_youtube_url(req.video_url):
            normalized_url = normalize_youtube_url(req.video_url)
        elif is_instagram_url(req.video_url):
            normalized_url = normalize_instagram_url(req.video_url)
        else:
            normalized_url = req.video_url
    except ValueError as e:
        raise HTTPException(400, f"잘못된 URL 형식: {e}")

    # 3) 동영상 다운로드 (YouTube=Playwright, TikTok/인스타=기존 로직)
    try:
        video_file, caption = await download_video(normalized_url, req.downloader_api_key)
    except Exception as e:
        raise HTTPException(500, f"동영상 다운로드 오류: {e}")

    # 4) 프레임 추출
    try:
        if req.extraction_type == "interval":
            if req.seconds_per_frame is None:
                raise HTTPException(400, "interval 방식을 위해 seconds_per_frame가 필요합니다.")
            frames_b64, timecodes = extract_frames_from_video(video_file, req.seconds_per_frame)
        elif req.extraction_type == "keyframe":
            if req.seconds_per_frame is None:
                raise HTTPException(400, "keyframe 방식을 위해 seconds_per_frame가 필요합니다.")
            frames_b64, timecodes = extract_keyframes_from_video(video_file, req.seconds_per_frame)
        else:
            raise HTTPException(400, "extraction_type은 'interval' 또는 'keyframe'이어야 합니다.")
    except Exception as e:
        if os.path.exists(video_file):
            os.remove(video_file)
        raise HTTPException(500, f"프레임 추출 오류: {e}")

    # 5) 필요 시 동영상 파일 삭제
    if os.path.exists(video_file):
        os.remove(video_file)

    # 6) 프레임 분석 & 요약
    try:
        analyzed_desc = await analyze_frames_with_gpt4(req.api_key, frames_b64, timecodes)
        summary_text = await summarize_descriptions(req.api_key, analyzed_desc)
        if caption:
            summary_text = f"[caption]: {caption}\n{summary_text}"
    except Exception as e:
        raise HTTPException(500, f"프레임 분석/요약 오류: {e}")
    finally:
        del frames_b64

    return {
        "analysis": "\n".join(analyzed_desc),
        "summary": summary_text
    }

# ------------------------------------------------
# 메인 실행
# ------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
