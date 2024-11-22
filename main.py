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

# Swagger 검사 설정
SWAGGER_HEADERS = {
    "title": "LINKBRICKS HORIZON-AI Video Frame Analysis API ENGINE",
    "version": "100.100.100",
    "description": (
        "## 비디오 프레임 분석 API 엔진 \n"
        "- API Swagger \n"
        "- 비디오에서 프레임을 출시하고 Linkbricks Horizon-Ai로 분석 \n"
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
    video_url: str  # 동영상 URL (유튜브 또는 틱톡 링크 포함)
    seconds_per_frame: int = None  # 프레임 출시 간격(초), interval 방식에서 사용
    downloader_api_key: str  # 동영상 다운로드를 위한 API 키
    extraction_type: str  # "interval" 또는 "keyframe"

# 유튜브 URL인지 확인하는 함수
def is_youtube_url(url: str) -> bool:
    return "youtube.com" in url or "youtu.be" in url

# 틱톡 URL인지 확인하는 함수
def is_tiktok_url(url: str) -> bool:
    return "tiktok.com" in url

# 인스타그램 URL인지 확인하는 함수
def is_instagram_url(url: str) -> bool:
    return "instagram.com/reel/" in url or "instagram.com/p/" in url

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
    
    # youtube.com/shorts 형식 처리
    if "youtube.com/shorts" in video_url:
        video_id = video_url.split('/')[-1].split('?')[0]
        return f"https://www.youtube.com/watch?v={video_id}"
    
    # youtube.com/watch 형식 (이미 표준화된 URL)
    if "youtube.com/watch" in video_url:
        return video_url.split('&')[0]  # 추가 쿼리 매개변수 제거
    
    # 예상치 못한 형식 처리
    raise ValueError("유효하지 않은 유튜브 URL 형식입니다.")

# 인스타그램 URL을 표준 형식으로 변환하는 함수
def normalize_instagram_url(video_url: str) -> str:
    if "/reel/" in video_url:
        video_id = video_url.split("/reel/")[-1].split("/")[0]
        return f"https://www.instagram.com/p/{video_id}/"
    return video_url

# URL로부터 동영상을 다운로드하는 함수
def download_video(video_url: str, downloader_api_key: str) -> str:
    if is_youtube_url(video_url):
        # 유튜브 동영상 처리
        video_id = video_url.split('v=')[-1] if 'v=' in video_url else video_url.split('/')[-1]
        api_url = f"https://zylalabs.com/api/3219/youtube+mp4+video+downloader+api/6812/youtube+downloader?videoId={video_id}"
        api_headers = {
            'Authorization': f'Bearer {downloader_api_key}'
        }

        response = requests.get(api_url, headers=api_headers)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="API로부터 동영상 정보를 가져오는 데 실패했습니다.")

        data = response.json()

        # 'videos' -> 'items' 리스트에서 'url' 추출
        video_items = data.get('videos', {}).get('items', [])
        if not video_items:
            raise HTTPException(status_code=500, detail="동영상 정보를 찾을 수 없습니다.")

        # 가능한 최고 화질의 동영상 URL 선택
        highest_resolution = 0
        highest_mp4_url = None

        for item in video_items:
            if item.get('mimeType', '').startswith('video/mp4'):
                width = item.get('width', 0)
                height = item.get('height', 0)
                resolution = width * height
                if resolution > highest_resolution:
                    highest_resolution = resolution
                    highest_mp4_url = item.get('url')

        if highest_mp4_url:
            video_response = requests.get(highest_mp4_url, stream=True)
            if video_response.status_code != 200:
                raise HTTPException(status_code=500, detail="동영상을 다운로드하는 데 실패했습니다.")

            video_file = os.path.join(VIDEO_DIR, f"{uuid.uuid4()}.mp4")
            with open(video_file, 'wb') as file:
                for chunk in video_response.iter_content(chunk_size=1024):
                    if chunk:
                        file.write(chunk)
            return video_file, None
        else:
            raise HTTPException(status_code=500, detail="적절한 MP4 파일을 찾을 수 없습니다.")

    elif is_tiktok_url(video_url):
        # TikTok 동영상 처리
        api_url = "https://zylalabs.com/api/4481/tiktok+video+retriever+api/5499/video+download"
        api_headers = {
            'Authorization': f'Bearer {downloader_api_key}'
        }
        
        # API 호출
        response = requests.get(f"{api_url}?url={video_url}", headers=api_headers)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="API로부터 TikTok 동영상 정보를 가져오는 데 실패했습니다.")
        
        # API 응답 파싱
        data = response.json()
        print("API Response:", json.dumps(data, indent=4))  # 응답 디버깅
        
        # 'data' 키에서 정보 가져오기
        response_data = data.get('data')
        if not response_data:
            raise HTTPException(status_code=500, detail="TikTok API 응답에서 데이터를 찾을 수 없습니다.")
        
        # 다운로드 URL 확인
        video_play_url = response_data.get('play')  # 워터마크 없는 URL
        if not video_play_url:
            video_play_url = response_data.get('wmplay')  # 워터마크 있는 URL
            if not video_play_url:
                raise HTTPException(status_code=500, detail="TikTok 동영상 URL을 찾을 수 없습니다.")
        
        print(f"Downloading from URL: {video_play_url}")
        
        # 세션 생성 및 설정
        session = requests.Session()
        
        # 향상된 헤더 설정
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'video/webm,video/mp4,video/*;q=0.9,application/ogg;q=0.7,audio/*;q=0.6,*/*;q=0.5',
            'Accept-Language': 'en-US,en;q=0.5',
            'Range': 'bytes=0-',
            'Connection': 'keep-alive',
            'Referer': 'https://www.tiktok.com/',
            'Sec-Fetch-Dest': 'video',
            'Sec-Fetch-Mode': 'no-cors',
            'Sec-Fetch-Site': 'cross-site',
            'Pragma': 'no-cache',
            'Cache-Control': 'no-cache',
        }
        
        # 캐시 우회를 위한 랜덤 쿼리 파라미터 추가
        if '?' in video_play_url:
            video_play_url += f'&_r={uuid.uuid4().hex}'
        else:
            video_play_url += f'?_r={uuid.uuid4().hex}'
        
        try:
            # HEAD 요청으로 URL 유효성 확인
            head_response = session.head(
                video_play_url, 
                headers=headers, 
                allow_redirects=True, 
                timeout=30
            )
            
            if head_response.status_code == 403:
                # 403 에러 시 워터마크 버전 시도
                video_play_url = response_data.get('wmplay', '')
                if not video_play_url:
                    raise HTTPException(status_code=500, detail="모든 다운로드 URL에 접근이 거부되었습니다.")
            
            # 실제 비디오 다운로드
            video_response = session.get(
                video_play_url,
                headers=headers,
                stream=True,
                timeout=30,
                allow_redirects=True
            )
            
            if video_response.status_code != 200:
                print(f"Failed to download video. Status code: {video_response.status_code}")
                raise HTTPException(status_code=500, detail=f"다운로드 실패. 상태 코드: {video_response.status_code}")
            
            # 동영상 파일 저장
            video_file = os.path.join(VIDEO_DIR, f"{uuid.uuid4()}.mp4")
            print(f"Saving video to: {video_file}")
            
            with open(video_file, 'wb') as file:
                for chunk in video_response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
            
            return video_file, None
            
        except requests.exceptions.RequestException as e:
            print(f"Download error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"다운로드 중 오류 발생: {str(e)}")
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"예상치 못한 오류 발생: {str(e)}")
        finally:
            session.close()


    elif is_instagram_url(video_url):
        # 인스타그램 동영상 처리
        normalized_url = normalize_instagram_url(video_url)
        api_url = f"https://zylalabs.com/api/1943/instagram+reels+downloader+api/2944/reel+downloader?url={normalized_url}"
        headers = {
            'Authorization': f'Bearer {downloader_api_key}'
        }

        response = requests.get(api_url, headers=headers)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="API로부터 동영상 정보를 가져오는 데 실패했습니다.")

        data = response.json()
        video_url = data.get("video")
        caption = data.get("caption", "")

        if not video_url:
            raise HTTPException(status_code=500, detail="적절한 MP4 파일을 찾을 수 없습니다.")

        video_response = requests.get(video_url, stream=True)
        video_file = os.path.join(VIDEO_DIR, f"{uuid.uuid4()}.mp4")
        with open(video_file, 'wb') as file:
            for chunk in video_response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)

        return video_file, caption

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

    return video_file, None

# 초를 hh:mm:ss 형식으로 변환하는 함수
def seconds_to_timecode(seconds: int) -> str:
    return str(datetime.timedelta(seconds=seconds))

# 키 프레임을 추출하는 함수
def extract_keyframes_from_video(video_file: str, seconds_per_frame: int, threshold: float = 0.6):
    frames = []
    timecodes = []

    video = cv2.VideoCapture(video_file)
    fps = video.get(cv2.CAP_PROP_FPS)
    if not fps or fps == 0.0:
        fps = 25  # 기본 FPS 설정
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_skip = int(fps * seconds_per_frame)
    curr_frame = 0

    success, prev_frame = video.read()
    if not success:
        raise ValueError("동영상을 읽는 데 실패했습니다.")

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    frames.append(base64.b64encode(cv2.imencode('.jpg', prev_frame)[1]).decode('utf-8'))
    timecodes.append(seconds_to_timecode(0))

    while curr_frame < total_frames - 1:
        curr_frame += frames_to_skip
        video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        success, curr_frame_data = video.read()
        if not success:
            break

        curr_gray = cv2.cvtColor(curr_frame_data, cv2.COLOR_BGR2GRAY)

        # 이전 프레임과 현재 프레임 간 차이를 계산 (구조적 유사도 또는 절대 차이)
        frame_diff = cv2.absdiff(prev_gray, curr_gray)
        non_zero_count = np.count_nonzero(frame_diff)
        diff_ratio = non_zero_count / frame_diff.size

        # 변화 비율이 임계값을 초과하면 키 프레임으로 선택
        if diff_ratio > threshold:
            frames.append(base64.b64encode(cv2.imencode('.jpg', curr_frame_data)[1]).decode('utf-8'))
            timecodes.append(seconds_to_timecode(int(curr_frame / fps)))
            prev_gray = curr_gray  # 현재 프레임을 이전 프레임으로 업데이트

    video.release()

    # 최대 250개의 프레임만 유지
    if len(frames) > 250:
        step = len(frames) // 250
        frames = frames[::step][:250]
        timecodes = timecodes[::step][:250]

    return frames, timecodes

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

    # 최대 250개의 프레임만 유지
    if len(frames) > 250:
        step = len(frames) // 250
        frames = frames[::step][:250]
        timecodes = timecodes[::step][:250]

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
                "detail": "high"  # 필요에 따라 "high"로 변경 가능
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
        elif is_instagram_url(request.video_url):
            normalized_video_url = normalize_instagram_url(request.video_url)
        else:
            normalized_video_url = request.video_url

        video_file, caption = download_video(normalized_video_url, request.downloader_api_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"동영상 다운로드 중 오류 발생: {str(e)}")

    try:
        # 프레임 출시 방식에 따라 분기 처리
        if request.extraction_type == "interval":
            if request.seconds_per_frame is None:
                raise HTTPException(status_code=400, detail="interval 방식에서는 seconds_per_frame 값이 필요합니다.")
            frames_base64, timecodes = extract_frames_from_video(video_file, request.seconds_per_frame)
        elif request.extraction_type == "keyframe":
            if request.seconds_per_frame is None:
                raise HTTPException(status_code=400, detail="keyframe 방식에서는 seconds_per_frame 값이 필요합니다.")
            frames_base64, timecodes = extract_keyframes_from_video(video_file, request.seconds_per_frame)
        else:
            raise HTTPException(status_code=400, detail="유효하지 않은 extraction_type 값입니다. 'interval' 또는 'keyframe'이어야 합니다.")
    except Exception as e:
        # 동영상 파일 삭제
        if os.path.exists(video_file):
            os.remove(video_file)
        raise HTTPException(status_code=500, detail=f"프레임 출시 중 오류 발생: {str(e)}")

    try:
        # 동영상 파일 삭제
        if os.path.exists(video_file):
            os.remove(video_file)

        analyzed_descriptions = await analyze_frames_with_gpt4(request.api_key, frames_base64, timecodes)
        summary_text = await summarize_descriptions(request.api_key, analyzed_descriptions)

        # Instagram 서적 추가
        if caption:
            summary_text = f"[caption]: {caption}\n" + summary_text

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
