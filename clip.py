import yt_dlp
import os
import json
import re
import time
import sys
import cv2
import glob
import subprocess
import shutil
import warnings
import datetime
import whisper
import google.generativeai as genai
from colorama import init, Fore, Style

# --- CONFIG & SETUP ---
init(autoreset=True)
warnings.filterwarnings("ignore")

if not shutil.which("ffmpeg"):
    print(Fore.RED + "[FATAL] FFmpeg tidak ditemukan!" + Style.RESET_ALL)
    sys.exit()

API_KEY_FILE = "gemini_key.txt"
MODEL_NAME = "models/gemini-2.5-flash"
TEMP_AUDIO_PREFIX = "temp_analysis"

# ==========================================
# KONFIGURASI ESTETIKA (KARAOKE STYLE)
# ==========================================
# Warna dalam format ASS (BGR)
COLOR_ACTIVE = "&H00FFFF" # Kuning (Saat kata diucapkan)
COLOR_INACTIVE = "&HFFFFFF" # Putih (Kata lainnya)
OUTLINE_COLOR = "&H000000" 

FONT_NAME = "Arial"
FONT_SIZE = "36"
BORDER_WIDTH = "2"
WORDS_PER_LINE = 5 # Jumlah kata per baris agar terlihat "Utuh"

# ==========================================
# 1. UTILS & ASS GENERATOR
# ==========================================
def format_time_ass(seconds):
    """Format waktu untuk ASS: H:MM:SS.cs (centiseconds)"""
    x = datetime.timedelta(seconds=float(seconds))
    hours = int(x.seconds // 3600)
    minutes = int((x.seconds % 3600) // 60)
    secs = int(x.seconds % 60)
    # ASS butuh centiseconds (2 digit), bukan millis
    centis = int((seconds - int(seconds)) * 100) 
    return f"{hours}:{minutes:02}:{secs:02}.{centis:02}"

def generate_karaoke_ass(audio_path, ass_path, margin_v):
    """
    Membuat subtitle ASS dengan efek Highlight (Karaoke).
    """
    print(f"   {Fore.CYAN}[Whisper] Membuat Karaoke Highlight Subtitle...{Style.RESET_ALL}")
    
    try:
        model = whisper.load_model("small")
        # Word timestamps wajib True untuk efek ini
        result = model.transcribe(audio_path, language="id", word_timestamps=True)
        
        # HEADER ASS (Style Definition)
        header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{FONT_NAME},{FONT_SIZE},{COLOR_INACTIVE},&H000000FF,{OUTLINE_COLOR},&H00000000,-1,0,1,{BORDER_WIDTH},0,2,10,10,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        
        all_words = []
        for segment in result["segments"]:
            for word in segment["words"]:
                text = word["word"].strip().upper()
                if text and text not in [".", ",", "?", "!"]:
                    all_words.append({
                        "text": text,
                        "start": word["start"],
                        "end": word["end"]
                    })

        with open(ass_path, "w", encoding="utf-8") as f:
            f.write(header)
            
            # LOGIKA GROUPING (Agar text utuh per baris)
            for i in range(0, len(all_words), WORDS_PER_LINE):
                chunk = all_words[i:i + WORDS_PER_LINE]
                if not chunk: continue
                
                # Waktu total baris ini muncul
                line_start = chunk[0]["start"]
                line_end = chunk[-1]["end"]
                
                # Untuk setiap kata dalam chunk, kita buat event highlight
                for j, active_word in enumerate(chunk):
                    # Waktu highlight kata ini
                    w_start = active_word["start"]
                    w_end = active_word["end"]
                    
                    # Bangun kalimat utuh dengan pewarnaan
                    formatted_line = ""
                    for k, w in enumerate(chunk):
                        word_txt = w["text"]
                        if k == j:
                            # KATA INI SEDANG DIUCAPKAN -> WARNA AKTIF (KUNING)
                            formatted_line += f"{{\\c{COLOR_ACTIVE}}}{word_txt}{{\\c{COLOR_INACTIVE}}} "
                        else:
                            # KATA LAIN -> WARNA BIASA (PUTIH)
                            formatted_line += f"{word_txt} "
                    
                    # Tulis Event ASS
                    # Start/End pake waktu kata agar warnanya pindah pas kata selesai
                    f.write(f"Dialogue: 0,{format_time_ass(w_start)},{format_time_ass(w_end)},Default,,0,0,0,,{formatted_line.strip()}\n")

        return True
    except Exception as e:
        print(f"{Fore.RED}   [Error Whisper] {e}{Style.RESET_ALL}")
        return False

def find_audio_file(prefix):
    for file in os.listdir("."):
        if file.startswith(prefix) and file.endswith(('.mp3', '.m4a', '.webm', '.opus', '.wav')):
            return file
    return None

def cleanup_files(patterns):
    for p in patterns:
        for f in glob.glob(p):
            try: os.remove(f)
            except: pass

def sanitize_filename(name):
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    name = name.encode('ascii', 'ignore').decode('ascii')
    return name.strip().replace(" ", "_")

def parse_seconds(time_str):
    try:
        parts = list(map(int, time_str.split(':')))
        if len(parts) == 3: return parts[0]*3600 + parts[1]*60 + parts[2]
        elif len(parts) == 2: return parts[0]*60 + parts[1]
        else: return parts[0]
    except: return 0

def clean_json(text):
    text = re.sub(r"```json", "", text)
    text = re.sub(r"```", "", text)
    return text.strip()

def setup_gemini():
    api_key = ""
    if os.path.exists(API_KEY_FILE):
        with open(API_KEY_FILE, "r") as f: api_key = f.read().strip()
    if not api_key:
        api_key = input("Paste Gemini API Key: ").strip()
        with open(API_KEY_FILE, "w") as f: f.write(api_key)
    genai.configure(api_key=api_key)
    return True

def get_mime_type(filename):
    if filename.endswith(".mp3"): return "audio/mp3"
    if filename.endswith(".m4a"): return "audio/mp4"
    if filename.endswith(".webm"): return "audio/webm"
    return "audio/mp3"

# ==========================================
# 2. DOWNLOADER
# ==========================================
def download_audio_raw(url):
    cleanup_files([f"{TEMP_AUDIO_PREFIX}*"]) 
    opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{TEMP_AUDIO_PREFIX}.%(ext)s',
        'quiet': True, 'no_warnings': True,
        'socket_timeout': 30, 'retries': 10
    }
    print(f"{Fore.YELLOW}[Network] Downloading Audio Source...{Style.RESET_ALL}")
    try:
        with yt_dlp.YoutubeDL(opts) as ydl: ydl.download([url])
        return True
    except Exception as e:
        print(f"{Fore.RED}Error Audio DL: {e}{Style.RESET_ALL}")
        return False

def download_video_segment(url, start, end, output_path):
    if os.path.exists(output_path): os.remove(output_path)
    safe_start = max(0, start - 15) 
    safe_end = end + 15
    opts = {
        'format': 'bestvideo+bestaudio/best',
        'merge_output_format': 'mkv',
        'outtmpl': output_path,
        'quiet': True, 'no_warnings': True,
        'download_ranges': lambda info, ydl: [{'start_time': safe_start, 'end_time': safe_end}],
        'force_keyframes_at_cuts': False,
        'concurrent_fragment_downloads': 4,
    }
    print(f"   {Fore.YELLOW}[Network] Downloading Segment (Buffered)...{Style.RESET_ALL}")
    try:
        with yt_dlp.YoutubeDL(opts) as ydl: ydl.download([url])
        if os.path.exists(output_path) and os.path.getsize(output_path) > 10000:
            return True, safe_start
        return False, 0
    except Exception as e:
        print(f"{Fore.RED}   [Fail] {e}{Style.RESET_ALL}")
        return False, 0

# ==========================================
# 3. SMART PROCESSING
# ==========================================
def get_video_info(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened(): return 0, 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h

def detect_face_info(video_path):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    check_points = [total//5, total//4, total//2, (total*3)//4, (total*4)//5]
    avg_center_x = []
    max_bottom_y = 0
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    for pt in check_points:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pt)
        ret, frame = cap.read()
        if not ret: continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
            avg_center_x.append(x + (w // 2))
            current_bottom = y + h
            if current_bottom > max_bottom_y: max_bottom_y = current_bottom
    cap.release()
    if not avg_center_x: return None
    return {'center_x': int(sum(avg_center_x) / len(avg_center_x)), 'bottom_y': max_bottom_y}

def process_clip_final(clip_data, url):
    title = sanitize_filename(clip_data['title']) or f"clip_{int(time.time())}"
    start = parse_seconds(clip_data['start'])
    end = parse_seconds(clip_data['end'])
    target_duration = end - start
    
    if not os.path.exists(title): os.makedirs(title)
    print(f"\n{Fore.MAGENTA}>>> Processing: {title}{Style.RESET_ALL}")

    raw_path = f"{title}/raw.mkv"
    final_path = f"{title}/final_karaoke.mp4"
    audio_export_path = f"{title}/audio_clean.wav"
    ass_path = f"{title}/subs.ass" # Format ASS
    temp_ass_root = "temp_subs_karaoke.ass"

    # 1. Download
    success, buffer_start = download_video_segment(url, start, end, raw_path)
    if not success: return

    seek_time = start - buffer_start
    if seek_time < 0: seek_time = 0

    # 2. Extract Audio
    print(f"   {Fore.CYAN}[Audio] Extracting Audio...{Style.RESET_ALL}")
    subprocess.run([
        "ffmpeg", "-y", "-ss", str(seek_time), "-i", raw_path, "-t", str(target_duration),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", 
        "-loglevel", "error", audio_export_path
    ])

    # 3. SMART LAYOUT (Hitung Margin Dulu)
    orig_w, orig_h = get_video_info(raw_path)
    target_w = int(orig_h * (9/16))
    if target_w % 2 != 0: target_w -= 1
    
    face_info = detect_face_info(raw_path)
    final_margin_v = "95" 

    if face_info:
        print(f"   {Fore.GREEN}[Smart-Layout] Wajah terdeteksi.{Style.RESET_ALL}")
        chin_y = face_info['bottom_y']
        # Scaling Margin karena ASS pakai resolusi referensi 1080x1920
        # Jika video asli 4K atau 720p, perbandingannya harus pas.
        # Kita asumsikan output akan 1080p, jadi kita mapping margin ke 1080p.
        scale_factor = 1920 / orig_h
        scaled_chin_y = chin_y * scale_factor
        
        calculated_margin = 1920 - (scaled_chin_y + 80) # +80px buffer
        final_margin_v = str(max(95, int(calculated_margin)))
        print(f"   {Fore.GREEN}[Smart-Layout] Margin Subtitle set ke: {final_margin_v}{Style.RESET_ALL}")

    # 4. Generate Karaoke ASS (Dengan Margin Dinamis)
    if not generate_karaoke_ass(audio_export_path, ass_path, final_margin_v):
        print(Fore.RED + "   [Skip] Gagal membuat subtitle.")
        return

    shutil.copy(ass_path, temp_ass_root)

    # 5. FFmpeg Burning
    crop_x = (orig_w // 2) - (target_w // 2)
    if face_info: crop_x = face_info['center_x'] - (target_w // 2)
    
    if crop_x < 0: crop_x = 0
    if (crop_x + target_w) > orig_w: crop_x = orig_w - target_w

    print(f"   {Fore.CYAN}[FFmpeg] Burning Karaoke Subtitle...{Style.RESET_ALL}")

    # Perhatikan filter: subtitles=file.ass (Tanpa force_style karena style ada di dalam file ASS)
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(seek_time),
        "-i", raw_path,
        "-t", str(target_duration),
        "-vf", f"crop={target_w}:{orig_h}:{crop_x}:0,format=yuv420p,subtitles={temp_ass_root}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-r", "30",
        "-map", "0:v:0", "-map", "0:a:0",
        "-c:a", "aac", "-b:a", "192k", "-ac", "2",
        "-loglevel", "error", "-stats",
        final_path
    ]

    try:
        subprocess.run(cmd, check=True)
        
        with open(f"{title}/caption.txt", "w", encoding="utf-8") as f:
            f.write(clip_data.get('caption', ''))
            f.write(f"\n\nHOOK STRATEGY:\n{clip_data.get('hook_reason', '-')}")
            
        print(f"{Fore.GREEN}   [SUCCESS] Video: {final_path}{Style.RESET_ALL}")
        
        if os.path.exists(raw_path): os.remove(raw_path)
        if os.path.exists(audio_export_path): os.remove(audio_export_path)
        if os.path.exists(temp_ass_root): os.remove(temp_ass_root)

    except Exception as e:
        print(f"{Fore.RED}   [Error FFmpeg] {e}{Style.RESET_ALL}")

# ==========================================
# 4. MAIN FLOW
# ==========================================
def main():
    if not setup_gemini(): return
    os.system('cls' if os.name == 'nt' else 'clear')

    print(f"{Fore.CYAN}{'='*60}")
    print(f"  YOUTUBE CLIPPER V22 (KARAOKE HIGHLIGHT EFFECT)")
    print(f"{'='*60}{Style.RESET_ALL}")
    
    url = input("Masukkan URL: ")

    if not download_audio_raw(url): return
    audio_path = find_audio_file(TEMP_AUDIO_PREFIX)
    if not audio_path: return print(f"{Fore.RED}Audio Error.{Style.RESET_ALL}")
    
    mime = get_mime_type(audio_path)
    print(f"{Fore.BLUE}[AI]{Style.RESET_ALL} Mencari Clip Viral...")

    model = genai.GenerativeModel(MODEL_NAME)
    prompt = """
    Bertindaklah sebagai Senior Video Editor & TikTok Strategist yang ahli dalam "Retention Editing".

    Tugas Anda adalah menganalisis konten video dari link URL yang saya berikan di bawah.
    Tujuan: Mengidentifikasi 3 Klip "GOLDEN MOMENT" yang berpotensi viral untuk TikTok/Reels/Shorts.

    KRITERIA KLIP (Wajib Patuh):
    1. **Durasi:** 90 - 180 detik (Long-form TikTok).
    2. **Context Closure:** Titik POTONG (Start & End) harus presisi. Jangan memotong saat narasumber sedang menarik napas, di tengah kalimat, atau saat gagasan belum selesai. Pastikan klip memiliki "Awal yang memancing" dan "Akhir yang tuntas/konklusif".
    3. **Substansi:** Klip harus berdiri sendiri (stand-alone) tanpa penonton perlu melihat video full untuk paham.
    Output JSON Only:
    [ { "title": "Judul_Viral", "start": "MM:SS", "end": "MM:SS", "caption": "...", "hook_reason": "..." } ]
    """
    
    try:
        audio_file = genai.upload_file(audio_path, mime_type=mime)
        while audio_file.state.name == "PROCESSING": 
            print(".", end="", flush=True); time.sleep(1)
        
        if audio_file.state.name == "FAILED": return print("AI Failed.")

        resp = model.generate_content([audio_file, prompt])
        clips = json.loads(clean_json(resp.text))
        valid = [c for c in clips if (parse_seconds(c['end']) - parse_seconds(c['start'])) >= 60]
        
        print(f"\n{Fore.GREEN}[AI] Menemukan {len(valid)} Clip.{Style.RESET_ALL}")
        audio_file.delete()
        cleanup_files([f"{TEMP_AUDIO_PREFIX}*"]) 

        if not valid: return print("Tidak ada clip > 60s.")

        for clip in valid:
            process_clip_final(clip, url)

        print(f"\n{Fore.GREEN}=== SELESAI ==={Style.RESET_ALL}")

    except Exception as e:
        print(f"{Fore.RED}Error Utama: {e}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()