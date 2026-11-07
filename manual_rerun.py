import os
import sys
import json
import time
import hashlib
import zipfile
import subprocess
import asyncio
import requests
import static_ffmpeg
import re
import select
from pathlib import Path
from dotenv import load_dotenv

def clipboard_copy(text: str) -> bool:
    """
    Attempts to copy text to the clipboard using a chain of fallbacks:
    1. copykitten (Rust-backed, no external system dependencies like xclip)
    2. pyperclip (uses xclip/xsel/pbcopy/pbpaste or python GUI frameworks)
    3. platform-specific shell tools via subprocess (pbcopy, clip, xclip, xsel, wl-copy)
    
    Returns True if successfully copied, False otherwise.
    """
    # ── Try copykitten ──
    try:
        import copykitten
        # Use detach=True so the clipboard content persists after the python process exits (only relevant on Linux).
        copykitten.copy(text, detach=True)
        return True
    except Exception as e:
        print(f"[DEBUG] copykitten failed: {e}")
        pass

    # ── Try pyperclip ──
    try:
        import pyperclip
        pyperclip.copy(text)
        return True
    except Exception as e:
        print(f"[DEBUG] pyperclip failed: {e}")
        pass

    # ── Try native subprocess shell utilities ──
    try:
        if sys.platform == "darwin":
            subprocess.run(["pbcopy"], input=text.encode('utf-8'), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        elif sys.platform == "win32":
            subprocess.run(["clip"], input=text.encode('utf-8'), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        elif sys.platform.startswith("linux"):
            for cmd in [
                ["xclip", "-selection", "clipboard"],
                ["xsel", "--clipboard", "--input"],
                ["wl-copy"]
            ]:
                try:
                    subprocess.run(cmd, input=text.encode('utf-8'), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    return True
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue
    except Exception as e:
        print(f"[DEBUG] subprocess failed: {e}")
        pass

    return False

def get_downloads_folder():
    if sys.platform == 'win32':
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r'Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders') as key:
                location = winreg.QueryValueEx(key, '{374DE290-123F-4565-9164-39C4925E467B}')[0]
                location = os.path.expandvars(location)
                return location
        except:
            pass
    return str(Path.home() / "Downloads")

def fetch_downloaded_audio(expected_count=1, start_time=0):
    downloads_dir = get_downloads_folder()
    
    if not os.path.exists(downloads_dir):
        print(f"   [!] Could not locate Downloads folder at {downloads_dir}")
        return []
        
    for attempt in range(3):
        valid_exts = {".mp3", ".wav", ".m4a"}
        recent_files = []
        for f in Path(downloads_dir).iterdir():
            if f.is_file() and f.suffix.lower() in valid_exts:
                if f.stat().st_mtime >= start_time:
                    recent_files.append(f)
                    
        if len(recent_files) >= expected_count:
            recent_files.sort(key=lambda x: x.stat().st_mtime)
            return recent_files[-expected_count:]
            
        if attempt < 2:
            print(f"   [!] Found {len(recent_files)} recent audio files, expected {expected_count}.")
            print("   Please ensure the download is complete.")
            input("   → Press Enter to retry... ")
        else:
            print("   [!] Max retries reached.")
            
    return []

# Load environment variables
load_dotenv()

def get_cross_platform_cache_dir(app_name):
    if sys.platform == "win32":
        base = os.getenv("LOCALAPPDATA") or os.getenv("APPDATA") or os.path.expanduser("~")
        return os.path.join(base, app_name)
    elif sys.platform == "darwin":
        return os.path.expanduser(f"~/Library/Caches/{app_name}")
    else:
        return os.path.expanduser(f"~/.cache/{app_name}")

# Initialize static FFmpeg binaries (adds ffmpeg and ffprobe to PATH)
import static_ffmpeg.run
platform_key = static_ffmpeg.run.get_platform_key()
static_ffmpeg.add_paths(download_dir=os.path.join(get_cross_platform_cache_dir("ffmpeg_cache"), platform_key))

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
NVIDIA_NIM_API_KEY = os.getenv("NVIDIA_NIM_API_KEY")

from openai import OpenAI
from EnvatoDownloader import EnvatoElementsDownloader, BASE_DOWNLOAD_DIR, human_delay

def check_and_install_whisper():
    try:
        import faster_whisper
        return True
    except ImportError:
        pass
    
    print("\n[!] faster-whisper is not installed. Checking system specs...")
    import platform
    print(f"   OS: {platform.system()} {platform.release()} ({platform.machine()})")
    
    # Check RAM if psutil is available
    try:
        import psutil
        total_ram_gb = psutil.virtual_memory().total / (1024**3)
        print(f"   Total RAM: {total_ram_gb:.1f} GB")
        if total_ram_gb < 3.5:
            print("   ⚠️ Warning: System has less than 4GB RAM. Whisper will be unstable. Falling back to old generation flow.")
            return False
    except ImportError:
        print("   Total RAM: Unknown (psutil not installed)")
        
    # Check GPU if torch is available
    try:
        import torch
        if torch.cuda.is_available():
            print(f"   GPU: CUDA available ({torch.cuda.get_device_name(0)})")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("   GPU: Apple MPS available")
        else:
            print("   GPU: None detected. Whisper will run on CPU.")
    except ImportError:
        print("   GPU: Unknown (torch not installed)")

    print("   Installing faster-whisper (this may take a minute)...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "faster-whisper"], check=True)
        print("   ✅ faster-whisper installed successfully!")
        return True
    except Exception as e:
        print(f"   ❌ Failed to install faster-whisper: {e}")
        return False

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Initialize NIM client
nim_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_NIM_API_KEY
)

# Initialize OpenRouter client
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
) if OPENROUTER_API_KEY else None

def get_llm_client(provider):
    if provider == "nvidia": return nim_client
    if provider == "openrouter": return openrouter_client
    return None

def get_script_and_segments(old_script_text, old_segments_data):
    print("Step 1: Topic & Script Generation (Manual Rerun)")
    script_text = ""
    if os.path.exists("CurrentScript.md"):
        with open("CurrentScript.md", "r") as f:
            script_text = f.read().strip()
            
    if not script_text:
        print("CurrentScript.md not found or empty.")
        return None
        
    import re
    # Extract dynamic sub-themes from the script
    dynamic_subthemes = re.findall(r'~([^~]+)~', script_text)
    
    # Clean script text (remove ~) for hashing and LLM processing
    clean_script_text = script_text.replace('~', '')
    
    script_hash = hashlib.md5(clean_script_text.encode('utf-8')).hexdigest()
    
    # Save a backup of the current script text indexed by its hash
    backup_script_path = f"script_{script_hash}.md"
    if not os.path.exists(backup_script_path):
        with open(backup_script_path, "w") as f:
            f.write(script_text)
    
    # Check if we already have valid segments for this exact script
    if os.path.exists("segments.json"):
        try:
            with open("segments.json", "r") as f:
                cache_data = json.load(f)
            
            if cache_data.get("script_hash") == script_hash:
                print("Found cached segments.json for this script. Skipping LLM segment generation.")
                theme_word = cache_data.get("theme_word", "Bio-concrete")
                theme_hex_color = cache_data.get("theme_hex_color", "00FF00")
                return cache_data.get("segments", []), theme_word, theme_hex_color, script_hash, dynamic_subthemes
        except Exception as e:
            print(f"Error checking cache: {e}")
            
    print("Segmenting modified script using previous segments as reference...")
    
    # Dump old segments for prompt
    old_segments_json_str = json.dumps(old_segments_data, indent=2)
    
    # Count sentences to set a minimum segment target for new fragments
    sentence_count = len([s for s in clean_script_text.replace('?', '.').replace('!', '.').split('.') if s.strip()])
    min_segments = sentence_count  # Aim for roughly 1-2 segments per sentence
    
    segment_prompt = f"""You are a B-roll video editor. The script has been slightly modified.
    
    Old Script:
    {old_script_text}
    
    Old Segments JSON:
    {old_segments_json_str}
    
    New Script:
    {clean_script_text}
    
    Your task is to break the NEW script down into distinct visual segments EXACTLY like the old script, but updated for the new text.
    - If a segment's text has NOT changed, keep its `segment_id` and `envato_search_query` EXACTLY the same.
    - If a segment's text is slightly modified, update its `voiceover_text`, but try to keep the same `segment_id` and `envato_search_query`.
    - For any completely NEW or heavily modified fragments, you must follow the STRICT SEGMENTATION RULES below.
    
    === SEGMENTATION RULES FOR NEW FRAGMENTS (MANDATORY) ===
    1. The unit of segmentation is the CLAUSE. Split compound sentences (e.g. joined by "and", "but") if they represent distinct visual actions.
    2. COMBINE FLUFF (CRITICAL): Micro-phrases that are purely rhetorical or conversational MUST be merged with the nearest meaningful sentence. NEVER create a standalone segment for: "And honestly?", "It's about time.", "The result?", "Let me explain.", "Here's the thing.", or any other phrase under 5 words that has no standalone visual meaning.
    3. Target number of segments: ~{min_segments} (approx 1 per sentence on average). Do not over-segment filler, but do not under-segment action sequences.
    4. CTA segments ("Let us know in the comments!", "Subscribe!", "Follow for more!") should use a search query that visualizes ENGAGEMENT, not social media icons. E.g. "audience watching" or "person commenting".

    === SEARCH QUERY RULES FOR NEW FRAGMENTS (MANDATORY) ===
    1. Each NEW envato_search_query MUST be EXACTLY 2 WORDS. Not 1 word, not 3 words. Exactly 2.
    2. The 2 words must be a LITERAL, CONCRETE, VISUAL noun phrase that a stock video site would have footage of.
    3. INHERIT CONTEXT & IGNORE IDIOMS: If a segment is purely figurative (e.g., "it's about time", "elephant in the room", "breath of fresh air") DO NOT search for literal clocks, elephants, or wind. Instead, infer what is actually being discussed (e.g., "drywall renovation", "construction planning") and use that.
    4. AVOID AMBIGUITY (CRITICAL): If a word has multiple meanings (e.g., "application" -> software app vs applying drywall, "network" -> internet vs networking event, "joints" -> smoking/medical vs drywall joints), you MUST use the physical/topical word instead. Never use "application" for a physical process.
    5. NEVER USE GENERIC TERMS: Do NOT use vague terms like "construction site", "social media", "thoughtful worker", "application", or "water". BE SPECIFIC (e.g., "drywall construction", "pouring concrete").
    6. GOOD examples: "mixing plaster", "cracked drywall", "cement truck", "drywall construction"
    7. BAD examples: "water", "cracked joints", "clock", "application", "fast application", "construction site", "social media", "thoughtful worker"
    === FEW-SHOT EXAMPLES ===
    EXAMPLE 1 (Combining micro-phrases with adjacent sentence):
    INPUT: "Dry powder is officially dying on construction sites. And honestly? It's about time."
    WRONG - creates 3 segments, floats micro-phrases alone:
      seg_1: "Dry powder is officially dying on construction sites." → "drywall powder"
      seg_2: "And honestly?" → "construction worker" (WRONG! Standalone micro-phrase!)
      seg_3: "It's about time." → "drywall renovation" (WRONG! Standalone micro-phrase!)
    CORRECT - merges them all:
      seg_1: "Dry powder is officially dying on construction sites. And honestly? It's about time." → "drywall powder"

    EXAMPLE 2 (Construction Theme, idiom handling):
    INPUT: "It's about time we addressed it. Workers have been mixing cement by hand for decades, but the result is cracked joints and toxic dust."
    CORRECT (Combined fluff, contextualized 2-word keywords):
      seg_1: "It's about time we addressed it. Workers have been mixing cement by hand for decades," → "mixing plaster"
      seg_2: "but the result is cracked joints" → "cracked drywall"
      seg_3: "and toxic dust." → "construction dust"

    EXAMPLE 3 (Fitness Theme, idiom handling):
    INPUT: "Don't throw in the towel if you miss a day. Consistency is the name of the game when building muscle."
    CORRECT (Ignored idioms, contextualized):
      seg_1: "Don't throw in the towel if you miss a day." → "exhausted athlete"
      seg_2: "Consistency is the name of the game when building muscle." → "weightlifting exercise"
    WRONG (Literal/ambiguous):
      seg_1: "Don't throw in the towel if you miss a day." → "gym towel" (WRONG! Figurative!)
      seg_2: "Consistency is the name of the game when building muscle." → "playing game" (WRONG! Figurative!)

    === OUTPUT FORMAT ===
    Output ONLY a JSON object:
    {{
      "theme_word": "core material/subject (e.g. drywall, concrete)",
      "theme_hex_color": "FF6600",
      "segments": [
        {{
          "segment_id": "seg_1",
          "voiceover_text": "exact clause text from script",
          "envato_search_query": "two words"
        }}
      ]
    }}
    """
    
    models_to_try = [
        {"provider": "nvidia", "model": "openai/gpt-oss-120b"},
        {"provider": "openrouter", "model": "openai/gpt-oss-120b:free"},
        {"provider": "nvidia", "model": "meta/llama-3.3-70b-instruct"},
        {"provider": "nvidia", "model": "meta/llama-3.1-70b-instruct"},
        {"provider": "nvidia", "model": "nvidia/nemotron-3-super-120b-a12b"},
        {"provider": "openrouter", "model": "google/gemma-4-31b-it:free"},
        {"provider": "openrouter", "model": "nvidia/nemotron-3-ultra-550b-a55b:free"}
    ]
    
    for attempt_idx, attempt_cfg in enumerate(models_to_try):
        try:
            provider = attempt_cfg["provider"]
            model_to_use = attempt_cfg["model"]
            client_to_use = get_llm_client(provider)
            
            if not client_to_use:
                continue
                
            print(f"   Calling {provider.upper()} API with model: {model_to_use}...")
            response = client_to_use.chat.completions.create(
                model=model_to_use,
                messages=[{"role": "user", "content": segment_prompt}],
                max_tokens=4096,
                temperature=0.2,
                timeout=90.0
            )
            
            raw_content = response.choices[0].message.content
            if not raw_content:
                # Reasoning models (e.g. gpt-oss-120b) put output in reasoning_content
                raw_content = getattr(response.choices[0].message, 'reasoning_content', None)
            if not raw_content:
                raise ValueError(f"API returned no content from model {model_to_use}.")
            
            content = raw_content.strip()
            
            # Try to parse JSON. Sometimes LLMs wrap in ```json
            import re
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                extracted = json_match.group(1)
            else:
                start_idx = content.find('{')
                end_idx = content.rfind('}')
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    extracted = content[start_idx:end_idx+1]
                else:
                    extracted = content
                    
            data = json.loads(extracted)
            segments = data.get("segments", [])
            theme_word = data.get("theme_word", "Concept")
            theme_hex_color = data.get("theme_hex_color", "00FFFF")
                
            # Post-LLM validation: enforce 2-word search queries
            for seg in segments:
                query = seg.get("envato_search_query", "").strip()
                words = query.split()
                if len(words) == 1:
                    # Pad single-word queries with a contextual word from the voiceover text
                    voiceover = seg.get("voiceover_text", "")
                    # Find a concrete noun from the voiceover that isn't the query itself
                    candidates = [w.strip('.,!?;:()"\'-') for w in voiceover.split() 
                                  if len(w.strip('.,!?;:()"\'-')) > 3 and w.strip('.,!?;:()"\'-').lower() != query.lower()]
                    if candidates:
                        seg["envato_search_query"] = f"{query} {candidates[0]}"
                    print(f"   ⚠️  Padded 1-word query '{query}' → '{seg['envato_search_query']}'")
                elif len(words) > 2:
                    # Truncate to first 2 words
                    seg["envato_search_query"] = " ".join(words[:2])
                    print(f"   ⚠️  Truncated query '{query}' → '{seg['envato_search_query']}'")
            # Deduplicate: remove segments with repeated voiceover_text (LLM looping)
            seen_voiceovers = set()
            deduped_segments = []
            for seg in segments:
                vt = seg.get("voiceover_text", "").strip().lower()
                if vt not in seen_voiceovers:
                    seen_voiceovers.add(vt)
                    deduped_segments.append(seg)
                else:
                    print(f"   ⚠️  Removed duplicate segment: '{seg.get('voiceover_text', '')[:50]}'")
            segments = deduped_segments

            print(f"   ✅ LLM returned {len(segments)} segments (minimum target was {min_segments})")
                
            # Deterministic Validation: Ensure theme_word is exactly in the script
            if theme_word.lower() not in clean_script_text.lower():
                import string
                # Find all words > 4 chars to pick a meaningful fallback
                valid_words = [w.strip(string.punctuation) for w in clean_script_text.split() if len(w.strip(string.punctuation)) > 4]
                if valid_words:
                    theme_word = max(valid_words, key=len) # Fallback to longest word
                else:
                    theme_word = clean_script_text.split()[0] if clean_script_text else "Concept"
                print(f"⚠️  LLM hallucinated theme word. Falling back to exact substring: '{theme_word}'")
                
            # Save to cache
            with open("segments.json", "w") as f:
                json.dump({
                    "script_hash": script_hash, 
                    "theme_word": theme_word,
                    "theme_hex_color": theme_hex_color,
                    "segments": segments
                }, f, indent=2)
                
            # Also save a backup specific to this hash
            import shutil
            shutil.copy2("segments.json", f"segments_{script_hash}.json")
                
            return segments, theme_word, theme_hex_color, script_hash, dynamic_subthemes
        except Exception as e:
            print(f"Attempt {attempt_idx+1} failed to parse JSON via {model_to_use}. Error: {e}. Retrying...")
            if attempt_idx == len(models_to_try) - 1:
                print("Final attempt failed. Raw output:")
                try:
                    print(raw_content)
                except NameError:
                    print("No content received.")
                raise e

def generate_voiceover_api(text, audio_path):
    # Check if audio already exists
    if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
        print(f"   Audio already exists -> {audio_path}")
        return True
        
    print(f"   Generating voiceover via ElevenLabs...")
    
    # Using a default pleasant voice: Rachel
    voice_id = "21m00Tcm4TlvDq8ikWAM" 
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }
    
    data = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }
    
    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        
        with open(audio_path, "wb") as f:
            f.write(response.content)
            
        return True
    except requests.exceptions.RequestException as e:
        print(f"   [!] ElevenLabs API failed (Error: {e})")
        return False

def generate_voiceover_edge_tts(text, audio_path, voice="en-US-GuyNeural"):
    """Generate voiceover using Microsoft Edge TTS (free, no API key)."""
    import edge_tts

    if os.path.exists(audio_path):
        print(f"   Audio already exists -> {audio_path}")
        return True

    print(f"   Generating voiceover via Edge TTS (voice: {voice})...")
    try:
        communicate = edge_tts.Communicate(text, voice)
        asyncio.run(communicate.save(audio_path))
        print(f"   ✓ Saved {audio_path}")
        return True
    except Exception as e:
        print(f"   [!] Edge TTS failed (Error: {e})")
        return False
        
def transcribe_and_align(audio_path, original_text):
    from faster_whisper import WhisperModel
    import string
    
    print("   Loading faster-whisper model (base.en)...")
    # Determine compute_type based on platform. cpu typically only supports int8 or float32.
    compute_type = "int8"
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            compute_type = "float16" # GPUs support float16
    except:
        device = "cpu"
        
    model = WhisperModel("base.en", device=device, compute_type=compute_type)
    print("   Extracting word-level timestamps...")
    transcribed_segments, _ = model.transcribe(audio_path, word_timestamps=True)
    
    whisper_words = []
    for segment in transcribed_segments:
        for word in segment.words:
            clean_w = word.word.strip().strip(string.punctuation).lower()
            if clean_w:
                whisper_words.append({
                    "word": word.word.strip(), 
                    "clean": clean_w,
                    "start": word.start, 
                    "end": word.end
                })
                
    # Sequence Alignment to Original Script
    orig_words = original_text.split()
    aligned_timestamps = []
    
    w_idx = 0
    for o_word in orig_words:
        clean_o = o_word.strip(string.punctuation).lower()
        if not clean_o:
            # If the original word is purely punctuation, give it a tiny duration mapped to previous word
            prev_end = aligned_timestamps[-1]["end"] if aligned_timestamps else 0.0
            aligned_timestamps.append({"word": o_word, "start": prev_end, "end": prev_end + 0.01})
            continue
            
        match = None
        for i in range(w_idx, min(w_idx + 8, len(whisper_words))):
            if whisper_words[i]["clean"] == clean_o:
                match = whisper_words[i]
                w_idx = i + 1
                break
                
        if match:
            aligned_timestamps.append({"word": o_word, "start": match["start"], "end": match["end"]})
        else:
            # Interpolate if whisper missed it
            prev_end = aligned_timestamps[-1]["end"] if aligned_timestamps else 0.0
            aligned_timestamps.append({"word": o_word, "start": prev_end + 0.05, "end": prev_end + 0.2})
            
    return aligned_timestamps

def get_audio_duration(audio_path):
    cmd = [
        "ffprobe", "-v", "error", 
        "-show_entries", "format=duration", 
        "-of", "default=noprint_wrappers=1:nokey=1", 
        audio_path
    ]
    output = subprocess.check_output(cmd).decode().strip()
    return float(output)

def is_video_horizontal(filepath):
    """
    Checks if a video file is horizontal (width > height).
    Returns True if it is horizontal, False if vertical or square.
    """
    if not os.path.exists(filepath):
        return False
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=s=x:p=0",
            filepath
        ]
        output = subprocess.check_output(cmd).decode(errors="replace").strip()
        if output and "x" in output:
            width, height = map(int, output.split("x", 1))
            return width > height
    except Exception as e:
        print(f"Error checking aspect ratio for {filepath}: {e}")
    return False

def has_high_motion(filepath):
    """
    Checks if a video has sufficient motion using FFmpeg's freezedetect filter.
    freezedetect is more reliable than mpdecimate across ffmpeg versions.
    Returns (has_motion_bool, freeze_fraction) where freeze_fraction is 0.0-1.0
    representing the proportion of the video that is frozen.
    """
    if not os.path.exists(filepath):
        return (False, 1.0)
    try:
        # Get video duration first
        dur_cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0", filepath
        ]
        duration = float(subprocess.check_output(dur_cmd).decode().strip())
        if duration <= 0:
            return (False, 1.0)
        
        # Run freezedetect: n=0.003 is noise threshold, d=0.5 means freeze periods >= 0.5s
        cmd = [
            "ffmpeg", "-i", filepath,
            "-vf", "freezedetect=n=0.003:d=0.5",
            "-f", "null", "-"
        ]
        result = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True)
        
        # Parse freeze_duration lines to sum total frozen time
        import re as _re
        total_frozen = 0.0
        for line in result.stderr.split('\n'):
            # Match lines like: lavfi.freezedetect.freeze_duration: 5.000000
            m = _re.search(r'freeze_duration:\s*([\d.]+)', line)
            if m:
                total_frozen += float(m.group(1))
        
        freeze_fraction = total_frozen / duration
        if freeze_fraction > 0.50:
            print(f"   [Motion Check] Video is mostly frozen ({freeze_fraction:.0%} frozen).")
            return (False, freeze_fraction)
            
        print(f"   [Motion Check] Video has good motion ({freeze_fraction:.0%} frozen).")
        return (True, freeze_fraction)
    except Exception as e:
        print(f"Error checking motion for {filepath}: {e}")
        return (True, 0.0)

def is_video_mostly_black(filepath):
    """
    Checks if a video file is mostly black/blank using FFmpeg's blackdetect filter.
    Returns True if more than 35% of the video duration is detected as black.
    """
    if not os.path.exists(filepath):
        return False
    try:
        # Get video duration first
        duration_cmd = [
            "ffprobe", "-v", "error", 
            "-show_entries", "format=duration", 
            "-of", "default=noprint_wrappers=1:nokey=1", 
            filepath
        ]
        duration = float(subprocess.check_output(duration_cmd).decode(errors="replace").strip())
        if duration <= 0:
            return True
            
        # Run blackdetect
        cmd = [
            "ffmpeg", "-i", filepath,
            "-vf", "blackdetect=d=1.0:pic_th=0.90:pix_th=0.10",
            "-f", "null",
            "-"
        ]
        res = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        output = res.stderr.decode(errors="replace")
        
        # Parse blackdetect output lines
        black_duration = 0.0
        for line in output.split("\n"):
            if "black_duration:" in line:
                match = re.search(r"black_duration:([\d.]+)", line)
                if match:
                    black_duration += float(match.group(1))
                    
        ratio = black_duration / duration
        if ratio > 0.35:
            print(f"   ⚠️  Video {os.path.basename(filepath)} rejected: {ratio:.1%} of duration is black ({black_duration:.1f}s / {duration:.1f}s)")
            return True
            
        return False
    except Exception as e:
        print(f"   ⚠️  Error checking video blackness: {e}")
        return False


def process_segment_video(segment, video_path, audio_path, duration, script_hash):
    seg_id = segment["segment_id"]
    output_path = f"temp_{script_hash}_{seg_id}.mp4"
    
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        print(f"Step 4: Processed video already exists for {seg_id} -> {output_path}")
        return output_path
        
    print(f"Step 4: Processing video for {seg_id}...")
    
    if audio_path is None:
        cmd = [
            "ffmpeg", "-y",
            "-stream_loop", "-1",
            "-i", video_path,
            "-t", str(duration),
            "-vf", "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,fps=30,setsar=1,format=yuv420p",
            "-c:v", "libx264",
            "-an",
            output_path
        ]
    else:
        original_duration = duration - 0.4
        fade_out_start = max(0, original_duration - 0.1)
        af_filter = f"afade=t=in:ss=0:d=0.05,afade=t=out:st={fade_out_start}:d=0.1,apad"
        
        cmd = [
            "ffmpeg", "-y",
            "-stream_loop", "-1",
            "-i", video_path,
            "-i", audio_path,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-t", str(duration),
            "-vf", "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,fps=30,setsar=1,format=yuv420p",
            "-c:v", "libx264",
            "-af", af_filter,
            "-c:a", "aac",
            output_path
        ]
        
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path

def stitch_videos(video_paths, script_hash, master_audio_path=None):
    print("Step 5: Stitching segments together (with crossfade transitions)...")
    output_path = f"merged_output_{script_hash}.mp4"
    
    CROSSFADE_DURATION = 0.3  # seconds of overlap between segments

    if len(video_paths) == 1:
        if master_audio_path:
            cmd = [
                "ffmpeg", "-y",
                "-i", video_paths[0],
                "-i", master_audio_path,
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", "aac",
                output_path
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            # Single segment — just copy it
            import shutil as _shutil
            _shutil.copy2(video_paths[0], output_path)
        return output_path

    # Build xfade filter chain for smooth transitions between segments
    # Each xfade needs the offset = (cumulative duration so far) - (crossfade * transition_index)
    # We need durations of each segment to compute offsets.
    durations = []
    for vp in video_paths:
        dur = float(subprocess.check_output([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            vp
        ]).decode().strip())
        durations.append(dur)

    # Build the filter_complex for N-1 crossfades
    n = len(video_paths)
    filter_parts = []
    cumulative_dur = durations[0]

    for i in range(n - 1):
        offset = cumulative_dur - CROSSFADE_DURATION
        offset = max(0, offset)  # safety clamp

        if i == 0:
            src_a = "[0:v]"
        else:
            src_a = f"[xf{i-1}]"

        src_b = f"[{i+1}:v]"

        if i < n - 2:
            out_label = f"[xf{i}]"
        else:
            out_label = "[vout]"

        filter_parts.append(
            f"{src_a}{src_b}xfade=transition=fade:duration={CROSSFADE_DURATION}:offset={offset:.3f}{out_label}"
        )

        cumulative_dur = offset + durations[i + 1]

    if master_audio_path:
        filter_complex = ";\n".join(filter_parts)
        cmd = ["ffmpeg", "-y"]
        for vp in video_paths:
            cmd.extend(["-i", vp])
        cmd.extend(["-i", master_audio_path])
        
        master_audio_idx = len(video_paths)
        cmd.extend([
            "-filter_complex", filter_complex,
            "-map", "[vout]",
            "-map", f"{master_audio_idx}:a:0",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-movflags", "+faststart",
            output_path
        ])
    else:
        # Audio: concat all audio streams
        audio_inputs = "".join(f"[{i}:a]" for i in range(n))
        filter_parts.append(f"{audio_inputs}concat=n={n}:v=0:a=1[aout]")
        filter_complex = ";\n".join(filter_parts)
        
        cmd = ["ffmpeg", "-y"]
        for vp in video_paths:
            cmd.extend(["-i", vp])
        cmd.extend([
            "-filter_complex", filter_complex,
            "-map", "[vout]",
            "-map", "[aout]",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-movflags", "+faststart",
            output_path
        ])

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        print("   ⚠️  Crossfade failed, falling back to simple concat...")
        concat_txt = f"concat_{script_hash}.txt"
        with open(concat_txt, "w") as f:
            for vp in video_paths:
                f.write(f"file '{vp}'\n")
                
        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_txt]
        if master_audio_path:
            cmd.extend(["-i", master_audio_path, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac"])
        else:
            cmd.extend(["-c", "copy"])
        cmd.append(output_path)
            
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return output_path

import re

def create_ass(segment_durations, theme_word, theme_hex_color, script_hash):
    theme_start = None
    theme_end = None

    if theme_hex_color and len(theme_hex_color) == 6:
        R, G, B = theme_hex_color[0:2], theme_hex_color[2:4], theme_hex_color[4:6]
        ass_color = f"&H00{B}{G}{R}"
    else:
        ass_color = "&H0000FFFF"

    ass_content = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,42,&H00FFFFFF,&H000000FF,&H00000000,&H90000000,1,0,0,0,100,100,0,0,3,18,0,2,100,100,250,1
Style: Highlight,Arial,55,{ass_color},&H000000FF,&H00000000,&H90000000,1,0,0,0,100,100,0,0,3,12,0,5,100,100,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    current_time = 0.0
    
    def format_time_ass(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        cs = int(round((seconds - int(seconds)) * 100))
        return f"{hours}:{minutes:02d}:{secs:02d}.{cs:02d}"
        
    def write_chunks(words_data, global_offset, max_end_time):
        nonlocal ass_content
        if not words_data: return
        CHUNK_SIZE = 4
        chunks = [words_data[i:i + CHUNK_SIZE] for i in range(0, len(words_data), CHUNK_SIZE)]
        for i, chunk in enumerate(chunks):
            # No pre-offset for perfect sync
            c_start = global_offset + chunk[0]["start"]
            c_end = global_offset + chunk[-1]["end"]
            
            # Make subtitle linger on screen during pauses
            if i + 1 < len(chunks):
                next_start = global_offset + chunks[i+1][0]["start"]
                # Extend until just before next subtitle, capped at 1.0s extension
                c_end = max(c_end, min(next_start - 0.05, c_end + 1.0))
            else:
                # Last chunk in segment: cap at max_end_time so it never overlaps with next segment
                c_end = max(c_end, min(max_end_time - 0.05, c_end + 1.0))
                
            text = " ".join([w["word"] for w in chunk])
            ass_content += f"Dialogue: 0,{format_time_ass(c_start)},{format_time_ass(c_end)},Default,,0,0,0,,{text}\n"

    for seg, duration in segment_durations:
        start_time = current_time
        end_time = current_time + duration
        # Align with the crossfade timeline by subtracting the 0.3s transition overlap
        current_time = end_time - 0.3
        
        words_data = seg.get("word_timestamps", [])
        if not words_data:
            # Fallback to character-level timing interpolation if Whisper is unavailable
            def char_weight(s):
                if not s: return 0
                weight = len(s)
                weight += s.count(',') * 8
                weight += s.count('.') * 12
                weight += s.count('?') * 12
                weight += s.count('!') * 12
                return weight
                
            words = seg["voiceover_text"].split()
            total_weight = max(1, sum(char_weight(w) for w in words))
            time_per_weight = duration / total_weight
            
            words_data = []
            curr_start = 0.0
            for w in words:
                # Minimum weight of 1 so short words don't flash instantly
                w_dur = max(1, char_weight(w)) * time_per_weight
                words_data.append({"word": w, "start": curr_start, "end": curr_start + w_dur})
                curr_start += w_dur
            
        theme_matched = False
        if theme_word and theme_word.lower() in seg["voiceover_text"].lower():
            # Find which word indices correspond to the theme word
            idx = seg["voiceover_text"].lower().find(theme_word.lower())
            if idx != -1:
                words_before = len(seg["voiceover_text"][:idx].split())
                theme_len = len(theme_word.split())
                
                before_words = words_data[:words_before]
                theme_words = words_data[words_before : words_before + theme_len]
                after_words = words_data[words_before + theme_len:]
                
                t_start = start_time + theme_words[0]["start"] if theme_words else end_time
                write_chunks(before_words, start_time, t_start)
                
                if theme_words:
                    t_dur = max(0.1, theme_words[-1]["end"] - theme_words[0]["start"])
                    
                    theme_text = " ".join([w["word"] for w in theme_words]).upper()
                    
                    # Calculate extended duration for the theme subtitle to persist
                    extended_dur = 1.5  # default/minimum persistence
                    if after_words:
                        words_to_keep = min(4, len(after_words))
                        words_dur = after_words[words_to_keep - 1]["end"] - after_words[0]["start"]
                        # Persist for the duration of the next 3-4 words, up to at most 1.5s, and at least 1.0s
                        extended_dur = max(1.0, min(1.5, words_dur))
                    else:
                        extended_dur = 1.5
                        
                    final_end_time = t_start + t_dur + extended_dur
                    
                    # Smooth fade-in (200ms) and fade-out (200ms) with no jarring pop/zoom animation
                    stylized_text = f"{{\\fad(200,200)}}{theme_text}"
                    ass_content += f"Dialogue: 0,{format_time_ass(t_start)},{format_time_ass(final_end_time)},Highlight,,0,0,0,,{stylized_text}\n"
                    
                    write_chunks(after_words, start_time, end_time)
                    
                    theme_start = t_start
                    theme_end = final_end_time
                    theme_word = None
                    theme_matched = True
        
        if not theme_matched:
            write_chunks(words_data, start_time, end_time)
        
    ass_path = f"subtitles_{script_hash}.ass"
    with open(ass_path, "w") as f:
        f.write(ass_content)
        
    return theme_start, theme_end, ass_path

def generate_and_burn_subtitles(video_path, theme_word, theme_hex_color, theme_start, theme_end, script_hash, ass_path):
    print("Step 6: Burning pixel-perfect ASS subtitles and overlaying theme images...")
    output_path = f"final_video_{script_hash}.mp4"
    
    # Setup theme images
    images_dir = Path(f"theme_images_{script_hash}")
    images_dir.mkdir(exist_ok=True)
    valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".tif"}
    theme_images = [p for p in images_dir.iterdir() if p.suffix.lower() in valid_exts][:3]
    
    try:
        if not theme_images or theme_start is None or theme_end is None:
            # Simple subtitle burn if no images exist
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-vf", f"subtitles={ass_path}",
                "-c:v", "libx264",
                "-c:a", "copy",
                output_path
            ]
        else:
            print(f"Overlaying {len(theme_images)} image(s) from {theme_start:.2f}s to {theme_end:.2f}s...")
            # Calculate explicit duration for image inputs to prevent infinite stream hangs
            # Must be long enough to reach theme_end because the stream timestamp starts at 0
            img_input_duration = theme_end + 1.0
            
            cmd = ["ffmpeg", "-y", "-i", video_path]
            for img in theme_images:
                # Limit looped image stream to only the needed duration to prevent FFmpeg hanging
                cmd.extend(["-loop", "1", "-framerate", "30", "-t", f"{img_input_duration:.2f}", "-i", str(img)])
                
            filter_complex = ""
            last_bg = "[0:v]"
            
            for i, img in enumerate(theme_images):
                img_idx = i + 1
                
                # Stack from top to bottom
                if len(theme_images) == 1:
                    y_pos = "(H-h)/2"
                elif len(theme_images) == 2:
                    y_pos = "(H-h)/3" if i == 0 else "2*(H-h)/3"
                else: # 3 images
                    y_pos = "(H-h)/4" if i == 0 else ("(H-h)/2" if i == 1 else "3*(H-h)/4")
                    
                x_pos = "(W-w)/2"
                
                t_s = f"{theme_start:.2f}"
                t_e = f"{theme_end:.2f}"
                
                # Scale cleanly, apply 45% alpha, and fade in/out
                img_filter = f"[{img_idx}:v]format=rgba,colorchannelmixer=aa=0.45,scale='min(800,iw)':'min(600,ih)':force_original_aspect_ratio=decrease,fade=t=in:st={t_s}:d=0.3:alpha=1,fade=t=out:st={t_e}:d=0.3:alpha=1[img{i}];"
                filter_complex += img_filter
                
                next_bg = f"[bg{i+1}]" if i < len(theme_images) - 1 else "[v_overlaid]"
                overlay_filter = f"{last_bg}[img{i}]overlay=x={x_pos}:y={y_pos}:enable='between(t,{t_s},{theme_end + 0.3:.2f})':eof_action=pass{next_bg};"
                filter_complex += overlay_filter
                
                last_bg = next_bg
                
            filter_complex += f"[v_overlaid]subtitles={ass_path}[final_v]"
            
            cmd.extend([
                "-filter_complex", filter_complex,
                "-map", "[final_v]",
                "-map", "0:a",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-c:a", "copy",
                output_path
            ])

        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Success! Final video saved to {output_path}")
    except Exception as e:
        print(f"Could not burn subtitles. Error: {e}")
        os.rename(video_path, output_path)

def rank_video_results(voiceover, links):
    """Uses the LLM to rank the video search results based on their titles."""
    if len(links) <= 1:
        return links
        
    # If Envato hid the titles (meaning they are all Unknown Title or stock-video.item.alt)
    # then ranking is pointless. Just return the original order.
    has_valid_titles = False
    for link in links:
        t = link.get('title', '').strip().lower()
        if t and t != "unknown title" and t != "stock-video.item.alt":
            has_valid_titles = True
            break
            
    if not has_valid_titles:
        print("   ⚠️  No valid video titles available on Envato search page. Bypassing LLM ranking.")
        return links
        
    prompt = f"""You are an expert video editor. I have a voiceover segment: "{voiceover}"
I searched for B-roll and found the following videos:
"""
    for i, link in enumerate(links):
        prompt += f"\nTitle {i+1}: {link['title']}"
        
    prompt += """\n
Rank these videos from most relevant to least relevant to the voiceover.
Consider that some titles might be ambiguous or completely unrelated (e.g. medical vs construction).
Output ONLY a comma-separated list of the numbers representing your ranking (e.g., '2, 1, 3' or '3, 2, 1').
"""
    models_to_try = [
        {"provider": "nvidia", "model": "openai/gpt-oss-120b"},
        {"provider": "openrouter", "model": "openai/gpt-oss-120b:free"},
        {"provider": "nvidia", "model": "meta/llama-3.3-70b-instruct"},
        {"provider": "nvidia", "model": "meta/llama-3.1-70b-instruct"},
        {"provider": "nvidia", "model": "nvidia/nemotron-3-super-120b-a12b"},
        {"provider": "openrouter", "model": "google/gemma-4-31b-it:free"},
        {"provider": "openrouter", "model": "nvidia/nemotron-3-ultra-550b-a55b:free"}
    ]
    
    for attempt_cfg in models_to_try:
        try:
            provider = attempt_cfg["provider"]
            model_to_use = attempt_cfg["model"]
            client_to_use = get_llm_client(provider)
            
            if not client_to_use:
                continue
                
            response = client_to_use.chat.completions.create(
                model=model_to_use,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.1,
                timeout=30.0
            )
            content = response.choices[0].message.content.strip()
            print(f"   [LLM Ranking ({provider.upper()}: {model_to_use})] Assessing titles against voiceover: '{voiceover}'")
            for i, link in enumerate(links):
                print(f"      {i+1}. {link['title']}")
            print(f"   [LLM Ranking] Chosen Order: {content}")
            
            import re
            indices = [int(idx) for idx in re.findall(r'\d+', content)]
            valid_indices = [idx - 1 for idx in indices if 1 <= idx <= len(links)]
            
            # Add any missing indices
            for i in range(len(links)):
                if i not in valid_indices:
                    valid_indices.append(i)
                    
            ranked_links = [links[i] for i in valid_indices[:len(links)]]
            return ranked_links
        except Exception as e:
            print(f"   ⚠️  Failed to rank videos via {model}: {e}.")
            
    print("   ⚠️  All models failed to rank videos. Falling back to default order.")
    return links

def main():
    if not NVIDIA_NIM_API_KEY:
        print("Please set NVIDIA_NIM_API_KEY in the .env file. It is required for script generation.")
        return

    # Prompt user for old hash
    old_hash_partial = input("Enter the hash key (or partial string) of the previous script to reuse downloads from: ").strip()
    
    # Find matching hash
    downloads_dir = BASE_DOWNLOAD_DIR
    matching_hashes = []
    if downloads_dir.exists():
        for d in downloads_dir.iterdir():
            if d.is_dir() and old_hash_partial in d.name:
                matching_hashes.append(d.name)
                
    if not matching_hashes:
        print(f"No matching hash folders found for '{old_hash_partial}' in {downloads_dir}")
        return
        
    old_hash = matching_hashes[0]
    if len(matching_hashes) > 1:
        print(f"Multiple matches found: {matching_hashes}. Choosing the first one: {old_hash}")
    else:
        print(f"Found match: {old_hash}")
        
    old_script_path = f"script_{old_hash}.md"
    old_segments_path = f"segments_{old_hash}.json"
    
    if not os.path.exists(old_script_path):
        print(f"Error: Could not find old script file {old_script_path}")
        return
        
    if not os.path.exists(old_segments_path):
        print(f"Error: Could not find old segments file {old_segments_path}")
        return
        
    with open(old_script_path, "r") as f:
        old_script_text = f.read().strip()
        
    with open(old_segments_path, "r") as f:
        old_segments_data = json.load(f)

    # Check and install Whisper for timestamp generation
    whisper_available = check_and_install_whisper()
    if not whisper_available:
        print("⚠️ Whisper is unavailable. System will use legacy fallback generation (segment-by-segment).")

    elevenlabs_available = bool(ELEVENLABS_API_KEY and ELEVENLABS_API_KEY.strip())
    if not elevenlabs_available:
        print("⚠️  ELEVENLABS_API_KEY is empty — will use fallback TTS for voiceovers.")
        
    segments_data = get_script_and_segments(old_script_text, old_segments_data)
    if not segments_data: return
    segments, theme_word, theme_hex_color, script_hash, dynamic_subthemes = segments_data
    
    print(f"Generated {len(segments)} segments. Theme: {theme_word} ({theme_hex_color})")
    
    # ---------------------------------------------------------
    # TAG & ISOLATE THEME IMAGES FOR THIS SCRIPT
    # ---------------------------------------------------------
    import shutil
    src_theme_dir = Path("theme_images")
    dst_theme_dir = Path(f"theme_images_{script_hash}")
    
    # Clean up old orphaned tagged directories from previous scripts
    base_dir = Path(".")
    for d in base_dir.glob("theme_images_*"):
        if d.is_dir() and d.name != f"theme_images_{script_hash}" and d.name != f"theme_images_{old_hash}":
            shutil.rmtree(str(d), ignore_errors=True)
            print(f"Cleaned up old script cache: {d.name}/")
        
    dst_theme_dir.mkdir(exist_ok=True)
    
    valid_img_exts = {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".tif"}
    
    old_dst_theme_dir = Path(f"theme_images_{old_hash}")
    
    # If the old theme dir exists and we don't have new source images to tag, copy the old ones over
    if old_dst_theme_dir.exists() and not src_theme_dir.exists():
        for f in old_dst_theme_dir.iterdir():
            if f.is_file() and f.suffix.lower() in valid_img_exts:
                if not (dst_theme_dir / f.name).exists():
                    shutil.copy2(str(f), str(dst_theme_dir / f.name))
                    print(f"Reused theme image: {f.name} from {old_hash}")
                
        old_subthemes = old_dst_theme_dir / "SubThemes"
        dst_subthemes = dst_theme_dir / "SubThemes"
        if old_subthemes.exists():
            dst_subthemes.mkdir(exist_ok=True)
            for f in old_subthemes.iterdir():
                if f.is_file() and f.suffix.lower() in valid_img_exts:
                    if not (dst_subthemes / f.name).exists():
                        shutil.copy2(str(f), str(dst_subthemes / f.name))
                        print(f"Reused SubTheme image: {f.name} from {old_hash}")

    
    valid_img_exts = {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".tif"}
    
    if src_theme_dir.exists():
        for f in src_theme_dir.iterdir():
            if f.is_file() and f.suffix.lower() in valid_img_exts:
                shutil.move(str(f), str(dst_theme_dir / f.name))
                print(f"Tagged theme image: {f.name} -> {dst_theme_dir.name}/")
                
        src_subthemes = src_theme_dir / "SubThemes"
        dst_subthemes = dst_theme_dir / "SubThemes"
        if src_subthemes.exists():
            dst_subthemes.mkdir(exist_ok=True)
            for f in src_subthemes.iterdir():
                if f.is_file() and f.suffix.lower() in valid_img_exts:
                    shutil.move(str(f), str(dst_subthemes / f.name))
                    print(f"Tagged SubTheme image: {f.name} -> {dst_subthemes.name}/")
        
        # Clean up the source theme_images/ directory if only .DS_Store or nothing remains
        remaining = [f for f in src_theme_dir.iterdir() if f.name != ".DS_Store" and not (f.is_dir() and not any(f.iterdir()))]
        if not remaining:
            shutil.rmtree(str(src_theme_dir), ignore_errors=True)
            print(f"Cleaned up empty source directory: {src_theme_dir}/")
    
    # ---------------------------------------------------------
    # PHASE 1: DOWNLOAD ALL VIDEOS (Single Browser Session)
    # ---------------------------------------------------------
    print("\n=== PHASE 1: ENVATO VIDEO DOWNLOADS ===")
    videos_to_download = []
    video_paths = {}
    global_used_urls = set()
    valid_exts = (".mp4", ".mov", ".webm", ".mkv", ".avi")
    
    for seg in segments:
        seg_id = seg["segment_id"]
        download_dir = BASE_DOWNLOAD_DIR / script_hash / seg_id
        download_dir.mkdir(parents=True, exist_ok=True)
        
        # Unzip any stranded zip files before checking
        zip_files = [f for f in os.listdir(download_dir) if f.lower().endswith('.zip')]
        for zf in zip_files:
            zip_path = download_dir / zf
            print(f"Unzipping existing archive {zf} in {seg_id}...")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(download_dir)
                os.remove(zip_path)
            except Exception as e:
                print(f"Failed to unzip {zf}: {e}")
                
        files = [f for f in os.listdir(download_dir) if f.lower().endswith(valid_exts)]
        is_existing_valid = False
        
        # Check new dir first
        if files:
            existing_file = download_dir / files[0]
            if is_video_mostly_black(str(existing_file)):
                print(f"   ⚠️  Existing cached video for {seg_id} is mostly black/blank. Deleting to re-download...")
                shutil.rmtree(str(download_dir), ignore_errors=True)
                download_dir.mkdir(parents=True, exist_ok=True)
            elif is_video_horizontal(str(existing_file)):
                print(f"   ⚠️  Existing cached video for {seg_id} is horizontal. Deleting to re-download vertical...")
                shutil.rmtree(str(download_dir), ignore_errors=True)
                download_dir.mkdir(parents=True, exist_ok=True)
            else:
                has_motion, drop_ratio = has_high_motion(str(existing_file))
                if not has_motion:
                    print(f"   ⚠️  Existing cached video for {seg_id} has very low motion (Drop ratio: {drop_ratio:.2f}). Deleting to re-download...")
                    shutil.rmtree(str(download_dir), ignore_errors=True)
                    download_dir.mkdir(parents=True, exist_ok=True)
                else:
                    video_paths[seg_id] = str(existing_file)
                    print(f"Video already exists and is valid for {seg_id} -> {video_paths[seg_id]}")
                    is_existing_valid = True
                
        # Check old dir if not found in new dir
        old_download_dir = BASE_DOWNLOAD_DIR / old_hash / seg_id
        if not is_existing_valid and old_download_dir.exists():
            old_files = [f for f in os.listdir(old_download_dir) if f.lower().endswith(valid_exts)]
            if old_files:
                existing_old_file = old_download_dir / old_files[0]
                has_motion, drop_ratio = has_high_motion(str(existing_old_file)) if not is_video_mostly_black(str(existing_old_file)) else (False, 1.0)
                if not is_video_mostly_black(str(existing_old_file)) and not is_video_horizontal(str(existing_old_file)) and has_motion:
                    video_paths[seg_id] = str(existing_old_file)
                    print(f"Reusing existing video from {old_hash} for {seg_id} -> {video_paths[seg_id]}")
                    is_existing_valid = True
                else:
                    print(f"   ⚠️  Existing cached video in {old_hash} for {seg_id} is invalid (black, horizontal, or low motion). Will download new.")
                
        if not is_existing_valid:
            videos_to_download.append(seg)
            
    if videos_to_download:
        with EnvatoElementsDownloader(download_dir=BASE_DOWNLOAD_DIR, media_type="video", headless=False) as downloader:
            downloader.ensure_logged_in()
            failed_downloads = videos_to_download.copy()
            max_retries = 3
            current_retry = 0
            unauthorized_attempts = 0
            
            while failed_downloads and current_retry <= max_retries:
                if current_retry > 0:
                    print(f"\n🔄 Auto-retry pass {current_retry}/{max_retries} for {len(failed_downloads)} failed segments...")
                    
                retry_queue = failed_downloads.copy()
                failed_downloads.clear()
                
                queue = [{"seg": seg, "link_idx": 0, "links": []} for seg in retry_queue]
                active_downloads = []
                max_concurrent = 5
                
                while queue or active_downloads:
                    # 1. Fill the window
                    while len(active_downloads) < max_concurrent and queue:
                        item = queue.pop(0)
                        seg = item["seg"]
                        link_idx = item["link_idx"]
                        seg_id = seg["segment_id"]
                        query = seg["envato_search_query"]
                        
                        if not item["links"]:
                            print(f"\nSearching for {seg_id}: {query}")
                            links = downloader.search_and_get_links(query, 5)
                            if links:
                                filtered_links = [l for l in links if l["url"] not in global_used_urls]
                                if not filtered_links:
                                    print("   ⚠️  All top search results have already been used by previous segments. Falling back to unfiltered list.")
                                    filtered_links = links
                                item["links"] = rank_video_results(seg.get("voiceover_text", query), filtered_links)
                                
                        links = item["links"]
                        
                        if links and link_idx < len(links):
                            seg_dir = BASE_DOWNLOAD_DIR / script_hash / seg_id
                            cache_dir = seg_dir / "temp_cache"
                            cache_dir.mkdir(parents=True, exist_ok=True)
                            
                            downloader.download_dir = cache_dir
                            link_obj = links[link_idx]
                            
                            res = downloader.trigger_download(link_idx + 1, link_obj["url"])
                            if res == "UNAUTHORIZED":
                                queue.clear()
                                active_downloads.clear()
                                failed_downloads.extend([q["seg"] for q in queue] + [a["queue_item"]["seg"] for a in active_downloads] + [item["seg"]])
                                unauthorized_attempts += 1
                                if unauthorized_attempts == 1:
                                    print("\n" + "!" * 60 + "\n [Recovery Mode 1] Log in to Envato. Will auto-retry in 5 mins.\n" + "!" * 60)
                                    timed_input("Press Enter to retry now... ", 300)
                                elif unauthorized_attempts == 2:
                                    print("\n" + "!" * 60 + "\n [Recovery Mode 2] Log in to Envato. Will auto-retry in 5 mins.\n" + "!" * 60)
                                    timed_input("Press Enter to retry now... ", 300)
                                else:
                                    print("\n" + "!" * 60 + "\n [Manual Override] Log in to Envato. Press Enter when done.\n" + "!" * 60)
                                    input("Press Enter to continue... ")
                                break
                            elif res:
                                global_used_urls.add(link_obj["url"])
                                active_downloads.append({
                                    "queue_item": item,
                                    "download_info": res,
                                    "seg_dir": seg_dir,
                                    "cache_dir": cache_dir
                                })
                            else:
                                item["link_idx"] += 1
                                queue.insert(0, item)
                        else:
                            print(f"❌ Download failed for all links for {seg_id}. Adding to retry queue.")
                            if seg not in failed_downloads:
                                failed_downloads.append(seg)
                                
                    if not active_downloads:
                        continue
                        
                    # 2. Process oldest active
                    active = active_downloads.pop(0)
                    dl_info = active["download_info"]
                    q_item = active["queue_item"]
                    cache_dir = active["cache_dir"]
                    seg_dir = active["seg_dir"]
                    link_idx = q_item["link_idx"]
                    seg_id = q_item["seg"]["segment_id"]
                    
                    success = False
                    
                    save_res = downloader.wait_and_save_download(dl_info["download_obj"], dl_info["filepath"])
                    if save_res:
                        downloaded_zips = [f for f in os.listdir(cache_dir) if f.lower().endswith('.zip')]
                        for zf in downloaded_zips:
                            try:
                                import zipfile
                                with zipfile.ZipFile(cache_dir / zf, 'r') as z: z.extractall(cache_dir)
                                os.remove(cache_dir / zf)
                            except: pass
                        
                        files = [f for f in os.listdir(cache_dir) if f.lower().endswith(valid_exts) and not f.startswith('cached_')]
                        if files:
                            candidate = cache_dir / files[0]
                            if is_video_mostly_black(str(candidate)) or is_video_horizontal(str(candidate)):
                                os.remove(candidate)
                            else:
                                has_motion, freeze_frac = has_high_motion(str(candidate))
                                cache_name = f"cached_{link_idx}_{freeze_frac:.4f}_{candidate.name}"
                                os.rename(str(candidate), str(cache_dir / cache_name))
                                if has_motion:
                                    shutil.move(str(cache_dir / cache_name), str(seg_dir / candidate.name))
                                    shutil.rmtree(str(cache_dir), ignore_errors=True)
                                    video_paths[seg_id] = str(seg_dir / candidate.name)
                                    success = True
                    
                    if not success:
                        q_item["link_idx"] += 1
                        queue.insert(0, q_item)

                current_retry += 1

    # Check if all videos are present
    missing_videos = [s for s in segments if s["segment_id"] not in video_paths]
    if missing_videos:
        print(f"Warning: Could not download videos for {len(missing_videos)} segments. Using fallback videos.")
        # Do not drop segments! Use the video from a previous valid segment.
        for i, seg in enumerate(segments):
            seg_id = seg["segment_id"]
            if seg_id not in video_paths:
                fallback_path = None
                # Try finding previous valid video
                for j in range(i - 1, -1, -1):
                    prev_id = segments[j]["segment_id"]
                    if prev_id in video_paths:
                        fallback_path = video_paths[prev_id]
                        break
                # If none before, find next valid video
                if not fallback_path:
                    for j in range(i + 1, len(segments)):
                        next_id = segments[j]["segment_id"]
                        if next_id in video_paths:
                            fallback_path = video_paths[next_id]
                            break
                if fallback_path:
                    print(f"   Using fallback video for {seg_id}")
                    video_paths[seg_id] = fallback_path
                else:
                    print(f"   CRITICAL ERROR: No fallback video available for {seg_id}!")
        
    if not segments:
        print("No segments have videos. Exiting.")
        return

    # ---------------------------------------------------------
    # PHASE 1.5: SUB-THEME PROCESSING
    # ---------------------------------------------------------
    print("\n=== PHASE 1.5: SUB-THEME PROCESSING ===")
    from subtheme_processor import (
        discover_subthemes, match_subthemes_to_segments,
        generate_subtheme_videos, compute_interleave_plan,
        build_interleaved_segment,
    )
    
    # Process dynamic subthemes (Option 2) before discovery
    dst_subthemes = dst_theme_dir / "SubThemes"
    dst_subthemes.mkdir(exist_ok=True)
    
    missing_dynamic = []
    for phrase in dynamic_subthemes:
        # Check if already cached (ignore extension)
        existing = list(dst_subthemes.glob(f"{phrase}.*"))
        if not existing:
            missing_dynamic.append(phrase)
            
    if missing_dynamic:
        print(f"Found {len(missing_dynamic)} dynamic sub-theme(s) needing download...")
        downloader = EnvatoElementsDownloader(download_dir=dst_subthemes, headless=False, media_type="photo")
        try:
            downloader.start()
            downloader.ensure_logged_in()
            for phrase in missing_dynamic:
                links = downloader.search_and_get_links(phrase, 1)
                if links:
                    success = downloader.download_item(1, links[0])
                    if success:
                        downloaded_photos = [f for f in os.listdir(dst_subthemes) if f.startswith("1_")]
                        for photo in downloaded_photos:
                            ext = os.path.splitext(photo)[1]
                            old_path = dst_subthemes / photo
                            new_path = dst_subthemes / f"{phrase}{ext}"
                            if new_path.exists():
                                os.remove(new_path)
                            os.rename(old_path, new_path)
                            print(f"✅ Renamed {photo} to {phrase}{ext}")
                human_delay(2, 4)
        finally:
            downloader.stop()

    subthemes = discover_subthemes(script_hash)
    subtheme_matches = {}
    subtheme_video_map = {}

    if subthemes:
        print(f"Found {len(subthemes)} sub-theme image(s):")
        for st in subthemes:
            print(f"   - {st.keywords_str} ({Path(st.image_path).name})")

        subtheme_matches = match_subthemes_to_segments(subthemes, segments)

        if subtheme_matches:
            print(f"\nMatched sub-themes to {len(subtheme_matches)} segment(s):")
            for seg_id, matches in subtheme_matches.items():
                for st, pos in matches:
                    print(f"   {seg_id}: '{st.keywords_str}' at char {pos}")

            subtheme_video_map = generate_subtheme_videos(subtheme_matches)
        else:
            print("No sub-theme keywords matched any segment text.")
    else:
        print("No sub-theme images found in theme_images/SubThemes/")

    # ---------------------------------------------------------
    # PHASE 2: GENERATE ALL AUDIO
    # ---------------------------------------------------------
    print("\n=== PHASE 2: AUDIO GENERATION ===")
    
    if whisper_available:
        full_audio_path = f"full_audio_{script_hash}.mp3"
        full_text = " ".join([seg["voiceover_text"] for seg in segments])
        
        audio_ready = False
        
        if os.path.exists(full_audio_path):
            print(f"   ✅ Using cached master audio: {full_audio_path}")
            audio_ready = True
            
        if not audio_ready and elevenlabs_available:
            audio_ready = generate_voiceover_api(full_text, full_audio_path)
        
        if not audio_ready:
            print("\n[!] ElevenLabs unavailable or failed. Switching to fallback.")
            print("\n" + "="*60)
            print(" FALLBACK — CHOOSE TTS METHOD")
            print("="*60)
            
            clipboard_works = clipboard_copy("test")
            if clipboard_works:
                print("  [1] 📋 Manual Voiceover (Full script copied to your clipboard)")
            else:
                print("  [1] 📋 Manual Voiceover (Full script printed to terminal)")
            print("  [2] 🤖 Auto-generate with Edge TTS (free, no API key)")
            choice = input("\nSelect option (1 or 2): ").strip()
            
            if choice == "2":
                print("\n🤖 Using Edge TTS...")
                audio_ready = generate_voiceover_edge_tts(full_text, full_audio_path)
                
            if not audio_ready:
                print("\n[!] Please generate manual audio for the full script.")
                print(f"\n   ── TEXT START ──\n   {full_text}\n   ── TEXT END ──\n")
                
                if clipboard_works:
                    clipboard_copy(full_text)
                    print("📋 Full script copied to clipboard!")
                    
                import time
                prompt_time = time.time()
                input("   → Press Enter when you've downloaded the TTS file from ElevenLabs...")
                
                fetched_files = fetch_downloaded_audio(expected_count=1, start_time=prompt_time)
                
                if fetched_files:
                    import shutil
                    shutil.move(str(fetched_files[0]), full_audio_path)
                    print(f"   ✅ Auto-moved {fetched_files[0].name} to {full_audio_path}")
                else:
                    manual_path = input("   → Auto-detect failed. Paste the absolute path to your MP3 for the full script: ").strip().strip("\"'")
                    if os.path.exists(manual_path):
                        import shutil
                        shutil.copy2(manual_path, full_audio_path)
            
        aligned_words = transcribe_and_align(full_audio_path, full_text)
        
        audio_dur = get_audio_duration(full_audio_path)
        
        import string
        print("\n=== Calculating Segment Durations ===")
        # Pre-calculate robust alignment using character matching
        def clean_chars(s):
            return "".join(c.lower() for c in s if c.isalnum())
            
        word_ptr = 0
        for i, seg in enumerate(segments):
            seg_text = seg["voiceover_text"]
            target_len = len(clean_chars(seg_text))
            
            seg_words = []
            collected_chars = ""
            
            while word_ptr < len(aligned_words):
                w = aligned_words[word_ptr]
                seg_words.append(w)
                collected_chars += clean_chars(w["word"])
                word_ptr += 1
                if len(collected_chars) >= target_len:
                    break
                    
            if i == len(segments) - 1 and word_ptr < len(aligned_words):
                while word_ptr < len(aligned_words):
                    seg_words.append(aligned_words[word_ptr])
                    word_ptr += 1
                    
            seg["word_timestamps_raw"] = seg_words
            
        for i, seg in enumerate(segments):
            seg_words = seg["word_timestamps_raw"]
            start_time = seg_words[0]["start"] if seg_words else 0.0
            
            # Calculate perfect display duration bridging to the next segment's first word
            if i + 1 < len(segments):
                next_seg_words = segments[i+1]["word_timestamps_raw"]
                next_start = next_seg_words[0]["start"] if next_seg_words else start_time + 1.0
                
                # For the first segment, measure display duration from 0.0 of the audio track
                # to align the video timeline with the absolute audio start.
                display_dur = next_start - (0.0 if i == 0 else start_time)
                raw_video_dur = display_dur + 0.3 # Provide extra 0.3s of video to feed the crossfade transition
            else:
                # Last segment should stretch to the end of the full audio track, 
                # ensuring the video doesn't end abruptly and all trailing subtitles are displayed.
                end_time = max(aligned_words[-1]["end"], audio_dur) if aligned_words else start_time + 1.0
                display_dur = end_time - (0.0 if i == 0 else start_time)
                raw_video_dur = display_dur
                
            seg["display_dur"] = display_dur
            seg["raw_video_dur"] = raw_video_dur
            
            # Save relative word timestamps for ASS subtitle generation
            seg["word_timestamps"] = []
            for w in seg_words:
                seg["word_timestamps"].append({
                    "word": w["word"],
                    "start": max(0.0, w["start"] - (0.0 if i == 0 else start_time)),
                    "end": max(0.0, w["end"] - (0.0 if i == 0 else start_time))
                })
    else:
        print("Using legacy segment-by-segment generation (whisper unavailable).")
        fallback_needed = not elevenlabs_available
        missing_audio_segments = []

        if fallback_needed:
            for seg in segments:
                seg_id = seg["segment_id"]
                if not os.path.exists(f"{script_hash}_{seg_id}_audio.mp3"):
                    missing_audio_segments.append(seg)
        else:
            for seg in segments:
                seg_id = seg["segment_id"]
                audio_path = f"{script_hash}_{seg_id}_audio.mp3"
                if not generate_voiceover_api(seg["voiceover_text"], audio_path):
                    print("\n[!] ElevenLabs API call failed. Switching to fallback.")
                    fallback_needed = True
                    if not os.path.exists(audio_path):
                        missing_audio_segments.append(seg)
                    for r_seg in segments[segments.index(seg)+1:]:
                        if not os.path.exists(f"{script_hash}_{r_seg['segment_id']}_audio.mp3"):
                            missing_audio_segments.append(r_seg)
                    break
                    
        if fallback_needed and missing_audio_segments:
            print("\n" + "="*60)
            print(" FALLBACK — CHOOSE TTS METHOD")
            print("="*60)
            print("  [1] 📋 Manual Voiceover (Segment by Segment)")
            print("  [2] 🤖 Auto-generate with Edge TTS (free, no API key)")
            choice = input("\nSelect option (1 or 2): ").strip()
            
            if choice == "2":
                print("\n🤖 Using Edge TTS...")
                for seg in missing_audio_segments:
                    generate_voiceover_edge_tts(seg["voiceover_text"], f"{script_hash}_{seg['segment_id']}_audio.mp3")
            else:
                for idx, seg in enumerate(missing_audio_segments, 1):
                    text = seg["voiceover_text"]
                    clipboard_copy(text)
                    print(f"\n📋 [{idx}/{len(missing_audio_segments)}] Segment '{seg['segment_id']}' copied to clipboard:")
                    print(f"   ── TEXT START ──\n   {text}\n   ── TEXT END ──")
                    
                    import time
                    prompt_time = time.time()
                    input("   → Press Enter when you've downloaded this segment's audio... ")
                    
                    fetched_files = fetch_downloaded_audio(expected_count=1, start_time=prompt_time)
                    dest_path = f"{script_hash}_{seg['segment_id']}_audio.mp3"
                    
                    if fetched_files:
                        import shutil
                        shutil.move(str(fetched_files[0]), dest_path)
                        print(f"   ✅ Auto-moved {fetched_files[0].name} to {dest_path}")
                    else:
                        manual_path = input("   → Auto-detect failed. Paste the absolute path to this segment's MP3: ").strip().strip("\"'")
                        if os.path.exists(manual_path):
                            import shutil
                            shutil.copy2(manual_path, dest_path)
                print("✅ Manual audio mapping complete!")

    # Verify all audio is present before proceeding
    if whisper_available:
        if not os.path.exists(f"full_audio_{script_hash}.mp3"):
            print("Full audio missing. Exiting.")
            return
    else:
        segments = [s for s in segments if os.path.exists(f"{script_hash}_{s['segment_id']}_audio.mp3")]
        if not segments:
            print("No segments have audio. Exiting.")
            return

    # ---------------------------------------------------------
    # PHASE 3: PROCESS & STITCH
    # ---------------------------------------------------------
    print("\n=== PHASE 3: VIDEO PROCESSING ===")
    processed_videos = []
    segment_durations = []
    
    for seg in segments:
        seg_id = seg["segment_id"]
        vid_path = video_paths[seg_id]
        
        if whisper_available:
            aud_path = None
            duration = seg["raw_video_dur"]
        else:
            aud_path = f"{script_hash}_{seg_id}_audio.mp3"
            # Add 0.4 seconds of padding for a natural pacing pause
            duration = get_audio_duration(aud_path) + 0.4
            
        segment_durations.append((seg, duration))
        
        if seg_id in subtheme_matches and subtheme_video_map:
            # Build interleaved segment with sub-theme clips
            clips = compute_interleave_plan(
                seg, subtheme_matches[seg_id], duration, subtheme_video_map,
            )
            has_subtheme = any(c.clip_type == 'subtheme' for c in clips)
            
            final_seg_path = None
            if has_subtheme:
                final_seg_path = build_interleaved_segment(
                    clips, vid_path, aud_path, duration, seg_id, script_hash
                )
                
            if final_seg_path is None:
                # Fallback to normal processing if interleaving failed or wasn't needed
                final_seg_path = process_segment_video(seg, vid_path, aud_path, duration, script_hash)
        else:
            final_seg_path = process_segment_video(seg, vid_path, aud_path, duration, script_hash)
        
        processed_videos.append(final_seg_path)
        
    master_audio = f"full_audio_{script_hash}.mp3" if whisper_available else None
    merged_video = stitch_videos(processed_videos, script_hash, master_audio)
    
    # Generate subtitles file based on timings
    theme_start, theme_end, ass_path = create_ass(segment_durations, theme_word, theme_hex_color, script_hash)
    generate_and_burn_subtitles(merged_video, theme_word, theme_hex_color, theme_start, theme_end, script_hash, ass_path)
    
    # Cleanup temp files
    print("\nCleaning up temporary files...")
    for vp in processed_videos:
        if os.path.exists(vp):
            os.remove(vp)
    if os.path.exists(f"concat_{script_hash}.txt"):
        os.remove(f"concat_{script_hash}.txt")
    if os.path.exists(f"merged_output_{script_hash}.mp4"):
        os.remove(f"merged_output_{script_hash}.mp4")
    if os.path.exists(f"full_audio_{script_hash}.wav"):
        os.remove(f"full_audio_{script_hash}.wav")
    # Cleanup ASS subtitles file
    if os.path.exists(f"subtitles_{script_hash}.ass"):
        os.remove(f"subtitles_{script_hash}.ass")
        print(f"Cleaned up subtitles: subtitles_{script_hash}.ass")
        
    print(f"Workflow complete! Saved as final_video_{script_hash}.mp4")

if __name__ == "__main__":
    main()
