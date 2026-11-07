# **Professional Video Production Workflow v2 (Automated AI Pipeline)**

## **Major Improvements in v2**
- **Forced Alignment Subtitles**: Replaced character-estimation math with `faster-whisper`. Extracts exact word-level timestamps directly from the audio file for zero-lag, pixel-perfect ASS subtitles.
- **Full-Script TTS Integration**: Consolidated voiceover generation from disjointed segment chunks into a single, fluid audio track to eliminate robotic pauses and improve natural pacing.
- **Robust Fallback Matrix**: Intelligent graceful degradation. The system checks system RAM/OS limits and API Key availability (ElevenLabs). Falls back to Edge TTS, clipboard manual generation, or legacy segment-by-segment math if capabilities are absent.
- **Deterministic Theme Validation**: Added a Python constraint layer to ensure the LLM-generated highlight theme word strictly exists as a substring in the actual script text.

---

## **Workflow Outline**

1. **Start**  
   * **Deep Research:** Identify constant topics to make videos on.  
2. **Topic Selection**  
3. **Script Generation (NVIDIA NIM)**
   * Generate raw script and deterministically validate the "Theme Word" highlight.
   * *At this point, the workflow splits into two parallel branches (Visuals and Audio):*

### **Branch A: Visuals**
1. **Segment Breakdown:** Perform semantic breakdown of the script. Each segment maps to a sentence / group of sentences.
2. **Asset Retrieval:** Search on Envato and download videos.
3. **Subtheme Processing:** Extract highly specific keywords to overlay short photo snippets on top of the main videos for visual retention.
4. **Ranking:** Select the #1 best video match for each semantic segment.

### **Branch B: Audio & Timing (Voiceover)**
1. **System Capability Check:** Validate >3.5GB RAM for `faster-whisper`.
2. **Voiceover Generation (Full-Script):** 
   * If ElevenLabs API Key is present: Generate entire script as one single high-quality audio track.
   * If Key is missing: Auto-fallback to free Edge TTS, or present user with manual clipboard generation prompt.
3. **Timestamping & Splicing (Forced Alignment):** 
   * Pass the single audio track through local `faster-whisper`.
   * Align Whisper's word-level timestamps sequentially to the original script text (bypassing AI transcription typos).
   * Slice the continuous audio into distinct segment audio files based on word timestamps.

### **Merge & Assembly**
1. **Combine Data:** Combine the Ranked Videos (Visuals) and exact Whisper segment durations (Audio).  
2. **Segment Processing Loop:** *Repeat for all segments (1 to n)*  
   * Use the selected video and subtheme photos.
   * Trim videos perfectly to the millisecond, derived from Whisper's word-level timestamp boundaries.
   * Apply crossfade transitions.
3. **Final Assembly (Stitch):** Stitch the seamless video segments and underlay the unified full-script audio track.  
4. **Post-Processing (Captions):** 
   * Generate `.ass` subtitle file consuming the Whisper word timestamps for perfect synchronization.
   * Apply highlight colors and text-pop animations to the verified Theme Word.
   * Burn subtitles into the final MP4.
5. **Complete**
