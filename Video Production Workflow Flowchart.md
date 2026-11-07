# **Professional Video Production Workflow (From AI Script)**

## **Workflow Outline**

1. **Start**  
   * **Deep Research:** Identify constant topics to make videos on.  
2. **Topic Selection**  
3. **Script Generation**  
   * *At this point, the workflow splits into two parallel branches (Visuals and Audio):*

### **Branch A: Visuals**

1. **Segment Breakdown:** Perform semantic breakdown of the script. Each segment maps to a sentence / group of sentences (2-3 max) that has the potential to be a separate video segment (visually different)  
2. **Asset Retrieval:** Search on Envato and download videos.

Q.I Improvement effort \#1: 

3. **Video Tagging:** Tag videos frame-by-frame.  
4. **Ranking:** Rank videos by adherence to the specific segment. Select only \#1

### **Branch B: Audio (Voiceover)**

Q.I Improvement Effort \#2:  
0\. Segment script / add punctuation such that it will be ideal for text to speech generation using eleven labs.

1. **Voiceover Generation:** Generate voiceover for the script (using Eleven Labs API).  
2. **Timestamping:** For each voiceover, identify timestamps for each "segment" (from start second to end second).

### **Merge & Assembly**

1. **Combine Data:** Bring together the Ranked Videos (Visuals) and Segment Timestamps (Audio).  
2. **Segment Processing Loop:** *Repeat for all segments (1 to n)*  
   * Use the selected video for that given segment  
   * Trim videos (Pre-Stitch Processing) such that:  
     1. The visual essence is not lost.  
     2. The video plays exactly and only within the identified voiceover timestamp.  
3. **Final Assembly (Stitch):** Stitch everything together in a manner that does not trigger YouTube's AI-generated content policies.  
4. **Post-Processing (Captions):** Add captions.  
   * *Style specifications:* Dark/grey shadow, white text.  
5. **Complete**