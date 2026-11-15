import os
import re
import json
import uuid
import time
import shutil
import asyncio
import fitz  # PyMuPDF
import edge_tts
import requests

COMFY_HOST = os.environ.get('COMFY_HOST', '127.0.0.1')
COMFY_PORT = os.environ.get('COMFY_PORT', '8188')
COMFY_OUTPUT_DIR = os.environ.get('COMFY_OUTPUT_DIR', r'D:\comfy\ComfyUI_windows_portable\ComfyUI\output')

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
COVERS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'covers')
AUDIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'audio')
UPLOADS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')

for d in [DATA_DIR, COVERS_DIR, AUDIO_DIR, UPLOADS_DIR]:
    os.makedirs(d, exist_ok=True)

BOOKS_FILE = os.path.join(DATA_DIR, 'books.json')

def load_books():
    if not os.path.exists(BOOKS_FILE):
        return []
    try:
        with open(BOOKS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []

def save_books(books):
    with open(BOOKS_FILE, 'w', encoding='utf-8') as f:
        json.dump(books, f, ensure_ascii=False, indent=2)

def extract_pdf_data(pdf_path, book_id):
    """Extracts text and renders first page as cover."""
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    
    # 1. Render Cover
    cover_filename = f"{book_id}.jpg"
    cover_path = os.path.join(COVERS_DIR, cover_filename)
    try:
        first_page = doc[0]
        pix = first_page.get_pixmap(dpi=150)
        pix.save(cover_path)
        cover_url = f"/covers/{cover_filename}"
    except Exception as e:
        print(f"Error rendering cover: {e}")
        cover_url = "/static/default_cover.svg"

    # 2. Extract Text Page by Page
    pages_text = []
    full_text = []
    for page_num in range(total_pages):
        text = doc[page_num].get_text("text").strip()
        if text:
            pages_text.append({"page": page_num + 1, "text": text})
            full_text.append(text)
            
    doc.close()
    return {
        "cover_url": cover_url,
        "total_pages": total_pages,
        "pages_text": pages_text,
        "full_text": "\n\n".join(full_text)
    }

def analyze_and_categorize(full_text, filename, api_key=None):
    """Uses Gemini 3.8 Flash to analyze and categorize book, or heuristic fallback."""
    sample_text = full_text[:4000] # First ~4000 chars contains intro/title/toc
    
    if api_key:
        try:
            prompt = f"""أنت خبير مكتبي ومفهرس كتب ذكي. حلل نص هذا الكتاب واستخرج بياناته بدقة بالصيغة المحددة.
اسم الملف: {filename}
بداية نص الكتاب:
---
{sample_text}
---

أجب فقط بكائن JSON صالح يحتوي على الحقول التالية:
{{
  "title": "عنوان الكتاب الدقيق والرسمي",
  "author": "اسم المؤلف أو مجهول إذا لم يذكر",
  "category": "تصنيف واحد من: (روايات، تطوير ذات، تقنية، دين، تاريخ، فكر وفلسفة، إدارة وأعمال، عام)",
  "summary": "نبذة موجزة وذكية ومشوقة عن الكتاب في سطرين أو ثلاثة",
  "chapters": ["عنوان الفصل أو الموضوع الأول", "عنوان الفصل الثاني", ...]
}}
"""
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent?key={api_key}"
            resp = requests.post(url, json={
                "contents": [{"parts": [{"text": prompt}]}]
            }, timeout=15)
            
            if resp.status_code == 200:
                data = resp.json()
                raw_out = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                # Clean json formatting
                if raw_out.startswith("```json"):
                    raw_out = raw_out[7:]
                if raw_out.startswith("```"):
                    raw_out = raw_out[3:]
                if raw_out.endswith("```"):
                    raw_out = raw_out[:-3]
                parsed = json.loads(raw_out.strip())
                return parsed
        except Exception as e:
            print(f"Gemini analysis error: {e}")

    # Fallback heuristic
    clean_title = os.path.splitext(filename)[0].replace("_", " ").replace("-", " ")
    return {
        "title": clean_title,
        "author": "غير محدد",
        "category": "عام",
        "summary": "كتاب صوتي تم استخراجه وفهرسته آلياً من ملف PDF.",
        "chapters": ["المقدمة وقراءة الكتاب"]
    }

def detect_language(text):
    """Accurately detects if book is Arabic or English."""
    arabic_chars = len(re.findall(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]', text))
    latin_chars = len(re.findall(r'[a-zA-Z]', text))
    # If the book contains Arabic content, it is ARABIC. Never translate!
    if arabic_chars >= 25:
        return "ar"
    elif latin_chars >= 40:
        return "en"
    return "ar"

def clean_raw_text_for_speech(text):
    """
    Advanced heuristic cleaner: Removes all unmeaningful characters,
    URLs, page numbers, footnote markers, publisher boilerplate, and noise.
    """
    if not text:
        return ""

    # 1. Remove URLs and emails
    text = re.sub(r'https?://\S+|www\.\S+|\S+@\S+', '', text)
    
    # 2. Remove private-use unicode, corrupted font symbols
    text = re.sub(r'[\ue000-\uf8ff\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    # 3. Process line by line
    cleaned_lines = []
    boilerplate_patterns = [
        r'حقوق (الطبع|النشر) محفوظة',
        r'رقم (الإيداع|الترخيص|الفسح)',
        r'ISBN',
        r'دار (النشر|التوزيع)',
        r'الطبعة (الأولى|الثانية|الثالثة)',
        r'شبكة\s+\S+\s+للكتب',
        r'قسم الكتب',
        r'فهرس المحتويات',
        r'صندوق بريد',
        r'هاتف\s*:',
        r'فاكس\s*:'
    ]
    
    for line in text.split('\n'):
        s = line.strip()
        if not s:
            continue
            
        # Skip isolated numbers or page numbers (e.g. "12", "صفحة 4", "Page 15")
        if re.match(r'^(صفحة|page|ص)?\s*\d+\s*$', s, re.IGNORECASE):
            continue
            
        # Skip repetitive symbols (e.g. "......", "------", "******", "======")
        if re.match(r'^[_\-=\.\*~#\s]{2,}$', s):
            continue
            
        # Skip lines that match publisher/copyright boilerplate
        if any(re.search(pat, s, re.IGNORECASE) for pat in boilerplate_patterns):
            continue
            
        # Skip lines that are just footnote markers like "(1)", "[2]", "*"
        if re.match(r'^[\(\[\*]\s*\d+\s*[\)\]]$', s):
            continue
            
        cleaned_lines.append(s)
        
    result = " ".join(cleaned_lines)
    
    # 4. Remove inline footnote brackets like [1] or (12)
    result = re.sub(r'\[\d+\]|\(\d+\)', '', result)
    
    # 5. Remove repetitive dots and dashes
    result = re.sub(r'\.{2,}', '.. ', result)
    result = re.sub(r'\-{2,}', ' ', result)
    result = re.sub(r'\_{2,}', ' ', result)
    result = re.sub(r'\*{2,}', ' ', result)
    
    # 6. Normalize whitespace
    result = re.sub(r'\s+', ' ', result).strip()
    return result

def polish_audiobook_script(text, api_key, is_english=False, progress_callback=None, task_id=None):
    """
    Uses Gemini 3.8 Flash to transform raw text into an authentic,
    professional audiobook narration script:
    - If Arabic: Keeps in Arabic! Strips all remaining noise, formats for natural narration.
    - If English: Translates to rich, literary Arabic.
    """
    # Pre-clean noise first
    cleaned = clean_raw_text_for_speech(text)
    if not cleaned or not api_key:
        return cleaned or text

    # Split into chunks of ~2500 characters to process
    paragraphs = cleaned.split('. ')
    chunks = []
    curr_chunk = []
    curr_len = 0
    
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if curr_len + len(p) > 2500:
            chunks.append(". ".join(curr_chunk) + ".")
            curr_chunk = [p]
            curr_len = len(p)
        else:
            curr_chunk.append(p)
            curr_len += len(p)
    if curr_chunk:
        chunks.append(". ".join(curr_chunk) + ".")
        
    if not chunks:
        chunks = [cleaned[:3000]]
        
    total_chunks = len(chunks)
    script_parts = []
    
    for idx, chunk in enumerate(chunks):
        if progress_callback:
            pct = 30 + int((idx / total_chunks) * 35) # 30% to 65%
            action_name = "ترجمة وصياغة" if is_english else "تهيئة وضبط السرد الصوتي"
            progress_callback(task_id, pct, f"جاري {action_name} باحترافية ({idx+1}/{total_chunks}) 🎙️")
            
        if is_english:
            prompt = f"""أنت مخرج ومترجم كتب صوتية محترف. ترجم النص الإنجليزي التالي إلى لغة عربية فصحى عذبة وسلسة وممتعة للاستماع الصوتي.
احذف تماماً أي أرقام صفحات، هوامش، روابط، أو تفاصيل غير مفيدة للسرد.
أخرج النص المترجم الجاهز للإلقاء الصوتي فقط دون أي مقدمات أو تعليقات منك:
---
{chunk}
---"""
        else:
            prompt = f"""أنت مخرج ومعد كتب صوتية محترف (Audiobook Producer).
أمامك نص مستخرج من كتاب عربي. مهمتك إعداده للإلقاء الصوتي الاحترافي:
1. احتفظ بالنص باللغة العربية الفصحى كما هو تماماً، ولا تترجمه إطلاقاً!
2. احذف كل ما ليس له معنى أو يشتت الاستماع: (أرقام الصفحات، الهوامش، روابط المواقع، فهارس النقاط، ورموز الطباعة).
3. اجعل النص يتدفق بسلاسة ووضوح بأحسن صياغة سردية، وضع الفواصل والنقاط في مكانها الصحيح ليتنفس القارئ الصوتي بشكل طبيعي.
4. أخرج النص الصافي الجاهز للقراءة الصوتية فقط، دون أي تعليقات منك:
---
{chunk}
---"""

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent?key={api_key}"
        try:
            resp = requests.post(url, json={
                "contents": [{"parts": [{"text": prompt}]}]
            }, timeout=25)
            if resp.status_code == 200:
                data = resp.json()
                out = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                script_parts.append(out)
            else:
                script_parts.append(chunk)
        except Exception as e:
            print(f"Script polishing error: {e}")
            script_parts.append(chunk)
            
    return "\n\n".join(script_parts)

async def generate_with_drama3(text, book_id, progress_callback=None, task_id=None):
    """
    Submits generation to local Fish Audio Drama-3 running on ComfyUI,
    polls for completion, and downloads the output audio file.
    """
    cleaned = clean_raw_text_for_speech(text)
    if not cleaned:
        cleaned = "مرحباً بكم في هذا الكتاب الصوتي."

    # Use first ~1200 characters for Drama-3 generation
    sample_text = cleaned[:1200]

    template = {
        "1": {
            "class_type": "T8DramaLoader",
            "inputs": {
                "model": "drama3-t8-w4a8.safetensors",
                "codec": "drama3-codec.safetensors",
                "expert_cache_gb": 32.0,
                "gdn_backend": "torch",
                "model_path": "",
                "codec_path": "",
                "text_head_path": "drama3-text-head.safetensors",
                "line_head_path": "drama3-line-head.safetensors",
                "scene_head_mode": "compact",
                "moe_backend": "torch",
                "cuda_graph": False,
                "expert_placement": "ram"
            }
        },
        "2": {
            "class_type": "LoadAudio",
            "inputs": {
                "audio": "makkah_voice_reference.wav"
            }
        },
        "3": {
            "class_type": "T8DramaReference",
            "inputs": {
                "engine": ["1", 0],
                "audio": ["2", 0],
                "transcript": "يا هلا والله ومسهلا يا سيدي، نورتنا وشرفتنا والله! إيش أخبارك وإيش مسوي؟ أبشر بالخير يا غالي، كل شيء تبغاه نسويه لك على كيف كيفك.",
                "name": "makkah_voice",
                "trim_silence": False,
                "normalize": False
            }
        },
        "4": {
            "class_type": "T8DramaSession",
            "inputs": {
                "direction": "[calm] [makkawi] [narrative]",
                "references": ["3", 0]
            }
        },
        "5": {
            "class_type": "T8DramaGenerate",
            "inputs": {
                "engine": ["1", 0],
                "session": ["4", 0],
                "mode": "speech",
                "text": sample_text,
                "lookahead": "",
                "chunk_bytes": 300,
                "seed": 42,
                "temperature": 0.7,
                "top_p": 0.7,
                "top_k": 30,
                "max_frames": 250,
                "sampling_preset": "expressive",
                "replace_delivery_tags": False
            }
        },
        "6": {
            "class_type": "T8DramaDecode",
            "inputs": {
                "engine": ["1", 0],
                "session": ["5", 0],
                "take_index": -1,
                "all_takes": False,
                "shorten_long_pauses": False,
                "max_pause_seconds": 0.75
            }
        },
        "7": {
            "class_type": "SaveAudio",
            "inputs": {
                "audio": ["6", 0],
                "filename_prefix": f"Drama3_Book_{book_id}"
            }
        }
    }

    base_url = f"http://{COMFY_HOST}:{COMFY_PORT}"
    try:
        resp = requests.post(f"{base_url}/prompt", json={"prompt": template}, timeout=15)
        if resp.status_code != 200:
            raise Exception(f"ComfyUI Error: {resp.text}")
        prompt_id = resp.json()["prompt_id"]
    except Exception as e:
        print(f"Error connecting to ComfyUI, falling back: {e}")
        return await generate_audiobook_files(text, book_id, voice='ar-SA-HamedNeural')

    # Poll ComfyUI
    start_time = time.time()
    audio_filename = None
    while time.time() - start_time < 1800:
        await asyncio.sleep(3)
        try:
            h_resp = requests.get(f"{base_url}/history/{prompt_id}", timeout=10)
            if h_resp.status_code == 200:
                h_data = h_resp.json()
                if prompt_id in h_data:
                    outputs = h_data[prompt_id].get("outputs", {})
                    node_7_out = outputs.get("7", {}).get("audio", [])
                    if node_7_out:
                        audio_filename = node_7_out[0]["filename"]
                        break
        except Exception:
            pass

        if progress_callback:
            progress_callback(task_id, 88, "موديل دراما 3 (Fish Audio Drama-3) جالس يولد الصوت المحلي الحين... 🎭")

    if not audio_filename:
        raise Exception("انتهت مهلة التوليد في موديل دراما 3")

    target_path = os.path.join(AUDIO_DIR, f"{book_id}.flac")
    local_path = os.path.join(COMFY_OUTPUT_DIR, audio_filename)
    if os.path.exists(local_path):
        shutil.copy2(local_path, target_path)
    else:
        # Download via /view API
        file_resp = requests.get(f"{base_url}/view?filename={audio_filename}&type=output", stream=True)
        with open(target_path, 'wb') as f:
            for chunk in file_resp.iter_content(chunk_size=8192):
                f.write(chunk)

    return f"/audio/{book_id}.flac", "00:45"

async def generate_audiobook_files(text, book_id, voice='ar-SA-HamedNeural', max_chars=12000):
    """Generates MP3 audio file from text using edge-tts."""
    cleaned = clean_raw_text_for_speech(text)
    if not cleaned:
        cleaned = "عذراً، هذا الملف لا يحتوي على نص قابل للقراءة صوتياً."
    
    # Slice if text is extremely huge to prevent timeouts on first preview
    text_to_speak = cleaned[:max_chars]
    
    audio_filename = f"{book_id}.mp3"
    audio_path = os.path.join(AUDIO_DIR, audio_filename)
    
    communicate = edge_tts.Communicate(text_to_speak, voice)
    await communicate.save(audio_path)
    
    # Calculate duration approximately
    estimated_seconds = max(10, int(len(text_to_speak) / 16))
    mins = estimated_seconds // 60
    secs = estimated_seconds % 60
    duration_str = f"{mins:02d}:{secs:02d}"
    
    return f"/audio/{audio_filename}", duration_str

async def process_pdf_pipeline(task_id, pdf_path, filename, api_key=None, voice='ar-SA-HamedNeural', auto_translate=True, progress_callback=None):
    """Full automated ingestion pipeline with intelligent Arabic preservation and professional script polishing."""
    try:
        book_id = str(uuid.uuid4())[:8]
        
        # Step 1: Extract (in thread)
        if progress_callback:
            progress_callback(task_id, 15, "جاري استخراج النصوص وتوليد غلاف الكتاب 📄")
        pdf_data = await asyncio.to_thread(extract_pdf_data, pdf_path, book_id)
        
        raw_text = pdf_data["full_text"]
        lang = detect_language(raw_text)
        is_english = (lang == "en")
        is_translated = False
        
        # Step 2: Professional Polishing & Language Handling
        if progress_callback:
            if is_english and auto_translate and api_key:
                progress_callback(task_id, 30, "تم رصد كتاب إنجليزي! جاري الترجمة الاحترافية للعربية الفصحى 🌐")
            else:
                progress_callback(task_id, 30, "كتاب عربي! جاري تنقية السرد وحذف الهوامش والأرقام غير المفيدة 📖✨")
                
        # Run polishing/translating in thread to keep event loop responsive
        processed_text = await asyncio.to_thread(
            polish_audiobook_script,
            raw_text,
            api_key,
            is_english and auto_translate,
            progress_callback,
            task_id
        )
        
        if is_english and auto_translate and api_key:
            is_translated = True
            
        # Ensure voice is Arabic if book is in Arabic (or translated)
        if not is_english or is_translated:
            if voice.startswith('en-'):
                voice = 'ar-SA-HamedNeural'

        # Step 3: Index & Categorize (in thread)
        if progress_callback:
            progress_callback(task_id, 70, "جاري فهرسة وتصنيف الكتاب وتوليد الملخص بالذكاء الاصطناعي 🧠")
        meta = await asyncio.to_thread(analyze_and_categorize, processed_text, filename, api_key)
        
        # Step 4: Audio Generation
        if progress_callback:
            voice_label = "موديل دراما 3 (Fish Audio Drama-3)" if voice == "drama3" else voice
            progress_callback(task_id, 85, f"جاري تسجيل الكتاب صوتياً بنبرة ({voice_label}) 🎙️")
            
        if voice == "drama3":
            audio_url, duration_str = await generate_with_drama3(processed_text, book_id, progress_callback, task_id)
        else:
            audio_url, duration_str = await generate_audiobook_files(processed_text, book_id, voice=voice)
        
        # Step 5: Save to Library
        book_record = {
            "id": book_id,
            "title": meta.get("title", filename),
            "author": meta.get("author", "غير محدد"),
            "category": meta.get("category", "عام"),
            "summary": meta.get("summary", ""),
            "chapters": meta.get("chapters", ["كامل الكتاب"]),
            "cover_url": pdf_data["cover_url"],
            "audio_url": audio_url,
            "duration": duration_str,
            "total_pages": pdf_data["total_pages"],
            "voice": voice,
            "filename": filename,
            "translated": is_translated,
            "original_lang": "en" if is_english else "ar"
        }
        
        books = load_books()
        books.insert(0, book_record) # Newest first
        save_books(books)
        
        if progress_callback:
            success_msg = "تمت ترجمة وتوليد الكتاب الصوتي وإضافته لمكتبتك بنجاح! 📚✨" if is_translated else "تمت تنقية وتوليد الكتاب الصوتي وإضافته لمكتبتك بنجاح! 📚✨"
            progress_callback(task_id, 100, success_msg, book=book_record)
            
        return book_record
    except Exception as e:
        print(f"PIPELINE ERROR: {e}")
        import traceback
        traceback.print_exc()
        if progress_callback:
            progress_callback(task_id, -1, f"حدث خطأ أثناء معالجة الكتاب: {str(e)}")
        raise e
    except Exception as e:
        if progress_callback:
            progress_callback(task_id, -1, f"حدث خطأ أثناء معالجة الكتاب: {str(e)}")
        raise e
