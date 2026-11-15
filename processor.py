import os
import re
import json
import uuid
import asyncio
import fitz  # PyMuPDF
import edge_tts
import requests

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

def is_predominantly_english(text):
    """Checks if the text is predominantly English."""
    latin_chars = len(re.findall(r'[a-zA-Z]', text))
    arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
    return latin_chars > (arabic_chars * 2) and latin_chars > 50

def translate_chunks_to_arabic(text, api_key, progress_callback=None, task_id=None):
    """Translates English text to fluent, literary Arabic using Gemini 3.8 Flash."""
    if not api_key:
        return text
    
    # Split text into chunks around paragraphs (~2500 chars)
    paragraphs = text.split('\n\n')
    chunks = []
    curr_chunk = []
    curr_len = 0
    
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if curr_len + len(p) > 2500:
            chunks.append("\n\n".join(curr_chunk))
            curr_chunk = [p]
            curr_len = len(p)
        else:
            curr_chunk.append(p)
            curr_len += len(p)
    if curr_chunk:
        chunks.append("\n\n".join(curr_chunk))
        
    if not chunks:
        chunks = [text[:3000]]
        
    translated_parts = []
    total_chunks = len(chunks)
    
    for idx, chunk in enumerate(chunks):
        if progress_callback:
            pct = 30 + int((idx / total_chunks) * 30) # 30% to 60%
            progress_callback(task_id, pct, f"جاري ترجمة الكتاب إلى لغة عربية فصيحة ({idx+1}/{total_chunks}) 🌐")
            
        prompt = f"""أنت مترجم أدبي محترف. ترجم النص التالي من كتاب إنجليزي إلى لغة عربية فصحى عذبة، سلسة، ورصينة جداً، بأسلوب سردي مشوق ومناسب للاستماع الصوتي في كتاب صوتي. حافظ على دقة الأفكار وتجنب الركاكة والترجمة الحرفية. أخرج النص المترجم فقط دون أي تعليقات أو هوامش إضافية:
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
                translated = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                translated_parts.append(translated)
            else:
                translated_parts.append(chunk)
        except Exception as e:
            print(f"Translation chunk error: {e}")
            translated_parts.append(chunk)
            
    return "\n\n".join(translated_parts)

def clean_arabic_text(text):
    """Cleans text for speech synthesis, removes page numbers and headers."""
    # Remove lines with just numbers (page numbers)
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.isdigit():
            continue
        # Remove repeated hyphens or underscores
        if re.match(r'^[-_=\.]{3,}$', stripped):
            continue
        cleaned_lines.append(stripped)
    
    result = " ".join(cleaned_lines)
    # Remove excessive spaces
    result = re.sub(r'\s+', ' ', result).strip()
    return result

async def generate_audiobook_files(text, book_id, voice='ar-SA-HamedNeural', max_chars=12000):
    """Generates MP3 audio file from text using edge-tts."""
    cleaned = clean_arabic_text(text)
    if not cleaned:
        cleaned = "عذراً، هذا الملف لا يحتوي على نص قابل للقراءة صوتياً."
    
    # Slice if text is extremely huge to prevent timeouts on first preview
    text_to_speak = cleaned[:max_chars]
    
    audio_filename = f"{book_id}.mp3"
    audio_path = os.path.join(AUDIO_DIR, audio_filename)
    
    communicate = edge_tts.Communicate(text_to_speak, voice)
    await communicate.save(audio_path)
    
    # Calculate duration approximately (or via mutagen if installed, or char estimate)
    # Average Arabic speech rate: ~15-18 chars per second
    estimated_seconds = max(10, int(len(text_to_speak) / 16))
    mins = estimated_seconds // 60
    secs = estimated_seconds % 60
    duration_str = f"{mins:02d}:{secs:02d}"
    
    return f"/audio/{audio_filename}", duration_str

async def process_pdf_pipeline(task_id, pdf_path, filename, api_key=None, voice='ar-SA-HamedNeural', auto_translate=True, progress_callback=None):
    """Full automated ingestion pipeline with intelligent English-to-Arabic translation."""
    try:
        book_id = str(uuid.uuid4())[:8]
        
        # Step 1: Extract
        if progress_callback:
            progress_callback(task_id, 15, "جاري استخراج النصوص وتوليد غلاف الكتاب 📄")
        pdf_data = extract_pdf_data(pdf_path, book_id)
        
        raw_text = pdf_data["full_text"]
        is_english = is_predominantly_english(raw_text)
        is_translated = False
        
        # Step 2: Language Check & Translation
        if is_english and auto_translate and api_key:
            if progress_callback:
                progress_callback(task_id, 30, "تم رصد كتاب إنجليزي! جاري الترجمة الاحترافية للعربية الفصحى 🌐")
            processed_text = translate_chunks_to_arabic(raw_text, api_key, progress_callback, task_id)
            is_translated = True
        else:
            processed_text = raw_text

        # If English book and not translated, switch to high-quality English voice
        if is_english and not is_translated and not voice.startswith('en-'):
            voice = 'en-US-GuyNeural'
        
        # Step 3: Index & Categorize
        if progress_callback:
            progress_callback(task_id, 65, "جاري فهرسة وتصنيف الكتاب وتوليد الملخص بالذكاء الاصطناعي 🧠")
        meta = analyze_and_categorize(processed_text, filename, api_key)
        
        # Step 4: Audio Generation
        if progress_callback:
            progress_callback(task_id, 80, f"جاري تسجيل الكتاب صوتياً بنبرة ({voice}) 🎙️")
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
            success_msg = "تمت ترجمة وتوليد الكتاب الصوتي وإضافته لمكتبتك بنجاح! 📚✨" if is_translated else "تمت إضافة الكتاب الصوتي إلى مكتبتك بنجاح! 📚✨"
            progress_callback(task_id, 100, success_msg, book=book_record)
            
        return book_record
    except Exception as e:
        if progress_callback:
            progress_callback(task_id, -1, f"حدث خطأ أثناء معالجة الكتاب: {str(e)}")
        raise e
