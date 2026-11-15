import os
import uuid
import asyncio
from aiohttp import web
from processor import (
    process_pdf_pipeline, load_books, save_books,
    DATA_DIR, COVERS_DIR, AUDIO_DIR, UPLOADS_DIR
)

PORT = 8086
TASKS = {}

def update_task_progress(task_id, progress, status, book=None):
    TASKS[task_id] = {
        "progress": progress,
        "status": status,
        "book": book
    }

async def handle_index(request):
    index_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'index.html')
    return web.FileResponse(index_path)

async def handle_get_books(request):
    books = load_books()
    return web.json_response(books)

async def handle_delete_book(request):
    book_id = request.match_info.get('id')
    books = load_books()
    updated = [b for b in books if b['id'] != book_id]
    save_books(updated)
    
    # Clean files if exist
    for dir_path, ext in [(COVERS_DIR, '.jpg'), (AUDIO_DIR, '.mp3')]:
        fpath = os.path.join(dir_path, f"{book_id}{ext}")
        if os.path.exists(fpath):
            try:
                os.remove(fpath)
            except Exception:
                pass
                
    return web.json_response({"success": True})

async def handle_upload(request):
    reader = await request.multipart()
    task_id = str(uuid.uuid4())[:8]
    TASKS[task_id] = {"progress": 5, "status": "جاري استلام الملف وحفظه... ⏳"}

    filename = "book.pdf"
    pdf_path = None
    voice = 'ar-SA-HamedNeural'
    api_key = None
    auto_translate = True

    while True:
        part = await reader.next()
        if part is None:
            break
        if part.name == 'pdf':
            filename = part.filename or "uploaded_book.pdf"
            pdf_path = os.path.join(UPLOADS_DIR, f"{task_id}_{filename}")
            with open(pdf_path, 'wb') as f:
                while True:
                    chunk = await part.read_chunk()
                    if not chunk:
                        break
                    f.write(chunk)
        elif part.name == 'voice':
            voice = (await part.text()).strip() or 'ar-SA-HamedNeural'
        elif part.name == 'api_key':
            api_key = (await part.text()).strip() or None
        elif part.name == 'auto_translate':
            auto_translate = (await part.text()).strip().lower() == 'true'

    if not pdf_path or not os.path.exists(pdf_path):
        return web.json_response({"error": "لم يتم العثور على ملف PDF مرفوع"}, status=400)

    # Spawn background processing task
    asyncio.create_task(
        process_pdf_pipeline(
            task_id=task_id,
            pdf_path=pdf_path,
            filename=filename,
            api_key=api_key,
            voice=voice,
            auto_translate=auto_translate,
            progress_callback=update_task_progress
        )
    )

    return web.json_response({"task_id": task_id})

async def handle_status(request):
    task_id = request.match_info.get('task_id')
    task_info = TASKS.get(task_id, {"progress": 0, "status": "قيد الانتظار..."})
    return web.json_response(task_info)

def create_app():
    app = web.Application(client_max_size=100 * 1024 * 1024) # 100MB PDF upload limit
    
    # Routes
    app.router.add_get('/', handle_index)
    app.router.add_get('/api/books', handle_get_books)
    app.router.add_delete('/api/books/{id}', handle_delete_book)
    app.router.add_post('/api/upload', handle_upload)
    app.router.add_get('/api/status/{task_id}', handle_status)

    # Static routes
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
    app.router.add_static('/static/', path=static_dir, name='static')
    app.router.add_static('/covers/', path=COVERS_DIR, name='covers')
    app.router.add_static('/audio/', path=AUDIO_DIR, name='audio')

    return app

if __name__ == '__main__':
    print(f"=====================================================")
    print(f"📚 المكتبة الصوتية الذكية تعمل الآن على الرابط:")
    print(f"👉 http://localhost:{PORT}")
    print(f"=====================================================")
    app = create_app()
    web.run_app(app, host='0.0.0.0', port=PORT)
