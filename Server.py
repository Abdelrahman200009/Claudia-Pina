import os, base64, aiohttp, asyncio, tempfile
import whisper, torch
from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pyngrok import ngrok
import uvicorn, threading
import os
from dotenv import load_dotenv
from pyngrok import ngrok
# ------------ Load env (ngrok auth) ------------


load_dotenv()

# Read token from environment
ngrok_authtoken = os.getenv("NGROK_AUTHTOKEN")

if not ngrok_authtoken:
    raise EnvironmentError("NGROK_AUTHTOKEN is not set in the environment")

ngrok.set_auth_token(ngrok_authtoken)

# ------------ CUDA / Device ------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    torch.backends.cudnn.benchmark = True  # small perf boost for variable-length inputs
else:
    print("Running on CPU")

# Load Whisper ONCE, pinned to device
# Try 'small' for a bit more quality, or 'tiny' for more speed.
model = whisper.load_model("base", device=DEVICE)

# ------------ FastAPI app ------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- helpers ----------
async def transcribe_path(path: str, language: str | None = None):
    # fp16 only speeds up on CUDA; CPU needs fp16=False
    result = model.transcribe(
        path,
        fp16=(DEVICE == "cuda"),
        language=None if (language in [None, "", "auto"]) else language
    )
    return {"text": (result.get("text") or "").strip()}

async def transcribe_bytes(raw: bytes, name: str = "audio.wav", language: str | None = None):
    # Keep extension if present to help ffmpeg
    suffix = os.path.splitext(name)[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name
    try:
        return await transcribe_path(tmp_path, language=language)
    finally:
        try: os.remove(tmp_path)
        except: pass

# ---------- health + preflight ----------
@app.options("/receiveAudio")
async def receive_audio_options():
    return JSONResponse(status_code=204, content=None)

@app.get("/Transcribe")
async def receive_audio_health():
    return {
        "status": "ok",
        "detail": "POST audio via multipart 'file' or JSON {audio_b64|audio_url}",
        "device": DEVICE
    }

@app.head("/Transcribe")
async def receive_audio_head():
    return JSONResponse(status_code=200, content=None)

# ---------- main POST (multipart OR JSON) ----------
@app.post("/Transcribe")
async def transcribe_audio(
    request: Request,
    file: UploadFile | None = File(default=None),
    language: str | None = Form(default=None)  # optional language hint from frontend
):
    try:
        # 1) multipart/form-data
        if file is not None:
            raw = await file.read()
            return await transcribe_bytes(raw, file.filename or "audio.wav", language=language)

        # 2) application/json fallback
        if request.headers.get("content-type", "").startswith("application/json"):
            data = await request.json()
            lang = data.get("language") or language
            if "audio_b64" in data:
                raw = base64.b64decode(data["audio_b64"])
                return await transcribe_bytes(raw, "audio_from_b64.wav", language=lang)
            if "audio_url" in data:
                async with aiohttp.ClientSession() as sess:
                    async with sess.get(data["audio_url"]) as resp:
                        if resp.status != 200:
                            return JSONResponse(status_code=400, content={"error": f"download failed: {resp.status}"})
                        raw = await resp.read()
                name = os.path.basename(data["audio_url"]) or "audio_from_url.wav"
                return await transcribe_bytes(raw, name, language=lang)
            return JSONResponse(status_code=400, content={"error": "Provide 'file' (multipart) or 'audio_b64'/'audio_url' (JSON)."})
        return JSONResponse(status_code=400, content={"error": "Unsupported content-type. Use multipart/form-data or application/json."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Also support trailing slash
@app.get("/Transcribe/")
async def slash_health():
    return await receive_audio_health()

@app.post("/Transcribe/")
async def slash_post(request: Request, file: UploadFile | None = File(default=None), language: str | None = Form(default=None)):
    return await transcribe_audio(request, file, language)

# ---------- run + ngrok ----------
def run_app():
    uvicorn.run(app, host="0.0.0.0", port=8000, proxy_headers=True, forwarded_allow_ips="*")

# Set your auth token in Colab env for stable tunnels (optional but recommended)
# from pyngrok.conf import PyngrokConfig
# ngrok.set_auth_token(os.environ.get("NGROK_AUTHTOKEN", ""))

threading.Thread(target=run_app, daemon=True).start()
public_url = ngrok.connect(8000)
print("🌍 Public URL:", public_url)
