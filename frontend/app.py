"""PDF chat frontend. Install: pip install 'streamlit[auth]>=1.67,<2' pypdf python-dotenv
Requires the project's backend package and its dependencies for ingestion and chat.
Run: streamlit run frontend/app.py. Set GROQ_API_KEY in .env or run Ollama with qwen2.5:latest.
Local library: ./recent_pdfs. Set PDF_SESSION_ONLY=1 and DISABLE_OLLAMA=1 for a shared cloud deployment.
Optional ALLOWED_EMAILS (comma-separated) restricts Google OAuth access.
Optional RAG_EVAL_ACCURACY (0–100); otherwise read RAGAS faithfulness from eval/reports/summary_full.json.
"""
import json
import os
import re
import sys
import tempfile
from contextlib import contextmanager
from datetime import date, datetime
from hashlib import sha256
from html import escape
from io import BytesIO
from pathlib import Path
from threading import Lock
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from pypdf.errors import PdfReadError

st.set_page_config(page_title="Ask your PDFs · RAG", page_icon="📑", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, str) and _k not in os.environ:
            os.environ[_k] = _v
except Exception:
    pass

BACKEND_IMPORT_ERROR = None
try:
    from backend.app.ingestion.pipeline import ingest_document
    from backend.app.retrieval.vector_store import collection_exists, get_collection_count
    from backend.app.generation.answer import answer_question
except ImportError as exc:
    BACKEND_IMPORT_ERROR = exc

ss = st.session_state
ROOT = PROJECT_ROOT / "recent_pdfs"
DB = ROOT / "db.json"


def init_session_state():
    for key, value in dict(active=None, library={"files": [], "chats": {}}, provider="Groq").items():
        ss.setdefault(key, value)


init_session_state()


@st.cache_resource
def disk_lock():
    return Lock()


def setting(key, default=""):
    val = os.getenv(key)
    if val:
        return val
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


LOCAL = setting("PDF_SESSION_ONLY") != "1"
OLLAMA_ENABLED = setting("DISABLE_OLLAMA") != "1"
MODELS = {"Groq": setting("GROQ_MODEL", "openai/gpt-oss-20b")}
if OLLAMA_ENABLED:
    MODELS["Ollama (offline)"] = setting("OLLAMA_MODEL", "qwen2.5:latest")


def current_email() -> str:
    return (getattr(st.user, "email", None) or "").strip().lower()


def is_allowed_user(email: str) -> bool:
    raw = setting("ALLOWED_EMAILS")
    if not raw:
        return True  # Open to any Google account if ALLOWED_EMAILS is unset
    allowed = {e.strip().lower() for e in raw.split(",") if e.strip()}
    return email.lower() in allowed


def user_db_path() -> Path:
    email = current_email()
    if not email:
        return DB
    uid = sha256(email.encode()).hexdigest()[:12]
    return ROOT / f"db_{uid}.json"


def read_library():
    if not LOCAL:
        return ss.library
    try:
        return json.loads(user_db_path().read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return {"files": [], "chats": {}}


@contextmanager
def edit_library():
    with disk_lock():
        db = read_library()
        yield db
        if LOCAL:
            ROOT.mkdir(exist_ok=True)
            db_path = user_db_path()
            tmp = db_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(db, ensure_ascii=False), encoding="utf-8")
            tmp.replace(db_path)


# Theme and component styles
CSS = """
:root{color-scheme:light dark;--bg:light-dark(#f2f6f7,#0c0a08);--fg:light-dark(#16120e,#fafaf9);
 --card:light-dark(#ffffffad,#28211cba);--solid:light-dark(#fffdfa,#201c19);--muted:light-dark(#16120e0b,#ffffff0d);
 --mfg:light-dark(#57524c,#c5bbb2);--line:color-mix(in srgb,var(--fg) 13%,transparent);
 --grid:color-mix(in srgb,var(--fg) 7%,transparent);--ring:light-dark(#a6460a,#ffae67);--orange:#f58c24;
 --accent:#ffcc34;--raised:light-dark(#fff,#312923);--user-bubble:light-dark(#fff3e8,#47352f);
 --selected:light-dark(#fff0e0,#453327);--mono-bg:light-dark(#fff,#39322e);--mono-fg:light-dark(#211d19,#fff);
 --shadow:light-dark(#22170c22,#0006);--glass:light-dark(#fffcf3bd,#221b18be);
 --sunrise:radial-gradient(125% 125% at 50% 103%,#ef5808 9%,#f58928 17%,#ffc294 29%,#edb7cd 44%,#cdbade 65%,#b6d9ee 94%)}
body{background:var(--bg);color:var(--fg);color-scheme:inherit}
:is(.stApp,[data-testid=stDialog]),:is(.stApp,[data-testid=stDialog]) *{color-scheme:inherit!important}
.stApp{color:var(--fg);background:linear-gradient(90deg,var(--grid) 1px,transparent 1px) 0 0/40px 40px,
 linear-gradient(var(--grid) 1px,transparent 1px) 0 0/40px 40px,radial-gradient(ellipse at 55% 60%,#f58c2440 0%,#f58c2422 25%,#ec489914 45%,transparent 72%),var(--bg);background-attachment:fixed}
[data-testid=stHeader],[data-testid=stAppViewContainer],[data-testid=stMain]{background:transparent!important;color:var(--fg)}
[data-testid=stAppDeployButton]{display:none!important}
[data-testid=stSidebar]{background:var(--solid)!important;color:var(--fg);border-right:1px solid var(--line)}
.stMainBlockContainer{max-width:1180px;padding-top:3.35rem;padding-bottom:2rem}
/* Streamlit widgets and the dialog portal share the same palette. */
:is(.stApp,[data-testid=stDialog]) :is(h1,h2,p,label,[data-testid=stMetricLabel],[data-testid=stMetricValue],
 [data-testid=stVerticalBlock],[data-testid=stElementContainer],[data-testid=stText],[data-testid=stText]>span,
 [data-testid=stChatMessageContent],button svg){color:inherit}
:is(.stApp,[data-testid=stDialog]) [data-testid=stCaptionContainer] p{color:var(--mfg)}
[data-testid=stText],[data-testid=stText]>span{white-space:pre-wrap;overflow-wrap:anywhere;max-width:100%}
:is(.stApp,[data-testid=stDialog]) button[kind=secondary]{background:var(--solid);color:var(--fg);border-color:var(--line)}
:is(.stApp,[data-testid=stDialog]) button[kind=secondary]:not(:disabled):hover{background:var(--raised);border-color:var(--ring)}
[role=tooltip]{background:var(--raised);color:var(--fg)}
.brand{display:flex;gap:.65rem;align-items:center;font:700 1.5rem Georgia,serif;letter-spacing:-.035em}
.mark{display:grid;place-items:center;width:36px;height:36px;border-radius:11px;background:var(--mono-bg);color:var(--mono-fg);border:1px solid var(--line)}
.mark:before{content:"";width:12px;height:16px;border:1.5px solid;border-radius:2px;background:linear-gradient(currentColor,currentColor) center 52%/6px 1px no-repeat,linear-gradient(currentColor,currentColor) center 74%/6px 1px no-repeat}
.kicker{font:700 .72rem ui-monospace,monospace;letter-spacing:.13em;text-transform:uppercase;color:var(--mfg)}
.hero{max-width:38rem}.hero h1{font-size:clamp(2.1rem,4.4vw,3.2rem);font-weight:600;letter-spacing:-.035em;line-height:1.18;margin:.55rem 0;padding:0}.hero p,.uh p{color:var(--mfg)}
.loop{display:inline-block;height:1.3em;line-height:1.3em;overflow:hidden;vertical-align:bottom;color:var(--ring);white-space:nowrap}.loop>span{display:flex;flex-direction:column;animation:roll 9s cubic-bezier(.7,0,.2,1) infinite}.loop b{height:1.3em;line-height:1.3em;font-weight:inherit}
@keyframes roll{0%,28%{transform:none}33%,61%{transform:translateY(-1.3em)}66%,94%{transform:translateY(-2.6em)}100%{transform:translateY(-3.9em)}}
.st-key-theme_icon button[kind=secondary]{position:relative;width:42px;height:42px;min-height:42px;padding:0;gap:0;
 border-radius:50%;background:var(--mono-bg);color:var(--mono-fg);box-shadow:0 3px 12px var(--shadow)}
.st-key-theme_icon button[kind=secondary]:hover{background:var(--solid);border-color:var(--ring)}
.st-key-theme_icon button [data-testid=stMarkdownContainer]{position:absolute;width:1px;height:1px;
 padding:0;clip-path:inset(50%);overflow:hidden;white-space:nowrap}
.accuracy-track{height:4px;border-radius:10px;background:var(--muted);overflow:hidden;margin:-.35rem 0 .15rem}
.accuracy-track span{display:block;height:100%;background:var(--orange);border-radius:inherit}
.st-key-center{align-items:center;gap:13px;margin-top:.4rem}
.st-key-chat_btn{width:max-content!important;margin-inline:auto}
.st-key-upload_card{margin-inline:auto;width:min(28rem,100%)!important;background:var(--card);border:1px solid var(--line);border-radius:15px;padding:1.3rem;backdrop-filter:blur(12px)}
.uh h2{font-size:1.12rem;font-weight:650;padding:0;margin:0}.uh p{font-size:.87rem;margin:.3rem 0 0}
[data-testid=stFileUploaderDropzone]{min-height:165px;flex-direction:column;justify-content:center;gap:.8rem;border:1px dashed var(--mfg);border-radius:11px;background:var(--muted);color:var(--fg)}
[data-testid=stFileUploaderDropzone] button{background:var(--mono-bg)!important;color:var(--mono-fg)!important;border-radius:999px}
[data-testid=stFileChip]{background:var(--raised);color:var(--fg);border:1px solid var(--line)}
[data-testid=stFileChipName],[data-testid=stFileUploaderFile]{color:var(--fg)}
[data-testid=stFileChip] small,[data-testid=stFileUploaderFile] small{color:var(--mfg)}
[data-testid=stFileChip]>div:has(>svg){background:var(--mono-bg);color:var(--mono-fg);border:1px solid var(--line)}
[data-testid=stFileChip] svg{color:inherit}
[data-testid=stFileUploaderDropzoneInstructions] small{color:var(--mfg)!important}
.how{display:flex;gap:.5rem;flex-wrap:wrap;justify-content:center;font-size:.82rem;color:var(--mfg)}
.how span,.pill{border:1px solid var(--line);background:var(--card);border-radius:10px;padding:.4rem .7rem}.how b{color:var(--ring);margin-right:.35rem}.pill{border-color:color-mix(in srgb,var(--orange) 75%,var(--line));border-radius:999px;overflow-wrap:anywhere;width:fit-content;max-width:min(28rem,100%);margin-inline:auto}
.st-key-chat_btn button{background:var(--accent)!important;border:0;border-radius:999px;padding:.48rem 1.25rem;color:#211609!important;box-shadow:0 5px 14px var(--shadow)}
.st-key-chat_btn button:disabled{opacity:.55;box-shadow:none}.st-key-chat_btn button:not(:disabled):hover{transform:translateY(-2px);filter:brightness(1.04)}.st-key-chat_btn button:active{transform:scale(.96)}
.st-key-recent{width:min(248px,100%)!important;margin:.45rem auto 0}
.st-key-recent details{border-color:var(--line);border-radius:15px;background:var(--card);backdrop-filter:blur(14px);box-shadow:0 6px 18px var(--shadow);animation:rise .6s both}
@keyframes rise{from{opacity:0;transform:translateY(18px)}}
.st-key-recent summary{position:relative;isolation:isolate;min-height:136px;align-items:flex-end;padding:1rem;color:var(--fg);perspective:800px}
.st-key-recent summary:before,.st-key-recent summary:after{content:"";position:absolute;z-index:-1;pointer-events:none;left:10px;right:10px;transition:transform .4s ease-out;border-radius:12px}
.st-key-recent summary:before{top:8px;bottom:14px;clip-path:polygon(0 0,35% 0,44% 17%,100% 17%,100% 100%,0 100%);background:var(--papers,linear-gradient(transparent,transparent)),radial-gradient(ellipse at 25% 25%,#c2433f,transparent 65%),radial-gradient(ellipse at 80% 85%,#22395e,transparent 73%),linear-gradient(125deg,#f1c9c4,#91576b 55%,#2a1e2a)}
.st-key-recent details:has(.st-key-recent_files button) summary{--papers:linear-gradient(#fff8e9,#e5d5ce) 71% 62%/58% 48% no-repeat,linear-gradient(#f3e6db,#d6adb9) 46% 48%/59% 47% no-repeat}
.st-key-recent summary:after{top:53%;bottom:0;left:0;right:0;background:var(--glass);backdrop-filter:blur(14px) saturate(1.2);border:1px solid var(--line);transform-origin:bottom}
.st-key-recent summary:is(:hover,:focus-visible):before,.st-key-recent details[open] summary:before{transform:translateY(-5px)}
.st-key-recent summary:is(:hover,:focus-visible):after,.st-key-recent details[open] summary:after{transform:rotateX(-18deg)}
.st-key-recent button{justify-content:flex-start;text-align:left}.st-key-recent button[kind=primary]{background:var(--selected);color:var(--fg);border-color:var(--orange)}
[data-testid=stDialog]{background:#160d0c60}
[data-testid=stDialog]>div:has(>[role=dialog]){background:transparent;box-shadow:none}
[data-testid=stDialog] [role=dialog]{color:var(--fg);border:1px solid var(--line);border-radius:23px;box-shadow:0 24px 80px #0005;animation:pop .35s cubic-bezier(.2,.9,.3,1.1);
 background:linear-gradient(light-dark(#ffffff19,#1c1512bc),light-dark(#ffffff19,#1c1512bc)),var(--sunrise)}
[data-testid=stDialog] [role=dialog] .st-key-chat_history{height:clamp(9rem,39dvh,22rem)!important;overflow-x:hidden;padding:.2rem .25rem .5rem;color:var(--fg)}
[data-testid=stDialog] :is([data-testid=stText],button[aria-label="Close"]){color:var(--fg)}
[data-testid=stDialog] small{color:var(--mfg)}
@keyframes pop{from{opacity:0;transform:scale(.94) translateY(12px)}to{opacity:1;transform:none}}
[data-testid=stChatMessage]{min-width:0;max-width:100%;background:var(--solid);border:1px solid var(--line);
 border-radius:16px;padding:.65rem .8rem;color:var(--fg)}
[data-testid=stChatMessage]:has([data-testid=stChatMessageAvatarUser]){width:fit-content;max-width:92%;margin-inline-start:auto;
 background:var(--user-bubble);flex-direction:row-reverse;align-items:center}
[data-testid=stChatMessageContent]{min-width:0;flex:1 1 auto}
[data-testid^=stChatMessageAvatar]{flex-shrink:0;background:var(--mono-bg);color:var(--mono-fg);border:1px solid var(--line)}
[data-testid^=stChatMessageAvatar] [data-testid=stIconMaterial]{color:inherit}
[data-testid=stChatMessage] details{background:var(--muted);color:var(--fg);border-color:var(--line);border-radius:10px}
[data-testid=stChatMessage] details summary{color:var(--fg)}
.st-key-chatbox{padding:.65rem .8rem;border-radius:18px;border:1px solid var(--line);color:var(--fg);
 background:color-mix(in srgb,var(--solid) 88%,transparent);box-shadow:0 3px 14px var(--shadow)}
.st-key-chatbox [data-testid=stChatInput]>div{background:var(--solid);color:var(--fg);border-color:var(--line);border-radius:22px}
.st-key-chatbox [data-testid=stChatInput]>div:focus-within{border-color:var(--ring)}
.st-key-chatbox [data-testid=stChatInput] textarea{background:transparent;color:var(--fg)}
.st-key-chatbox textarea::placeholder{color:var(--mfg)}
.st-key-chatbox button[data-testid=stChatInputSubmitButton]{background:var(--mono-bg);color:var(--mono-fg);
 border:1px solid var(--line);border-radius:50%}
.st-key-chatbox button[data-testid=stChatInputSubmitButton]:disabled{opacity:.5}
.st-key-chatbox button[role=radio]{background:var(--solid);color:var(--fg);border-color:var(--line)}
.st-key-chatbox button[role=radio][aria-checked=true]{background:var(--selected);color:var(--fg);border-color:var(--ring)}
.st-key-login_wrap{min-height:72vh;justify-content:center;align-items:center}
.st-key-login_card{margin-inline:auto;width:min(26rem,100%)!important;background:var(--card);border:1px solid var(--line);
 border-radius:20px;padding:2rem 1.75rem;backdrop-filter:blur(14px);box-shadow:0 18px 50px var(--shadow);text-align:center}
.login-sub{color:var(--mfg);font-size:.92rem;margin:.5rem 0 1.25rem}
.st-key-google_btn button{background:var(--accent)!important;border:0;border-radius:999px;padding:.58rem 1.35rem;
 color:#211609!important;font-weight:600;box-shadow:0 5px 14px var(--shadow)}
.st-key-google_btn button:hover{transform:translateY(-2px);filter:brightness(1.04)}
.user-card{display:flex;flex-direction:column;gap:.15rem;padding:.55rem .7rem;border:1px solid var(--line);
 border-radius:12px;background:var(--muted);margin-bottom:.5rem;overflow-wrap:anywhere}
.user-card strong{font-size:.88rem}.user-card span{font-size:.76rem;color:var(--mfg)}
:is(.stApp,[data-testid=stDialog]) :is(button,summary):focus-visible{outline:2px solid var(--ring);outline-offset:3px}
@media(max-width:640px){.stMainBlockContainer{padding-top:3.2rem}.st-key-upload_card{padding:1rem}.hero h1{min-height:2.36em}}
@media(prefers-reduced-motion:reduce){.stApp *,[data-testid=stDialog] *{animation:none!important;transition:none!important}}
"""


def toggle_theme():
    current = ss.get("theme_mode") or st.context.theme.type
    ss.theme_mode = "light" if current == "dark" else "dark"


# PDF indexing and model calls
def require_backend():
    if BACKEND_IMPORT_ERROR is not None:
        raise ValueError(
            "The PDF backend is unavailable. Add the project’s backend/ package and install its dependencies."
        ) from BACKEND_IMPORT_ERROR


def collection_for(pid: str) -> str:
    return f"pdf_{pid[:16]}"


def validate_pdf(data: bytes):
    pdf = PdfReader(BytesIO(data))
    if pdf.is_encrypted or len(pdf.pages) > 300:
        raise ValueError("Use an unlocked PDF with up to 300 pages.")


def ingest_bytes(pid: str, data: bytes) -> int:
    require_backend()
    col = collection_for(pid)
    if collection_exists(col):
        return get_collection_count(col)

    if LOCAL:
        ROOT.mkdir(exist_ok=True)
        pdf_path = ROOT / f"{pid}.pdf"
        if not pdf_path.exists():
            pdf_path.write_bytes(data)
        return ingest_document(str(pdf_path), collection_name=col)

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(data)
        return ingest_document(str(tmp_path), collection_name=col)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def store_upload(file):
    data = file.getvalue()
    pid = sha256(data).hexdigest()
    with st.status("Reading, embedding, and indexing your PDF in Qdrant…") as status:
        try:
            validate_pdf(data)
            chunk_count = ingest_bytes(pid, data)
            meta = dict(id=pid, name=file.name, size=len(data), at=datetime.now().timestamp())
            with edit_library() as db:
                if LOCAL:
                    ROOT.mkdir(exist_ok=True)
                    path = ROOT / f"{pid}.pdf"
                    if not path.exists():
                        path.write_bytes(data)
                db["files"] = [meta] + [f for f in db["files"] if f["id"] != pid][:29]
        except (ValueError, PdfReadError, OSError) as exc:
            print(f"[store_upload] Validation/IO error: {exc}")
            status.update(label="Couldn’t index this PDF", state="error", expanded=True)
            st.error(
                str(exc)
                if isinstance(exc, ValueError)
                else "The PDF may be damaged or the backend or library may be unavailable."
            )
            return
        except Exception as exc:
            print(f"[store_upload] Backend indexing error: {exc}")
            status.update(label="Couldn’t index this PDF", state="error", expanded=True)
            st.error("The indexing backend failed. Check that the embedding service and Qdrant are running, then try again.")
            return
        ss.active = pid
        status.update(label=f"Ready · {chunk_count} passages indexed in Qdrant", state="complete")


def ask(pid, provider, question):
    try:
        require_backend()
        col = collection_for(pid)
        if not collection_exists(col):
            if not LOCAL or not re.fullmatch(r"[a-f0-9]{10}|[a-f0-9]{64}", pid):
                return dict(role="assistant", text="PDF index not found. Try uploading the PDF again.")
            pdf_path = ROOT / f"{pid}.pdf"
            if not pdf_path.exists():
                return dict(role="assistant", text="The saved PDF file is missing. Try uploading it again.")
            ingest_document(str(pdf_path), collection_name=col)

        result = answer_question(
            query=question,
            collection_name=col,
            provider_override="groq" if provider == "Groq" else "ollama",
        )
        hits = [(source["page"], source["text"]) for source in result.get("sources", [])]
        return dict(
            role="assistant",
            text=result["answer"],
            hits=hits,
            score=result.get("top_score"),
            model=result.get("model"),
            id=uuid4().hex,
        )
    except Exception as exc:
        print(f"[ask] Retrieval/generation error: {exc}")
        if BACKEND_IMPORT_ERROR is not None:
            return dict(
                role="assistant",
                text="The PDF backend is unavailable. Add the project’s backend/ package and install its dependencies.",
            )
        return dict(
            role="assistant",
            text="Couldn’t search this PDF. Check that the backend and model services are running, then try again.",
        )


def rate(pid, message_id, key):
    with edit_library() as db:
        for message in db["chats"].get(pid, []):
            if message.get("id") == message_id:
                message["fb"] = ss[key]
                break


def close_chat():
    ss.pop("chat_pdf", None)


@st.dialog("Chat with your PDF", width="medium", on_dismiss=close_chat)
def chat(pid, filename):
    st.text(filename)
    history = st.container(key="chat_history", height=320, border=False, autoscroll=True)
    with st.container(key="chatbox"):
        if len(MODELS) > 1:
            default_provider = ss.provider if ss.provider in MODELS else "Groq"
            provider = st.segmented_control(
                "Answer provider",
                list(MODELS),
                key="provider_picker",
                default=default_provider,
                required=True,
                label_visibility="collapsed",
            )
            ss.provider = provider
        else:
            provider = "Groq"
            ss.provider = provider

        question = st.chat_input("Ask anything about this PDF…", key=f"question_{pid}", max_chars=1000)
        if provider == "Groq":
            caption = (
                "Groq receives the question and matched passages (falls back to local Ollama if unreachable)."
                if OLLAMA_ENABLED
                else "Groq receives the question and matched passages."
            )
        else:
            caption = "Ollama runs on the machine hosting this app."
        st.caption(caption)

    messages = list(read_library()["chats"].get(pid, []))
    with history:
        def show(message):
            with st.chat_message(message["role"]):
                if message["role"] == "assistant":
                    st.markdown(message["text"])
                else:
                    st.text(message["text"])
                hits = message.get("hits", [])
                if hits:
                    pages = ", ".join(map(str, sorted({page for page, _ in hits})))
                    label = "pages" if "," in pages else "page"
                    with st.expander(f"Retrieved passages · PDF {label} {pages}", icon=":material/menu_book:"):
                        for page, text in hits:
                            st.caption(f"PAGE {page}")
                            st.text(text)
                elif message.get("src"):
                    st.caption("Retrieved pages: " + ", ".join(str(p) for p, _ in message["src"]))
                if message.get("model") and "id" in message:
                    key = f"fb_{message['id']}"
                    ss.setdefault(key, message.get("fb"))
                    st.feedback("thumbs", key=key, on_change=rate, args=(pid, message["id"], key))

        for message in messages:
            show(message)
        if not (messages or question):
            st.caption("Try asking for the main takeaways, a definition, or a specific detail.")
        if question and question.strip():
            user = dict(role="user", text=question.strip())
            show(user)
            with st.spinner("Finding relevant passages and preparing a reply…"):
                reply = ask(pid, provider, user["text"])
            with edit_library() as db:
                db["chats"].setdefault(pid, []).extend((user, reply))
            show(reply)


def evaluation_accuracy():
    raw = setting("RAG_EVAL_ACCURACY")
    if raw not in (None, ""):
        try:
            value = float(raw)
            if 0 <= value <= 100:
                return value, False
        except (TypeError, ValueError):
            pass
    report = PROJECT_ROOT / "eval" / "reports" / "summary_full.json"
    try:
        data = json.loads(report.read_text(encoding="utf-8"))
        faith = data.get("ragas_faithfulness") if isinstance(data, dict) else None
        if isinstance(faith, list):
            faith = faith[0] if faith else None
        value = float(faith)
        if 0 <= value <= 1:
            return round(value * 100, 1), True
    except (ValueError, TypeError, OSError):
        pass
    return None, False


def clear_chat():
    with edit_library() as db:
        db["chats"].pop(ss.active, None)


def reset():
    # Leave the on-disk library and its chat history intact.
    for key in ("active", "chat_pdf", "upload", "pending_upload", "library", "provider"):
        ss.pop(key, None)


def render_login():
    mode = ss.get("theme_mode")
    # Hide sidebar on the login page for a clean full-screen view
    hide_sidebar = "[data-testid=stSidebar],[data-testid=stSidebarCollapsedControl]{display:none!important}"
    st.html(f"<style>{CSS}{hide_sidebar}:root{{color-scheme:{mode or 'light dark'}}}</style>")

    with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center", wrap=False):
        st.html('<div class="kicker">PDF / RAG Chat</div>', width="content")
        target = "light" if (mode or st.context.theme.type) == "dark" else "dark"
        st.button(f"Switch to {target} mode", key="theme_icon", icon=f":material/{target}_mode:", on_click=toggle_theme)

    with st.container(key="login_wrap"):
        with st.container(key="login_card"):
            st.html('''
                <div class="brand" style="justify-content:center">
                    <span class="mark" aria-hidden="true"></span>PDF Workspace
                </div>
                <p class="login-sub">Sign in with your Google account to upload PDFs and chat with source-backed answers.</p>
            ''')
            with st.container(key="google_btn"):
                if st.button("Continue with Google", icon=":material/login:", width="stretch"):
                    try:
                        st.login()
                    except Exception as exc:
                        st.error("Google OAuth isn't configured yet. Add your [auth] section to .streamlit/secrets.toml.")
                        print(f"[AUTH] Login error: {exc}")
            st.caption("Free Google OAuth · Your files and chats stay private to your account.")


def render_access_denied(email: str):
    mode = ss.get("theme_mode")
    hide_sidebar = "[data-testid=stSidebar],[data-testid=stSidebarCollapsedControl]{display:none!important}"
    st.html(f"<style>{CSS}{hide_sidebar}:root{{color-scheme:{mode or 'light dark'}}}</style>")
    with st.container(key="login_wrap"):
        with st.container(key="login_card"):
            st.html(f'''
                <div class="brand" style="justify-content:center">Access Restricted</div>
                <p class="login-sub"><strong>{escape(email)}</strong> is not on the authorized user list.</p>
            ''')
            st.button("Sign out", icon=":material/logout:", width="stretch", on_click=st.logout)


def render_workspace():
    mode = ss.get("theme_mode")
    st.html(f"<style>{CSS}:root{{color-scheme:{mode or 'light dark'}}}</style>")
    with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center", wrap=False):
        st.html('<div class="kicker">PDF / RAG Chat</div>', width="content")
        target = "light" if (mode or st.context.theme.type) == "dark" else "dark"
        st.button(f"Switch to {target} mode", key="theme_icon", icon=f":material/{target}_mode:", on_click=toggle_theme)

    st.html('''<div class="hero"><h1>Ask your PDFs
<span class="loop" aria-hidden="true"><span><b>anything</b><b>instantly</b><b>with sources</b><b>anything</b></span></span></h1>
<p>Upload a document, then chat with it. Check the source pages behind each answer.</p></div>''')
    with st.container(key="center"):
        with st.container(key="upload_card"):
            st.html('<div class="uh"><h2>PDF Upload</h2><p>Text PDFs · Up to 50 MB and 300 pages</p></div>')
            file = st.file_uploader(
                "PDF",
                type="pdf",
                key="upload",
                label_visibility="collapsed",
                max_upload_size=50,
                on_change=ss.update,
                args=({"pending_upload": True},),
            )
            if ss.pop("pending_upload", False) and file:
                store_upload(file)
        db = read_library()
        selected = next((f for f in db["files"] if f["id"] == ss.active), None)
        if selected:
            st.html(f'<div class="pill">📄 {escape(selected["name"])}</div>')
        else:
            st.html('<div class="how"><span><b>1</b>Upload a PDF</span><span><b>2</b>Open chat</span><span><b>3</b>Check the cited pages</span></div>')
        with st.container(key="chat_btn"):
            if st.button("Open chat", icon=":material/chat:", disabled=not selected):
                ss.chat_pdf = selected["id"]

    with st.sidebar:
        st.html('<div class="brand"><span class="mark" aria-hidden="true"></span>PDF Workspace</div>')
        st.caption("YOUR PDF READING ROOM")
        user_name = getattr(st.user, "name", None) or "Signed-in User"
        user_email = current_email()
        st.html(f'<div class="user-card"><strong>{escape(user_name)}</strong><span>{escape(user_email)}</span></div>')
        st.button("Sign out", icon=":material/logout:", width="stretch", on_click=st.logout)
        st.divider()
        st.html('<div class="kicker">Session insights</div>')
        messages = db["chats"].get(ss.active, [])
        last = next((m for m in reversed(messages) if m["role"] == "assistant"), {})
        score = last.get("score")
        st.metric(
            "Retrieval score",
            "—" if score is None else f"{score:.2f}",
            help="Top retrieved passage vector similarity (reranked by Cross-Encoder), not answer accuracy.",
        )
        accuracy, from_ragas = evaluation_accuracy()
        st.metric(
            "Answer accuracy",
            "Not evaluated" if accuracy is None else f"{accuracy:g}%",
            help="Uses a configured evaluation score or RAGAS faithfulness. Faithfulness measures grounding, not answer correctness.",
        )
        st.html(f'<div class="accuracy-track" aria-hidden="true"><span style="width:{accuracy or 0:g}%"></span></div>')
        if from_ragas:
            st.caption("RAGAS faithfulness · grounding, not answer correctness")
        st.caption("LAST ANSWER MODEL" if last else "SELECTED MODEL")
        selected_model = MODELS.get(ss.provider, MODELS["Groq"])
        st.text(
            (last.get("model") or ("Passage retrieval only" if score is not None else "No model used"))
            if last
            else selected_model
        )
        st.divider()
        st.button("Clear this PDF’s chat", icon=":material/delete_sweep:", width="stretch", disabled=not messages, on_click=clear_chat)
        st.button("Reset session", icon=":material/restart_alt:", width="stretch", disabled=not (selected or file), on_click=reset)
        st.caption("Recent PDFs and chats are stored on this machine." if LOCAL else "Files and chats are kept in this browser session only.")

    files = db["files"]
    count = len(files)
    if count:
        with st.container(key="recent"), st.expander(f"Recent PDFs · {count} {'file' if count == 1 else 'files'}", key="recent_folder"):
            with st.container(key="recent_files", height=245 if count > 5 else "content", border=False):
                last_day = None
                for item in files:
                    day = datetime.fromtimestamp(item["at"]).date()
                    if day != last_day:
                        st.caption("Today" if day == date.today() else day.strftime("%a, %d %b %Y"))
                        last_day = day
                    pid = item["id"]
                    label = item["name"].replace("[", r"\[").replace("!", r"\!")
                    st.button(
                        label,
                        icon=":material/description:",
                        width="stretch",
                        key=f"pdf_{pid}",
                        type="primary" if ss.active == pid else "secondary",
                        on_click=ss.update,
                        args=({"active": pid},),
                    )

    if selected and ss.get("chat_pdf") == selected["id"]:
        chat(selected["id"], selected["name"])
    else:
        close_chat()


if not getattr(st.user, "is_logged_in", False):
    render_login()
    st.stop()

email = current_email()
if not is_allowed_user(email):
    print(f"[AUTH] Blocked unauthorized login attempt: {email}")
    render_access_denied(email)
    st.stop()

if ss.get("_logged_user") != email:
    reset()  # Clear any previous user's in-memory session state
    init_session_state()
    print(f"[AUTH] User signed in: {email} at {datetime.now().isoformat(timespec='seconds')}")
    ss._logged_user = email

render_workspace()