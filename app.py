"""
app.py — Streamlit frontend for AskFirst chat app.

Features
────────
• Sidebar: list of persistent threads, create / rename / delete
• Temporary chat toggle (ghost icon in sidebar)
• Chat window with role-tagged messages + model badge
• Persistent threads → messages stored in SQLite via FastAPI
• Temporary threads → purely in-memory; nothing persisted
"""

import time
import requests
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000"
PAGE_TITLE = "AskFirst"

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0f0f0f;
    border-right: 1px solid #1e1e1e;
}
[data-testid="stSidebar"] * {
    color: #e8e8e8 !important;
}

/* ── Main area ── */
.main .block-container {
    padding-top: 1.5rem;
    max-width: 860px;
}

/* ── Chat bubbles ── */
.msg-user {
    background: #1a1a2e;
    border-left: 3px solid #6c63ff;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    color: #e8e8e8;
}
.msg-assistant {
    background: #0d1117;
    border-left: 3px solid #00d4aa;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    color: #e8e8e8;
}
.role-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 4px;
    opacity: 0.55;
}
.model-badge {
    display: inline-block;
    font-size: 10px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 20px;
    background: #1e1e2e;
    color: #6c63ff;
    border: 1px solid #2e2e4e;
    margin-top: 6px;
}

/* ── Temp chat banner ── */
.temp-banner {
    background: #1a1208;
    border: 1px solid #a0621a;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 12px;
    color: #f0b060 !important;
    font-size: 13px;
}

/* ── Thread button highlight ── */
div[data-testid="stButton"] > button:hover {
    border-color: #6c63ff !important;
    color: #6c63ff !important;
}

/* ── Input box ── */
.stTextInput > div > div > input,
.stChatInputContainer textarea {
    background: #0d1117 !important;
    color: #e8e8e8 !important;
    border: 1px solid #2a2a3a !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)


# ── API helpers ───────────────────────────────────────────────────────────────

def api_get(path: str, timeout: int = 10) -> dict | list | None:
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot connect to backend. Make sure `uvicorn main:app` is running.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"API error {e.response.status_code}: {e.response.text}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None


def api_post(path: str, payload: dict, timeout: int = 60) -> dict | None:
    try:
        r = requests.post(f"{API_BASE}{path}", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot connect to backend.")
        return None
    except requests.exceptions.HTTPError as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", e.response.text)
        except Exception:
            detail = e.response.text
        st.error(f"API error {e.response.status_code}: {detail}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None


def api_delete(path: str, timeout: int = 10) -> bool:
    try:
        r = requests.delete(f"{API_BASE}{path}", timeout=timeout)
        r.raise_for_status()
        return True
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot connect to backend.")
        return False
    except requests.exceptions.HTTPError as e:
        st.error(f"Delete error {e.response.status_code}: {e.response.text}")
        return False
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return False


def api_patch(path: str, payload: dict, timeout: int = 10) -> dict | None:
    try:
        r = requests.patch(f"{API_BASE}{path}", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Rename error: {e}")
        return None


# ── Session state defaults ────────────────────────────────────────────────────

def init_state():
    defaults = {
        "active_thread_id":    None,   # int or None
        "is_temporary":        False,  # True → temp chat mode
        "temp_messages":       [],     # [{role, content, model_used?}]
        "rename_thread_id":    None,   # which thread is being renamed
        "rename_value":        "",
        "delete_confirm_id":   None,   # which thread is awaiting confirm
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 💬 AskFirst")
    st.markdown("---")

    # Temporary chat toggle
    temp_toggle = st.toggle(
        "👻 Temporary chat",
        value=st.session_state.is_temporary,
        help="Messages are NOT saved to the database. Memory is not updated.",
    )
    if temp_toggle != st.session_state.is_temporary:
        st.session_state.is_temporary = temp_toggle
        if temp_toggle:
            # Switching to temp — clear temp buffer
            st.session_state.temp_messages = []
            st.session_state.active_thread_id = None
        st.rerun()

    st.markdown("---")

    # New persistent chat button
    if not st.session_state.is_temporary:
        if st.button("＋  New chat", use_container_width=True):
            data = api_post("/threads", {"title": "New Chat"})
            if data:
                st.session_state.active_thread_id = data["id"]
                st.rerun()

        st.markdown("**Threads**")

        threads = api_get("/threads") or []

        if not threads:
            st.caption("No threads yet. Start a chat!")

        for t in threads:
            tid = t["id"]
            title = t["title"] or "Untitled"
            count = t.get("message_count", 0)
            is_active = (tid == st.session_state.active_thread_id)

            # Rename mode
            if st.session_state.rename_thread_id == tid:
                new_name = st.text_input(
                    "Rename",
                    value=st.session_state.rename_value or title,
                    key=f"rename_input_{tid}",
                    label_visibility="collapsed",
                )
                col_save, col_cancel = st.columns([1, 1])
                with col_save:
                    if st.button("✓", key=f"save_{tid}", use_container_width=True):
                        api_patch(f"/threads/{tid}/rename", {"title": new_name})
                        st.session_state.rename_thread_id = None
                        st.rerun()
                with col_cancel:
                    if st.button("✗", key=f"cancel_{tid}", use_container_width=True):
                        st.session_state.rename_thread_id = None
                        st.rerun()
                continue

            # Delete confirm mode
            if st.session_state.delete_confirm_id == tid:
                st.warning(f"Delete **{title}**?")
                col_yes, col_no = st.columns([1, 1])
                with col_yes:
                    if st.button("Yes, delete", key=f"del_yes_{tid}", use_container_width=True):
                        if api_delete(f"/threads/{tid}"):
                            if st.session_state.active_thread_id == tid:
                                st.session_state.active_thread_id = None
                            st.session_state.delete_confirm_id = None
                            st.rerun()
                with col_no:
                    if st.button("Cancel", key=f"del_no_{tid}", use_container_width=True):
                        st.session_state.delete_confirm_id = None
                        st.rerun()
                continue

            # Normal thread row
            col_btn, col_edit, col_del = st.columns([6, 1, 1])
            with col_btn:
                label = f"{'▶ ' if is_active else ''}{title[:28]}{'…' if len(title)>28 else ''}"
                if st.button(label, key=f"thread_{tid}", use_container_width=True,
                             help=f"{count} messages"):
                    st.session_state.active_thread_id = tid
                    st.session_state.is_temporary = False
                    st.rerun()
            with col_edit:
                if st.button("✏️", key=f"edit_{tid}", help="Rename"):
                    st.session_state.rename_thread_id = tid
                    st.session_state.rename_value = title
                    st.rerun()
            with col_del:
                if st.button("🗑️", key=f"delete_{tid}", help="Delete thread"):
                    st.session_state.delete_confirm_id = tid
                    st.rerun()

    else:
        st.info("👻 Temporary chat active.\nMessages won't be saved.")

    # Health status in sidebar footer
    st.markdown("---")
    health = api_get("/health")
    if health:
        groq_ok = health.get("groq_available", False)
        gem_ok  = health.get("gemini_available", False)
        st.caption(
            f"{'🟢' if groq_ok else '🔴'} Groq  "
            f"{'🟢' if gem_ok else '🔴'} Gemini"
        )


# ── Main content ──────────────────────────────────────────────────────────────

def render_message(role: str, content: str, model_used: str | None = None):
    if role == "user":
        st.markdown(
            f'<div class="msg-user">'
            f'<div class="role-label">You</div>'
            f'{content}'
            f'</div>',
            unsafe_allow_html=True,
        )
    elif role == "assistant":
        badge = f'<span class="model-badge">{model_used}</span>' if model_used else ""
        st.markdown(
            f'<div class="msg-assistant">'
            f'<div class="role-label">Assistant</div>'
            f'{content}'
            f'{badge}'
            f'</div>',
            unsafe_allow_html=True,
        )


# ── TEMPORARY CHAT view ───────────────────────────────────────────────────────
if st.session_state.is_temporary:
    st.markdown(
        '<div class="temp-banner">👻 <strong>Temporary chat</strong> — '
        'Nothing you say here will be saved or added to memory.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("### 👻 Temporary Chat")

    for msg in st.session_state.temp_messages:
        render_message(msg["role"], msg["content"], msg.get("model_used"))

    prompt = st.chat_input("Message (not saved)…")
    if prompt:
        st.session_state.temp_messages.append({"role": "user", "content": prompt})

        with st.spinner("Thinking…"):
            # Build history excluding the message we just appended (it's sent separately)
            history = st.session_state.temp_messages[:-1]
            resp = api_post("/chat", {
                "message":      prompt,
                "is_temporary": True,
                "temp_history": history,
            })

        if resp:
            st.session_state.temp_messages.append({
                "role":       "assistant",
                "content":    resp["reply"],
                "model_used": resp.get("model_used"),
            })
        st.rerun()

    if st.session_state.temp_messages:
        if st.button("🗑 Clear temporary chat"):
            st.session_state.temp_messages = []
            st.rerun()


# ── PERSISTENT CHAT view ──────────────────────────────────────────────────────
elif st.session_state.active_thread_id:
    tid = st.session_state.active_thread_id
    thread = api_get(f"/threads/{tid}")
    if not thread:
        st.session_state.active_thread_id = None
        st.rerun()

    st.markdown(f"### 💬 {thread['title']}")
    st.caption(f"Thread #{tid} · {thread.get('message_count', 0)} messages")
    st.markdown("---")

    messages = api_get(f"/threads/{tid}/messages") or []

    for msg in messages:
        render_message(msg["role"], msg["content"], msg.get("model_used"))

    prompt = st.chat_input("Message…")
    if prompt:
        # Optimistic render of user message
        render_message("user", prompt)

        with st.spinner("Thinking…"):
            resp = api_post("/chat", {
                "thread_id":    tid,
                "message":      prompt,
                "is_temporary": False,
            })

        if resp:
            render_message("assistant", resp["reply"], resp.get("model_used"))
            # Force re-fetch to stay in sync
            time.sleep(0.1)
            st.rerun()


# ── WELCOME / NO THREAD SELECTED ──────────────────────────────────────────────
else:
    st.markdown("## Welcome to AskFirst 👋")
    st.markdown(
        "Start a **new chat** from the sidebar, pick an existing thread, "
        "or enable **👻 Temporary chat** for a private, unsaved session."
    )
    st.markdown("---")
    st.markdown("**Features**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("🧠 **Universal memory**\nThe AI remembers context from all your past threads.")
    with col2:
        st.markdown("🔀 **Multiple threads**\nOrganise different topics in separate chats.")
    with col3:
        st.markdown("👻 **Temporary chats**\nEphemeral sessions — nothing saved, nothing remembered.")

    st.markdown("---")
    health = api_get("/health")
    if health:
        st.markdown("**LLM Status**")
        c1, c2 = st.columns(2)
        with c1:
            icon = "🟢" if health.get("groq_available") else "🔴"
            st.metric("Groq", f"{icon} {health.get('groq_model', 'N/A')}")
        with c2:
            icon = "🟢" if health.get("gemini_available") else "🔴"
            st.metric("Gemini (fallback)", f"{icon} {health.get('gemini_model', 'N/A')}")
