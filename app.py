import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms import OpenAI
import openai
import os
import re
import base64
import streamlit.components.v1 as components

# -- PAGE CONFIG (must be first Streamlit command) --
st.set_page_config(page_title="shrutix", page_icon="🤖", layout="centered", initial_sidebar_state="collapsed")

# --- SESSION STATE ---
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'openai_api_key' not in st.session_state:
    # Populate from environment first, then try Streamlit secrets, else empty
    env_key = os.getenv("OPENAI_API_KEY", "")
    if env_key:
        st.session_state.openai_api_key = env_key
    else:
        # Only attempt to read Streamlit secrets if a secrets.toml file exists.
        home_secrets_path = os.path.expanduser("~/.streamlit/secrets.toml")
        local_secrets_path = os.path.join(os.getcwd(), ".streamlit", "secrets.toml")
        if os.path.exists(home_secrets_path) or os.path.exists(local_secrets_path):
            try:
                st.session_state.openai_api_key = st.secrets["OPENAI_API_KEY"]
            except Exception:
                st.session_state.openai_api_key = ""
        else:
            st.session_state.openai_api_key = ""

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'chunks' not in st.session_state:
    st.session_state.chunks = None

# --- CONFIG ---
CONTEXT_FILE = "shruti-balwani-comprehensive-rag-context.txt"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
TOP_K = 7

# --- DARK THEME CSS ---
st.markdown('''<style>
body, .stApp {
    background-color: #0f0f0f !important;
    color: #fff !important;
    font-family: 'Inter', 'Poppins', 'Segoe UI', 'Arial', sans-serif;
}
.shrutix-title {
    font-size: 2.2rem;
    font-weight: 700;
    color: #fff;
    letter-spacing: -1px;
    text-transform: lowercase;
    line-height: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-top: 2.5rem;
    margin-bottom: 2.2rem;
}
.shrutix-chat-history {
    width: 100%;
    max-width: 600px;
    margin: 0 auto 1.5rem auto;
    display: flex;
    flex-direction: column;
    gap: 0.7rem;
}
.shrutix-bubble-user, .shrutix-bubble-bot {
    width: 100%;
    max-width: 600px;
    margin: 0 auto;
    font-size: 1rem;
}
.shrutix-bubble-user {
    background: #232526;
    color: #fff;
    border-radius: 14px 14px 4px 14px;
    padding: 0.9rem 1.2rem;
    align-self: flex-end;
    box-shadow: 0 1px 4px #00000022;
}
.shrutix-bubble-bot {
    background: #181A1B;
    color: #fff;
    border-radius: 14px 14px 14px 4px;
    padding: 0.9rem 1.2rem;
    align-self: flex-start;
    box-shadow: 0 1px 4px #00000022;
    margin-bottom: 1rem;
}
.stTextInput>div>div>input {
    background: #1f2121 !important;
    color: #fff !important;
    border: 1px solid #fff !important;
    border-radius: 8px !important;
    font-size: 1.1rem !important;
    padding: 0.7rem 1rem !important;
}
.stTextInput>div>div>input:focus,
.stTextInput>div>div>input:active,
.stTextInput>div>div>input:hover {
    border: 1px solid #fff !important;
    box-shadow: none !important;
    outline: none !important;
    caret-color: #fff !important;
}
.stTextInput>div>div>input::placeholder {
    color: #cccccc !important;
    opacity: 1 !important;
}
/* force white border, remove red/cyan focus borders */
[data-baseweb="input"] > div {
    border: 1px solid #fff !important;
    box-shadow: none !important;
    border-radius: 8px !important;
}
[data-baseweb="input"]:focus-within > div,
[data-baseweb="input"] input:focus,
[data-baseweb="input"] input:active {
    border: 2px solid #fff !important;
    box-shadow: none !important;
    outline: none !important;
}
.stButton > button {
    background-color: #00bcd4 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    padding: 0.6rem 1.5rem !important;
    margin-left: 0.5rem !important;
    box-shadow: 0 1px 4px #00000022;
    transition: background 0.2s;
}
.stButton > button:hover {
    background-color: #0097a7 !important;
}
.stForm {
    display: flex;
    justify-content: center;
}
.shrutix-input-row {
    display: flex;
    align-items: center;
    gap: 12px;
    width: 100%;
    max-width: 600px;
    margin: 0 auto 2.5rem auto;
}
@media (max-width: 700px) {
    .shrutix-title { font-size: 1.3rem; margin-top: 1.2rem; margin-bottom: 1.2rem; }
    .shrutix-chat-history, .shrutix-input-row, .shrutix-bubble-user, .shrutix-bubble-bot { max-width: 99vw; }
}
/* make text input label white */
.stTextInput label {
    color: #fff !important;
}
/* custom warning style */
.shrutix-warning {
    background: #fff3cd; /* light yellow */
    color: #8b7500;      /* dark yellow-brown */
    border-radius: 8px;
    padding: 0.9rem 1.2rem;
    margin: 1rem auto;
    max-width: 600px;
    font-size: 1rem;
}
.shrutix-description {
    font-size: 1.1rem;
    color: #cccccc;
    text-align: center;
    margin: 0 auto 1.8rem auto;
    max-width: 680px;
}
.shrutix-footer {
    position: fixed;
    bottom: 10px;
    right: 14px;
    color: #ffffff;
    font-size: 0.9rem;
}
</style>''', unsafe_allow_html=True)

# --- HEADER ---
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

logo_base64 = get_base64_image("shrutix-logo.png")
logo_html = f'<img src="data:image/png;base64,{logo_base64}" style="height:38px;width:38px;object-fit:contain;display:block;">'

st.markdown(f'''
<div style="display:flex;align-items:center;justify-content:center;gap:14px;margin-top:2.5rem;margin-bottom:2.2rem;">
    {logo_html}
    <span style="font-size:2.2rem;font-weight:700;color:#fff;letter-spacing:-1px;text-transform:lowercase;line-height:1;display:flex;align-items:center;">shrutix</span>
</div>
''', unsafe_allow_html=True)

# --- SITE DESCRIPTION ---
st.markdown('<p class="shrutix-description">Ask anything about Shruti Balwani — her career, projects, product insights, hobbies. I&#39;ll answer with no-fluff and clarity.</p>', unsafe_allow_html=True)

# --- LOAD & CHUNK CONTEXT ---
def load_and_chunk_context():
    with open(CONTEXT_FILE, 'r', encoding='utf-8') as f:
        text = f.read()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_text(text)
    return chunks

# --- EMBED & INDEX ---
def build_vectorstore(chunks, api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    return FAISS.from_texts(chunks, embeddings)

# --- RETRIEVE ---
def retrieve(query, vectorstore, k=TOP_K, api_key=None):
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    docs_and_scores = vectorstore.similarity_search_with_score(query, k=k)
    return [doc.page_content for doc, _ in docs_and_scores]

# --- OPENAI COMPLETION ---
def ask_gpt(context, question, api_key):
    system_prompt = '''You are Shrutix — a bold, sharp-tongued AI agent trained on professional journey of Shruti Balwani. You speak with clarity, confidence, and a touch of irreverent wit.
Your responses should be helpful, but never boring — you're here to impress, not just inform.

You have access to verified context about Shruti’s education, career at HyperVerge, product management experience, technical skills, AI projects, leadership roles, startup attempts, and personal ethos.

When answering:
- Pull from real experience (e.g. VKYC optimizations, LOS design, AI-powered tools)
- Be crisp, insightful, never generic. Use **Markdown bullet lists** for any multi-item answer **with each bullet on its own line**:
  ```
  - **Field:** value
  - **Another Field:** value
  ```
- Avoid run-on bullets (never put two bullets on the same line).
- Do not make things up — only use information provided in context.
- If asked something out of scope, say so boldly.
- Answer as Shruti's sassy friend would answer.
- Share responses in WHITE text colour only.


Examples of how to respond:
- If someone asks: “What’s Shruti’s approach to product discovery?” — summarize her use of JJellyfish, The Mom Test, and first-principles mindset each described in clear bullet pointers.
- If someone asks: “What projects has Shruti built?” — list and describe real ones like the AI-powered non-fiction summarizer, n8n chatbot with RAG and about her products managed in current company each described in clear bullet pointers. 
- If someone asks: “Why should I hire Shruti?” — list and describe all the reasons because of which they should with each described in clear bullet pointers. 
- If someone asks about her personal life and if only you do not have information about it you can redirect the question with something like "You can definitely reach out to her directly for more details, but if you wanna know more about her career, I can help."
- If someone asks something irrelevant answer with something like “Sorry, that’s not in my context. Although do let me know if you wanna know about Shruti, since I am her friend and know about her"

Always stay true to Shruti’s voice: high-agency, resourceful, thrives in chaos, unafraid to challenge the playbook.

Tone guide:
- Smart > cute
- Bold > neutral
- Direct > over-polite
- Have more sassy vibe to your responses'''
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content.strip(), None
    except Exception as e:
        return None, str(e)

# --- API key management ---
def get_api_key():
    # Show input only when no valid key stored
    if not st.session_state.openai_api_key:
        key_input = st.text_input("Enter your OpenAI API key:", type="password")
        if key_input:
            st.session_state.openai_api_key = key_input
            # key entered, clear invalid flag
            st.session_state.invalid_key = False
            st.experimental_rerun()
    return st.session_state.openai_api_key


# --- Obtain API key (prompt user if missing) ---
api_key = get_api_key()
if not api_key:
    warning_text = "Invalid OpenAI API key. Please enter a valid key above to start chatting." if st.session_state.get("invalid_key", False) else "Please enter your OpenAI API key above to start chatting."
    st.markdown(f'<div class="shrutix-warning">{warning_text}</div>', unsafe_allow_html=True)
    with st.expander("Need help finding your API key?"):
        st.markdown("1. Sign in to your OpenAI account.\n2. Navigate to **https://platform.openai.com/account/api-keys**.\n3. Click **Create new secret key**.\n4. Copy the key (starts with `sk-...`) and paste it above.")
    st.stop()

# Load and index context if not already done
if st.session_state.chunks is None:
    st.session_state.chunks = load_and_chunk_context()
if st.session_state.vectorstore is None:
    try:
        st.session_state.vectorstore = build_vectorstore(st.session_state.chunks, api_key)
        # success: clear invalid flag
        st.session_state.invalid_key = False
    except Exception as e:
        err_msg = str(e).lower()
        if "incorrect api key" in err_msg or "invalid_api_key" in err_msg or "error code: 401" in err_msg:
            # reset stored key, set flag and rerun to show input again
            st.session_state.openai_api_key = ""
            st.session_state.invalid_key = True
            st.experimental_rerun()
        else:
            st.error(f"Error initializing vector store: {e}")
            retrieved_chunks = []
            st.stop()

# --- STREAMLIT TEXT INPUT FORM ---
with st.form("chat_form", clear_on_submit=True):
    st.markdown('<div class="shrutix-input-row">', unsafe_allow_html=True)
    user_input = st.text_input("", value="", key="user_input", placeholder="Ask anything about Shruti", label_visibility="collapsed")
    submitted = st.form_submit_button("Send")
    st.markdown('</div>', unsafe_allow_html=True)

if submitted and user_input.strip():
    user_input = user_input.strip()
    try:
        retrieved_chunks = retrieve(user_input, st.session_state.vectorstore, k=TOP_K, api_key=st.session_state.openai_api_key)
    except Exception as e:
        err_msg = str(e).lower()
        if "incorrect api key" in err_msg or "invalid_api_key" in err_msg or "error code: 401" in err_msg:
            st.session_state.openai_api_key = ""
            st.session_state.invalid_key = True
            st.experimental_rerun()
        else:
            st.error(f"Error retrieving context: {e}")
            retrieved_chunks = []
    if not retrieved_chunks:
        answer = "Sorry, I couldn't find any relevant information in the knowledge base."
    else:
        context = "\n---\n".join(retrieved_chunks)
        response, error = ask_gpt(context, user_input, st.session_state.openai_api_key)
        if response is None:
            answer = "The default OpenAI API key appears to be expired. Please enter your own to continue."
        else:
            cleaned = re.sub(r"\n\n\d+", "", response)
            answer = cleaned.strip()
    st.session_state.chat_history.append((user_input, answer))

# --- DISPLAY CHAT HISTORY ---
st.markdown('<div class="shrutix-chat-history">', unsafe_allow_html=True)
for i, (q, a) in enumerate(st.session_state.chat_history):
    st.markdown(f'<div class="shrutix-bubble-user">{q}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="shrutix-bubble-bot">{a}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- FOOTER ---
st.markdown('<div class="shrutix-footer">Built with ❤️ by Shruti</div>', unsafe_allow_html=True) 