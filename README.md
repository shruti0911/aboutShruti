# Shrutix – Ask Shruti Balwani Anything

A Streamlit chat app powered by LangChain + OpenAI + FAISS that answers questions about Shruti Balwani’s professional journey.

## Local setup

```bash
# clone
pip install -r requirements.txt
streamlit run app.py
```
Add your OpenAI key via the UI prompt or by creating `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY="sk-..."
```

## Deploy on Streamlit Cloud

1. Push this repo to GitHub.
2. In Streamlit Cloud → **New app**, select the repo and `app.py`.
3. In **Settings → Secrets**, add:
   ```
   OPENAI_API_KEY = "sk-..."
   ```
4. Click **Deploy**.

That’s it! The app will build and be live at `https://<username>-<repo>-<hash>.streamlit.app`. 