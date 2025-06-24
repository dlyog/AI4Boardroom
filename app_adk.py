import os
import requests
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from werkzeug.utils import secure_filename
import pytesseract
from PIL import Image, UnidentifiedImageError
import json
import logging
import hashlib
import uuid
import threading
import functools
import base64
import json

from dotenv import load_dotenv
import asyncio
from agents.core_agents import ALL_AGENTS
from agents.core_agents import build_router_agent
agent_router = build_router_agent()

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

# Load environment variables
load_dotenv()

import google.generativeai as genai
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


logging.basicConfig(level=logging.DEBUG)
app = Flask(__name__)
app.secret_key = 'supersecretkey'

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'pdf'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

session_service = InMemorySessionService()
classify_runner = Runner(
    agent=ALL_AGENTS["document_classifier"],
    app_name="classify_app",
    session_service=session_service
)


def login_required(f):
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def load_users():
    try:
        raw = os.environ.get("USERS_JSON")
        if not raw:
            logging.warning("USERS_JSON env var is missing.")
            return {"users": []}
        users = json.loads(raw)
        for user in users.get("users", []):
            user["password"] = base64.b64decode(user["password"]).decode()
        return users
    except Exception as e:
        logging.error(f"Failed to parse USERS_JSON: {e}")
        return {"users": []}




def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def compute_content_hash(content):
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def extract_text_from_file(filepath, filename):
    try:
        ext = filename.rsplit('.', 1)[1].lower()
        if ext in ['png', 'jpg', 'jpeg', 'gif']:
            return pytesseract.image_to_string(Image.open(filepath))
        elif ext == 'pdf':
            return extract_text_from_pdf(filepath)
    except Exception as e:
        logging.error(f"Text extraction error: {e}")
    return ''

def extract_text_from_pdf(filepath):
    try:
        import PyPDF2
        with open(filepath, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            return ''.join(page.extract_text() for page in reader.pages if page.extract_text())
    except Exception as e:
        logging.error(f"PDF extraction error: {e}")
    return ''

async def _async_classify(text):
    # ensure session exists
    await asyncio.run(session_service.create_session(
        app_name="classify_app",
        user_id="user",
        session_id="session"
    ))
    # prepare message
    from google.genai import types
    msg = types.Content(role="user", parts=[types.Part(text=prefixed_text)])
    result_doc_type = "Other"
    async for event in classify_runner.run_async(user_id="user", session_id="session", new_message=msg):
        if event.is_final_response():
            # extract JSON output
            try:
                data = json.loads(event.content.parts[0].text)
                result_doc_type = data.get("document_type", "Other")
            except Exception:
                pass
    return result_doc_type

def classify_document(text):
    try:
        return asyncio.run(_async_classify(text))
    except Exception as e:
        logging.error(f"Classification failed: {e}")
        return "Other"


def get_agents_for_document_type(doc_type, text):
    try:
        from google.genai import types
        runner = Runner(agent=agent_router, app_name="router_app", session_service=session_service)
        asyncio.run(session_service.create_session(app_name="router_app", user_id="user", session_id="router-session"))
        msg = types.Content(role="user", parts=[types.Part(text=f"Document type: {doc_type}\n\n{text}")])

        agents = []
        for event in runner.run(user_id="user", session_id="router-session", new_message=msg):
            if event.is_final_response():
                raw = event.content.parts[0].text.strip()
                logging.debug(f"[router] Raw agent_router response: {raw}")
                if raw.startswith("{") or raw.startswith("```"):
                    try:
                        cleaned = _strip_fences(raw)
                        parsed = json.loads(cleaned)
                        agents = parsed.get("agents_required", [])
                    except json.JSONDecodeError as e:
                        logging.warning(f"[router] JSON parse error: {e}")
        valid = [a for a in agents if a in ALL_AGENTS and a != "document_classifier" and a != "agent_router"]
        logging.debug(f"[router] Parsed and validated agent list: {valid}")
        return valid
    except Exception as e:
        logging.error(f"Router agent failed: {e}")
        return ["tax_agent"]




def update_document_status(doc_id, status):
    try:
        with open('data.json', 'r') as f:
            docs = json.load(f)
        for doc in docs:
            if doc.get('document_id') == doc_id:
                doc['status'] = status
                break
        with open('data.json', 'w') as f:
            json.dump(docs, f, indent=4)
    except Exception as e:
        logging.error(f"Status update error: {e}")

import re

def _strip_fences(raw: str) -> str:
    # remove leading ```json and trailing ```
    return re.sub(r"^```(?:json)?\s*|```$", "", raw.strip(), flags=re.IGNORECASE)

from agents.company_utils import load_company_profile, reduce_company_profile


def process_document(document_id, text, filename):
    import json, logging, asyncio
    from google.genai import types
    from google.adk.runners import Runner

    logging.debug(f"[{document_id}] Starting document processing.")

    update_document_status(document_id, 'Processing Data')

    company_name = "dlyog"  # TODO: Replace this with session.get('company') in future
    profile = load_company_profile(company_name)
    profile_context = reduce_company_profile(profile)
    prefixed_text = f"Company Profile:\n{profile_context}\n\n{text}"

    # Classification
    asyncio.run(session_service.create_session(
        app_name="classify_app",
        user_id="user",
        session_id=document_id
    ))

    classify_runner = Runner(
        agent=ALL_AGENTS["document_classifier"],
        app_name="classify_app",
        session_service=session_service
    )

    classify_msg = types.Content(role="user", parts=[types.Part(text=prefixed_text)])
    doc_type = "Other"

    for event in classify_runner.run(user_id="user", session_id=document_id, new_message=classify_msg):
        if event.is_final_response():
            raw = event.content.parts[0].text
            logging.debug(f"[{document_id}] Classifier raw response: {raw}")
            try:
                payload = json.loads(_strip_fences(raw))
                doc_type = payload.get("document_type", "Other")
                logging.debug(f"[{document_id}] Detected doc_type: {doc_type}")
            except Exception as e:
                logging.warning(f"[{document_id}] Classification JSON parse error: {e}")
            break

    # SPECIAL CASE: Decision logging only
    if doc_type == "Decision":
        logging.debug(f"[{document_id}] Routing to decision_logger_agent")

        agent = ALL_AGENTS.get("decision_logger_agent")
        asyncio.run(session_service.create_session(
            app_name="decision_logger_agent",
            user_id="user",
            session_id=document_id
        ))

        runner = Runner(agent=agent, app_name="decision_logger_agent", session_service=session_service)
        msg = types.Content(role="user", parts=[types.Part(text=text)])

        result = {}
        for event in runner.run(user_id="user", session_id=document_id, new_message=msg):
            if event.is_final_response():
                summary = event.content.parts[0].text.strip()
                result = {
                    "agent_name": "decision_logger_agent",
                    "comments": summary,
                    "decision": "recorded"
                }
                break

        try:
            with open('data.json', 'r') as f:
                docs = json.load(f)
            for d in docs:
                if d['document_id'] == document_id:
                    d.update({
                        'document_type': "Decision",
                        'agents_required': ["decision_logger_agent"],
                        'agents': [result],
                        'irs_expense_category': "N/A"
                    })
                    break
            with open('data.json', 'w') as f:
                json.dump(docs, f, indent=4)
        except Exception as e:
            logging.error(f"[{document_id}] Error saving decision log: {e}")

        update_document_status(document_id, "Logged")
        return  # 🚫 Skip rest of the pipeline

    # Normal document processing continues below

    agents_required = get_agents_for_document_type(doc_type, text)
    logging.debug(f"[{document_id}] Agents required: {agents_required}")

    try:
        with open('data.json', 'r') as f:
            docs = json.load(f)
        for d in docs:
            if d['document_id'] == document_id:
                d.update({
                    'document_type': doc_type,
                    'agents_required': agents_required,
                    'agents': [],
                    'irs_expense_category': "Unknown"
                })
                break
        with open('data.json', 'w') as f:
            json.dump(docs, f, indent=4)
    except Exception as e:
        logging.error(f"[{document_id}] Error saving classification data: {e}")
        return

    seen_agents = set()
    pending_agents = list(agents_required)

    while pending_agents:
        agent_key = pending_agents.pop(0)
        if agent_key in seen_agents:
            continue
        seen_agents.add(agent_key)

        update_document_status(document_id, f"Review by {agent_key} In Progress")
        logging.debug(f"[{document_id}] Running agent: {agent_key}")

        agent = ALL_AGENTS.get(agent_key)
        if not agent:
            logging.warning(f"[{document_id}] Agent {agent_key} not found, skipping.")
            continue

        asyncio.run(session_service.create_session(
            app_name=agent_key,
            user_id="user",
            session_id=document_id
        ))

        runner = Runner(agent=agent, app_name=agent_key, session_service=session_service)

        msg = types.Content(role="user", parts=[types.Part(text=prefixed_text)])
        result = {}

        for event in runner.run(user_id="user", session_id=document_id, new_message=msg):
            if event.is_final_response():
                raw = event.content.parts[0].text
                logging.debug(f"[{document_id}] Agent {agent_key} raw response: {raw}")
                stripped = _strip_fences(raw)
                try:
                    parsed = json.loads(stripped)
                    result.update(parsed)
                except json.JSONDecodeError:
                    result["comments"] = stripped
                    result["decision"] = result.get("decision", "reject")
                break

        if "comments" not in result:
            result["comments"] = "No additional remarks."
        if "decision" not in result:
            result["decision"] = "approve"

        result["agent_name"] = agent_key

        irs_cat = result.get("irs_expense_category") if agent_key == "tax_agent" else None

        for a in result.get("forward_to_agents", []):
            if a in ALL_AGENTS and a not in seen_agents and a not in pending_agents:
                logging.debug(f"[{document_id}] Agent {agent_key} forwarding to {a}")
                pending_agents.append(a)

        try:
            with open('data.json', 'r') as f:
                docs = json.load(f)
            for d in docs:
                if d['document_id'] == document_id:
                    d['agents'].append(result)
                    if irs_cat:
                        d['irs_expense_category'] = irs_cat
                    break
            with open('data.json', 'w') as f:
                json.dump(docs, f, indent=4)
        except Exception as e:
            logging.error(f"[{document_id}] Error saving agent result: {e}")

    logging.debug(f"[{document_id}] All agents done.")
    update_document_status(document_id, "Processing Complete")








from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.agents import LlmAgent
# in core_agents.py, define a vision agent:
vision_agent = LlmAgent(
    name="vision_tax_agent",
    model="gemini-2.0-flash",
    instruction="""
You are a TAX Expert for Small Businesses in California, specializing in AI Startups.
Look at the provided image input and:
1. If unclear or irrelevant, reply: "No relevant input provided."
2. Else summarize amount, category, date, and include emojis:
😊 if deductible, 😞 if non-deductible, ⚠️ if critical business document.
Return plain English.
"""
)
ALL_AGENTS["vision_tax_agent"] = vision_agent

# then in app_adk.py, create a runner:
session_service = InMemorySessionService()
vision_runner = Runner(agent=vision_agent, app_name="vision_app", session_service=session_service)

def reimburse_with_adk(image_bytes, mimetype):
    import asyncio
    from google.genai import types

        # create the session before streaming
    asyncio.run(session_service.create_session(
        app_name="vision_app",
        user_id="user",
        session_id="session"
    ))

    msg = types.Content(role="user", parts=[ ... ])  # your existing parts

    async def run_it():
        advice = "No response"
        async for event in vision_runner.run_async(user_id="user", session_id="session", new_message=msg):
            if event.is_final_response():
                advice = event.content.parts[0].text
        return advice

    advice = asyncio.run(run_it())

    return advice or "No relevant input provided"

@app.route('/reimbursement', methods=['GET','POST'])
@login_required
def reimbursement():
    if request.method=='GET':
        return render_template('reimbursement.html')
    file = request.files.get('file')
    if not file or not allowed_file(file.filename):
        return jsonify({"status":"error","message":"No valid file"}), 400
    image_data = file.read()
    advice = reimburse_with_adk(image_data, file.mimetype)
    return jsonify({"status":"success","advice": advice}), 200



@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'GET':
        return render_template('upload.html')

    try:
        if 'text' in request.form and request.form['text'].strip():
            text = request.form['text'].strip()
            doc_id = str(uuid.uuid4())
            filename = f"{doc_id}.txt"
            path = os.path.join(UPLOAD_FOLDER, filename)
            with open(path, 'w') as f:
                f.write(text)

            with open('data.json', 'r') as f:
                docs = json.load(f)

            if any(compute_content_hash(text) == d.get("content_hash") for d in docs):
                return jsonify({"status": "duplicate", "message": "Duplicate content"}), 409

            docs.append({
                "document_id": doc_id,
                "filename": filename,
                "original_filename": filename,
                "text": text,
                "status": "Upload in Progress",
                "agents": [],
                "document_type": "Unknown",
                "content_hash": compute_content_hash(text)
            })

            with open('data.json', 'w') as f:
                json.dump(docs, f, indent=4)

            threading.Thread(target=process_document, args=(doc_id, text, None)).start()

            return jsonify({"status": "processing", "document_id": doc_id})

        elif 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                ext = file.filename.rsplit('.', 1)[1].lower()
                doc_id = str(uuid.uuid4())
                unique_filename = f"{doc_id}.{ext}"
                path = os.path.join(UPLOAD_FOLDER, unique_filename)
                file.save(path)

                text = extract_text_from_file(path, unique_filename)
                if not text.strip():
                    return jsonify({"status": "error", "message": "Could not extract text"}), 400

                with open('data.json', 'r') as f:
                    docs = json.load(f)

                if any(compute_content_hash(text) == d.get("content_hash") for d in docs):
                    return jsonify({"status": "duplicate", "message": "Duplicate content"}), 409

                docs.append({
                    "document_id": doc_id,
                    "filename": unique_filename,
                    "original_filename": file.filename,
                    "text": text,
                    "status": "Upload in Progress",
                    "agents": [],
                    "document_type": "Unknown",
                    "content_hash": compute_content_hash(text)
                })

                with open('data.json', 'w') as f:
                    json.dump(docs, f, indent=4)

                threading.Thread(target=process_document, args=(doc_id, text, unique_filename)).start()

                return jsonify({"status": "processing", "document_id": doc_id})

        return jsonify({"status": "error", "message": "No valid input"}), 400

    except Exception as e:
        logging.error(f"Upload error: {e}")
        return jsonify({"status": "error", "message": "Unexpected error"}), 500

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        if 'username' in session:
            return redirect(url_for('dashboard'))
        return render_template('login.html')
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')

        users_data = load_users()
        for user in users_data.get("users", []):
            if user["username"] == username and user["password"] == password:
                session['username'] = username
                session['role'] = user.get('role', 'user')
                return jsonify({"status": "success"})

        return jsonify({"status": "error", "message": "Invalid credentials"}), 401

    except Exception as e:
        logging.error(f"Login error: {e}")
        return jsonify({"status": "error", "message": "Unexpected error"}), 500

@app.route('/login')
def login():
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('index.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/status/<document_id>', methods=['GET'])
@login_required
def get_status(document_id):
    try:
        with open('data.json', 'r') as f:
            docs = json.load(f)
        for doc in docs:
            if doc.get('document_id') == document_id:
                return jsonify({
                    "status": doc.get("status", "Unknown"),
                    "document_id": doc.get("document_id"),
                    "document_type": doc.get("document_type", "Unknown"),
                    "agents": doc.get("agents", []),
                    "irs_expense_category": doc.get("irs_expense_category", "Unknown"),
                    "original_text": doc.get("text", "[Text not available]")
                })
        return jsonify({"status": "error", "message": "Not found"}), 404
    except Exception as e:
        logging.error(f"Status check error: {e}")
        return jsonify({"status": "error"}), 500

    


@app.route('/audit', methods=['GET'])
@login_required
def audit():
    """Renders the audit page, displaying all processed documents and allowing search."""
    try:
        json_file = 'data.json'
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                data = json.load(f)
        else:
            data = [] # Empty list if data.json doesn't exist

        # Ensure each entry has 'agents' and 'document_type' for consistent display
        for entry in data:
            entry.setdefault('agents', [])
            entry.setdefault('document_type', 'Unknown')
            entry.setdefault('irs_expense_category', 'Unknown') # Ensure IRS category is present

        # Implement search functionality
        query = request.args.get('q', '').lower()
        if query:
            data = [entry for entry in data if query in entry['text'].lower() or query in entry['document_type'].lower() or query in entry['original_filename'].lower()]
    except Exception as e:
        logging.error(f"Error loading audit data: {e}")
        data = [] # Return empty data in case of error

    return render_template('audit.html', data=data, query=query)

@app.route('/delete_document', methods=['POST'])
@login_required
def delete_document():
    try:
        data = request.get_json()
        doc_id = data.get('document_id')
        with open('data.json', 'r') as f:
            docs = json.load(f)
        docs = [d for d in docs if d.get('document_id') != doc_id]
        with open('data.json', 'w') as f:
            json.dump(docs, f, indent=4)
        return jsonify({"status": "success"})
    except Exception as e:
        logging.error(f"Delete error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)


