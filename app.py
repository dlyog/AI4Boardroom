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
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.DEBUG)









logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'pdf'}

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# LLAMA API Endpoint
LLAMA_API_URL = "http://192.168.86.42:8893/llama3-completion"



# Login required decorator
def login_required(f):
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def load_users():
    """Load users from users.json file."""
    try:
        with open('users.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error("users.json file not found")
        return {"users": []}
    except json.JSONDecodeError:
        logging.error("Error decoding users.json")
        return {"users": []}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def compute_content_hash(content):
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def extract_text_from_file(filepath, filename):
    try:
        extension = filename.rsplit('.', 1)[1].lower()
        if extension in ['png', 'jpg', 'jpeg', 'gif']:
            text = pytesseract.image_to_string(Image.open(filepath))
        elif extension == 'pdf':
            text = extract_text_from_pdf(filepath)
        else:
            text = ''
        return text
    except Exception as e:
        logging.error(f"Error extracting text from file {filename}: {e}")
        return ''


def extract_text_from_pdf(filepath):
    try:
        import PyPDF2
        text = ''
        with open(filepath, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()
        return text
    except Exception as e:
        logging.error(f"Error extracting text from PDF file: {e}")
        return ''


def extract_field(response_string, field_name):
    try:
        start_index = response_string.find(f'"{field_name}":')
        if start_index == -1:
            return None

        start_index += len(f'"{field_name}":')
        end_index = response_string.find(",", start_index)
        if end_index == -1:
            end_index = response_string.find("\n", start_index)
        if end_index == -1:
            end_index = len(response_string)

        value = response_string[start_index:end_index].strip()
        return value.strip('" ')
    except Exception as e:
        logging.error(f"Error extracting field {field_name}: {e}")
        return None


def classify_document(text):
    try:
        prompt = f"""
You are an AI assistant. Analyze the provided text and classify the type of document it is. The possible document types are:

- Expense Report
- Requirement Document
- Idea
- Other

Output your response in the following JSON format:
{{
    "document_type": "<Document Type>"
}}

Please output your response as valid JSON, and nothing else.

Now process this text:
{text}
"""
        payload = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that formats data precisely in JSON."},
                {"role": "user", "content": prompt}
            ],
            "max_gen_len": 150,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True
        }
        response = requests.post(LLAMA_API_URL, json=payload)
        response.raise_for_status()

        result = response.json()
        logging.info(f"LLM Response for classification: {result}")

        response_content_raw = result.get("response", "{}")
        try:
            response_content = json.loads(response_content_raw)
            document_type = response_content.get("document_type", "Other")
        except json.JSONDecodeError:
            logging.warning("LLM response is not valid JSON. Attempting to extract fields with string manipulation.")
            document_type = extract_field(response_content_raw, "document_type")
        return document_type if document_type else "Other"
    except Exception as e:
        logging.error(f"Error classifying document: {e}")
        return "Other"


def get_agents_for_document_type(document_type):
    agent_mapping = {
        "Expense Report": ["CFO Agent", "Ethics Agent", "Tax Agent"],
        "Requirement Document": ["CTO Agent", "Tax Agent"],
        "Idea": ["CEO Agent", "Chief Human Resource Officer", "Tax Agent"],
        "Other": ["Chief Human Resource Officer", "Tax Agent"]
    }
    return agent_mapping.get(document_type, ["Chief Human Resource Officer", "Tax Agent"])



def process_with_agent(agent_name, text):
    try:
        if agent_name == "Tax Agent":
            prompt = f"""
You are {agent_name}. Please review the following document and classify it into an IRS Expense Category. The possible categories are:

- Cost of Goods Sold (COGS)
- Operating Expenses
- Business Use of Home
- Depreciation
- Travel Expenses
- Meals and Entertainment
- Advertising and Marketing
- Professional Fees
- Vehicle Expenses
- Employee Benefits
- Interest
- Taxes
- Miscellaneous Expenses
- Income

Document text:
{text}

Provide your classification in the following JSON format:
{{
    "decision": "<approve/reject>",
    "irs_expense_category": "<IRS Expense Category>",
    "comments": "<Your comments here>"
}}

Please output your response as valid JSON, and nothing else.
"""
        else:
            prompt = f"""
You are {agent_name}. Please review the following document and decide whether to approve or reject it.

Document text:
{text}

Provide your decision in the following JSON format:
{{
    "decision": "<approve/reject>",
    "comments": "<Your comments here>"
}}

Please output your response as valid JSON, and nothing else.
"""

        payload = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that formats data precisely in JSON."},
                {"role": "user", "content": prompt}
            ],
            "max_gen_len": 150,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True
        }
        response = requests.post(LLAMA_API_URL, json=payload)
        response.raise_for_status()

        result = response.json()
        logging.info(f"LLM Response from {agent_name}: {result}")

        response_content_raw = result.get("response", "{}")
        try:
            response_content = json.loads(response_content_raw)
        except json.JSONDecodeError:
            logging.warning(f"LLM response from {agent_name} is not valid JSON. Attempting to extract fields.")
            decision = extract_field(response_content_raw, "decision")
            comments = extract_field(response_content_raw, "comments")
            if agent_name == "Tax Agent":
                irs_expense_category = extract_field(response_content_raw, "irs_expense_category")
            else:
                irs_expense_category = None
            return {
                "agent_name": agent_name,
                "decision": decision or "reject",
                "comments": comments or "",
                "irs_expense_category": irs_expense_category
            }
        else:
            decision = response_content.get("decision", "reject")
            comments = response_content.get("comments", "")
            if agent_name == "Tax Agent":
                irs_expense_category = response_content.get("irs_expense_category", "Unknown")
            else:
                irs_expense_category = None
            return {
                "agent_name": agent_name,
                "decision": decision,
                "comments": comments,
                "irs_expense_category": irs_expense_category
            }
    except Exception as e:
        logging.error(f"Error processing with {agent_name}: {e}")
        return {
            "agent_name": agent_name,
            "decision": "reject",
            "comments": "Error processing",
            "irs_expense_category": None
        }

def update_document_status(document_id, status):
    """Update the status of a document in data.json."""
    try:
        json_file = 'data.json'
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                data_list = json.load(f)
            for data in data_list:
                if data.get('document_id') == document_id:
                    data['status'] = status
                    break
            with open(json_file, 'w') as f:
                json.dump(data_list, f, indent=4)
    except Exception as e:
        logging.error(f"Error updating status in data.json for document {document_id}: {e}")


def process_document(document_id, text, filename):
    """Process the document by classifying it and processing with agents."""
    # Update status to 'Processing Data'
    update_document_status(document_id, 'Processing Data')

    document_type = classify_document(text)
    agents_required = get_agents_for_document_type(document_type)

    # Update the data with document_type and agents_required
    try:
        json_file = 'data.json'
        with open(json_file, 'r') as f:
            data_list = json.load(f)
        for data in data_list:
            if data.get('document_id') == document_id:
                data['document_type'] = document_type
                data['agents_required'] = agents_required
                data['agents'] = []
                # Add IRS Expense Category field
                data['irs_expense_category'] = "Unknown"
                # Status remains 'Processing Data'
                break
        else:
            logging.error(f"Document ID {document_id} not found in data.json")
            return
        with open(json_file, 'w') as f:
            json.dump(data_list, f, indent=4)
    except Exception as e:
        logging.error(f"Error updating data.json for document {document_id}: {e}")

    # Process with each agent and update progress
    for agent_name in agents_required:
        # Update status to indicate which agent is processing
        update_document_status(document_id, f'Review by {agent_name} In Progress')

        agent_result = process_with_agent(agent_name, text)

        # Update data.json with the agent's result
        try:
            with open(json_file, 'r') as f:
                data_list = json.load(f)
            for data in data_list:
                if data.get('document_id') == document_id:
                    data['agents'].append(agent_result)
                    # If agent is Tax Agent, update the IRS Expense Category
                    if agent_name == "Tax Agent":
                        data['irs_expense_category'] = agent_result.get('irs_expense_category', "Unknown")
                    break
            with open(json_file, 'w') as f:
                json.dump(data_list, f, indent=4)
        except Exception as e:
            logging.error(f"Error updating data.json with agent result for document {document_id}: {e}")

    # After processing all agents, update the overall status
    try:
        with open(json_file, 'r') as f:
            data_list = json.load(f)
        for data in data_list:
            if data.get('document_id') == document_id:
                if all(agent['decision'] == 'approve' for agent in data['agents']):
                    data['status'] = 'Processing Complete'
                else:
                    data['status'] = 'Processing Failed'
                break
        with open(json_file, 'w') as f:
            json.dump(data_list, f, indent=4)
    except Exception as e:
        logging.error(f"Error updating overall status in data.json for document {document_id}: {e}")

@app.route('/update_status', methods=['POST'])
@login_required
def update_status():
    try:
        data = request.get_json()
        document_id = data.get('document_id')
        new_status = data.get('status')

        json_file = 'data.json'
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                data_list = json.load(f)
            for entry in data_list:
                if entry.get('document_id') == document_id:
                    entry['status'] = new_status
                    break
            with open(json_file, 'w') as f:
                json.dump(data_list, f, indent=4)
            return jsonify({"status": "success"})
        else:
            return jsonify({"status": "error", "message": "Data file not found"}), 404
    except Exception as e:
        logging.error(f"Error updating status for document {document_id}: {e}")
        return jsonify({"status": "error", "message": "An error occurred"}), 500

@app.route('/delete_document', methods=['POST'])
@login_required
def delete_document():
    try:
        data = request.get_json()
        document_id = data.get('document_id')

        json_file = 'data.json'
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                data_list = json.load(f)
            data_list = [entry for entry in data_list if entry.get('document_id') != document_id]
            with open(json_file, 'w') as f:
                json.dump(data_list, f, indent=4)
            return jsonify({"status": "success"})
        else:
            return jsonify({"status": "error", "message": "Data file not found"}), 404
    except Exception as e:
        logging.error(f"Error deleting document {document_id}: {e}")
        return jsonify({"status": "error", "message": "An error occurred"}), 500

@app.route('/reimbursement', methods=['GET', 'POST'])
@login_required
def reimbursement():
    if request.method == 'GET':
        logging.info('Rendering reimbursement.html')
        return render_template('reimbursement.html')

    if request.method == 'POST':
        logging.info('Received POST request to /reimbursement')
        response = None
        try:
            if 'file' not in request.files or not request.files['file'].filename:
                logging.error("No file provided in the request.")
                return jsonify({"status": "error", "message": "No file provided."}), 400

            file = request.files['file']
            if not allowed_file(file.filename):
                logging.error("File type not allowed.")
                return jsonify({"status": "error", "message": "File type not allowed."}), 400

            # Read and encode image
            image_data = file.read()
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            # Load credentials and prepare request
            load_dotenv()
            API_URL = "http://dlyog05:8000/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "X-API-Key": os.getenv('API_PASSWORD'),
                "X-Username": os.getenv('API_USERNAME')
            }

            
            # Updated instruction for simpler responses
            instruction = """
You are a TAX Expert for Small Businesses in California, specializing in AI Startups. The startup's name is DLYog Lab. Examine the provided image, which could be a receipt, document, or other business-related file. Follow these steps:

1. If the image is unclear or irrelevant, respond with: "No relevant input provided."
2. If it's relevant to the business:
   - Explain briefly in simple English.
   - Highlight key details such as the amount, category, or date.
   - Use emojis for clarity:
     - 😊 for tax-deductible items.
     - 😞 for non-deductible items.
     - ⚠️ for significant business documents or decision points.

The response should look like valid english senetences.
"""


            # Prepare payload
            payload = {
                "model": "vision_llm",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": instruction}
                        ]
                    }
                ],
                "temperature": 1.0,
                "max_tokens": 150,
                "image": image_base64
            }

            # Make request
            response = requests.post(API_URL, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
            # Extract response content
            advice = result['choices'][0]['message']['content']
            logging.info(f"Tax advice response: {advice}")
            
            # Return the plain text response
            return jsonify({
                "status": "success",
                "advice": advice.strip()
            }), 200

        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed: {str(e)}")
            return jsonify({
                "status": "error",
                "message": "Failed to communicate with API"
            }), 500
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}", exc_info=True)
            return jsonify({
                "status": "error",
                "message": "An unexpected error occurred"
            }), 500


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        if 'username' in session:
            return redirect(url_for('dashboard'))
        return render_template('login.html')

    if request.method == 'POST':
        try:
            data = request.get_json()
            username = data.get('username')
            password = data.get('password')

            users_data = load_users()
            
            for user in users_data.get('users', []):
                if user['username'] == username and user['password'] == password:
                    session['username'] = username
                    session['role'] = user['role']
                    return jsonify({"status": "success", "message": "Login successful"})
            
            return jsonify({"status": "error", "message": "Invalid credentials"}), 401
        
        except Exception as e:
            logging.error(f"Login error: {e}")
            return jsonify({"status": "error", "message": "An error occurred"}), 500

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('index.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

def is_duplicate_text(content, existing_data):
    new_content_hash = compute_content_hash(content)
    for entry in existing_data:
        if entry.get("content_hash") == new_content_hash:
            return True, "Content is similar to an existing document"
    return False, ""


def is_duplicate(filename, content):
    json_file = 'data.json'
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            existing_data = json.load(f)

        new_content_hash = compute_content_hash(content)

        for entry in existing_data:
            existing_hash = entry.get("content_hash")
            if existing_hash and existing_hash == new_content_hash:
                return True, "Content is similar to an existing document"
    return False, ""


def sanitize_text(text):
    """Sanitize text to remove problematic characters."""
    return text.replace('"', '').replace("'", '').replace("`", '').strip() if text else ''

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_file():
    if request.method == 'GET':
        return render_template('upload.html')

    if request.method == 'POST':
        try:
            if 'text' in request.form and request.form['text'].strip():
                # Process text input
                text = sanitize_text(request.form['text'].strip())
                logging.info(f"Received text input: {text}")

                # Generate a unique document ID and filename
                document_id = str(uuid.uuid4())
                unique_filename = f"{document_id}.txt"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

                # Save the text to a .txt file
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(text)
                logging.info(f"Text saved to {filepath}")

                data = {
                    'document_id': document_id,
                    'filename': unique_filename,
                    'original_filename': unique_filename,
                    'text': text,
                    'status': 'Upload in Progress',
                    'agents': [],
                    'document_type': 'Unknown',
                    'content_hash': compute_content_hash(text)
                }

                json_file = 'data.json'
                if os.path.exists(json_file):
                    with open(json_file, 'r') as f:
                        existing_data = json.load(f)
                else:
                    existing_data = []

                # Perform duplicate content check
                is_dup, message = is_duplicate_text(text, existing_data)
                if is_dup:
                    return {"status": "duplicate", "message": message}, 409

                existing_data.append(data)

                with open(json_file, 'w') as f:
                    json.dump(existing_data, f, indent=4)

                threading.Thread(target=process_document, args=(document_id, text, None)).start()

                return {
                    "status": "processing",
                    "message": "Document is being processed.",
                    "document_id": document_id
                }, 200

            elif 'file' in request.files and request.files['file'].filename:
                file = request.files['file']
                if file and allowed_file(file.filename):
                    # Generate a unique filename
                    original_filename = secure_filename(file.filename)
                    extension = original_filename.rsplit('.', 1)[1].lower() if '.' in original_filename else ''
                    unique_filename = f"{uuid.uuid4()}.{extension}"
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                    file.save(filepath)
                    logging.info(f"File saved at {filepath}")

                    
                    text = sanitize_text(extract_text_from_file(filepath, unique_filename))

                    # Check for empty text extraction for all file types
                    if not text.strip():
                        logging.warning(f"No text extracted from the file: {original_filename}")
                        return {"status": "error", "message": "Failed to extract text from the uploaded file."}, 400
                    else:
                        logging.info(f"Extracted text: {text}")

                    # Perform duplicate check
                    is_dup, message = is_duplicate(unique_filename, text)
                    if is_dup:
                        return {"status": "duplicate", "message": message}, 409

                    document_id = str(uuid.uuid4())

                    data = {
                        'document_id': document_id,
                        'filename': unique_filename,
                        'original_filename': original_filename,
                        'text': text,
                        'status': 'Upload in Progress',
                        'agents': [],
                        'document_type': 'Unknown',
                        'content_hash': compute_content_hash(text)
                    }

                    json_file = 'data.json'
                    if os.path.exists(json_file):
                        with open(json_file, 'r') as f:
                            existing_data = json.load(f)
                    else:
                        existing_data = []

                    existing_data.append(data)

                    with open(json_file, 'w') as f:
                        json.dump(existing_data, f, indent=4)

                    threading.Thread(target=process_document, args=(document_id, text, unique_filename)).start()

                    return {
                        "status": "processing",
                        "message": "Document is being processed.",
                        "document_id": document_id
                    }, 200
                else:
                    return {"status": "error", "message": "File type not allowed."}, 400
            else:
                logging.error("No file or text in the request.")
                return {"status": "error", "message": "No file or text provided."}, 400

        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            return {"status": "error", "message": "An error occurred while processing your input."}, 500


@app.route('/status/<document_id>', methods=['GET'])
@login_required
def get_status(document_id):
    try:
        json_file = 'data.json'
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                data_list = json.load(f)
            for data in data_list:
                if data.get('document_id') == document_id:
                    return data
        return {"status": "error", "message": "Document not found"}, 404
    except Exception as e:
        logging.error(f"Error getting status for document {document_id}: {e}")
        return {"status": "error", "message": "An error occurred"}, 500


@app.route('/audit', methods=['GET'])
@login_required
def audit():
    try:
        json_file = 'data.json'
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                data = json.load(f)
        else:
            data = []

        # Ensure each entry has 'agents' and 'document_type' defined
        for entry in data:
            entry.setdefault('agents', [])
            entry.setdefault('document_type', 'Unknown')

        query = request.args.get('q', '').lower()
        if query:
            data = [entry for entry in data if query in entry['text'].lower() or query in entry['document_type'].lower()]
    except Exception as e:
        logging.error(f"Error loading audit data: {e}")
        data = []

    return render_template('audit.html', data=data, query=query)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8872, debug=True)
