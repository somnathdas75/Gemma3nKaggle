# ====================================================================
# LuckStone Offline Gemma3n Assistant - FIXED VERSION WITH CAMERA
# ====================================================================
import os
import uuid
import sqlite3
import hashlib
import re
import json
import pickle
from glob import glob
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, simpledialog

import ollama
import requests
from PyPDF2 import PdfReader
from docx import Document
import openpyxl
import pytesseract
from PIL import Image, ImageTk, ImageGrab, ImageEnhance, ImageFilter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pyttsx3
import platform
import cv2  # Add this import for camera functionality
import numpy as np

# ========== Global Configuration and Directory Setup ==========
MANUALS_DIR = "./Documents"  # Fixed directory path
DB_FILE = "./luckstone.db"
CACHE_FILE = "./doc_cache.pkl"
GEMA_MODEL_NAME = "gemma3n:latest"
LOGO_PATH = "luck_logo.png"
CAPTURE_DIR = "./Captures"

# Create necessary directories
os.makedirs(MANUALS_DIR, exist_ok=True)
os.makedirs(CAPTURE_DIR, exist_ok=True)

# Global variables for caching
all_docs_text = []
vectorizer = None
docs_vectors = None
docs_paths = []
is_gemma_running = False
tts_engine = None
doc_cache_timestamp = 0

# ========== Database Functions with Connection Pooling ==========
_db_connection = None
_db_lock = threading.Lock()

def get_db_connection():
    global _db_connection
    with _db_lock:
        if _db_connection is None:
            _db_connection = sqlite3.connect(DB_FILE, check_same_thread=False)
            _db_connection.execute("PRAGMA journal_mode=WAL")
        return _db_connection

def setup_database():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            userid TEXT NOT NULL UNIQUE,
            phonenumber_hash TEXT NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            user_id INTEGER NOT NULL,
            query TEXT NOT NULL,
            response TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    conn.commit()

def log_conversation(user_id, conversation_id, query, response):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        with _db_lock:
            cursor.execute("""
                INSERT INTO chat_history (conversation_id, user_id, query, response)
                VALUES (?, ?, ?, ?)
            """, (conversation_id, user_id, query, response))
            conn.commit()
    except Exception as e:
        print(f"❌ Conversation logging error: {e}")

def get_user_chat_history(user_id, limit=50):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        with _db_lock:
            cursor.execute("""
                SELECT query, response, timestamp FROM chat_history
                WHERE user_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (user_id, limit))
            history = cursor.fetchall()
        return history[::-1]
    except Exception as e:
        print(f"❌ Failed to retrieve chat history: {e}")
        return []

def hash_phonenumber(phone_number):
    return hashlib.sha256(phone_number.encode()).hexdigest()

def get_user_id(userid, phonenumber):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        phone_hash = hash_phonenumber(phonenumber)

        with _db_lock:
            cursor.execute("SELECT id FROM users WHERE userid = ? AND phonenumber_hash = ?", (userid, phone_hash))
            user = cursor.fetchone()
            if user:
                return user[0]
            else:
                cursor.execute("INSERT INTO users (userid, phonenumber_hash) VALUES (?, ?)", (userid, phone_hash))
                conn.commit()
                return cursor.lastrowid
    except Exception as e:
        print(f"❌ Database error: {e}")
        return None

# ========== Enhanced Image Processing ==========
def enhance_image_for_ocr(image_path):
    """Enhance image for better OCR results"""
    try:
        img = Image.open(image_path)
        
        # Convert to grayscale
        if img.mode != 'L':
            img = img.convert('L')
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(2.0)
        
        # Apply slight gaussian blur to reduce noise
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        # Resize if too small (OCR works better on larger images)
        if img.width < 800 or img.height < 600:
            scale_factor = max(800 / img.width, 600 / img.height)
            new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        return img
        
    except Exception as e:
        print(f"⚠️ Image enhancement failed: {e}")
        return Image.open(image_path)

def extract_text_from_image(image_path):
    """Enhanced text extraction from images with better OCR"""
    try:
        # Enhance image first
        img = enhance_image_for_ocr(image_path)
        
        # Configure Tesseract for better results
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,;:!?()-_/\+= '
        
        # Extract text with custom configuration
        text = pytesseract.image_to_string(img, config=custom_config)
        
        # Also try with different PSM modes for better results
        if len(text.strip()) < 10:
            custom_config_alt = r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,;:!?()-_/\+= '
            text_alt = pytesseract.image_to_string(img, config=custom_config_alt)
            if len(text_alt.strip()) > len(text.strip()):
                text = text_alt
        
        img.close()
        return text.strip()
        
    except Exception as e:
        print(f"⚠️ OCR failed for {image_path}: {e}")
        return ""

# ========== Optimized Document Processing ==========
def extract_text_from_file(file_path):
    """Extract text from various file formats with enhanced image processing"""
    text = ""
    try:
        file_ext = file_path.lower().split('.')[-1]

        if file_ext == "pdf":
            reader = PdfReader(file_path)
            max_pages = min(len(reader.pages), 10)
            for page_num in range(max_pages):
                page_text = reader.pages[page_num].extract_text()
                if page_text:
                    text += page_text + " "

        elif file_ext == "docx":
            doc = Document(file_path)
            text = " ".join([para.text for para in doc.paragraphs if para.text.strip()])

        elif file_ext in ["xlsx", "xls"]:
            workbook = openpyxl.load_workbook(file_path, read_only=True)
            for sheet in workbook.worksheets:
                for row in sheet.iter_rows(max_row=100):
                    row_text = " ".join([str(cell.value) for cell in row if cell.value is not None])
                    if row_text.strip():
                        text += row_text + " "
            workbook.close()

        elif file_ext in ["png", "jpg", "jpeg", "tiff", "bmp"]:
            text = extract_text_from_image(file_path)

        elif file_ext in ["txt"]:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()

    except Exception as e:
        print(f"⚠️ Could not extract text from {file_path}: {e}")

    return text.strip()

def get_files_modified_time():
    """Get the latest modification time of all files in documents directory"""
    try:
        all_files = []
        for ext in ['*.pdf', '*.docx', '*.doc', '*.xlsx', '*.xls', '*.png', '*.jpg', '*.jpeg', '*.tiff', '*.txt', '*.bmp']:
            all_files.extend(glob(os.path.join(MANUALS_DIR, '**', ext), recursive=True))

        if not all_files:
            return 0

        return max(os.path.getmtime(f) for f in all_files)
    except:
        return 0

def load_doc_cache():
    """Load cached documents if available and up to date"""
    global all_docs_text, docs_paths, vectorizer, docs_vectors, doc_cache_timestamp

    try:
        if not os.path.exists(CACHE_FILE):
            return False

        current_mod_time = get_files_modified_time()

        with open(CACHE_FILE, 'rb') as f:
            cache_data = pickle.load(f)

        if cache_data.get('timestamp', 0) >= current_mod_time:
            all_docs_text = cache_data.get('docs_text', [])
            docs_paths = cache_data.get('docs_paths', [])
            vectorizer = cache_data.get('vectorizer', None)
            docs_vectors = cache_data.get('docs_vectors', None)
            doc_cache_timestamp = cache_data.get('timestamp', 0)
            return len(all_docs_text) > 0

    except Exception as e:
        print(f"⚠️ Failed to load cache: {e}")

    return False

def save_doc_cache():
    """Save processed documents to cache"""
    try:
        cache_data = {
            'docs_text': all_docs_text,
            'docs_paths': docs_paths,
            'vectorizer': vectorizer,
            'docs_vectors': docs_vectors,
            'timestamp': time.time()
        }

        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    except Exception as e:
        print(f"⚠️ Failed to save cache: {e}")

def initialize_retriever():
    """Initialize document retriever with improved error handling"""
    global vectorizer, docs_vectors, all_docs_text, docs_paths

    # Try to load from cache first
    if load_doc_cache():
        print(f"✅ Loaded {len(all_docs_text)} documents from cache")
        return len(all_docs_text) > 0

    # If cache loading fails, process documents
    if not os.path.exists(MANUALS_DIR):
        os.makedirs(MANUALS_DIR, exist_ok=True)
        return False

    all_docs_text = []
    docs_paths = []

    # Get all supported files
    all_files = []
    for ext in ['*.pdf', '*.docx', '*.doc', '*.xlsx', '*.xls', '*.png', '*.jpg', '*.jpeg', '*.tiff', '*.txt', '*.bmp']:
        all_files.extend(glob(os.path.join(MANUALS_DIR, '**', ext), recursive=True))

    if not all_files:
        print("⚠️ No documents found in", MANUALS_DIR)
        return False

    print(f"Processing {len(all_files)} files...")

    # Process files with threading
    max_workers = min(4, len(all_files))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(extract_text_from_file, all_files))

    # Filter valid texts
    for i, text in enumerate(results):
        if text and len(text.strip()) > 20:  # Reduced threshold for better inclusion
            all_docs_text.append(text)
            docs_paths.append(all_files[i])

    if not all_docs_text:
        print("⚠️ No valid text content found in documents")
        return False

    try:
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        docs_vectors = vectorizer.fit_transform(all_docs_text)

        # Save to cache
        save_doc_cache()

        print(f"✅ Processed and cached {len(all_docs_text)} documents")
        return True

    except Exception as e:
        print(f"❌ Error creating vectors: {e}")
        return False

def find_relevant_docs(query, top_k=3):
    """Find relevant documents using TF-IDF similarity with FMI code special handling"""
    if vectorizer is None or docs_vectors is None or docs_vectors.shape[0] == 0:
        return []

    try:
        # Transform query
        query_vector = vectorizer.transform([query])
        
        # Calculate similarities
        cosine_similarities = cosine_similarity(query_vector, docs_vectors).flatten()

        # Get top-k most similar documents
        related_docs_indices = cosine_similarities.argsort()[:-top_k-1:-1]

        relevant_texts = []
        
        # Check if this is an FMI query for special handling
        fmi_patterns = [
            r'FMI\s*(\d+)',           # FMI 6, FMI6
            r'fmi\s*(\d+)',           # fmi 6, fmi6
            r'error\s*code\s*(\d+)',  # error code 6
            r'fault\s*(\d+)',         # fault 6
            r'code\s*(\d+)'           # code 6
        ]
        
        fmi_code = None
        for pattern in fmi_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                fmi_code = match.group(1)
                break
        
        # If FMI code detected, prioritize documents containing that code
        if fmi_code:
            fmi_specific_docs = []
            other_docs = []
            
            for i in related_docs_indices:
                if cosine_similarities[i] > 0.01:  # Very low threshold for FMI queries
                    text = all_docs_text[i]
                    
                    # Check if this document contains the specific FMI code
                    fmi_in_doc = False
                    for pattern in fmi_patterns:
                        if re.search(pattern.replace(r'(\d+)', fmi_code), text, re.IGNORECASE):
                            fmi_in_doc = True
                            break
                    
                    # Limit text length for better performance
                    if len(text) > 2000:
                        text = text[:2000] + "..."
                    
                    if fmi_in_doc:
                        fmi_specific_docs.append(text)
                    else:
                        other_docs.append(text)
            
            # Prioritize FMI-specific documents, then add others
            relevant_texts = fmi_specific_docs[:2] + other_docs[:max(1, 3-len(fmi_specific_docs))]
        else:
            # Regular similarity-based retrieval
            for i in related_docs_indices:
                if cosine_similarities[i] > 0.03:  # Slightly higher threshold for regular queries
                    text = all_docs_text[i]
                    if len(text) > 1500:
                        text = text[:1500] + "..."
                    relevant_texts.append(text)

        return relevant_texts

    except Exception as e:
        print(f"❌ Error in document retrieval: {e}")
        return []

# ========== Ollama Functions ==========
def check_ollama_and_gemma():
    """Check if Ollama server and Gemma3n model are available"""
    try:
        response = requests.get("http://localhost:11434", timeout=5)
        response.raise_for_status()

        ollama_models = ollama.list()
        models_names = [model.get('model', '') for model in ollama_models.get('models', [])]

        return GEMA_MODEL_NAME in models_names

    except Exception as e:
        print(f"❌ Ollama check failed: {e}")
        return False

def query_gemma(prompt):
    """Query Gemma3n model through Ollama"""
    try:
        full_response = ""
        response = ollama.chat(
            model=GEMA_MODEL_NAME,
            messages=[{'role': 'user', 'content': prompt}],
            stream=True,
            options={
                'temperature': 0.7,
                'top_k': 40,
                'top_p': 0.9,
                'num_ctx': 4096  # Increased context window
            }
        )

        for chunk in response:
            if 'message' in chunk and 'content' in chunk['message']:
                full_response += chunk['message']['content']

        return full_response.strip()

    except Exception as e:
        return f"❌ Error querying Gemma3n: {e}. Please check if the model is running."

def extract_fmi_codes_from_text(text):
    """Extract all FMI codes from text using multiple patterns"""
    fmi_patterns = [
        r'FMI\s*(\d+)',           # FMI 6, FMI6
        r'fmi\s*(\d+)',           # fmi 6, fmi6
        r'error\s*code\s*(\d+)',  # error code 6
        r'fault\s*(\d+)',         # fault 6
        r'failure\s*mode\s*identifier\s*(\d+)',  # failure mode identifier 6
    ]
    
    found_codes = set()
    for pattern in fmi_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        found_codes.update(matches)
    
    return list(found_codes)

def get_gemma_response(user_query, relevant_docs, image_text=None):
    """Enhanced response generation with better FMI code handling"""
    context = "\n\n".join(relevant_docs) if relevant_docs else ""
    
    if image_text:
        context += f"\n\nContext from Image (OCR):\n{image_text}"
    
    # Enhanced FMI code detection with multiple patterns
    fmi_patterns = [
        r'FMI\s*(\d+)',           # FMI 6, FMI6
        r'fmi\s*(\d+)',           # fmi 6, fmi6
        r'error\s*code\s*(\d+)',  # error code 6
        r'fault\s*(\d+)',         # fault 6
        r'failure\s*mode\s*(\d+)', # failure mode 6
        r'diagnostic\s*code\s*(\d+)', # diagnostic code 6
    ]
    
    fmi_code = None
    detected_pattern = None
    
    # Check query for FMI codes
    for pattern in fmi_patterns:
        match = re.search(pattern, user_query, re.IGNORECASE)
        if match:
            fmi_code = match.group(1)
            detected_pattern = pattern
            break
    
    # Also check image text if available
    if not fmi_code and image_text:
        for pattern in fmi_patterns:
            match = re.search(pattern, image_text, re.IGNORECASE)
            if match:
                fmi_code = match.group(1)
                detected_pattern = pattern
                break
    
    if fmi_code:
        print(f"🔍 Detected FMI code: {fmi_code}")
        
        # Check if the FMI code exists in the context using all possible patterns
        fmi_found_in_context = False
        context_lower = context.lower()
        
        # Check for various representations of the FMI code in context
        fmi_representations = [
            f"fmi {fmi_code}",
            f"fmi{fmi_code}",
            f"error code {fmi_code}",
            f"fault {fmi_code}",
            f"failure mode {fmi_code}",
            f"diagnostic code {fmi_code}",
            f"code {fmi_code}",
        ]
        
        for representation in fmi_representations:
            if representation in context_lower:
                fmi_found_in_context = True
                break
        
        if fmi_found_in_context or (context and fmi_code in context):
            prompt = f"""You are a technical assistant for LuckStone equipment diagnostics.
The user is asking about FMI (Failure Mode Identifier) code {fmi_code}.

Context from technical documents:
{context}

User Question: {user_query}

Instructions:
1. Focus specifically on FMI code {fmi_code} information from the provided context.
2. Provide a detailed explanation of what this error code means.
3. Include troubleshooting steps if available in the context.
4. Mention any related diagnostic information.
5. Be technical but clear in your explanation.
6. If the context contains information about causes, symptoms, or solutions, include them.

Provide a comprehensive response about FMI code {fmi_code}:"""
        else:
            # If no context found, but we have documents, inform the user
            if len(all_docs_text) > 0:
                prompt = f"""I found your query about FMI code {fmi_code}, but I couldn't locate specific information about this error code in the currently loaded technical documents.

This could mean:
1. The FMI code {fmi_code} might not be covered in the available documents
2. The documents might use different terminology or formatting
3. The OCR might have missed some text if this came from an image

To help you better:
1. Please verify the FMI code number is correct
2. Check if you have the relevant diagnostic or service manuals loaded in the Documents folder
3. Try rephrasing your question with additional context about the equipment or symptoms

If you have additional technical manuals or diagnostic guides, please add them to the Documents folder and restart the application."""
            else:
                prompt = f"""I cannot find information about FMI code {fmi_code} because no technical documents are currently loaded.

Please:
1. Add your technical manuals, diagnostic guides, or service documentation to the './Documents' folder
2. Restart the application to process the new documents
3. Then ask about FMI code {fmi_code} again

The system supports PDF, Word documents, Excel files, and image files containing technical information."""
    else:
        # Handle non-FMI queries
        if context:
            prompt = f"""You are a helpful technical assistant for LuckStone equipment and operations.
Use the following context to answer the user's question accurately and professionally.

Context from technical documents:
{context}

User Question: {user_query}

Instructions:
1. Answer based on the provided context when possible.
2. Be helpful, professional, and technically accurate.
3. If the question cannot be fully answered from the context, clearly state what information is missing.
4. Focus on providing practical and actionable information.
5. Use technical terminology appropriately but explain complex concepts clearly.

Response:"""
        else:
            prompt = f"""I don't have access to relevant technical documents for your question: "{user_query}". 

Please ensure that:
1. Your technical manuals and documentation are placed in the './Documents' folder
2. The documents are in supported formats (PDF, Word, Excel, or image files)
3. Restart the application after adding new documents to process them

Once the documents are loaded, I'll be able to help you with technical questions, troubleshooting, and equipment information."""

    return query_gemma(prompt)

# ========== Camera Capture Functions ==========
class CameraCapture:
    def __init__(self, callback):
        self.callback = callback
        self.camera = None
        self.capture_window = None
        
    def start_capture(self):
        """Start camera capture"""
        try:
            # Try to open camera (0 is usually the default camera)
            self.camera = cv2.VideoCapture(0)
            
            if not self.camera.isOpened():
                # Try other camera indices if default doesn't work
                for i in range(1, 5):
                    self.camera = cv2.VideoCapture(i)
                    if self.camera.isOpened():
                        break
                else:
                    raise Exception("No camera found")
            
            # Set camera properties for better quality
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Create camera window
            self.create_camera_window()
            
        except Exception as e:
            self.callback(None, f"Failed to access camera: {e}")
            self.cleanup()
    
    def create_camera_window(self):
        """Create camera preview window"""
        self.capture_window = tk.Toplevel()
        self.capture_window.title("Camera Capture")
        self.capture_window.geometry("680x580")
        self.capture_window.resizable(False, False)
        
        # Camera frame
        camera_frame = ttk.LabelFrame(self.capture_window, text="Camera Preview", padding="10")
        camera_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Video label
        self.video_label = ttk.Label(camera_frame)
        self.video_label.pack()
        
        # Button frame
        button_frame = ttk.Frame(self.capture_window)
        button_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        # Capture button
        ttk.Button(button_frame, text="📸 Capture Photo", 
                  command=self.capture_photo,
                  style='Accent.TButton').pack(side="left", padx=5)
        
        # Cancel button
        ttk.Button(button_frame, text="❌ Cancel", 
                  command=self.cancel_capture).pack(side="left", padx=5)
        
        # Instructions
        ttk.Label(button_frame, text="Position your document/error screen in the camera view and click Capture Photo",
                 font=("Helvetica", 9)).pack(side="right", padx=5)
        
        # Start video feed
        self.update_video_feed()
        
        # Handle window close
        self.capture_window.protocol("WM_DELETE_WINDOW", self.cancel_capture)
    
    def update_video_feed(self):
        """Update camera video feed"""
        if self.camera and self.capture_window and self.capture_window.winfo_exists():
            try:
                ret, frame = self.camera.read()
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Convert to PIL Image and then to PhotoImage
                    pil_image = Image.fromarray(frame_rgb)
                    photo = ImageTk.PhotoImage(pil_image)
                    
                    # Update label
                    self.video_label.configure(image=photo)
                    self.video_label.image = photo  # Keep a reference
                
                # Schedule next update
                self.capture_window.after(30, self.update_video_feed)  # ~33 FPS
                
            except Exception as e:
                print(f"Camera feed error: {e}")
                self.cleanup()
    
    def capture_photo(self):
        """Capture a photo from the camera with enhanced processing"""
        if not self.camera:
            return
            
        try:
            # Capture frame
            ret, frame = self.camera.read()
            if not ret:
                raise Exception("Failed to capture frame")
            
            # Save captured image
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"camera_capture_{timestamp}.png"
            save_path = os.path.join(CAPTURE_DIR, filename)
            
            # Convert BGR to RGB before saving
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            pil_image.save(save_path)
            
            # Call callback with success
            self.callback(save_path, None)
            
        except Exception as e:
            self.callback(None, f"Failed to capture photo: {e}")
        finally:
            self.cleanup()
    
    def cancel_capture(self):
        """Cancel camera capture"""
        self.cleanup()
    
    def cleanup(self):
        """Clean up camera resources"""
        if self.camera:
            self.camera.release()
            self.camera = None
        
        if self.capture_window:
            self.capture_window.destroy()
            self.capture_window = None

# ========== Screen Capture Functions ==========
class ScreenCapture:
    def __init__(self, callback):
        self.callback = callback
        self.start_x = None
        self.start_y = None
        self.rect_id = None
        self.overlay = None
        
    def start_capture(self):
        """Start the screen capture process"""
        # Create overlay window
        self.overlay = tk.Toplevel()
        self.overlay.attributes('-fullscreen', True)
        self.overlay.attributes('-alpha', 0.3)
        self.overlay.attributes('-topmost', True)
        self.overlay.configure(bg='black')
        
        # Create canvas for drawing selection rectangle
        canvas = tk.Canvas(self.overlay, highlightthickness=0, cursor='cross')
        canvas.configure(bg='black')
        canvas.pack(fill='both', expand=True)
        
        # Bind mouse events
        canvas.bind('<Button-1>', self.on_click)
        canvas.bind('<B1-Motion>', self.on_drag)
        canvas.bind('<ButtonRelease-1>', self.on_release)
        
        # Bind escape key to cancel
        self.overlay.bind('<Escape>', self.cancel_capture)
        self.overlay.focus_set()
        
    def on_click(self, event):
        """Handle mouse click start"""
        self.start_x = event.x
        self.start_y = event.y
        
    def on_drag(self, event):
        """Handle mouse drag to show selection"""
        if self.start_x is not None and self.start_y is not None:
            # Delete previous rectangle
            if self.rect_id:
                event.widget.delete(self.rect_id)
            
            # Draw new rectangle
            self.rect_id = event.widget.create_rectangle(
                self.start_x, self.start_y, event.x, event.y,
                outline='red', width=2, fill='', stipple='gray50'
            )
            
    def on_release(self, event):
        """Handle mouse release to capture area"""
        if self.start_x is None or self.start_y is None:
            return
            
        end_x, end_y = event.x, event.y
        
        # Calculate capture area
        x1 = min(self.start_x, end_x)
        y1 = min(self.start_y, end_y)
        x2 = max(self.start_x, end_x)
        y2 = max(self.start_y, end_y)
        
        # Check if area is large enough
        if abs(x2 - x1) < 20 or abs(y2 - y1) < 20:
            messagebox.showwarning("Invalid Selection", "Selection area too small (minimum 20x20 pixels)")
            self.cancel_capture()
            return
            
        # Hide overlay before capture
        self.overlay.withdraw()
        
        try:
            # Wait a moment for overlay to hide
            time.sleep(0.1)
            
            # Capture screen area
            screenshot = ImageGrab.grab(bbox=(x1, y1, x2, y2))
            
            # Save screenshot
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"screen_capture_{timestamp}.png"
            save_path = os.path.join(CAPTURE_DIR, filename)
            screenshot.save(save_path)
            
            # Call callback with saved image path
            self.callback(save_path)
            
        except Exception as e:
            messagebox.showerror("Capture Error", f"Failed to capture screen: {e}")
        finally:
            self.cleanup()
            
    def cancel_capture(self, event=None):
        """Cancel the capture process"""
        self.cleanup()
        
    def cleanup(self):
        """Clean up overlay window"""
        if self.overlay:
            self.overlay.destroy()
            self.overlay = None

# ========== Main Application ==========
class LuckStoneApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("LuckStone Offline Gemma3n Assistant")
        self.geometry("900x700")
        self.style = ttk.Style(self)
        self.style.theme_use('clam')

        # User data
        self.user_id = ""
        self.phone_number = ""
        self.user_db_id = None
        self.conversation_id = str(uuid.uuid4())
        self.current_image_path = None

        # TTS engine
        self.tts_engine = None
        self.init_tts_async()

        # Logo
        self.logo = None
        self.load_logo_async()

        # Response queue
        self.gemma_response_queue = queue.Queue()
        self.after(100, self.process_queue)

        # Configure styles
        self.configure_styles()

        # Start with login
        self.create_login_frame()

    def configure_styles(self):
        """Configure custom styles"""
        self.style.configure('Header.TFrame', background='#000080')
        self.style.configure('Header.TLabel', 
                           background='#000080', 
                           foreground='white',
                           font=('Helvetica', 14, 'bold'))
        self.style.configure('Accent.TButton', 
                           background='#f46e1e',
                           foreground='white', 
                           font=('Helvetica', 10, 'bold'))

    def init_tts_async(self):
        """Initialize TTS in background thread"""
        def init_tts():
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 180)
            except Exception as e:
                print(f"⚠️ TTS initialization failed: {e}")
                self.tts_engine = None
        
        threading.Thread(target=init_tts, daemon=True).start()

    def load_logo_async(self):
        """Load logo in background thread"""
        def load_logo():
            try:
                if os.path.exists(LOGO_PATH):
                    original_image = Image.open(LOGO_PATH)
                    resized_image = original_image.resize((100, 60), Image.Resampling.LANCZOS)
                    self.logo = ImageTk.PhotoImage(resized_image)
                    original_image.close()
            except Exception as e:
                print(f"⚠️ Logo loading failed: {e}")

        threading.Thread(target=load_logo, daemon=True).start()

    def create_login_frame(self):
        """Create login/registration interface"""
        self.login_frame = ttk.Frame(self, padding="20")
        self.login_frame.pack(expand=True, fill="both")

        # Header
        header = ttk.Frame(self.login_frame, style='Header.TFrame')
        header.pack(fill="x", pady=(0, 20))
        ttk.Label(header, text="LuckStone Login / Register",
                  style='Header.TLabel').pack(pady=10)

        # Content frame
        content_frame = ttk.Frame(self.login_frame)
        content_frame.pack(expand=True)

        # User ID input
        ttk.Label(content_frame, text="User ID:", font=("Helvetica", 12)).pack(pady=5)
        self.userid_entry = ttk.Entry(content_frame, width=30, font=("Helvetica", 12))
        self.userid_entry.pack(pady=5)

        # Phone number input
        ttk.Label(content_frame, text="Phone Number:", font=("Helvetica", 12)).pack(pady=5)
        self.phone_entry = ttk.Entry(content_frame, width=30, font=("Helvetica", 12))
        self.phone_entry.pack(pady=5)

        # Buttons
        button_frame = ttk.Frame(content_frame)
        button_frame.pack(pady=20)

        ttk.Button(button_frame, text="Login", command=self.attempt_login,
                   style='Accent.TButton').pack(side="left", padx=10)
        ttk.Button(button_frame, text="Register", command=self.attempt_login,
                   style='Accent.TButton').pack(side="left", padx=10)

    def attempt_login(self):
        """Attempt to login/register user"""
        user_id = self.userid_entry.get().strip()
        phone_number = self.phone_entry.get().strip()

        if not user_id or not phone_number:
            messagebox.showwarning("Input Required", "Please fill in both User ID and Phone Number.")
            return

        self.userid_entry.config(state='disabled')
        self.phone_entry.config(state='disabled')

        def login_worker():
            user_db_id = get_user_id(user_id, phone_number)
            self.after(0, lambda: self.login_complete(user_db_id, user_id, phone_number))

        threading.Thread(target=login_worker, daemon=True).start()

    def login_complete(self, user_db_id, user_id, phone_number):
        """Complete login process"""
        if user_db_id:
            self.user_db_id = user_db_id
            self.user_id = user_id
            self.phone_number = phone_number
            self.login_frame.destroy()
            self.create_main_frame()
        else:
            self.userid_entry.config(state='normal')
            self.phone_entry.config(state='normal')
            messagebox.showerror("Login Failed", "An error occurred with the database.")

    def create_main_frame(self):
        """Create main application interface"""
        self.main_frame = ttk.Frame(self, padding="10")
        self.main_frame.pack(expand=True, fill="both")

        # Header with logo and title
        self.create_header()
        
        # Initialize retriever in background
        self.init_retriever_async()
        
        # Create chat interface
        self.create_chat_interface()
        
        # Load chat history
        self.load_chat_history_async()

    def create_header(self):
        """Create header with logo and title"""
        header = ttk.Frame(self.main_frame, style='Header.TFrame')
        header.pack(fill="x", pady=(0, 10))

        header_content = ttk.Frame(header)
        header_content.pack(expand=True, fill="both", padx=10, pady=10)

        # Logo (if available)
        if self.logo:
            logo_label = ttk.Label(header_content, image=self.logo, style='Header.TLabel')
            logo_label.pack(side="left", padx=(0, 10))

        # Title and user info
        title_frame = ttk.Frame(header_content, style='Header.TFrame')
        title_frame.pack(side="left", fill="both", expand=True)

        ttk.Label(title_frame, text="LuckStone Technical Assistant",
                  style='Header.TLabel', font=('Helvetica', 16, 'bold')).pack(anchor="w")
        ttk.Label(title_frame, text=f"User: {self.user_id}",
                  style='Header.TLabel', font=('Helvetica', 10)).pack(anchor="w")

        # Status indicator
        self.status_label = ttk.Label(header_content, text="🔄 Initializing...",
                                    style='Header.TLabel', font=('Helvetica', 10))
        self.status_label.pack(side="right")

    def init_retriever_async(self):
        """Initialize document retriever in background"""
        def init_worker():
            success = initialize_retriever()
            gemma_available = check_ollama_and_gemma()
            
            if success and gemma_available:
                status = "✅ Ready"
            elif success:
                status = "⚠️ Documents loaded, Gemma3n not available"
            elif gemma_available:
                status = "⚠️ Gemma3n ready, no documents found"
            else:
                status = "❌ Not ready"
            
            self.after(0, lambda: self.status_label.config(text=status))

        threading.Thread(target=init_worker, daemon=True).start()

    def create_chat_interface(self):
        """Create the main chat interface"""
        # Chat display area
        chat_frame = ttk.LabelFrame(self.main_frame, text="Chat", padding="10")
        chat_frame.pack(fill="both", expand=True, pady=(0, 10))

        # Chat history display
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame, 
            wrap=tk.WORD,
            width=80, 
            height=20,
            font=('Helvetica', 11),
            state='disabled'
        )
        self.chat_display.pack(fill="both", expand=True, pady=(0, 10))

        # Configure text tags for formatting
        self.chat_display.configure(state='normal')
        self.chat_display.tag_configure("user", foreground="blue", font=('Helvetica', 11, 'bold'))
        self.chat_display.tag_configure("assistant", foreground="green", font=('Helvetica', 11))
        self.chat_display.tag_configure("system", foreground="red", font=('Helvetica', 10, 'italic'))
        self.chat_display.configure(state='disabled')

        # Input area
        input_frame = ttk.LabelFrame(self.main_frame, text="Your Question", padding="10")
        input_frame.pack(fill="x", pady=(0, 10))

        # Text input
        input_text_frame = ttk.Frame(input_frame)
        input_text_frame.pack(fill="x", pady=(0, 10))

        self.query_text = tk.Text(
            input_text_frame,
            height=3,
            width=80,
            font=('Helvetica', 11),
            wrap=tk.WORD
        )
        self.query_text.pack(side="left", fill="both", expand=True)

        # Scrollbar for text input
        input_scrollbar = ttk.Scrollbar(input_text_frame, orient="vertical", command=self.query_text.yview)
        input_scrollbar.pack(side="right", fill="y")
        self.query_text.config(yscrollcommand=input_scrollbar.set)

        # Button frame
        button_frame = ttk.Frame(input_frame)
        button_frame.pack(fill="x")

        # Send button
        self.send_button = ttk.Button(
            button_frame, 
            text="📤 Send Query",
            command=self.send_query,
            style='Accent.TButton'
        )
        self.send_button.pack(side="left", padx=(0, 5))

        # Image buttons
        ttk.Button(
            button_frame,
            text="📷 Camera",
            command=self.capture_from_camera
        ).pack(side="left", padx=5)

        ttk.Button(
            button_frame,
            text="🖼️ Screen",
            command=self.capture_screen
        ).pack(side="left", padx=5)

        ttk.Button(
            button_frame,
            text="📁 File",
            command=self.select_image_file
        ).pack(side="left", padx=5)

        # Image indicator
        self.image_indicator = ttk.Label(button_frame, text="", font=('Helvetica', 9, 'italic'))
        self.image_indicator.pack(side="left", padx=10)

        # Utility buttons
        ttk.Button(
            button_frame,
            text="🗑️ Clear",
            command=self.clear_chat
        ).pack(side="right", padx=5)

        ttk.Button(
            button_frame,
            text="🔄 Refresh Docs",
            command=self.refresh_documents
        ).pack(side="right", padx=5)

        # Bind Enter key to send
        self.query_text.bind('<Control-Return>', lambda e: self.send_query())
        self.query_text.bind('<Return>', self.handle_enter)

    def handle_enter(self, event):
        """Handle Enter key press"""
        if event.state & 0x4:  # Control+Enter
            self.send_query()
        else:
            # Allow normal newline
            return None

    def add_message(self, sender, message, tag=""):
        """Add a message to the chat display"""
        self.chat_display.configure(state='normal')
        
        timestamp = time.strftime("%H:%M:%S")
        
        if sender == "You":
            self.chat_display.insert(tk.END, f"[{timestamp}] You: ", "user")
            self.chat_display.insert(tk.END, f"{message}\n\n")
        elif sender == "Assistant":
            self.chat_display.insert(tk.END, f"[{timestamp}] Assistant: ", "assistant")
            self.chat_display.insert(tk.END, f"{message}\n\n")
        else:
            self.chat_display.insert(tk.END, f"[{timestamp}] {sender}: ", "system")
            self.chat_display.insert(tk.END, f"{message}\n\n")
        
        self.chat_display.configure(state='disabled')
        self.chat_display.see(tk.END)

    def capture_from_camera(self):
        """Capture image from camera"""
        def camera_callback(image_path, error):
            if error:
                self.add_message("System", f"Camera error: {error}")
            elif image_path:
                self.current_image_path = image_path
                self.image_indicator.config(text=f"📷 {os.path.basename(image_path)}")
                self.add_message("System", f"Camera image captured: {os.path.basename(image_path)}")

        camera_capture = CameraCapture(camera_callback)
        camera_capture.start_capture()

    def capture_screen(self):
        """Capture screen area"""
        def screen_callback(image_path):
            if image_path:
                self.current_image_path = image_path
                self.image_indicator.config(text=f"🖼️ {os.path.basename(image_path)}")
                self.add_message("System", f"Screen capture saved: {os.path.basename(image_path)}")

        screen_capture = ScreenCapture(screen_callback)
        screen_capture.start_capture()

    def select_image_file(self):
        """Select an image file"""
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.gif"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            self.current_image_path = file_path
            self.image_indicator.config(text=f"📁 {os.path.basename(file_path)}")
            self.add_message("System", f"Image file selected: {os.path.basename(file_path)}")

    def send_query(self):
        """Send query to Gemma3n"""
        query = self.query_text.get("1.0", tk.END).strip()
        
        if not query:
            return

        # Disable send button during processing
        self.send_button.config(state='disabled', text="🔄 Processing...")
        
        # Add user message to chat
        if self.current_image_path:
            self.add_message("You", f"{query} [📷 Image: {os.path.basename(self.current_image_path)}]")
        else:
            self.add_message("You", query)

        # Clear input
        self.query_text.delete("1.0", tk.END)

        # Process query in background
        def query_worker():
            try:
                # Extract text from image if available
                image_text = ""
                if self.current_image_path:
                    try:
                        image_text = extract_text_from_image(self.current_image_path)
                        if image_text:
                            print(f"🖼️ Extracted text from image: {image_text[:100]}...")
                    except Exception as e:
                        print(f"⚠️ Failed to extract text from image: {e}")

                # Find relevant documents
                search_query = query
                if image_text:
                    search_query += " " + image_text
                
                relevant_docs = find_relevant_docs(search_query, top_k=3)
                
                # Get response from Gemma3n
                response = get_gemma_response(query, relevant_docs, image_text)
                
                # Log conversation
                if self.user_db_id:
                    log_conversation(self.user_db_id, self.conversation_id, query, response)

                # Add to queue for main thread processing
                self.gemma_response_queue.put(('response', response))

            except Exception as e:
                self.gemma_response_queue.put(('error', str(e)))

        # Start worker thread
        threading.Thread(target=query_worker, daemon=True).start()

        # Clear current image after sending
        self.current_image_path = None
        self.image_indicator.config(text="")

    def process_queue(self):
        """Process responses from background threads"""
        try:
            while not self.gemma_response_queue.empty():
                msg_type, content = self.gemma_response_queue.get_nowait()
                
                if msg_type == 'response':
                    self.add_message("Assistant", content)
                elif msg_type == 'error':
                    self.add_message("System", f"Error: {content}")

                # Re-enable send button
                self.send_button.config(state='normal', text="📤 Send Query")

        except queue.Empty:
            pass

        # Schedule next check
        self.after(100, self.process_queue)

    def clear_chat(self):
        """Clear chat history"""
        self.chat_display.configure(state='normal')
        self.chat_display.delete("1.0", tk.END)
        self.chat_display.configure(state='disabled')
        self.conversation_id = str(uuid.uuid4())

    def refresh_documents(self):
        """Refresh document cache"""
        self.status_label.config(text="🔄 Refreshing documents...")
        
        def refresh_worker():
            try:
                # Clear cache
                if os.path.exists(CACHE_FILE):
                    os.remove(CACHE_FILE)
                
                # Reinitialize
                success = initialize_retriever()
                gemma_available = check_ollama_and_gemma()
                
                if success and gemma_available:
                    status = "✅ Ready"
                    message = f"Documents refreshed successfully! Loaded {len(all_docs_text)} documents."
                elif success:
                    status = "⚠️ Documents loaded, Gemma3n not available"
                    message = f"Documents refreshed! Loaded {len(all_docs_text)} documents, but Gemma3n is not available."
                else:
                    status = "❌ No documents found"
                    message = "No valid documents found in the Documents folder."
                
                self.after(0, lambda: (
                    self.status_label.config(text=status),
                    self.add_message("System", message)
                ))
                
            except Exception as e:
                self.after(0, lambda: (
                    self.status_label.config(text="❌ Refresh failed"),
                    self.add_message("System", f"Document refresh failed: {e}")
                ))

        threading.Thread(target=refresh_worker, daemon=True).start()

    def load_chat_history_async(self):
        """Load chat history in background"""
        def load_history():
            if self.user_db_id:
                history = get_user_chat_history(self.user_db_id, limit=10)
                for query, response, timestamp in history:
                    self.after(0, lambda q=query, r=response: (
                        self.add_message("You", q),
                        self.add_message("Assistant", r)
                    ))

        threading.Thread(target=load_history, daemon=True).start()

# ========== Main Execution ==========
def main():
    """Main function to run the application"""
    try:
        # Setup database
        setup_database()
        
        # Create and run application
        app = LuckStoneApp()
        app.mainloop()
        
    except KeyboardInterrupt:
        print("\n👋 Application interrupted by user")
    except Exception as e:
        print(f"❌ Application error: {e}")
        messagebox.showerror("Application Error", f"An error occurred: {e}")
    finally:
        # Cleanup
        if _db_connection:
            _db_connection.close()

if __name__ == "__main__":
    main()