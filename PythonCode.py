# BigQuery_AI_Appr2_IT_Triage_System_SomnathDas_OpenAI.py
import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
import os
import tempfile
from typing import List, Dict, Any
import time
import PyPDF2
import docx
from pathlib import Path
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import hashlib

# Hardcoded OpenAI API Key
OPENAI_API_KEY = "APIKeYpaste"  # Replace with your actual key

class OpenAIEmbeddingSystem:
    def __init__(self):
        self.api_key = OPENAI_API_KEY
        self.client = OpenAI(api_key=self.api_key)
        
        # Initialize knowledge base
        self.knowledge_base = self._initialize_knowledge_base()
        self.documents_base = []
        self.embeddings_cache = {}
        
        st.success("✅ OpenAI Embedding System Initialized!")
    
    def _initialize_knowledge_base(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": 1,
                "category": "Password Issues",
                "symptoms": ["forgot password", "can't login", "password reset", "locked out", "invalid password"],
                "solution": "Go to the password reset portal at https://passwordreset.company.com or contact IT helpdesk at ext. 5555",
                "priority": "High",
                "escalation": "IT Helpdesk",
                "search_text": "password issues forgot password can't login password reset locked out invalid password"
            },
            {
                "id": 2,
                "category": "Network Connectivity",
                "symptoms": ["no internet", "wifi not working", "can't connect", "network issues", "slow connection"],
                "solution": "Check physical connections, restart router, or contact network support at ext. 5560",
                "priority": "High",
                "escalation": "Network Team",
                "search_text": "network connectivity no internet wifi not working can't connect network issues slow connection"
            },
            {
                "id": 3,
                "category": "Software Installation",
                "symptoms": ["need to install software", "admin rights required", "installation failed", "access denied"],
                "solution": "Submit software request form at https://itportal.company.com/software-request",
                "priority": "Medium",
                "escalation": "Software Team",
                "search_text": "software installation need to install software admin rights required installation failed access denied"
            },
            {
                "id": 4,
                "category": "Email Problems",
                "symptoms": ["outlook not working", "can't send emails", "email sync issues", "attachment problems"],
                "solution": "Clear Outlook cache, restart application, or contact email support at ext. 5570",
                "priority": "Medium",
                "escalation": "Email Team",
                "search_text": "email problems outlook not working can't send emails email sync issues attachment problems"
            },
            {
                "id": 5,
                "category": "Hardware Issues",
                "symptoms": ["computer slow", "printer not working", "keyboard broken", "monitor issues", "hardware failure"],
                "solution": "Contact hardware support at ext. 5580 or submit a ticket at https://itportal.company.com/hardware",
                "priority": "Medium",
                "escalation": "Hardware Team",
                "search_text": "hardware issues computer slow printer not working keyboard broken monitor issues hardware failure"
            },
            {
                "id": 6,
                "category": "VPN Access",
                "symptoms": ["vpn not connecting", "remote access issues", "can't access network remotely"],
                "solution": "Update VPN client, check credentials, or contact remote access team at ext. 5590",
                "priority": "High",
                "escalation": "Network Team",
                "search_text": "vpn access vpn not connecting remote access issues can't access network remotely"
            },
            {
                "id": 7,
                "category": "Database Issues",
                "symptoms": ["database connection", "sql errors", "can't access database", "query performance"],
                "solution": "Check connection strings, verify permissions, or contact DBA team at ext. 5600",
                "priority": "High",
                "escalation": "Database Team",
                "search_text": "database issues database connection sql errors can't access database query performance"
            },
            {
                "id": 8,
                "category": "Security Concerns",
                "symptoms": ["suspicious activity", "phishing email", "malware detected", "security alert"],
                "solution": "Immediately contact security team at ext. 5610 or security@company.com",
                "priority": "Critical",
                "escalation": "Security Team",
                "search_text": "security concerns suspicious activity phishing email malware detected security alert"
            }
        ]
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embeddings using OpenAI API"""
        if text in self.embeddings_cache:
            return self.embeddings_cache[text]
        
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            embedding = response.data[0].embedding
            self.embeddings_cache[text] = embedding
            return embedding
        except Exception as e:
            st.warning(f"OpenAI embedding failed: {e}. Using fallback embedding.")
            return self._fallback_embedding(text)
    
    def _fallback_embedding(self, text: str) -> List[float]:
        """Fallback embedding generation"""
        # Create a simple hash-based embedding
        hash_obj = hashlib.md5(text.encode())
        hash_int = int(hash_obj.hexdigest()[:8], 16)
        
        # Create a 256-dimensional embedding based on the hash
        embedding = []
        for i in range(256):
            val = (hash_int + i * 13) % 100 / 100.0  # Normalize to 0-1
            embedding.append(val)
        
        return embedding
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            # Ensure embeddings are the same length
            min_len = min(len(embedding1), len(embedding2))
            emb1 = np.array(embedding1[:min_len]).reshape(1, -1)
            emb2 = np.array(embedding2[:min_len]).reshape(1, -1)
            
            similarity = cosine_similarity(emb1, emb2)[0][0]
            return max(0, min(1, similarity))  # Ensure between 0 and 1
        except:
            return 0.0
    
    def search_knowledge_base(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search knowledge base using semantic similarity"""
        try:
            query_embedding = self.generate_embedding(query)
            results = []
            
            for item in self.knowledge_base:
                item_embedding = self.generate_embedding(item['search_text'])
                similarity = self.calculate_similarity(query_embedding, item_embedding)
                
                # Check for keyword matches in symptoms
                matched_symptoms = [
                    symptom for symptom in item['symptoms'] 
                    if symptom.lower() in query.lower()
                ]
                
                # Boost similarity if keywords match
                if matched_symptoms:
                    similarity = min(1.0, similarity + 0.2)
                
                result_item = item.copy()
                result_item['similarity'] = similarity
                result_item['matched_symptoms'] = matched_symptoms
                results.append(result_item)
            
            # Sort by similarity and return top_k
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            st.error(f"Search error: {e}")
            return []
    
    def search_documents(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """Search processed documents using semantic similarity"""
        if not self.documents_base:
            return []
        
        try:
            query_embedding = self.generate_embedding(query)
            results = []
            
            for doc in self.documents_base:
                doc_embedding = self.generate_embedding(doc['content'][:500])  # Use first 500 chars
                similarity = self.calculate_similarity(query_embedding, doc_embedding)
                
                result_item = doc.copy()
                result_item['similarity'] = similarity
                results.append(result_item)
            
            # Sort by similarity and return top_k
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            st.error(f"Document search error: {e}")
            return []
    
    def process_luckstone_documents(self, folder_path: str):
        """Process LuckStone PDF and DOCX documents from folder"""
        folder = Path(folder_path)
        if not folder.exists():
            st.error(f"Folder not found: {folder_path}")
            return
        
        supported_files = list(folder.glob("**/*.pdf")) + list(folder.glob("**/*.docx"))
        
        if not supported_files:
            st.warning("No PDF or DOCX files found in the specified folder")
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        self.documents_base = []  # Reset documents base
        processed_count = 0
        
        for i, file_path in enumerate(supported_files):
            status_text.text(f"Processing {file_path.name}...")
            
            try:
                content = self._extract_text_from_file(file_path)
                if content and len(content.strip()) > 50:  # Ensure meaningful content
                    doc_item = {
                        "doc_id": f"{file_path.stem}_{i}",
                        "filename": file_path.name,
                        "content": content[:3000],  # Store first 3000 chars
                        "file_type": file_path.suffix,
                        "file_path": str(file_path)
                    }
                    self.documents_base.append(doc_item)
                    processed_count += 1
                
                progress_bar.progress((i + 1) / len(supported_files))
                
            except Exception as e:
                st.error(f"Error processing {file_path.name}: {e}")
        
        status_text.text("Document processing completed!")
        st.success(f"✅ Successfully processed {processed_count}/{len(supported_files)} documents")
        
        # Generate embeddings for all documents
        if self.documents_base:
            with st.spinner("Generating embeddings for documents..."):
                for doc in self.documents_base:
                    doc['embedding'] = self.generate_embedding(doc['content'][:500])
    
    def _extract_text_from_file(self, file_path: Path) -> str:
        """Extract text from PDF or DOCX files"""
        try:
            if file_path.suffix.lower() == '.pdf':
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            text += page_text.strip() + "\n"
                    return text.strip()
            
            elif file_path.suffix.lower() == '.docx':
                doc = docx.Document(file_path)
                text = ""
                for paragraph in doc.paragraphs:
                    if paragraph.text and paragraph.text.strip():
                        text += paragraph.text.strip() + "\n"
                return text.strip()
            
            else:
                return ""
                
        except Exception as e:
            st.error(f"Error extracting text from {file_path}: {e}")
            return ""
    
    def triage_issue(self, user_query: str) -> Dict[str, Any]:
        """Main triage function using OpenAI embeddings"""
        if not user_query.strip():
            return {"error": "Please provide a description of your issue"}
        
        start_time = time.time()
        
        # Search in knowledge base
        knowledge_matches = self.search_knowledge_base(user_query)
        
        # Search in LuckStone documents
        doc_matches = self.search_documents(user_query, top_k=2)
        
        processing_time = time.time() - start_time
        
        if not knowledge_matches:
            return {
                "query": user_query,
                "matches": [],
                "recommendation": "No specific match found. Please contact general IT helpdesk at ext. 5000",
                "priority": "Unknown",
                "processing_time": f"{processing_time:.2f}s",
                "document_matches": doc_matches
            }
        
        best_match = knowledge_matches[0]
        similarity_score = best_match.get('similarity', 0)
        confidence = "High" if similarity_score > 0.7 else "Medium" if similarity_score > 0.4 else "Low"
        
        return {
            "query": user_query,
            "matches": knowledge_matches,
            "best_match": best_match,
            "recommendation": best_match.get('solution', 'Contact IT helpdesk'),
            "priority": best_match.get('priority', 'Unknown'),
            "escalation": best_match.get('escalation', 'IT Helpdesk'),
            "confidence": confidence,
            "similarity_score": f"{similarity_score:.3f}",
            "processing_time": f"{processing_time:.2f}s",
            "document_matches": doc_matches,
            "matched_symptoms": best_match.get('matched_symptoms', [])
        }

def main():
    st.set_page_config(
        page_title="OpenAI IT Triage System - LuckStone Docs",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 OpenAI IT Triage System with Semantic Search developed by Somnath Das (M.S. Data Analytics: VCU, VA)")
    st.markdown("### AI-Powered Issue Classification using OpenAI Embeddings")
    
    # Initialize system
    if 'openai_system' not in st.session_state:
        with st.spinner("Initializing OpenAI Embedding System..."):
            try:
                st.session_state.openai_system = OpenAIEmbeddingSystem()
            except Exception as e:
                st.error(f"Failed to initialize OpenAI system: {e}")
                st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 System Operations")
        
        st.subheader("🔐 API Status")
        st.success(f"**OpenAI API Key:** Configured")
        st.info(f"**Embedding Model:** text-embedding-3-small")
        
        st.subheader("📂 Process LuckStone Documents")
        folder_path = st.text_input("LuckStone Documents Folder Path:", 
                                   value=r"C:\Users\somna\OneDrive\LuckStone\ImageProcessing_LuckStone\data")
        
        if st.button("📂 Process Documents"):
            with st.spinner("Processing LuckStone documents..."):
                st.session_state.openai_system.process_luckstone_documents(folder_path)
        
        st.subheader("📊 System Statistics")
        if st.button("Refresh Stats"):
            system = st.session_state.openai_system
            kb_count = len(system.knowledge_base)
            doc_count = len(system.documents_base)
            cache_count = len(system.embeddings_cache)
            
            st.write(f"**Knowledge Base Entries:** {kb_count}")
            st.write(f"**LuckStone Documents:** {doc_count}")
            st.write(f"**Cached Embeddings:** {cache_count}")
        
        if st.button("🔄 Clear Cache"):
            st.session_state.openai_system.embeddings_cache = {}
            st.success("Embedding cache cleared!")
        
        st.subheader("🚨 Emergency Contacts")
        st.error("""
        **Security Issues:** ext. 5610 (24/7)
        **System Outages:** ext. 5620 (24/7)
        **Critical Systems:** ext. 5630
        """)
        
        st.subheader("💡 Supported File Types")
        st.info("""
        - **PDF Documents** (.pdf)
        - **Word Documents** (.docx)
        - **Text Extraction** with embeddings
        """)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔍 IT Issue Triage")
        user_query = st.text_area(
            "Describe your IT issue:",
            placeholder="e.g., I can't login to my account because I forgot my password...",
            height=100
        )
        
        # Example queries
        st.write("**Try these examples:**")
        examples = [
            "I forgot my password and can't login to the system",
            "My internet connection is very slow today",
            "I need to install Photoshop but don't have admin rights",
            "Outlook keeps crashing when I try to send emails",
            "VPN connection keeps dropping every few minutes",
            "Database queries are running very slow",
            "I received a suspicious email asking for password"
        ]
        
        # Create buttons for examples
        cols = st.columns(3)
        example_buttons = []
        for i, example in enumerate(examples):
            with cols[i % 3]:
                if st.button(example[:30] + "..." if len(example) > 30 else example, 
                           key=f"example_{i}", use_container_width=True):
                    user_query = example
        
        if st.button("🔍 Analyze Issue", type="primary", use_container_width=True):
            if user_query.strip():
                with st.spinner("Analyzing your issue with OpenAI embeddings..."):
                    result = st.session_state.openai_system.triage_issue(user_query)
                
                st.subheader("🎯 Analysis Results")
                
                if "error" in result:
                    st.error(result["error"])
                else:
                    # Display main result
                    priority_colors = {
                        "Critical": "red", "High": "orange", 
                        "Medium": "blue", "Low": "green", "Unknown": "gray"
                    }
                    priority_color = priority_colors.get(result['priority'], "gray")
                    
                    st.markdown(f"""
                    <div style="background-color:#f0f2f6;padding:20px;border-radius:10px;margin:10px 0">
                        <h3>🎯 Best Match: {result['best_match']['category']}</h3>
                        <p><strong>Confidence:</strong> <span style="color:{'green' if result['confidence'] == 'High' else 'orange' if result['confidence'] == 'Medium' else 'red'}">{result['confidence']}</span> ({result['similarity_score']})</p>
                        <p><strong>Priority:</strong> <span style="color:{priority_color}">{result['priority']}</span></p>
                        <p><strong>Recommended Solution:</strong> {result['recommendation']}</p>
                        <p><strong>Escalate to:</strong> {result['escalation']}</p>
                        <p><strong>Processing Time:</strong> {result['processing_time']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display matched symptoms
                    if result.get('matched_symptoms'):
                        st.success(f"**✅ Identified Symptoms:** {', '.join(result['matched_symptoms'])}")
                    
                    # Display document matches
                    if result.get('document_matches'):
                        st.subheader("📚 Relevant LuckStone Documents")
                        for i, doc_match in enumerate(result['document_matches']):
                            similarity = doc_match.get('similarity', 0)
                            with st.expander(f"📄 {doc_match.get('filename', 'Document')} (Similarity: {similarity:.3f})"):
                                st.write(f"**File Type:** {doc_match.get('file_type', 'Unknown')}")
                                st.write(f"**Content Preview:** {doc_match.get('content', '')[:300]}...")
                                if st.button("View Full Content", key=f"view_{i}"):
                                    st.text_area("Full Content", doc_match.get('content', ''), height=200)
                    
                    # Alternative matches
                    if len(result['matches']) > 1:
                        with st.expander("🔍 Alternative Matches"):
                            for i, match in enumerate(result['matches'][1:], 1):
                                similarity = match.get('similarity', 0)
                                similarity_color = "green" if similarity > 0.7 else "orange" if similarity > 0.4 else "red"
                                st.markdown(f"**{i}. {match['category']}** (<span style='color:{similarity_color}'>{similarity:.3f}</span>)", unsafe_allow_html=True)
                                st.write(f"*Solution:* {match.get('solution', 'N/A')}")
                                st.progress(float(similarity))
            else:
                st.warning("Please describe your issue before analyzing.")
    
    with col2:
        st.subheader("💡 How It Works")
        st.info("""
        **OpenAI Embeddings:**
        - Uses text-embedding-3-small model
        - Converts text to 1536-dimensional vectors
        - Semantic understanding of IT issues
        
        **Cosine Similarity:**
        - Measures similarity between query and knowledge base
        - Returns most relevant matches
        - Confidence scoring based on similarity
        """)
        
        st.subheader("📈 Confidence Levels")
        st.markdown("""
        - **High (> 0.7):** Strong semantic match
        - **Medium (0.4-0.7):** Good contextual match  
        - **Low (< 0.4):** General recommendation
        """)
        
        st.subheader("🔍 Search Features")
        st.success("""
        ✅ Semantic IT issue matching
        ✅ LuckStone document search  
        ✅ Priority-based escalation
        ✅ Real-time OpenAI embeddings
        ✅ Fallback mechanisms
        """)
        
        st.subheader("⚡ Performance")
        st.metric("Embedding Model", "text-embedding-3-small")
        st.metric("Vector Dimensions", "1536")
        st.metric("Search Method", "Cosine Similarity")

if __name__ == "__main__":
    main()
