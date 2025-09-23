
# Intelligent IT Triage System with Semantic Document Search

## Approach 2: The Semantic Detective - BigQuery AI Hackathon Submission

### Project Overview
This project implements an intelligent IT triage system using BigQuery's vector search capabilities to semantically match support tickets with solutions and documentation.

### Key Features
- Semantic understanding of IT issues using ML.GENERATE_EMBEDDING
- Vector-based search across knowledge base and technical documentation
- Real-time processing with confidence scoring
- Integration with LuckStone manuals and documents

### Technology Stack
- **BigQuery ML**: ML.GENERATE_EMBEDDING, VECTOR_SEARCH
- **BigFrames Python**: TextEmbeddingGenerator, vector_search
- **Streamlit**: Web application interface
- **Google Cloud**: Infrastructure and services

### Setup Instructions
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Configure Google Cloud credentials
4. Set up BigQuery dataset and ML models
5. Run the application: `streamlit run app.py`

### Demo Video
The demo video showcases:
- Semantic matching of IT issues
- Document retrieval from LuckStone manuals
- BigQuery console operations
- Business impact demonstration

### Repository Structure
- `/src` - Source code implementation
- `/docs` - Documentation and setup guides
- `/data` - Sample datasets and test cases
- `/notebooks` - Jupyter notebooks with examples

### Contact Information
**Developer**: Somnath Das
**Submission Date**: September 22, 2025
**Competition**: BigQuery AI Hackathon - Approach 2: The Semantic Detective
    
