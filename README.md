--PDF Chatbot with Groq and Chainlit:

A fast and interactive PDF chatbot that allows users to upload PDF documents and ask questions about their content using conversational AI.

-- Features:

- Upload and process PDF documents
- Interactive chat interface with Chainlit
- Powered by Groq's Mixtral-8x7b model for fast responses
- Vector-based document retrieval with ChromaDB
- Conversation memory for contextual responses
- Source citation for answers

--Prerequisites

- Python 3.12+
- Groq API key (get one at https://console.groq.com)

--Installation

1. Clone the repository:

git clone <https://github.com/sahar549/DocChat>
cd DocChat


2. Install dependencies:
pip install langchain==0.1.20 langchain-community==0.0.38 langchain-text-splitters langchain-groq chainlit pypdf2 chromadb python-dotenv sentence-transformers


3. Create a `.env` file in the project root:

GROQ_API_KEY=your_groq_api_key_here



--Usage

Run the application:
chainlit run app.py -w

The app will open in your browser at `http://localhost:8000`

1. Upload a PDF file when prompted
2. Wait for the document to be processed
3. Start asking questions about the PDF content
