# README for ChromaDB and RAG Pipeline Application

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Setup and Installation](#setup-and-installation)
4. [Usage](#usage)
5. [File Descriptions](#file-descriptions)
6. [Technical Details](#technical-details)
7. [License](#license)

---

## Introduction
This project demonstrates an implementation of a chatbot that uses ChromaDB, a vector database optimized for AI applications, and Ollama llama3.2 for a Retrieval-Augmented Generation (RAG) pipeline.

The web interface is built using Streamlit to allow users to input their prompts and receive intelligent responses augmented by a knowledge base stored in ChromaDB.

---

## Features
- Vector Indexing and Efficient Querying: Leverages ChromaDB for scalable, high-performance vector data retrieval.
- RAG Pipeline: Combines contextual knowledge retrieval and powerful language modeling using Ollama llama3.2.
- User-Friendly Interface: Includes a web interface with Streamlit for easy interaction.

---

## Setup and Installation

### Prerequisites
Make sure you have the following installed:
- Python 3.9 or later
- pip
- A running instance of the Ollama server (adjustable URL)

### Steps
1. Clone the repository:
   ```bash
   git clone <repository-link>
   cd <repository-name>
2. Install the required Python libraries:
   ```bash
      pip install -r requierments.txt
3. Run the Streamlit application:
   ```bash
      streamlit run app.py
4. Open the application in your browser (Streamlit will provide a local URL).

---

## Usage
- Enter your query in the provided text area.
- Press the Generate button.
- View the AI-generated response based on the retrieved knowledge.

---

## File Descriptions

1. app.py ([source](src/app.py))  
   - Implements the backend logic for the RAG pipeline and Streamlit interface.  
   - Uses the ChromaDB vector database and Ollama llama3.2.

2. page.html ([source](src/page.html))  
   - Contains the HTML for a supplementary web interface describing ChromaDB.  

3. style.css ([source](src/style.css))  
   - Styles the content of page.html.  

4. requirements.txt ([source](requirements.txt))  
   - Lists the necessary Python packages.  


## Technical Details

### ChromaDB Initialization
The project uses chromadb.PersistentClient for database interactions, initialized with:
- A specific database path
- A custom embedding function based on the langchain_ollama library.

### RAG Pipeline Workflow
1. Knowledge Retrieval: Queries ChromaDB for documents matching the user prompt.
2. Prompt Augmentation: Enhances the input to the Ollama model with retrieved knowledge.
3. LLM Response Generation: Processes the augmented prompt via Ollama llama3.2.

---

## License
This project is released under the MIT License. Feel free to use, modify, and distribute the code, provided proper attribution is given.
   
