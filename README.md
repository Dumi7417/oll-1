# README for ChromaDB and RAG Pipeline Application (Assignment 3)

## Table of Contents
1. [Introduction](#introduction)  
2. [Features](#features)  
3. [Setup and Installation](#setup-and-installation)  
4. [Usage](#usage)  
5. [File Descriptions](#file-descriptions)  
6. [Technical Details](#technical-details)  
7. [Examples](#examples)  
8. [License](#license)  

---

## Introduction
This project builds upon the functionality from **Assignment 2**, enhancing the chatbot application to serve as an AI Assistant capable of answering questions about the **Constitution of the Republic of Kazakhstan**. The application integrates a **Retrieval-Augmented Generation (RAG)** pipeline with **ChromaDB** and **Ollama llama3.2** to provide accurate, context-aware responses.

Using a user-friendly interface built with **Streamlit**, the assistant allows users to interact by querying the constitution and receiving detailed answers, including specific citations of articles and sections for clarity.

---

## Features
- **Constitution Querying**: The AI Assistant is preloaded with the full text of the Constitution of the Republic of Kazakhstan in English, enabling users to ask detailed questions about it.  
- **Citations and Contextual Answers**: Responses are backed by specific articles and sections, ensuring accuracy and interpretability.  
- **Document Upload and Processing**: Users can also upload custom `.txt` files for additional queries, providing flexibility.  
- **Vector Indexing for Fast Retrieval**: Leverages **ChromaDB** to retrieve relevant information efficiently.  
- **RAG Pipeline**: Combines retrieval from preloaded documents (Constitution) and custom uploads with language modeling using **Ollama llama3.2**.  
- **Streamlit Interface**: Provides an intuitive web interface for uploading documents, querying the assistant, and viewing responses.  

---

## Setup and Installation

### Prerequisites
Ensure the following are installed:
- Python 3.9 or later  
- pip  
- A running instance of the **Ollama server** (adjustable URL)

### Steps
1. Clone the repository:  
   ```bash
   git clone <repository-link>
   cd <repository-name>
   ```

2. Install the required Python libraries:  
   ```bash
   pip install -r requirements.txt
   ```

3. Download the English version of the Constitution of the Republic of Kazakhstan from [this link](https://www.akorda.kz/en/constitution-of-the-republic-of-kazakhstan-50912).  
   Save it as `constitution.txt` in the project directory.

4. Run the **Streamlit** application:  
   ```bash
   streamlit run app.py
   ```

5. Open the application in your browser (Streamlit will provide a local URL).

---

## Usage
1. **Preloaded Constitution Querying**:  
   - Ask questions related to the Constitution of the Republic of Kazakhstan.  
   - The assistant will retrieve relevant information, citing specific articles and sections.  

2. **Custom Document Upload**:  
   - Upload one or more `.txt` files to query their content alongside the Constitution.  
   - The application will process the uploaded documents and include them in the RAG pipeline.

3. **Interaction**:  
   - Input your query in the provided text area and receive a response based on either the Constitution or uploaded files.

---

## File Descriptions
1. **app.py** ([source](src/app.py))  
   - Implements the RAG pipeline, including file upload handling, Constitution querying, and Streamlit interface logic.  

2. **constitution.txt**  
   - Contains the full text of the Constitution of the Republic of Kazakhstan in English.  

3. **requirements.txt** ([source](requirements.txt))  
   - Lists the necessary Python libraries for the project.  

4. **style.css** ([source](src/style.css))  
   - Styles the Streamlit interface for improved usability.  

---

## Technical Details

### Constitution Integration
The Constitution of the Republic of Kazakhstan is preprocessed and stored in **ChromaDB** as a vector index. This enables fast and accurate retrieval of articles and sections relevant to user queries.

### RAG Pipeline Workflow
1. **Knowledge Retrieval**: Queries **ChromaDB** for Constitution content or uploaded documents matching the user prompt.  
2. **Prompt Augmentation**: Combines retrieved context with the user's query to form an augmented prompt.  
3. **LLM Response Generation**: Processes the augmented prompt via **Ollama llama3.2** to generate context-aware responses.  

### Citing Articles and Sections
Responses include references to specific articles and sections of the Constitution for clarity and reliability. For example:  
> "As stated in Article 2, Section 3 of the Constitution, ..."

---

## Examples

### Example 1: Constitution Query
**Input Query**: What is the official language of the Republic of Kazakhstan?  
**AI Response**: Based on Article 7, Section 1 of the Constitution, the official language of the Republic of Kazakhstan is Kazakh.

### Example 2: Custom Document Upload
**Uploaded Document**:  
`economy.txt` contains the following text:  
> "The economy of Kazakhstan is largely based on oil production."

**Input Query**: What drives Kazakhstan's economy?  
**AI Response**: Based on the uploaded document, the economy of Kazakhstan is largely based on oil production.

---

## License
This project is released under the **MIT License**. You are free to use, modify, and distribute the code with proper attribution.
