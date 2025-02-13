# README for AI Assistant MVP (Assignment 4)

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

This project is the **Minimum Viable Product (MVP)** of an **AI Assistant** designed to answer all questions related to the **Constitution of the Republic of Kazakhstan**. It builds upon the work from **Assignment 3** by integrating **Multi Query, RAG Fusion, and Chat Functionality**.

The application leverages **Retrieval-Augmented Generation (RAG) Fusion**, **Multi Query Expansion**, and a **Vector Store** to provide accurate and context-aware responses. Users can query the **Constitution** and upload additional documents for extended information retrieval.

The assistant is implemented with **Streamlit** for an intuitive UI and supports various **LLMs (Ollama, Groq, Gemini, OpenAI)** for response generation.

---

## Features

- **Multi Query Expansion**: Generates multiple query variations to retrieve diverse relevant documents.
- **RAG Fusion**: Aggregates multiple retrieved contexts for improved accuracy.
- **Constitution Querying**: Preloaded with the **Constitution of the Republic of Kazakhstan**.
- **Chat Functionality**: Enables interactive question-answering.
- **Custom Document Upload**: Supports **single or multiple** `.txt` file uploads for personalized queries.
- **Vector Store for Query and Answer Storage**: Uses **MongoDB/ChromaDB** to store queries and responses.
- **Context-Based Responses**: Provides answers with citations from uploaded documents or the Constitution.
- **Streamlit UI**: Offers an easy-to-use web-based interface.

---

## Setup and Installation

### Prerequisites

Ensure the following are installed:

- Python 3.9 or later
- pip
- A running instance of an **LLM API (Ollama/Groq/Gemini/OpenAI)**
- A vector database such as **MongoDB or ChromaDB**

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/Dumi7417/ollamaAssignment3part1and2.git
   cd ollamaAssignment3part1and2
   ```
2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the **English version of the Constitution of the Republic of Kazakhstan** and save it as `constitution.txt` in the project directory.
4. Start the application:
   ```bash
   streamlit run app.py
   ```
5. Open the application in your browser (Streamlit will provide a local URL).

---

## Usage

### 1. **Multi Query Expansion & Constitution Querying**

- The assistant generates multiple versions of the input query for broader retrieval.
- Responses cite specific articles and sections for accuracy.

### 2. **Custom Document Upload with RAG Fusion**

- Users can upload one or more `.txt` files.
- Queries retrieve information from both the Constitution and uploaded documents.

### 3. **Chat Functionality**

- Users interact with the AI assistant in a conversational manner.
- Previous interactions are stored for context-aware responses.

### 4. **Storage of Queries and Responses**

- All queries and responses are stored in **MongoDB/ChromaDB** for reference and future improvements.

---

## File Descriptions

1. **app.py** ([source](src/app.py))

   - Implements the **Multi Query Expansion**, **RAG Fusion**, **Chat Functionality**, and **File Upload** features using Streamlit.

2. **constitution_kazakhstan.txt**

   - Contains the full text of the **Constitution of the Republic of Kazakhstan**.

3. **requirements.txt** ([source](requirements.txt))

   - Lists required Python libraries.

4. **style.css** ([source](src/style.css))
   - Styles the Streamlit interface.

---

## Technical Details

### Multi Query Expansion

- Generates multiple queries from a single user input.
- Captures different phrasings and contexts for better retrieval.

### RAG Fusion Workflow

1. **Multi Query Generation**: Expands user queries into multiple versions.
2. **Knowledge Retrieval**: Queries **ChromaDB/MongoDB** for Constitution and document content.
3. **Fusion of Retrieved Contexts**: Aggregates multiple retrieved responses.
4. **LLM Response Generation**: Uses **Ollama/Groq/Gemini/OpenAI** for answer synthesis.

### Vector Store for Queries and Answers

- Stores **all queries and answers** for future improvements and context-aware responses.

### Citing Articles and Sections

- Responses include specific references to the **Constitution**.

Example:

> "According to Article 5, Section 2 of the Constitution, ..."

---

## Examples

### Example 1: Constitution Query

**Input Query**: What are the rights of citizens in Kazakhstan?  
**AI Response**: Based on Article 12, Section 1 of the Constitution, citizens of Kazakhstan have the right to freedom, equality, and personal dignity.

### Example 2: Multi Query in Action

**Input Query**: What is the official language of Kazakhstan?  
**Generated Queries**:

- What language is officially recognized in Kazakhstan?
- What is the state language of Kazakhstan?
- Which language is used for official purposes in Kazakhstan?

**AI Response**: Based on Article 7, Section 1 of the Constitution, the official language of Kazakhstan is Kazakh.

### Example 3: RAG Fusion with Custom Documents

**Uploaded Document**:  
`economy.txt` contains:

> "The economy of Kazakhstan is largely based on oil production."

**Input Query**: What drives Kazakhstan's economy?  
**AI Response**: Based on the uploaded document, the economy of Kazakhstan is largely based on oil production.

---

## License

This project is released under the **MIT License**. You are free to use, modify, and distribute the code with proper attribution.
