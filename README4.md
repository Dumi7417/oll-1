# README for ChromaDB and RAG Fusion Application (Assignment 4)

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

This project builds upon the functionality from **Assignment 3**, further enhancing the chatbot application by implementing **Multi Query and RAG Fusion**. These features improve the retrieval and response accuracy of the AI Assistant when answering questions about the **Constitution of the Republic of Kazakhstan**.

The application integrates a **Retrieval-Augmented Generation (RAG) Fusion** pipeline with **ChromaDB** and **Ollama llama3.2**, leveraging **Multi Query techniques** to generate diverse queries for more comprehensive document retrieval.

Using a user-friendly interface built with **Streamlit**, the assistant enables users to interact by querying the constitution and receiving detailed answers, backed by multiple retrieved documents for a richer response.

---

## Features

- **Multi Query Expansion**: Generates multiple variations of a user query to retrieve a broader range of relevant documents.
- **RAG Fusion**: Aggregates information from multiple document retrievals to enhance response quality.
- **Constitution Querying**: Preloaded with the full text of the **Constitution of the Republic of Kazakhstan**, ensuring precise answers.
- **Citations and Contextual Answers**: Responses include references to specific articles and sections of the Constitution.
- **Custom Document Upload**: Users can upload `.txt` files for personalized queries.
- **Vector Indexing for Fast Retrieval**: Uses **ChromaDB** for optimized document search.
- **Streamlit Interface**: Provides an intuitive UI for querying and document uploads.

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
3. Download the English version of the Constitution of the Republic of Kazakhstan from this link.Save it as constitution.txt in the project directory.
4. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
5. Open the application in your browser (Streamlit will provide a local URL).

---

## Usage

1. **Multi Query Expansion for Constitution Querying:**:

   - The assistant generates multiple versions of the input query for broader retrieval.
   - Responses cite specific articles and sections for accuracy.

2. **Custom Document Upload with RAG Fusion:**:

   - Upload .txt files to extend queries beyond the Constitution.
   - The application integrates responses from multiple retrieved documents.

3. **Interaction**:
   - Input a query in the provided text area and receive a refined response based on retrieved knowledge.

---

## File Descriptions

1. **app.py** ([source](src/app.py))

   - Implements the Multi Query and RAG Fusion pipeline, file uploads, and Streamlit UI logic.

2. **constitution.txt**

   - Contains the full text of the Constitution of the Republic of Kazakhstan.

3. **requirements.txt** ([source](requirements.txt))

   - Lists required Python libraries.

4. **style.css** ([source](src/style.css))
   - Styles the Streamlit interface.

---

## Technical Details

### Multi Query Expansion

The system generates multiple queries from a single user input, improving retrieval by capturing different phrasings and contexts. This results in a more thorough search across stored documents.

### RAG Fusion Workflow

1. **Multi Query Generation**: Expands a user query into multiple versions.
2. **Knowledge Retrieval**: Queries **ChromaDB** for Constitution content or uploaded documents.
3. **Fusion of Retrieved Contexts**: Aggregates multiple document retrievals for a more complete answer.
4. **LLM Response Generation**: Uses **Ollama llama3.2** to generate context-aware responses.

### Citing Articles and Sections

Responses reference specific articles and sections of the Constitution for clarity. Example:

> "According to Article 5, Section 2 of the Constitution, ..."

---

## Examples

### Example 1: Constitution Query

**Input Query**: What are the rights of citizens in Kazakhstan?  
**AI Response**: Based on Article 12, Section 1 of the Constitution, citizens of Kazakhstan have the right to freedom, equality, and personal dignity.

### Example 2: Multi Query in Action

**Input Query**: What is the official language of Kazakhstan?
**Generated Queries**:
What language is officially recognized in Kazakhstan?

What is the state language of Kazakhstan?

Which language is used for official purposes in Kazakhstan?  
**AI Response**: Based on Article 7, Section 1 of the Constitution, the official language of Kazakhstan is Kazakh.

### Example 3: RAG Fusion with Custom Documents

**Uploaded Document**:  
`economy.txt` contains the following text:

> "The economy of Kazakhstan is largely based on oil production."

**Input Query**: What drives Kazakhstan's economy?  
**AI Response**: Based on the uploaded document, the economy of Kazakhstan is largely based on oil production.

---

## License

This project is released under the MIT License. You are free to use, modify, and distribute the code with proper attribution.
