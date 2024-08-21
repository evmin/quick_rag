# Quick RAG Tutorial


## Overview

This tutorial provides a step-by-step guide to setting up a Retrieval-Augmented Generation (RAG) system using Azure's OpenAI service. The goal is to demonstrate how to configure and utilize a search index with vector embeddings, and how to integrate this with a large language model (LLM) to generate responses based on the indexed knowledge.

## Prerequisites

Before you start, ensure you have the following installed:

- Python 3.10 or later
- Azure SDK for Python
- The required Python packages as listed in `requirements.txt`
- An Azure account with access to OpenAI service

You will also need to configure environment variables for your Azure OpenAI resources. These variables should be stored in a `.env` file or set directly in your environment. Please check the `.env.sample`, copy it, rename it to `.env` and fill in all empty values for the OpenAI resource and the AI Search resource.

## Tutorial Breakdown

The tutorial is structured into several steps:

1. **Loading Libraries and Configurations**:  
   We begin by loading necessary Python libraries and configuring our environment. This includes setting up the OpenAI client using Azure's services.

2. **Setting Up the OpenAI Clients**:  
   We initialize the OpenAI client to interact with the Azure OpenAI API for both general use and for generating embeddings.

3. **Creating a Search Index**:  
   We define and create a search index using Azure Cognitive Search. This index will store the vector embeddings and metadata that will be used in the RAG system.

4. **Populating the Search Index**:  
   We populate the search index with synthetic data (e.g., facts about the Tesla Model S), generating vector embeddings for each piece of data.

5. **Creating an Orchestrator**:  
   We set up an orchestrator that combines the capabilities of the LLM and the search index to answer queries by retrieving relevant information from the index.

6. **Querying the Knowledge Base**:  
   We demonstrate how to use the orchestrator to query the knowledge base and generate responses, showcasing the power of the RAG approach.


## Data and RAG Topic

To properly run this tutorial, and for the orchestrator to know which topic to decide to use RAG with, the user will have to supply a knowledge base (in the same format as the synthetic data in the data folder `Tesla_Model_S.txt`), and provide the topic name as well. The topic will help the orchestrator decide whether to reply right away or to use RAG to formulate an answer. 

To change the current topic about the Tesla Model S, then change the environment variable `KB_TOPIC` in the `.env` file, as well potentially change the name of the index `KB_INDEX_NAME`, and provide a differnt knowledge base as a text file under the `data` folder.

## Install the required Python packages:

   ```bash
    pip install -r requirements.txt
   ```

Ensure your environment variables are properly configured. You can set them in a .env file or export them in your shell.

Open the Jupyter notebook provided in the repository and follow along with the tutorial. Execute each cell in sequence to build and run the RAG system.


## Running the Tutorial

To run the tutorial, follow these steps:

1. Clone the repository and navigate to the tutorial directory:

   ```bash
   git clone <repository_url>
   cd <tutorial_directory>
   ```

2. Open and run the `tutorial.ipynb` notebook.

## User Interface (UI)
A user-friendly UI is available to interact with the RAG system. You can use the UI by running the following command from the PowerShell command line:

   ```bash
   chainlit run chat.py
   ```

This will start a local web server where you can interact with the RAG system through a simple chat interface.
