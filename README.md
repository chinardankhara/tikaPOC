# Socratic RAG Agent

A research topic exploration tool powered by Azure OpenAI and vector similarity search.

## Table of Contents
1. [Running Instructions](#running-instructions)
2. [Technical Details](#technical-details)
   - [Topic Data Pipeline](#a-topic-data-pipeline)
   - [Database & Embeddings](#b-database--embeddings)
   - [Topic Agent & Search](#c-topic-agent--search)
   - [Chat Manager](#d-chat-manager)
   - [User Interface](#e-user-interface)
   - [Cloud Infrastructure](#f-cloud-infrastructure)

## 1. Running Instructions

### Prerequisites
- Python 3.8 or higher
- Azure CLI.  Install from https://learn.microsoft.com/en-us/cli/azure/install-azure-cli
- Azure subscription with OpenAI access
- **SSL Certificates for Azure PostgreSQL:** Make sure you have installed the **recommended SSL certificates** to secure connections to **our dedicated Azure PostgreSQL server**. These certificates adhere to general industry standards and are **not specific to this project**. For detailed instructions, refer to the [Azure PostgreSQL SSL/TLS Documentation](https://learn.microsoft.com/en-us/azure/postgresql/flexible-server/concepts-networking-ssl-tls).
- **Azure Subscription Access:** To utilize our Azure Subscription ID `0bdb7994-618e-43ab-9dc4-e5510263d104`, you must obtain access through **Professor Lew Lefton**. Please request access by visiting the [Cloudbank Billing Account Access](https://www.cloudbank.org/billing-account-access) portal or by contacting Professor Lew Lefton directly at [lew.lefton@gatech.edu](mailto:lew.lefton@gatech.edu).

### a. Azure Authentication
```bash
# Login to Azure - Be sure to use the account from cloudbank.org
az login

# Set the correct subscription
az account set --subscription 0bdb7994-618e-43ab-9dc4-e5510263d104
```

### b. Create a Virtual Environment and Install Dependencies
```bash
python3 -m venv
source venv/bin/activate
pip install -r requirements.txt
```

### c. Run the Application
```bash
streamlit run src/ui/streamlit_app.py
```

The application will be available at `http://localhost:8501`

## 2. Technical Details

### a. Topic Data Pipeline
The system fetches research topics from the OpenAlex API using cursor-based pagination:
- Data validation and cleaning of topics and keywords
- Automatic keyword extraction and normalization
- JSON storage of raw data and keyword mappings

### b. Database & Embeddings
PostgreSQL database with pgvector extension for similarity search:
- Topics table storing basic research topic information
- Keywords table with 768-dimensional SciBERT embeddings
- Many-to-many relationship table (topic_keywords)
- IVFFlat index for efficient vector similarity search
- Batch processing for embedding generation using SciBERT

### c. Topic Agent & Search
Intelligent topic search system with:
- Multi-turn query understanding using conversation history
- Query rewriting for better context preservation
- Vector similarity search using pgvector extension
- Topic exclusion mechanism to avoid repetition
- Ranking based on keyword matches and similarity scores

### d. Chat Manager
Manages the conversation flow with:
- Azure OpenAI usage for natural language processing
- Fallback mechanisms for API failures
- Conversation state handling


### e. User Interface
Streamlit-based chat interface providing real-time interaction with the topic search system.

### f. Cloud Infrastructure
All services are hosted in Azure under the OKN Project Cloudbank subscription:
- **Resource Group**: tikabox
- **Services**:
  - Azure PostgreSQL: Stores topics, keywords, and embeddings
  - Azure Key Vault: Manages service credentials and connection strings
  - Azure OpenAI Service: GPT-4 deployment for query processing and response generation
- **Authentication**: Managed through Azure DefaultAzureCredential
