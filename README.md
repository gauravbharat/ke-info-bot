# Khata-Easy info AI bot

### Info
- This bot is intended to be added below the FAQs section on the KE landing page.
- No sign in required
- After the Google ReCaptcha verification to eliminate autobots, it should show a prompt to query the website for its features and terms.
- JSON files are already uploaded in the data folder for this project.
- Whenever a JSON file is modified, DO RUN the _injestion.py_ to update Pinecone vector database index.
- This project implements RAG to first -
  - load the JSON data as text
  - split into semantic chunks
  - embed data to create vector stores
  - store vectors into Pinecone
  - retrieve based on user query, keeping the history
- UI would be generated using StreamLit
---
### Financials
- **Embedding model**: Uses the OpenAI embedding model "text-embedding-3-small" with API for _pgmendse@gmail.com_
- **AI/LLM model**: Uses Google AI studio gemini-2.5-flash-lite model with API for _gaurav.mendse@gmail.com_
- **Pinecone vector storage**: Uses API for _garydsouza80@gmail.com_
---
