"""
Ra7al AI Concierge - Emergency Vector Database Pipeline
Module: src/db/emergency_db.py
Story Reference: Story 2.2 - Emergency Assistance (Week 2, Dev 3)
"""

import os
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Default local persistence path defined in project file structure
EMERGENCY_DB_DIR = os.getenv("EMERGENCY_DB_PATH", "data/emergency_db")

# Pre-compiled static emergency directory for Egypt
DEFAULT_EGYPT_EMERGENCY_DATA: List[Dict[str, Any]] = [
    {
        "service": "Tourist Police (شرطة السياحة)",
        "number": "126",
        "category": "police",
        "coverage": "Nationwide / Major Tourist Hubs",
        "description": "Assistance for international tourists, harassment reports, scam resolution, and travel safety."
    },
    {
        "service": "General Police & Emergency (الشرطة)",
        "number": "122",
        "category": "police",
        "coverage": "Nationwide",
        "description": "Standard emergency police response, accidents, and security incidents across Egypt."
    },
    {
        "service": "Ambulance / Medical Emergency (الإسعاف)",
        "number": "123",
        "category": "medical",
        "coverage": "Nationwide",
        "description": "Urgent medical transport and first responders across all governorates."
    },
    {
        "service": "Fire Department & Civil Protection (المطافئ)",
        "number": "180",
        "category": "fire",
        "coverage": "Nationwide",
        "description": "Fire emergencies, rescue operations, and structural hazards."
    },
    {
        "service": "Traffic Police (مرور مصر)",
        "number": "128",
        "category": "traffic",
        "coverage": "Nationwide / Highways",
        "description": "Highway breakdowns, towing, and roadside traffic incidents."
    },
    {
        "service": "Gas Emergency (طوارئ الغاز)",
        "number": "129",
        "category": "utility",
        "coverage": "Nationwide",
        "description": "Gas leak reports and residential utility hazards."
    },
    {
        "service": "Electricity Emergency (طوارئ الكهرباء)",
        "number": "121",
        "category": "utility",
        "coverage": "Nationwide",
        "description": "Power outages and electrical hazard interventions."
    }
]


class EmergencyDBPipeline:
    """
    Dedicated Vector Store Pipeline for Tourist Emergency Queries.
    Provides vector indexing, retrieval, and metadata filtering.
    """

    def __init__(self, persist_directory: str = EMERGENCY_DB_DIR):
        self.persist_directory = persist_directory
        # Uses lightweight, free local embeddings accessible in Egypt without VPN
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_store: Optional[Chroma] = None
        self._initialize_or_load()

    def _initialize_or_load(self) -> None:
        """Initializes existing vector database or creates a new one."""
        os.makedirs(self.persist_directory, exist_ok=True)
        self.vector_store = Chroma(
            collection_name="emergency_assistance",
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )

        # Ingest pre-compiled data if vector store is unpopulated
        if self.vector_store._collection.count() == 0:
            self.ingest_default_data()

    def ingest_default_data(self) -> None:
        """Ingests baseline emergency numbers and service contacts into ChromaDB."""
        documents = []
        for entry in DEFAULT_EGYPT_EMERGENCY_DATA:
            content = (
                f"Emergency Service: {entry['service']}\n"
                f"Direct Dial Number: {entry['number']}\n"
                f"Service Category: {entry['category']}\n"
                f"Geographic Coverage: {entry['coverage']}\n"
                f"Details: {entry['description']}"
            )
            metadata = {
                "service": entry["service"],
                "number": entry["number"],
                "category": entry["category"],
                "coverage": entry["coverage"],
                "source": "official_egypt_emergency_registry"
            }
            documents.append(Document(page_content=content, metadata=metadata))

        self.vector_store.add_documents(documents)
        self.vector_store.persist()

    def ingest_documents(self, documents: List[Document]) -> None:
        """Ingests external parsed Markdown/PDF/CSV emergency documents."""
        if documents:
            self.vector_store.add_documents(documents)
            self.vector_store.persist()

    def search_emergency(self, query: str, top_k: int = 3, category_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieves relevant emergency contacts based on the user's situation.
        
        Args:
            query: The situational query (e.g., 'I was robbed in Cairo', 'Need ambulance').
            top_k: Number of candidate documents to return.
            category_filter: Optional filter ('police', 'medical', 'fire', 'utility').
            
        Returns:
            List of structured emergency records for the Help Agent and Verification Node.
        """
        filter_dict = {"category": category_filter} if category_filter else None
        
        results = self.vector_store.similarity_search(
            query=query,
            k=top_k,
            filter=filter_dict
        )

        structured_results = []
        for doc in results:
            structured_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
            
        return structured_results

    def as_retriever(self, top_k: int = 3):
        """Returns standard LangChain retriever interface for subgraph RAG nodes."""
        return self.vector_store.as_retriever(search_kwargs={"k": top_k})


# Singleton instance for direct import across subgraphs
emergency_pipeline = EmergencyDBPipeline()