"""
LLM-Powered Intelligent Query-Retrieval System
Components:
1. Input Documents (PDF Blob URL)
2. LLM Parser (Extract structured query)
3. Embedding Search (Scikit-learn similarity retrieval)
4. Clause Matching (Semantic similarity)
5. Logic Evaluation (Decision processing)
6. Text Output
"""

import os
import json
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
import re
from datetime import datetime

# Core libraries
import requests
import PyPDF2
from docx import Document
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from io import BytesIO
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import configuration
from config import Config

# Use configuration from config.py
config = Config()

@dataclass
class DocumentChunk:
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None

class Component1_InputDocuments:
    """Component 1: Input Documents - PDF Blob URL Processing"""
    
    @staticmethod
    def download_pdf_from_url(url: str) -> str:
        """Download and extract text from PDF URL"""
        try:
            logger.info(f"Downloading PDF from: {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            pdf_file = BytesIO(response.content)
            reader = PyPDF2.PdfReader(pdf_file)
            
            text = ""
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                text += f"\n--- Page {page_num + 1} ---\n{page_text}"
            
            logger.info(f"Successfully extracted {len(text)} characters from PDF")
            return text.strip()
        except Exception as e:
            logger.error(f"Error downloading/extracting PDF: {e}")
            raise Exception(f"Error processing PDF: {str(e)}")
    
    @staticmethod
    def validate_document_url(url: str) -> bool:
        """Validate if URL points to a valid document"""
        try:
            response = requests.head(url, timeout=10)
            content_type = response.headers.get('content-type', '').lower()
            return 'pdf' in content_type or url.lower().endswith('.pdf')
        except:
            return False

class Component2_LLMParser:
    """Component 2: LLM Parser - Extract structured query understanding"""
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def parse_query_structure(self, query: str) -> Dict[str, Any]:
        """Parse and understand the structure of the query using Gemini Pro"""
        prompt = f"""
        Analyze this user query about a document and help optimize the search:

        User Query: "{query}"
        
        Please provide:
        1. What TYPE of information they want (definition, amount, coverage, process, condition, etc.)
        2. MAIN KEYWORDS that should be searched in the document (be comprehensive)
        3. ALTERNATIVE TERMS that might be used in the document for the same concepts
        4. What SPECIFIC INFORMATION they expect in the answer
        
        Think about what terms a document might use. For example:
        - "NCD" might also be called "No Claim Bonus", "loyalty discount", "claim-free discount"
        - "Hospital" might be defined with terms like "medical institution", "healthcare facility"
        - "Coverage" might be described as "benefits", "reimbursement", "payable amount"
        
        Return in this format:
        Query Type: [definition/amount/coverage/process/condition/etc]
        Main Keywords: [primary search terms from the query]
        Alternative Terms: [other ways the document might refer to these concepts]
        Search Strategy: [expanded keywords for better semantic search]
        Intent: [what the user wants to know]
        Expected Answer: [type of information they expect]
        """
        
        try:
            response = self.model.generate_content(prompt)
            parsed_result = self._parse_llm_response(response.text)
            logger.info(f"Enhanced query parsing: {parsed_result}")
            return parsed_result
        except Exception as e:
            logger.error(f"Error parsing query: {e}")
            return self._default_parse_structure(query)
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response into structured data"""
        result = {}
        lines = response.strip().split('\n')
        
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                value = value.strip()
                
                if key == 'key_entities' or key == 'search_keywords':
                    # Convert comma-separated values to list
                    result[key] = [item.strip() for item in value.split(',') if item.strip()]
                else:
                    result[key] = value
        
        return result
    
    def _default_parse_structure(self, query: str) -> Dict[str, Any]:
        """Fallback parsing if LLM fails"""
        return {
            'query_type': 'general',
            'key_entities': query.split(),
            'intent': 'find information',
            'domain': 'insurance',
            'answer_type': 'detailed explanation',
            'search_keywords': query.split()
        }

class Component3_EmbeddingSearch:
    """Component 3: Embedding Search - Scikit-learn Similarity Retrieval"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(model_name)
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        self.embeddings = None
        self.chunks = []
        self.similarity_model = None
    
    def create_document_chunks(self, text: str, chunk_size: int = 800, overlap: int = 100) -> List[DocumentChunk]:
        """Split document into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            
            chunk_metadata = {
                "chunk_id": len(chunks),
                "start_word": i,
                "end_word": min(i + chunk_size, len(words)),
                "word_count": len(chunk_words)
            }
            
            chunks.append(DocumentChunk(
                content=chunk_text,
                metadata=chunk_metadata
            ))
        
        logger.info(f"Created {len(chunks)} chunks from document")
        return chunks
    
    def build_search_index(self, chunks: List[DocumentChunk]):
        """Build search index from document chunks using scikit-learn"""
        if not chunks:
            raise ValueError("No chunks provided for indexing")
        
        self.chunks = chunks
        
        # Generate embeddings
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
        
        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        
        # Store embeddings for similarity search
        self.embeddings = embeddings
        
        # Initialize NearestNeighbors for fast similarity search
        self.similarity_model = NearestNeighbors(
            n_neighbors=min(len(chunks), 20),  # Search up to 20 neighbors
            metric='cosine',
            algorithm='brute'  # More reliable for smaller datasets
        )
        self.similarity_model.fit(embeddings)
        
        logger.info(f"Built similarity search index with {len(chunks)} chunks")
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Perform enhanced semantic search using scikit-learn"""
        if self.embeddings is None:
            raise ValueError("Search index not built. Call build_search_index first.")
        
        # Search with original query
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        
        # Get more results initially
        extended_k = min(top_k * 3, len(self.chunks))  # Get 3x more results
        distances, indices = self.similarity_model.kneighbors(query_embedding, n_neighbors=extended_k)
        
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            chunk = self.chunks[idx]
            # Convert cosine distance to similarity score
            similarity_score = 1 - distance
            results.append({
                "content": chunk.content,
                "metadata": chunk.metadata,
                "similarity_score": float(similarity_score),
                "rank": len(results) + 1
            })
        
        # Also try searching for individual key terms
        query_words = query.lower().split()
        for word in query_words:
            if len(word) > 3:  # Skip short words
                word_embedding = self.embedding_model.encode([word], convert_to_numpy=True)
                word_distances, word_indices = self.similarity_model.kneighbors(word_embedding, n_neighbors=5)
                
                for distance, idx in zip(word_distances[0], word_indices[0]):
                    similarity_score = 1 - distance
                    if similarity_score > 0.3:  # Decent similarity threshold
                        chunk = self.chunks[idx]
                        # Check if we already have this chunk
                        existing = next((r for r in results if r['metadata']['chunk_id'] == chunk.metadata['chunk_id']), None)
                        if not existing:
                            results.append({
                                "content": chunk.content,
                                "metadata": chunk.metadata,
                                "similarity_score": float(similarity_score * 0.8),  # Slightly lower weight for word-only matches
                                "rank": len(results) + 1
                            })
        
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        final_results = results[:top_k]
        
        logger.info(f"Enhanced search retrieved {len(final_results)} relevant chunks from {len(results)} candidates")
        return final_results

class Component4_ClauseMatching:
    """Component 4: Clause Matching - Semantic Similarity Analysis"""
    
    def __init__(self, gemini_api_key: str):
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def identify_relevant_clauses(self, query_structure: Dict[str, Any], search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify and rank relevant clauses based on semantic similarity"""
        matched_clauses = []
        
        # Get search keywords from enhanced parsing
        main_keywords = query_structure.get('main_keywords', [])
        alternative_terms = query_structure.get('alternative_terms', [])
        search_strategy = query_structure.get('search_strategy', [])
        
        for result in search_results:
            clause_analysis = self._analyze_clause_relevance(
                query_structure, 
                result["content"],
                result["similarity_score"],
                main_keywords,
                alternative_terms
            )
            
            if clause_analysis["is_relevant"]:
                matched_clauses.append({
                    **result,
                    "clause_analysis": clause_analysis,
                    "relevance_score": clause_analysis["relevance_score"]
                })
        
        # Sort by relevance score
        matched_clauses.sort(key=lambda x: x["relevance_score"], reverse=True)
        logger.info(f"Identified {len(matched_clauses)} relevant clauses using enhanced matching")
        return matched_clauses
    
    def _analyze_clause_relevance(self, query_structure: Dict[str, Any], clause_content: str, similarity_score: float, main_keywords: List[str], alternative_terms: List[str]) -> Dict[str, Any]:
        """Enhanced clause relevance analysis"""
        
        # Check for keyword presence (simple but effective boost)
        keyword_boost = 0.0
        content_lower = clause_content.lower()
        
        for keyword in main_keywords:
            if keyword.lower() in content_lower:
                keyword_boost += 0.2
        
        for alt_term in alternative_terms:
            if alt_term.lower() in content_lower:
                keyword_boost += 0.1
        
        # Cap the boost
        keyword_boost = min(keyword_boost, 0.5)
        
        prompt = f"""
        Analyze if this document section contains information relevant to the user's query:
        
        User wants to know: {query_structure.get('intent', 'unknown')}
        Query type: {query_structure.get('query_type', 'unknown')}
        Looking for keywords: {main_keywords}
        Alternative terms: {alternative_terms}
        
        Document Section: "{clause_content[:800]}..."
        
        Questions:
        1. Does this section contain information that could answer the user's question?
        2. How relevant is this content (0.0 to 1.0)?
        3. What specific information does this section provide?
        4. Are there any conditions or limitations mentioned?
        
        Return in format:
        Relevant: [yes/no]
        Score: [0.0-1.0]
        Key Info: [what information this section provides]
        Conditions: [any conditions or limitations]
        """
        
        try:
            response = self.model.generate_content(prompt)
            analysis = self._parse_clause_analysis(response.text, similarity_score + keyword_boost)
            
            # Apply keyword boost to final score
            analysis['relevance_score'] = min(analysis['relevance_score'] + keyword_boost, 1.0)
            
            return analysis
        except Exception as e:
            logger.error(f"Error analyzing clause relevance: {e}")
            return {
                "is_relevant": (similarity_score + keyword_boost) > 0.3,
                "relevance_score": similarity_score + keyword_boost,
                "key_information": "Analysis unavailable",
                "conditions": "Unknown"
            }
    
    def _parse_clause_analysis(self, response: str, similarity_score: float) -> Dict[str, Any]:
        """Parse clause analysis response"""
        lines = response.strip().split('\n')
        analysis = {}
        
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                
                if key == 'relevant':
                    analysis['is_relevant'] = value.lower() in ['yes', 'true']
                elif key == 'score':
                    try:
                        analysis['relevance_score'] = float(value)
                    except:
                        analysis['relevance_score'] = similarity_score
                elif key == 'key info':
                    analysis['key_information'] = value
                elif key == 'conditions':
                    analysis['conditions'] = value
        
        # Defaults
        analysis.setdefault('is_relevant', similarity_score > 0.3)
        analysis.setdefault('relevance_score', similarity_score)
        analysis.setdefault('key_information', 'Information extracted')
        analysis.setdefault('conditions', 'No specific conditions identified')
        
        return analysis

class Component5_LogicEvaluation:
    """Component 5: Logic Evaluation - Decision Processing with LLM"""
    
    def __init__(self, gemini_api_key: str):
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def process_decision_logic(self, query: str, query_structure: Dict[str, Any], matched_clauses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process decision logic using LLM reasoning"""
        
        # Prepare context from matched clauses
        context = self._prepare_context(matched_clauses)
        
        # Generate comprehensive answer
        decision_result = self._generate_decision(query, query_structure, context, matched_clauses)
        
        logger.info("Decision processing completed")
        return decision_result
    
    def _prepare_context(self, matched_clauses: List[Dict[str, Any]]) -> str:
        """Prepare context from matched clauses with enhanced location information"""
        context_parts = []
        for i, clause in enumerate(matched_clauses[:5], 1):  # Top 5 clauses
            content = clause['content']
            
            # Try to extract section/clause numbers from the content
            section_info = self._extract_section_info(content)
            
            context_parts.append(f"""
Clause {i} (Relevance: {clause['relevance_score']:.2f}):
{section_info}
Content: {content}
Key Information: {clause['clause_analysis']['key_information']}
Conditions: {clause['clause_analysis']['conditions']}
---
""")
        return "\n".join(context_parts)
    
    def _extract_section_info(self, content: str) -> str:
        """Extract section/clause information from content"""
        import re
        
        # Look for section patterns
        section_patterns = [
            r'Section\s+(\d+\.?\d*)',
            r'Clause\s+(\d+\.?\d*)',
            r'Article\s+(\d+\.?\d*)',
            r'(\d+\.?\d*)\s*[\.:]',
            r'Page\s+(\d+)',
            r'(\d+\.\d+\.\d+)',  # Pattern like 7.2.1
            r'(\d+\.\d+)',       # Pattern like 7.2
        ]
        
        found_sections = []
        for pattern in section_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                found_sections.extend(matches)
        
        if found_sections:
            return f"Document Location: {', '.join(set(found_sections[:3]))}"
        else:
            return "Document Location: Not specified"
    
    def _generate_decision(self, query: str, query_structure: Dict[str, Any], context: str, matched_clauses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate final decision using Gemini Pro"""
        
        prompt = f"""
        You are an expert document analyst. A user has asked you a question about a document. Your job is to read their question, understand what they want to know, and provide a helpful answer using the information from the document.

        USER'S QUESTION: "{query}"

        DOCUMENT INFORMATION:
        {context}

        Please read the user's question carefully and respond naturally. Answer exactly what they're asking for. If they want definitions, give definitions. If they want amounts, give amounts. If they want to know if something is covered, tell them yes or no with details. Just be helpful and natural.

        IMPORTANT: Even if the document sections don't seem directly related, look carefully for any information that might answer the user's question. Sometimes the information they want is mentioned in a different context or using different terminology.

        If you truly cannot find the answer in the provided document sections, say so clearly and suggest what type of information would be needed to answer their question.

        Format your response as:
        ANSWER: [Your natural response to their question]
        DETAILS: [Any additional helpful information you found]
        CONDITIONS: [Any limitations or special conditions]
        CONFIDENCE: [How confident you are: High/Medium/Low]
        REFERENCES: [Quote the relevant parts from the document, even if indirect]
        """
        
        try:
            response = self.model.generate_content(prompt)
            decision_result = self._parse_decision_response(response.text, matched_clauses)
            return decision_result
        except Exception as e:
            logger.error(f"Error generating decision: {e}")
            return self._generate_fallback_decision(query, matched_clauses)
    
    def _parse_decision_response(self, response: str, matched_clauses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Parse the decision response from Gemini Pro"""
        result = {
            "answer": "",
            "details": "",
            "conditions": "",
            "confidence": "Medium",
            "references": "",
            "supporting_clauses": len(matched_clauses),
            "processing_timestamp": datetime.now().isoformat()
        }
        
        lines = response.strip().split('\n')
        current_section = None
        
        for line in lines:
            if ':' in line and line.split(':')[0].strip().upper() in ['ANSWER', 'DETAILS', 'CONDITIONS', 'CONFIDENCE', 'REFERENCES']:
                key, value = line.split(':', 1)
                current_section = key.strip().lower()
                result[current_section] = value.strip()
            elif current_section and line.strip():
                result[current_section] += " " + line.strip()
        
        return result
    
    def _generate_fallback_decision(self, query: str, matched_clauses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fallback decision generation if LLM fails"""
        if matched_clauses:
            top_clause = matched_clauses[0]
            answer = f"Based on the document analysis, relevant information was found regarding your query. {top_clause['clause_analysis']['key_information']}"
        else:
            answer = "No relevant information found in the provided documents for this query."
        
        return {
            "answer": answer,
            "details": "Fallback response generated due to processing error",
            "conditions": "Please verify with original document",
            "confidence": "Low",
            "references": "Auto-generated response",
            "supporting_clauses": len(matched_clauses),
            "processing_timestamp": datetime.now().isoformat()
        }

class Component6_TextOutput:
    """Component 6: Text Output - Format final response"""
    
    @staticmethod
    def format_text_response(decision_result: Dict[str, Any], query: str) -> str:
        """Format the final text response"""
        
        formatted_response = f"""
=== INTELLIGENT DOCUMENT QUERY SYSTEM RESPONSE ===

Query: {query}

ANSWER:
{decision_result.get('answer', 'No answer generated')}

SUPPORTING DETAILS:
{decision_result.get('details', 'No additional details available')}

CONDITIONS & LIMITATIONS:
{decision_result.get('conditions', 'No specific conditions identified')}

CONFIDENCE LEVEL: {decision_result.get('confidence', 'Unknown')}

DOCUMENT REFERENCES:
{decision_result.get('references', 'No specific references')}

ANALYSIS METADATA:
- Supporting Clauses Found: {decision_result.get('supporting_clauses', 0)}
- Processing Time: {decision_result.get('processing_timestamp', 'Unknown')}

=== END RESPONSE ===
"""
        return formatted_response.strip()
    
    @staticmethod
    def format_json_response(decision_result: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Format as JSON response (for API)"""
        return {
            "query": query,
            "response": {
                "answer": decision_result.get('answer', ''),
                "details": decision_result.get('details', ''),
                "conditions": decision_result.get('conditions', ''),
                "confidence": decision_result.get('confidence', 'Unknown'),
                "references": decision_result.get('references', '')
            },
            "metadata": {
                "supporting_clauses": decision_result.get('supporting_clauses', 0),
                "processing_timestamp": decision_result.get('processing_timestamp', ''),
                "system_version": "1.0.0"
            }
        }

# Main System Integration
class IntelligentQueryRetrievalSystem:
    """Main system that orchestrates all components"""
    
    def __init__(self, gemini_api_key: str):
        self.component1 = Component1_InputDocuments()
        self.component2 = Component2_LLMParser(gemini_api_key)
        self.component3 = Component3_EmbeddingSearch()
        self.component4 = Component4_ClauseMatching(gemini_api_key)
        self.component5 = Component5_LogicEvaluation(gemini_api_key)
        self.component6 = Component6_TextOutput()
        
        self.document_processed = False
    
    def process_document(self, document_url: str) -> bool:
        """Process document through the system"""
        try:
            logger.info("=== STARTING DOCUMENT PROCESSING ===")
            
            # Component 1: Input Documents
            logger.info("Component 1: Processing input document...")
            document_text = self.component1.download_pdf_from_url(document_url)
            
            # Component 3: Embedding Search (Document indexing)
            logger.info("Component 3: Building embedding search index...")
            chunks = self.component3.create_document_chunks(document_text)
            self.component3.build_search_index(chunks)
            
            self.document_processed = True
            logger.info("=== DOCUMENT PROCESSING COMPLETE ===")
            return True
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return False
    
    def query_system(self, query: str, output_format: str = "text") -> str:
        """Process a query through the entire system"""
        try:
            if not self.document_processed:
                return "Error: No document has been processed. Please process a document first."
            
            logger.info(f"=== PROCESSING QUERY: {query} ===")
            
            # Component 2: LLM Parser
            logger.info("Component 2: Parsing query structure...")
            query_structure = self.component2.parse_query_structure(query)
            
            # Component 3: Embedding Search
            logger.info("Component 3: Performing semantic search...")
            
            # Use enhanced keywords from query parsing for better search
            search_keywords = query_structure.get('search_strategy', [])
            if search_keywords:
                # Create enhanced search query
                enhanced_query = f"{query} {' '.join(search_keywords)}"
                logger.info(f"Enhanced search query: {enhanced_query}")
                search_results = self.component3.semantic_search(enhanced_query, top_k=config.TOP_K_RESULTS)
            else:
                search_results = self.component3.semantic_search(query, top_k=config.TOP_K_RESULTS)
            
            # Log search results for debugging
            logger.info(f"Search found {len(search_results)} results with scores: {[r['similarity_score'] for r in search_results]}")
            for i, result in enumerate(search_results[:2]):  # Show top 2 results
                logger.info(f"Result {i+1} preview: {result['content'][:200]}...")
            
            # Component 4: Clause Matching
            logger.info("Component 4: Matching relevant clauses...")
            matched_clauses = self.component4.identify_relevant_clauses(query_structure, search_results)
            
            # Component 5: Logic Evaluation
            logger.info("Component 5: Processing decision logic...")
            decision_result = self.component5.process_decision_logic(query, query_structure, matched_clauses)
            
            # Component 6: Text Output
            logger.info("Component 6: Formatting output...")
            if output_format.lower() == "json":
                final_response = json.dumps(
                    self.component6.format_json_response(decision_result, query), 
                    indent=2
                )
            else:
                final_response = self.component6.format_text_response(decision_result, query)
            
            logger.info("=== QUERY PROCESSING COMPLETE ===")
            return final_response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"Error processing query: {str(e)}"

# FastAPI Application
from fastapi import APIRouter

app = FastAPI(
    title="LLM-Powered Intelligent Query-Retrieval System",
    description="Advanced document analysis system with semantic search and LLM reasoning",
    version="1.0.0"
)

# Create API router with v1 prefix
api_router = APIRouter(prefix="/api/v1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

# Global system instance
intelligent_system = None

# Pydantic models
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

class DocumentRequest(BaseModel):
    document_url: str

class SingleQueryRequest(BaseModel):
    query: str
    output_format: str = "text"

class SingleQueryResponse(BaseModel):
    response: str

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify bearer token"""
    if credentials.credentials != config.BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials.credentials

@api_router.post("/hackrx/run", response_model=QueryResponse)
async def run_query_system(request: QueryRequest, token: str = Depends(verify_token)):
    """Main API endpoint for the hackathon challenge"""
    global intelligent_system
    
    try:
        # Initialize system if not already done
        if intelligent_system is None:
            intelligent_system = IntelligentQueryRetrievalSystem(config.GEMINI_API_KEY)
        
        # Process document
        logger.info(f"Processing document: {request.documents}")
        if not intelligent_system.process_document(request.documents):
            raise HTTPException(status_code=400, detail="Failed to process document")
        
        # Process all questions
        answers = []
        for question in request.questions:
            logger.info(f"Processing question: {question}")
            answer_result = intelligent_system.query_system(question, output_format="text")
            
            # Extract just the answer part for the API response
            if "ANSWER:" in answer_result:
                answer = answer_result.split("ANSWER:")[1].split("SUPPORTING DETAILS:")[0].strip()
            else:
                answer = answer_result
            
            answers.append(answer)
        
        return QueryResponse(answers=answers)
        
    except Exception as e:
        logger.error(f"Error in API endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@api_router.post("/process_document")
async def process_document_endpoint(request: DocumentRequest, token: str = Depends(verify_token)):
    """Process a document for querying"""
    global intelligent_system
    
    try:
        # Initialize system if not already done
        if intelligent_system is None:
            intelligent_system = IntelligentQueryRetrievalSystem(config.GEMINI_API_KEY)
        
        # Process document
        logger.info(f"Processing document: {request.document_url}")
        success = intelligent_system.process_document(request.document_url)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to process document")
        
        return {"status": "success", "message": "Document processed successfully"}
        
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@api_router.post("/query", response_model=SingleQueryResponse)
async def query_endpoint(request: SingleQueryRequest, token: str = Depends(verify_token)):
    """Query the processed document"""
    global intelligent_system
    
    try:
        if intelligent_system is None:
            raise HTTPException(status_code=400, detail="System not initialized. Please process a document first.")
        
        if not intelligent_system.document_processed:
            raise HTTPException(status_code=400, detail="No document processed. Please process a document first.")
        
        # Process query
        logger.info(f"Processing query: {request.query}")
        response = intelligent_system.query_system(request.query, request.output_format)
        
        return SingleQueryResponse(response=response)
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# Include the API router
app.include_router(api_router)

# Root endpoint
@app.get("/")
async def root():
    return {
        "status": "healthy",
        "message": "LLM-Powered Intelligent Query-Retrieval System",
        "version": "1.0.0",
        "endpoints": {
            "main": "/api/v1/hackrx/run",
            "process_document": "/api/v1/process_document", 
            "query": "/api/v1/query",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "system": "LLM-Powered Query-Retrieval System", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    
    # Set your Gemini API key here
    if config.GEMINI_API_KEY == "your-gemini-api-key-here":
        print("⚠️  Please set your GEMINI_API_KEY environment variable or update config.GEMINI_API_KEY in the code")
    
    print("🚀 Starting LLM-Powered Intelligent Query-Retrieval System...")
    print("📋 System Components:")
    print("   1. ✅ Input Documents (PDF Blob URL)")
    print("   2. ✅ LLM Parser (Gemini Pro)")
    print("   3. ✅ Embedding Search (Scikit-learn Similarity)")
    print("   4. ✅ Clause Matching (Semantic Similarity)")
    print("   5. ✅ Logic Evaluation (LLM Decision Processing)")
    print("   6. ✅ Text Output")
    print("\n🌐 Server starting on http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
