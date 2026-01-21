"""
RAG Pipeline for Seismic Data Analysis

Implements Retrieval-Augmented Generation (RAG) using:
- sklearn TF-IDF for text embeddings (no heavy dependencies)
- FAISS vector store for efficient similarity search
- Ollama LLM for generation
- Document chunking and indexing

This demonstrates LLMOps capabilities for the MLOps project.
"""
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import pickle

# sklearn for embeddings (lightweight alternative)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Vector store
FAISS_AVAILABLE = False
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    print("Warning: FAISS not available. Using sklearn cosine similarity instead.")

# LLM
OLLAMA_AVAILABLE = False
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    print("Warning: Ollama not available.")


@dataclass
class Document:
    """Document chunk for RAG."""
    content: str
    metadata: Dict[str, Any]
    doc_id: str
    embedding: Optional[np.ndarray] = None


class SeismicDocumentLoader:
    """
    Load and chunk seismic-related documents for RAG.
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize document loader.
        
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def load_from_text(self, text: str, source: str = "unknown") -> List[Document]:
        """
        Load and chunk text content.
        
        Args:
            text: Text content to chunk
            source: Source identifier
            
        Returns:
            List of Document chunks
        """
        chunks = self._chunk_text(text)
        documents = []
        
        for i, chunk in enumerate(chunks):
            doc = Document(
                content=chunk,
                metadata={'source': source, 'chunk_index': i},
                doc_id=f"{source}_{i}"
            )
            documents.append(doc)
        
        return documents
    
    def load_from_file(self, file_path: Path) -> List[Document]:
        """
        Load and chunk file content.
        
        Args:
            file_path: Path to file
            
        Returns:
            List of Document chunks
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return self.load_from_text(content, source=file_path.name)
    
    def load_seismic_metadata(self, metadata_path: Path) -> List[Document]:
        """
        Load seismic dataset metadata as documents.
        
        Args:
            metadata_path: Path to metadata JSON file
            
        Returns:
            List of Document chunks
        """
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Convert metadata to text
        text_parts = []
        
        text_parts.append(f"Dataset: {metadata.get('source_file', 'Unknown')}")
        text_parts.append(f"Total traces: {metadata.get('total_traces', 'Unknown')}")
        text_parts.append(f"Sampled traces: {metadata.get('sampled_traces', 'Unknown')}")
        text_parts.append(f"Samples per trace: {metadata.get('samples_per_trace', 'Unknown')}")
        text_parts.append(f"Sample rate: {metadata.get('sample_rate_ms', 'Unknown')} ms")
        
        if 'class_distribution' in metadata:
            text_parts.append("\nClass Distribution:")
            for class_name, count in metadata['class_distribution'].items():
                text_parts.append(f"  - {class_name}: {count}")
        
        text = "\n".join(text_parts)
        return self.load_from_text(text, source="dataset_metadata")
    
    def load_validation_results(self, validation_path: Path) -> List[Document]:
        """
        Load validation results as documents.
        
        Args:
            validation_path: Path to validation JSON file
            
        Returns:
            List of Document chunks
        """
        with open(validation_path, 'r') as f:
            validation = json.load(f)
        
        text_parts = []
        text_parts.append("Data Validation Results:")
        text_parts.append(f"Total traces: {validation.get('total_traces', 'Unknown')}")
        text_parts.append(f"Total files: {validation.get('total_files', 'Unknown')}")
        text_parts.append(f"Validation passed: {validation.get('validation_passed', 'Unknown')}")
        text_parts.append(f"Sample rate consistent: {validation.get('sample_rate_consistency', 'Unknown')}")
        text_parts.append(f"Num samples consistent: {validation.get('num_samples_consistency', 'Unknown')}")
        
        if validation.get('anomalies'):
            text_parts.append("\nAnomalies detected:")
            for anomaly in validation['anomalies']:
                text_parts.append(f"  - {anomaly}")
        else:
            text_parts.append("\nNo anomalies detected.")
        
        if 'class_label_distribution' in validation:
            text_parts.append("\nClass Label Distribution:")
            for label, count in validation['class_label_distribution'].items():
                class_name = ['Normal', 'Anomaly', 'Boundary'][int(label)]
                text_parts.append(f"  - {class_name} (class {label}): {count}")
        
        text = "\n".join(text_parts)
        return self.load_from_text(text, source="validation_results")
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence end
                for sep in ['. ', '.\n', '\n\n', '\n']:
                    last_sep = text[start:end].rfind(sep)
                    if last_sep > self.chunk_size // 2:
                        end = start + last_sep + len(sep)
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - self.chunk_overlap
        
        return chunks


class TfidfEmbeddingModel:
    """
    Text embedding model using TF-IDF (lightweight, no heavy dependencies).
    """
    
    def __init__(self, max_features: int = 1000):
        """
        Initialize embedding model.
        
        Args:
            max_features: Maximum number of TF-IDF features
        """
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.embedding_dim = max_features
        self.is_fitted = False
    
    def fit(self, texts: List[str]):
        """Fit the vectorizer on texts."""
        self.vectorizer.fit(texts)
        self.embedding_dim = len(self.vectorizer.get_feature_names_out())
        self.is_fitted = True
        print(f"TF-IDF vectorizer fitted with {self.embedding_dim} features")
    
    def encode(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Args:
            texts: List of texts to encode
            show_progress: Whether to show progress
            
        Returns:
            Array of embeddings (n_texts, embedding_dim)
        """
        if not self.is_fitted:
            self.fit(texts)
        
        embeddings = self.vectorizer.transform(texts).toarray()
        return embeddings.astype(np.float32)


class SimpleVectorStore:
    """
    Simple vector store using sklearn cosine similarity.
    Falls back to this if FAISS is not available.
    """
    
    def __init__(self):
        self.documents: List[Document] = []
        self.embeddings: Optional[np.ndarray] = None
    
    def add_documents(self, documents: List[Document], embeddings: np.ndarray):
        """Add documents with embeddings."""
        self.documents = documents
        self.embeddings = embeddings
        print(f"Added {len(documents)} documents to vector store")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[Document, float]]:
        """Search for similar documents using cosine similarity."""
        if self.embeddings is None or len(self.documents) == 0:
            return []
        
        # Calculate cosine similarity
        query = query_embedding.reshape(1, -1)
        similarities = cosine_similarity(query, self.embeddings)[0]
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_indices:
            results.append((self.documents[idx], float(similarities[idx])))
        
        return results
    
    def save(self, path: Path):
        """Save vector store to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings
        np.save(path / "embeddings.npy", self.embeddings)
        
        # Save documents
        docs_data = [
            {
                'content': doc.content,
                'metadata': doc.metadata,
                'doc_id': doc.doc_id
            }
            for doc in self.documents
        ]
        with open(path / "documents.json", 'w') as f:
            json.dump(docs_data, f, indent=2)
        
        print(f"Vector store saved to: {path}")
    
    def load(self, path: Path):
        """Load vector store from disk."""
        path = Path(path)
        
        # Load embeddings
        self.embeddings = np.load(path / "embeddings.npy")
        
        # Load documents
        with open(path / "documents.json", 'r') as f:
            docs_data = json.load(f)
        
        self.documents = [
            Document(
                content=d['content'],
                metadata=d['metadata'],
                doc_id=d['doc_id']
            )
            for d in docs_data
        ]
        
        print(f"Vector store loaded: {len(self.documents)} documents")


class FAISSVectorStore:
    """
    FAISS-based vector store for efficient similarity search.
    """
    
    def __init__(self, embedding_dim: int = 1000):
        """
        Initialize FAISS vector store.
        
        Args:
            embedding_dim: Dimension of embeddings
        """
        self.embedding_dim = embedding_dim
        self.index = None
        self.documents: List[Document] = []
        
        if FAISS_AVAILABLE:
            self._create_index()
    
    def _create_index(self):
        """Create FAISS index."""
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product (cosine after normalization)
        print(f"Created FAISS index with dimension {self.embedding_dim}")
    
    def add_documents(self, documents: List[Document], embeddings: np.ndarray):
        """Add documents with embeddings to the index."""
        if not FAISS_AVAILABLE:
            raise RuntimeError("FAISS not available")
        
        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        embeddings_normalized = embeddings / norms
        
        # Add to index
        self.index.add(embeddings_normalized.astype(np.float32))
        
        # Store documents
        self.documents = documents
        
        print(f"Added {len(documents)} documents to FAISS index")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[Document, float]]:
        """Search for similar documents."""
        if not FAISS_AVAILABLE:
            raise RuntimeError("FAISS not available")
        
        # Normalize query
        query = query_embedding.reshape(1, -1).astype(np.float32)
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm
        
        # Search
        scores, indices = self.index.search(query, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents) and idx >= 0:
                results.append((self.documents[idx], float(score)))
        
        return results
    
    def save(self, path: Path):
        """Save vector store to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(path / "faiss.index"))
        
        # Save documents
        docs_data = [
            {
                'content': doc.content,
                'metadata': doc.metadata,
                'doc_id': doc.doc_id
            }
            for doc in self.documents
        ]
        with open(path / "documents.json", 'w') as f:
            json.dump(docs_data, f, indent=2)
        
        print(f"FAISS vector store saved to: {path}")
    
    def load(self, path: Path):
        """Load vector store from disk."""
        path = Path(path)
        
        # Load FAISS index
        self.index = faiss.read_index(str(path / "faiss.index"))
        
        # Load documents
        with open(path / "documents.json", 'r') as f:
            docs_data = json.load(f)
        
        self.documents = [
            Document(
                content=d['content'],
                metadata=d['metadata'],
                doc_id=d['doc_id']
            )
            for d in docs_data
        ]
        
        print(f"FAISS vector store loaded: {len(self.documents)} documents")


class SeismicRAGPipeline:
    """
    Complete RAG pipeline for seismic data analysis.
    """
    
    def __init__(self, 
                 llm_model: str = "llama3.1:8b",
                 data_dir: str = "data",
                 max_features: int = 1000):
        """
        Initialize RAG pipeline.
        
        Args:
            llm_model: Ollama model name
            data_dir: Base data directory
            max_features: Max TF-IDF features
        """
        self.data_dir = Path(data_dir)
        self.llm_model = llm_model
        
        # Initialize components
        self.document_loader = SeismicDocumentLoader()
        self.embedding_model = TfidfEmbeddingModel(max_features=max_features)
        self.vector_store = None
    
    def build_knowledge_base(self, include_reports: bool = True) -> int:
        """
        Build knowledge base from seismic data files.
        
        Args:
            include_reports: Whether to include quality reports
            
        Returns:
            Number of documents indexed
        """
        print("=" * 60)
        print("Building RAG Knowledge Base")
        print("=" * 60)
        
        all_documents = []
        
        # Load dataset metadata
        metadata_path = self.data_dir / "raw" / "dataset_metadata.json"
        if metadata_path.exists():
            print(f"Loading: {metadata_path}")
            docs = self.document_loader.load_seismic_metadata(metadata_path)
            all_documents.extend(docs)
            print(f"  Added {len(docs)} chunks")
        
        # Load validation results
        validation_path = self.data_dir / "bronze" / "validation_results.json"
        if validation_path.exists():
            print(f"Loading: {validation_path}")
            docs = self.document_loader.load_validation_results(validation_path)
            all_documents.extend(docs)
            print(f"  Added {len(docs)} chunks")
        
        # Load LLM schema analysis
        llm_analysis_path = self.data_dir / "bronze" / "llm_schema_analysis.txt"
        if llm_analysis_path.exists():
            print(f"Loading: {llm_analysis_path}")
            docs = self.document_loader.load_from_file(llm_analysis_path)
            all_documents.extend(docs)
            print(f"  Added {len(docs)} chunks")
        
        # Load quality reports
        if include_reports:
            bronze_dir = self.data_dir / "bronze"
            for report_file in bronze_dir.glob("quality_report_*.txt"):
                print(f"Loading: {report_file}")
                docs = self.document_loader.load_from_file(report_file)
                all_documents.extend(docs)
                print(f"  Added {len(docs)} chunks")
        
        # Load feature summary
        feature_summary_path = self.data_dir / "silver" / "feature_summary.json"
        if feature_summary_path.exists():
            print(f"Loading: {feature_summary_path}")
            with open(feature_summary_path, 'r') as f:
                summary = json.load(f)
            
            text = f"Feature Engineering Summary:\n"
            text += f"Total features: {summary.get('num_features', 'Unknown')}\n"
            text += f"Handcrafted features: {summary.get('handcrafted_features', 'Unknown')}\n"
            text += f"Embedding features: {summary.get('embedding_features', 'Unknown')}\n"
            text += f"Samples: {summary.get('num_samples', 'Unknown')}\n"
            text += f"Feature names: {', '.join(summary.get('feature_names', []))}\n"
            
            docs = self.document_loader.load_from_text(text, source="feature_summary")
            all_documents.extend(docs)
            print(f"  Added {len(docs)} chunks")
        
        # Load statistical comparison
        stats_path = self.data_dir / "bronze" / "f3_vs_test_statistical_comparison.txt"
        if stats_path.exists():
            print(f"Loading: {stats_path}")
            docs = self.document_loader.load_from_file(stats_path)
            all_documents.extend(docs)
            print(f"  Added {len(docs)} chunks")
        
        # Add seismic domain knowledge
        domain_knowledge = self._get_seismic_domain_knowledge()
        docs = self.document_loader.load_from_text(domain_knowledge, source="domain_knowledge")
        all_documents.extend(docs)
        print(f"Added {len(docs)} domain knowledge chunks")
        
        print(f"\nTotal documents: {len(all_documents)}")
        
        # Generate embeddings
        print("\nGenerating TF-IDF embeddings...")
        texts = [doc.content for doc in all_documents]
        embeddings = self.embedding_model.encode(texts)
        
        # Create vector store
        if FAISS_AVAILABLE:
            self.vector_store = FAISSVectorStore(embedding_dim=embeddings.shape[1])
        else:
            self.vector_store = SimpleVectorStore()
        
        self.vector_store.add_documents(all_documents, embeddings)
        
        # Save vector store
        vector_store_path = self.data_dir / "rag" / "vector_store"
        self.vector_store.save(vector_store_path)
        
        # Save vectorizer
        vectorizer_path = self.data_dir / "rag" / "tfidf_vectorizer.pkl"
        vectorizer_path.parent.mkdir(parents=True, exist_ok=True)
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.embedding_model.vectorizer, f)
        print(f"TF-IDF vectorizer saved to: {vectorizer_path}")
        
        return len(all_documents)
    
    def _get_seismic_domain_knowledge(self) -> str:
        """Get seismic domain knowledge text."""
        return """
        Seismic Data Analysis Domain Knowledge:
        
        1. SEG-Y Format:
        - Standard format for seismic data storage
        - Contains trace data (amplitude values) and headers (metadata)
        - Headers include inline, crossline, CDP coordinates
        - Sample rate typically 1-4 ms
        
        2. Seismic Traces:
        - Each trace represents amplitude over time at a location
        - Traces are organized by inline and crossline numbers
        - Amplitude values indicate subsurface reflections
        
        3. Data Quality Indicators:
        - Consistent sample rates across traces
        - No dead traces (zero variance)
        - Reasonable amplitude ranges
        - Complete spatial coverage
        
        4. Feature Engineering for Seismic:
        - Statistical features: mean, std, min, max, RMS amplitude
        - Frequency features: dominant frequency from FFT
        - Energy features: total energy, zero crossings
        - Embeddings: compressed representations using PCA or autoencoders
        
        5. Classification Tasks:
        - Normal: typical seismic response
        - Anomaly: unusual patterns (faults, bright spots)
        - Boundary: geological boundaries
        
        6. F3 Netherlands Dataset:
        - Public seismic dataset from North Sea
        - 3D seismic survey
        - Contains ~600,000 traces
        - 462 samples per trace at 4ms sample rate
        """
    
    def query(self, question: str, k: int = 5, use_llm: bool = True) -> Dict[str, Any]:
        """
        Query the RAG system.
        
        Args:
            question: User question
            k: Number of context documents to retrieve
            use_llm: Whether to use LLM for generation
            
        Returns:
            Dictionary with answer and context
        """
        if self.vector_store is None:
            # Try to load existing vector store
            vector_store_path = self.data_dir / "rag" / "vector_store"
            vectorizer_path = self.data_dir / "rag" / "tfidf_vectorizer.pkl"
            
            if vector_store_path.exists() and vectorizer_path.exists():
                # Load vectorizer
                with open(vectorizer_path, 'rb') as f:
                    self.embedding_model.vectorizer = pickle.load(f)
                self.embedding_model.is_fitted = True
                
                # Load vector store
                if FAISS_AVAILABLE and (vector_store_path / "faiss.index").exists():
                    self.vector_store = FAISSVectorStore()
                    self.vector_store.load(vector_store_path)
                else:
                    self.vector_store = SimpleVectorStore()
                    self.vector_store.load(vector_store_path)
            else:
                raise RuntimeError("Vector store not built. Run build_knowledge_base() first.")
        
        print(f"\nQuery: {question}")
        
        # Encode question
        query_embedding = self.embedding_model.encode([question], show_progress=False)[0]
        
        # Retrieve relevant documents
        results = self.vector_store.search(query_embedding, k=k)
        
        # Build context
        context_parts = []
        sources = []
        for doc, score in results:
            context_parts.append(f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.content}")
            sources.append({
                'source': doc.metadata.get('source', 'unknown'),
                'score': float(score),
                'content_preview': doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
            })
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Generate answer with LLM
        answer = None
        if use_llm and OLLAMA_AVAILABLE:
            prompt = f"""Based on the following context about seismic data analysis, answer the question.

Context:
{context}

Question: {question}

Provide a clear, concise answer based on the context. If the context doesn't contain enough information, say so.

Answer:"""
            
            try:
                response = ollama.generate(
                    model=self.llm_model,
                    prompt=prompt,
                    options={"temperature": 0.3}
                )
                answer = response.get('response', '')
            except Exception as e:
                print(f"LLM generation failed: {e}")
                answer = f"LLM unavailable. Retrieved context:\n{context}"
        else:
            answer = f"Retrieved context (LLM disabled):\n{context}"
        
        return {
            'question': question,
            'answer': answer,
            'sources': sources,
            'num_sources': len(sources)
        }
    
    def interactive_session(self):
        """Run interactive Q&A session."""
        print("\n" + "=" * 60)
        print("Seismic RAG Interactive Session")
        print("=" * 60)
        print("Type 'quit' to exit, 'rebuild' to rebuild knowledge base")
        print()
        
        while True:
            try:
                question = input("Question: ").strip()
                
                if question.lower() == 'quit':
                    break
                elif question.lower() == 'rebuild':
                    self.build_knowledge_base()
                    continue
                elif not question:
                    continue
                
                result = self.query(question)
                
                print("\nAnswer:")
                print(result['answer'])
                print(f"\n[Retrieved from {result['num_sources']} sources]")
                print("-" * 40)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("\nSession ended.")


def main():
    """Demonstrate RAG pipeline."""
    print("=" * 60)
    print("RAG Pipeline for Seismic Data Analysis")
    print("=" * 60)
    
    # Check dependencies
    print("\nDependency check:")
    print(f"  FAISS: {'Available' if FAISS_AVAILABLE else 'Using sklearn fallback'}")
    print(f"  Ollama: {'Available' if OLLAMA_AVAILABLE else 'NOT AVAILABLE'}")
    print(f"  TF-IDF: Available (sklearn)")
    
    # Initialize pipeline
    rag = SeismicRAGPipeline(
        llm_model="llama3.1:8b",
        data_dir="data",
        max_features=1000
    )
    
    # Build knowledge base
    num_docs = rag.build_knowledge_base()
    print(f"\nIndexed {num_docs} document chunks")
    
    # Example queries
    print("\n" + "=" * 60)
    print("Example Queries")
    print("=" * 60)
    
    example_questions = [
        "What is the sample rate of the seismic data?",
        "How many traces are in the dataset?",
        "What features were extracted from the seismic traces?",
        "Were there any data quality issues detected?"
    ]
    
    for question in example_questions:
        result = rag.query(question, k=3)
        print(f"\nQ: {question}")
        print(f"A: {result['answer'][:500]}...")
        print("-" * 40)
    
    print("\n[SUCCESS] RAG pipeline demonstration complete!")
    print(f"Vector store saved to: data/rag/vector_store")


if __name__ == "__main__":
    main()
