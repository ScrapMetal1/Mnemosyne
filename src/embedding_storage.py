# embedding_storage.py
from datetime import datetime
import json
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
from pathlib import Path
import torch

# Try importing sentence_transformers, handle if missing
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False





##
class EmbeddingStore:
    """
    Stores embeddings + metadata, supports adding, deleting, searching.
    Designed to be lightweight and portable for NPU/MPU.
    """

    def __init__(self, storage_dir: str = "./embeddings_db"):
        """ 
        Initialize the store:
        - storage_dir: folder for persistence
        - embedding_dim: expected embedding size
        - max_items: pre-allocate space for embeddings
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # In-memory storage
        self.embeddings = []
        self.ids: List[str] = []
        self.metadata: List[Dict] = []
        self.num_items = 0
        
        self.load()

    def load(self):
        embeddings_file = self.storage_dir / "embeddings.npy"
        metadata_file = self.storage_dir / "metadata.json"

        if embeddings_file.exists():
            # Load embeddings as numpy array
            self.embeddings = np.load(embeddings_file, allow_pickle=True).tolist()
            self.embeddings = [np.array(e) for e in self.embeddings]
        
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                data = json.load(f)
                self.metadata = data.get('metadata', [])
                self.ids = data.get('ids', [])
                
    def add(self, embedding: np.ndarray, text: str, metadata: Optional[Dict] = None, memory_id: Optional[str] = None) -> str:
        """
        Add an embedding to the store.
        Returns the unique memory_id.
        """
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.detach().cpu().numpy()
        embedding = np.array(embedding).flatten()

        # Normalize embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        if memory_id is None:
            memory_id = f"memory_{datetime.now().timestamp()}_{len(self.ids)}"
        
        # Store the embedding now
        self.embeddings.append(embedding)
        self.ids.append(memory_id)
        self.metadata.append({
            'text': text,
            'timestamp': datetime.now().isoformat(),
            'id': memory_id,
            **(metadata or {})
        })
        self.save()
        return memory_id

    def delete(self, memory_id: str) -> bool:
        """
        Delete a stored embedding by ID.
        Returns True if deleted.
        """
        if memory_id not in self.ids:
            return False
        
        idx = self.ids.index(memory_id)
        del self.embeddings[idx]
        del self.metadata[idx]
        del self.ids[idx]

        self.save()
        return True

    def search(self, query_embedding: np.ndarray, top_k: int = 5, threshold: float = 0.0) -> List[Dict]:
        """
        Search for similar embeddings using cosine similarity.
        
        Args:
            query_embedding: numpy array of query embedding
            top_k: number of results to return
            threshold: minimum similarity threshold (0.0 to 1.0)
            metadata_filter: optional metadata filter dict
        
        Returns:
            List of result dicts with 'text', 'metadata', 'similarity', 'id'
        """
        
        #check if the storage is not empty
        if len(self.embeddings) == 0:
            return []
        

        #ensure the querys embedding is in numpy format
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.detach().cpu().numpy()
            #squash into 1d array
            query_embedding = np.array(query_embedding).flatten()
            #normalise
            norm = np.linalg.norm(query_embedding)
            if norm > 0:
                query_embedding = query_embedding / norm
            else:
                # Handle zero vector edge case
                return [] 



        #turns all the embeddings into a 2d numpy matrix. 
        embeddings_array = np.array(self.embeddings)



        # use cosine similarty to match against embedding int the database. returns a nmpy array with all the similiarty scores for each embedding. 
        similarities = np.dot(embeddings_array, query_embedding)


        valid_indices = np.arange(len(similarities))



        #TODO later
        #Metadata filter for TIME. Only have soft filter for time.  Start with all indicies as valid. 

        # valid_indicies = np.arange(len(similarities))

        # if metadata_filter:
        #     filtered_indicies = []
        #     for idx in valid_indicies:
        #          meta = self.metadata[idx]
        #          match = True
        #          for key, value in metadata_filter.items():


        
        # Filter by threshold
        above_threshold = similarities >= threshold
        valid_indices = valid_indices[above_threshold]
        similarities = similarities[above_threshold] 
        
        # Get top_k results
        if len(similarities) == 0:
            return []
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Build results
        results = []
        for idx in top_indices:
            original_idx = valid_indices[idx]
            results.append({
                'id': self.ids[original_idx],
                'text': self.metadata[original_idx]['text'],
                #'metadata': self.metadata[original_idx], just repeats text and id
                'similarity': float(similarities[idx])
            })
        
        return results


    def save(self):
        """
        Persist embeddings and metadata to disk.
        """
        embeddings_file = self.storage_dir / "embeddings.npy"
        metadata_file = self.storage_dir / "metadata.json"

        np.save(embeddings_file, np.array(self.embeddings, dtype=object), allow_pickle=True)

        with open(metadata_file, 'w') as f:
            json.dump({
                'metadata': self.metadata,
                'ids': self.ids
            }, f, indent=2)

    def get_by_id(self, memory_id: str) -> Optional[Dict]:
        """
        Retrieve embedding and metadata by memory_id.
        """
        if memory_id not in self.ids:
            return None
        idx = self.ids.index(memory_id)
        return {
            "id": self.ids[idx],
            "metadata": self.metadata[idx],
            "embedding": self.embeddings[idx],
            "text": self.metadata[idx]["text"]
        }

    def count(self) -> int:
        """
        Return total number of stored embeddings.
        """
        return len(self.ids)

    def clear(self):
        """
        Remove all stored embeddings.
        """
        self.embeddings.clear()
        self.metadata.clear()
        self.ids.clear()
        self.save()


class EmbeddingExtractor:
    """
    Responsible for generating embeddings from text.
    Decoupled from storage so you can swap models (FastVLM, Qwen embeddings, etc.).
    """

    def __init__(self, model_name="all-MiniLM-L6-v2", model=None, tokenizer=None, device="cuda"):
        """
        Args:
            model_name: Name of sentence-transformer model (default: all-MiniLM-L6-v2)
            model: Optional existing LLM (fallback)
            tokenizer: Optional existing tokenizer (fallback)
            device: Device to run on
        """

        self.use_sentence_transformer = False
        self.device = device
        
        if HAS_SENTENCE_TRANSFORMERS and model_name: 
            try: 
                print(f"Loading Embedding model: {model_name}...")
                self.st_model = SentenceTransformer(model_name, device=device)
                self.use_sentence_transformer = True
                print(" Embedding Model Loaded" )
                return
            except Exception as e: 
                print(f"Warning: Could not load {model_name}")


        #fallback to using the LLM as an embedding model. Not recommended.         
        self.model = model  # this is an argument you pass in in the CLI
        self.tokenizer = tokenizer

        if not self.model:
            print("Warning: No embedding model available. Pass 'model' and 'tokenizer' or install sentence-transformers.")


    def extract_embeddings(self, text: str) -> np.ndarray:
        """
        Extract embedding from text.
        """

        # Use all-MiniLM embedding model
        if self.use_sentence_transformer:
            embedding = self.st_model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
            return embedding
        

        #fall back on the LLM if not working: 


        #tokenize text, tokens is a dictionary
        if self.model is None or self.tokenizer is None:
             raise ValueError("No model available for embedding extraction")

        tokens = self.tokenizer(
            text,
            return_tensors = 'pt',
            padding = True,   # if less than 512 tokens then the space will be padded
            truncation = True, # if there is more than 512 tokens then it will be cut off
            max_length = 2048,  # or 4096, or 8192 if this is too short. Need to check how many tokens our generated text desc is taking. 
        ).to(self.device)
        
        # Extract embeddings using model's hidden states for contextual embeddings
        with torch.no_grad():
            outputs = self.model(
                input_ids=tokens["input_ids"], #this is just the tokens that represent the text.
                attention_mask=tokens["attention_mask"], #mask
                output_hidden_states=True,
                return_dict=True
            )
            # Use the last hidden state
            last_hidden_state = outputs.hidden_states[-1] # [batch, seq_len, hidden_dim]
            
            # Use the embedding of the LAST token (EOS or last word) for decoder models
            # This is often better than mean pooling for GPT-style models
            # We gather the vector at the index of the last non-padding token
            # 1. Calculate the index of the last real token for each sentence in the batch
            # attention_mask is [1, 1, 1, 0, 0] -> sum is 3 -> last index is 2 (0, 1, 2)
            sequence_lengths = tokens["attention_mask"].sum(dim=1) - 1

            # 2. Create indices for the batch dimension [0, 1, 2, ...]
            batch_indices = torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device)

            # 3. Select the specific vector at [batch_idx, last_token_idx]
            sentence_embedding = last_hidden_state[batch_indices, sequence_lengths] # returns the embeddings for each batch's last token in a 2d array
            
            # Squeeze if batch size is 1
            sentence_embedding = sentence_embedding.squeeze(0) #skips if the batch size is greater than one. 
            
            # Convert to numpy and normalize
            embedding = sentence_embedding.cpu().float().numpy()
            
            # L2 Normalization
            if embedding.ndim == 1: # if only one batch. 
                norm = np.linalg.norm(embedding) # magnitude
                if norm > 0:
                    embedding = embedding / norm #normalise - unit vector
            else:
                norm = np.linalg.norm(embedding, axis=1, keepdims=True) # calculates the lenght of each batch respectively. 
                embedding = np.divide(embedding, norm, out=np.zeros_like(embedding), where=norm!=0) #divides. 
        
        return embedding





    
        


# Integration example
def create_memory_system(model, tokenizer, storage_dir="./embeddings_db"):
    """
    Create a complete memory system with FastVLM and embedding storage.
    """
    extractor = EmbeddingExtractor(model, tokenizer)
    store = EmbeddingStore(storage_dir)
    
    return extractor, store


# Usage example
if __name__ == "__main__":
    import sys
    import os
    import argparse
    import tempfile
    import shutil
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path
    from llava.utils import disable_torch_init

    print("=" * 70)
    print("COMPREHENSIVE EMBEDDING STORAGE TEST")
    print("=" * 70)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None, help="Path to model checkpoint for VLM fallback testing")
    args = parser.parse_args()
    
    # Create test directory in current folder
    temp_dir = "./test_embeddings_db"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    print(f"\nUsing storage directory: {temp_dir}")
    
    try:
        # Initialize Extractor - Prefer SentenceTransformers
        if HAS_SENTENCE_TRANSFORMERS:
             print("\n[Init] Using SentenceTransformer (all-MiniLM-L6-v2)...")
             extractor = EmbeddingExtractor(model_name="all-MiniLM-L6-v2")
        elif args.model_path:
            print(f"\n[Init] Loading VLM from {args.model_path} for embeddings...")
            disable_torch_init()
            model_name = get_model_name_from_path(args.model_path)
            tokenizer, model, image_processor, context_len = load_pretrained_model(
                args.model_path, 
                None, 
                model_name
            )
            extractor = EmbeddingExtractor(model=model, tokenizer=tokenizer)
            print("✓ Real model loaded")
        else:
            print("\n[Error] No SentenceTransformers installed and no --model-path provided.")
            print("        Cannot run tests with real embeddings.")
            sys.exit(1)

        # Initialize components
        store = EmbeddingStore(storage_dir=temp_dir)
        
        print("\n" + "-" * 70)
        print("STEP 1: Adding distinct text descriptions to storage")
        print("-" * 70)
        
        # Define distinct text descriptions with varying detail levels
        descriptions = [
            "House keys located on the marble island countertop next to the fruit bowl.",
            "A large crimson chesterfield sofa facing a flat-screen television mounted above the fireplace. A cinema room",
            "An open MacBook Pro sitting on a vintage oak writing desk near the bay window.",
            "A modern standing desk equipped with dual monitors and a mechanical keyboard.",
            "A ceramic basin with a large vanity mirror and toothbrush holder.",
            # Distractors
            "Outdoor area with an grey shed locked with a masterlock padlock", # Distractor for keys
            "A wooden bench in the hallway.", # Distractor for sitting
            "A broken calculator in the trash." # Distractor for tech
        ]
        
        memory_ids = []
        # Store expected index for simple verification
        stored_texts = [] 
        
        for i, text in enumerate(descriptions):
            print(f"\n[{i+1}] Adding: {text}")
            embedding = extractor.extract_embeddings(text) #get the embedding 
            memory_id = store.add(
                embedding=embedding,
                text=text,
                metadata={"index": i} # Store index to verify later
            )
            memory_ids.append(memory_id)
            stored_texts.append(text)
            print(f"    ✓ Stored with ID: {memory_id}")
        
        print(f"\n✓ Total memories stored: {store.count()}")
        
        print("\n" + "-" * 70)
        print("STEP 2: Testing search functionality with queries")
        print("-" * 70)
        
        # Define test queries mapped to the expected index of the description
        test_queries = [
            # Semantic query for keys (avoiding word "keys" if possible, or context specific)
            ("I need to unlock the front door to my house", 0), 
            # Semantic query for sofa (avoiding "sofa", using "relax", "watch")
            ("Where can I sit down and watch a movie?", 1),
            # Semantic query for laptop (using "portable computer", "work")
            ("Where is my portable computer?", 2),
            # Semantic query for office (using "workstation", "screens")
            ("Where is the workstation with multiple displays?", 3),
            # Semantic query for bathroom (using function "brush teeth", "wash face")
            ("Where can I brush my teeth?", 4)
        ]
        
        all_passed = True
        for i, (query, expected_index) in enumerate(test_queries):
            print(f"\n[{i+1}] Query: \"{query}\"")
            
            # Extract query embedding
            query_embedding = extractor.extract_embeddings(query)
            
            # Search with top_k=1
            results = store.search(query_embedding, top_k=1)
            
            if len(results) == 0:
                print("    ✗ FAILED: No results returned")
                all_passed = False
                continue
            
            result = results[0]
            # Get the original index we stored in metadata
            found_index = result['metadata'].get('index')
            
            print(f"    ✓ Found match:")
            print(f"      Text: {result['text']}")
            print(f"      Similarity: {result['similarity']:.4f}")
            
            if found_index == expected_index:
                 print(f"    ✓ PASSED: Correctly matched index {expected_index}")
            else:
                 print(f"    ✗ FAILED: Expected index {expected_index} ({descriptions[expected_index]}), but got {found_index} ({result['text']})")
                 all_passed = False

            # Verify that we got a result with reasonable similarity
            if result['similarity'] <= 0.0:
                print(f"    ✗ FAILED: Similarity too low: {result['similarity']:.4f}")
                all_passed = False
        
        print("\n" + "-" * 70)
        print("STEP 3: Testing edge cases")
        print("-" * 70)
        
        # Test empty search
        print("\n[Edge Case 1] Searching empty store...")
        empty_store = EmbeddingStore(storage_dir=tempfile.mkdtemp())
        empty_results = empty_store.search(query_embedding, top_k=1)
        assert len(empty_results) == 0, "Empty store should return no results"
        print("    ✓ PASSED: Empty store returns no results")
        
        # Test get_by_id
        print(f"\n[Edge Case 2] Retrieving memory by ID...")
        retrieved = store.get_by_id(memory_ids[0])
        assert retrieved is not None, "Should retrieve memory by ID"
        assert retrieved['text'] == descriptions[0], "Retrieved text should match"
        print(f"    ✓ PASSED: Retrieved memory: {retrieved['text'][:60]}...")
        
        # Test delete
        print(f"\n[Edge Case 3] Deleting a memory...")
        count_before = store.count()
        deleted = store.delete(memory_ids[-1])
        assert deleted, "Delete should return True"
        assert store.count() == count_before - 1, "Count should decrease after delete"
        print(f"    ✓ PASSED: Memory deleted, count: {count_before} → {store.count()}")
        
        # Test persistence
        print(f"\n[Edge Case 4] Testing persistence...")
        count_before_reload = store.count()
        reloaded_store = EmbeddingStore(storage_dir=temp_dir)
        assert reloaded_store.count() == count_before_reload, "Count should persist"
        print(f"    ✓ PASSED: Store persisted, count: {reloaded_store.count()}")
        
        print("\n" + "=" * 70)
        if all_passed:
            print("✓ ALL TESTS PASSED!")
        else:
            print("✗ SOME TESTS FAILED - Check output above")
        print("=" * 70)
        
    finally:
        # Cleanup
        # shutil.rmtree(temp_dir)
        print(f"\nTest database left at: {temp_dir}")
