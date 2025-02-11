#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 23:15:24 2025

@author: sintu
"""

import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import uuid
import shutil
import chromadb
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoModelForVision2Seq
from sentence_transformers import SentenceTransformer
from unstructured.partition.pdf import partition_pdf

@dataclass
class DocumentElement:
    content: str
    element_type: str
    image_path: str = None
    image_summary: str = None

class MultimodalRAG:
    def __init__(self, model_path: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"):
        # Initialize embeddings model
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Initialize LLM
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Initialize image captioning model
        self.image_processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.image_model = AutoModelForVision2Seq.from_pretrained("Salesforce/blip-image-captioning-base")
        
        # Initialize vector store
        self.chroma_client = chromadb.PersistentClient(path="multimodal_rag_db")
        self.collection = self.chroma_client.get_or_create_collection(
            name="multimodal_documents",
            metadata={"hnsw:space": "cosine"}
        )

    def extract_content(self, file_path: str) -> Tuple[List[str], List[str]]:
        """Extract text and images from PDF."""
        # Create figures directory if it doesn't exist
        figures_dir = "figures"
        if os.path.exists(figures_dir):
            shutil.rmtree(figures_dir)
        os.makedirs(figures_dir)
        
        # Extract chunks from PDF
        chunks = partition_pdf(
            filename=file_path,
            infer_table_structure=True,
            strategy="hi_res",
            extract_image_block_types=["Image"],
        )
        
        # Extract text content
        texts = [chunk.text for chunk in chunks if hasattr(chunk, 'text') and chunk.text]
        
        # Get list of extracted images
        image_paths = []
        if os.path.exists(figures_dir):
            image_paths = [os.path.join(figures_dir, f) for f in os.listdir(figures_dir) 
                         if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        return texts, image_paths

    def generate_image_summary(self, image_path: str) -> str:
        """Generate a summary description for an image."""
        try:
            image = Image.open(image_path)
            inputs = self.image_processor(images=image, return_tensors="pt")
            
            outputs = self.image_model.generate(
                **inputs,
                max_length=50,
                num_beams=4,
                early_stopping=True
            )
            
            return self.image_processor.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Error generating summary for {image_path}: {str(e)}")
            return "Unable to generate image summary"

    def process_document(self, document_path: str) -> List[DocumentElement]:
        """Process document and extract text and images."""
        texts, image_paths = self.extract_content(document_path)
        processed_elements = []
        
        # Process text chunks
        for text in texts:
            if text.strip():  # Only process non-empty text
                processed_elements.append(DocumentElement(
                    content=text,
                    element_type="text"
                ))
        
        # Process images
        for image_path in image_paths:
            try:
                # Generate image summary
                image_summary = self.generate_image_summary(image_path)
                
                # Create a copy in our persistent storage
                persistent_path = os.path.join("stored_images", os.path.basename(image_path))
                os.makedirs("stored_images", exist_ok=True)
                shutil.copy2(image_path, persistent_path)
                
                processed_elements.append(DocumentElement(
                    content=image_summary,
                    element_type="image",
                    image_path=persistent_path,
                    image_summary=image_summary
                ))
            except Exception as e:
                print(f"Error processing image {image_path}: {str(e)}")
        
        return processed_elements

    def store_elements(self, elements: List[DocumentElement]):
        """Store processed elements in the vector database."""
        for element in elements:
            # Generate embedding for the content
            embedding = self.embedding_model.encode(element.content)
            
            # Prepare metadata
            metadata = {
                "element_type": element.element_type,
                "image_path": element.image_path if element.image_path else "",
                "image_summary": element.image_summary if element.image_summary else ""
            }
            
            # Add to vector store
            self.collection.add(
                embeddings=[embedding.tolist()],
                documents=[element.content],
                metadatas=[metadata],
                ids=[f"element_{uuid.uuid4()}"]
            )

    def query(self, question: str, n_results: int = 3) -> Dict[str, Any]:
        """Query the RAG system with a question."""
        # Generate embedding for the question
        question_embedding = self.embedding_model.encode(question)
        
        # Query vector store
        results = self.collection.query(
            query_embeddings=[question_embedding.tolist()],
            n_results=n_results
        )
        
        # Construct prompt with retrieved context
        prompt = "Context:\n"
        retrieved_images = []
        
        for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
            prompt += f"{doc}\n"
            if metadata['element_type'] == 'image' and metadata['image_path']:
                retrieved_images.append(metadata['image_path'])
        
        prompt += f"\nQuestion: {question}\nAnswer:"
        
        # Generate response using LLM
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.llm.device)
        outputs = self.llm.generate(**inputs, max_length=500, num_beams=4)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {
            "response": response,
            "retrieved_images": retrieved_images
        }
