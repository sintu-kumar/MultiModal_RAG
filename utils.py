#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 23:15:20 2025

@author: sintu
"""
import matplotlib.pyplot as plt
import os 
from PIL import Image
import chromadb

def display_stored_images(chroma_client_path: str = "multimodal_rag_db"):
    """Display all images stored in the database."""
    
    
    # Initialize Chroma client
    client = chromadb.PersistentClient(path=chroma_client_path)
    collection = client.get_collection(name="multimodal_documents")
    
    # Get all items from the collection
    results = collection.get(
        include=['metadatas', 'documents']
    )
    
    # Filter for image elements
    image_entries = []
    for metadata, document in zip(results['metadatas'], results['documents']):
        if metadata.get('element_type') == 'image':
            image_entries.append({
                'image_path': metadata.get('image_path'),
                'image_summary': metadata.get('image_summary'),
                'content': document
            })
    
    # Display images with their summaries
    if image_entries:
        print(f"Found {len(image_entries)} images in the database:\n")
        
        n_images = len(image_entries)
        n_cols = min(3, n_images)
        n_rows = (n_images + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, 5*n_rows))
        
        for idx, entry in enumerate(image_entries):
            image_path = entry['image_path']
            
            if image_path and os.path.exists(image_path):
                plt.subplot(n_rows, n_cols, idx + 1)
                img = Image.open(image_path)
                plt.imshow(img)
                plt.axis('off')
                plt.title(f"Summary: {entry['image_summary']}\n", 
                         wrap=True, 
                         fontsize=10)
                
                print(f"Image {idx + 1}:")
                print(f"Path: {image_path}")
                print(f"Summary: {entry['image_summary']}")
                print(f"Content: {entry['content']}\n")
            else:
                print(f"Warning: Image file not found at {image_path}")
        
        plt.tight_layout()
        plt.show()
    else:
        print("No images found in the database.")
    
    return image_entries