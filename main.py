#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 23:15:24 2025

@author: sintu
"""
from utils import display_stored_images
from RAG_MM import MultimodalRAG
import argparse
import sys
from pathlib import Path


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Multimodal RAG system for processing documents with text and images'
    )
    
    # Add arguments
    parser.add_argument(
        '--document_path',
        type=str,
        required=True,
        help='Path to the document file (PDF)'
    )
    parser.add_argument(
        '--query',
        type=str,
        required=True,
        help='The query to run on the processed document'
    )
    
    # Parse arguments
    try:
        args = parser.parse_args()
        
        # Validate file path
        document_path = Path(args.document_path)
        if not document_path.exists():
            print(f"Error: File '{document_path}' does not exist")
            sys.exit(1)
        if document_path.suffix.lower() != '.pdf':
            print(f"Error: File '{document_path}' is not a PDF file")
            sys.exit(1)
            
        # Initialize the RAG system
        print("Initializing RAG system...")
        rag = MultimodalRAG()
        
        # Process document
        print(f"\nProcessing document: {document_path}")
        elements = rag.process_document(str(document_path))
        print(f"Extracted {len(elements)} elements")
        
        # Store elements
        print("\nStoring elements in vector database...")
        rag.store_elements(elements)
        
        # # Display stored images
        # print("\nDisplaying stored images...")
        # stored_images = display_stored_images()
        # print(f"Total images stored: {len(stored_images)}")
        
        # Use query from arguments
        query = args.query
        print(f"\nTrying query: '{query}'")
        result = rag.query(query)
        
        print("\nResponse:", result["response"])
        if result["retrieved_images"]:
            print("\nRetrieved Images:")
            for img_path in result["retrieved_images"]:
                print(f"- {img_path}")
                
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
