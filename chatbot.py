from typing import List, Dict

from google import genai
import os
from dotenv import load_dotenv

from db_service import retrieve_context

# Load environment variables from .env at project root
load_dotenv()

# Load API key from environment (GEMINI_API_KEY in .env)
gemini_api_key = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(
    api_key=gemini_api_key
)


def build_prompt(query: str, context_chunks: List[Dict]) -> str:
    """
    Build a prompt for the LLM with context and query.

    Args:
        query: User's question
        context_chunks: Retrieved relevant chunks

    Returns:
        Formatted prompt string
    """
    # Build context from chunks
    context = "\n\n".join([
        f"[Source {i + 1} - Similarity: {chunk['similarity']:.2f}]\n{chunk['contextualized_text']}"
        for i, chunk in enumerate(context_chunks)
    ])

    # Create the prompt
    prompt = f"""You are a helpful assistant that answers questions based on the provided context.
            
            Context Information:
            {context}
            
            User Question: {query}
            
            Instructions:
            - Answer the question based ONLY on the information provided in the context above
            - If the context doesn't contain enough information to answer the question, say "I don't have enough information in the provided documents to answer this question."
            - If the question is about greeting or similar, respond politely without using the context
            - Keep your answer clear and concise
            - Do not make up information that is not in the context
            
            Answer:"""

    return prompt


def generate_response_stream(prompt: str) -> str:
    """
    Generate response using Gemini API with streaming.

    Args:
        prompt: Formatted prompt with context and question

    Returns:
        Complete generated response text
    """
    try:
        full_response = ""
        print("\nAssistant: ", end="", flush=True)

        # Stream the response
        for chunk in gemini_client.models.generate_content_stream(
                model="gemma-3-27b-it",
                contents=prompt
        ):
            if chunk.text:
                print(chunk.text, end="", flush=True)
                full_response += chunk.text

        print()  # New line after streaming completes
        return full_response

    except Exception as e:
        error_msg = f"Sorry, I encountered an error while generating a response: {e}"
        print(error_msg)
        return error_msg


def chat(user_query: str) -> Dict[str, any]:
    """
    Main chat method that handles the full RAG pipeline.

    Args:
        user_query: User's question

    Returns:
        Dictionary with answer and metadata
    """
    # Step 1: Retrieve relevant context
    print(f"Searching for relevant information...")
    context_chunks = retrieve_context(user_query)

    if not context_chunks:
        answer = "I couldn't find any relevant information in the database to answer your question."
        print(f"\nAssistant: {answer}\n")
        return {
            'answer': answer,
            'sources': [],
            'num_sources': 0
        }

    print(f"Found {len(context_chunks)} relevant chunks")

    # Step 2: Build prompt with context
    prompt = build_prompt(user_query, context_chunks)

    # Step 3: Generate response with streaming
    print("Generating response...")
    answer = generate_response_stream(prompt)

    # Step 4: Format response
    result = {
        'answer': answer,
        'sources': context_chunks,
        'num_sources': len(context_chunks)
    }

    return result


print("\n" + "=" * 60)
print("RAG Chatbot - Ask questions about your documents!")
print("Type 'quit' or 'exit' to end the conversation")
print("=" * 60 + "\n")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() in ['quit', 'exit', 'q']:
        print("Goodbye!")
        break

    if not user_input:
        continue

    # Get response (answer is now printed during streaming)
    result = chat(user_input)
    print()  # Extra line for spacing