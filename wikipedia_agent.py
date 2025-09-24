#!/usr/bin/env python3
"""
Wikipedia Agent - A simple command line agent to answer natural language questions
using Wikipedia as knowledge source and Google Gemini for natural language processing.
"""

import os
import sys
import argparse
import wikipedia
import google.generativeai as genai
from dotenv import load_dotenv
from typing import List, Optional, Tuple


class WikipediaAgent:
    """A simple agent that answers questions using Wikipedia and Gemini LLM."""
    
    def __init__(self, api_key: str, max_iterations: int = 3):
        """
        Initialize the Wikipedia agent.
        
        Args:
            api_key: Google Gemini API key
            max_iterations: Maximum number of search iterations
        """
        self.max_iterations = max_iterations
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Wikipedia configuration
        wikipedia.set_lang("en")
    
    def generate_search_terms(self, question: str, previous_attempts: List[str] = None) -> List[str]:
        """
        Generate Wikipedia search terms for a given question using Gemini.
        
        Args:
            question: The user's question
            previous_attempts: List of previously tried search terms
            
        Returns:
            List of search terms to try
        """
        previous_info = ""
        if previous_attempts:
            previous_info = f"\nPrevious search terms that didn't work: {', '.join(previous_attempts)}"
        
        prompt = f"""
You are an expert at finding information on Wikipedia. Given a question, suggest 3-5 specific Wikipedia search terms that are most likely to contain the answer.

Question: {question}{previous_info}

Instructions:
- Focus on key entities, concepts, and topics mentioned in the question
- Use specific names, places, events, or technical terms when possible
- Avoid overly broad or generic terms
- If previous terms failed, try different angles or more specific/general variations
- Return only the search terms, one per line, without numbering or bullets

Search terms:"""

        try:
            response = self.model.generate_content(prompt)
            search_terms = [term.strip() for term in response.text.strip().split('\n') if term.strip()]
            return search_terms[:5]  # Limit to 5 terms
        except Exception as e:
            print(f"Error generating search terms: {e}")
            # Fallback: extract key words from the question
            return [question]
    
    def search_wikipedia(self, search_term: str) -> Optional[str]:
        """
        Search Wikipedia for a given term and return the content.
        
        Args:
            search_term: Term to search for
            
        Returns:
            Wikipedia page content or None if not found
        """
        try:
            # Search for pages
            search_results = wikipedia.search(search_term, results=3)
            if not search_results:
                return None
            
            # Try to get the first result
            page = wikipedia.page(search_results[0])
            
            # Return summary + first few paragraphs (limit content size)
            content = page.summary
            if len(page.content) > len(page.summary):
                # Add more content but limit total size
                additional_content = page.content[len(page.summary):2000]
                content = page.summary + "\n\n" + additional_content
            
            return content
            
        except wikipedia.exceptions.DisambiguationError as e:
            # Try the first disambiguation option
            try:
                page = wikipedia.page(e.options[0])
                return page.summary
            except:
                return None
        except wikipedia.exceptions.PageError:
            return None
        except Exception as e:
            print(f"Error searching Wikipedia for '{search_term}': {e}")
            return None
    
    def answer_question(self, question: str, context: str) -> str:
        """
        Generate an answer to the question using the Wikipedia context.
        
        Args:
            question: The user's question
            context: Wikipedia content as context
            
        Returns:
            Generated answer
        """
        prompt = f"""
Based on the following Wikipedia content, answer the user's question accurately and concisely.

Wikipedia Content:
{context}

Question: {question}

Instructions:
- Answer directly and factually based only on the provided Wikipedia content
- If the content doesn't contain enough information to answer the question, say so
- Keep the answer concise but complete
- Cite specific facts from the Wikipedia content when relevant

Answer:"""

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error generating answer: {e}"
    
    def process_query(self, question: str) -> Tuple[str, List[str]]:
        """
        Process a user query through iterative Wikipedia search.
        
        Args:
            question: The user's question
            
        Returns:
            Tuple of (answer, search_terms_used)
        """
        print(f"🤔 Processing question: {question}")
        
        previous_attempts = []
        search_terms_used = []
        
        for iteration in range(self.max_iterations):
            print(f"\n📍 Iteration {iteration + 1}/{self.max_iterations}")
            
            # Generate search terms
            search_terms = self.generate_search_terms(question, previous_attempts)
            print(f"🔍 Generated search terms: {', '.join(search_terms)}")
            
            # Try each search term
            for term in search_terms:
                if term in previous_attempts:
                    continue
                    
                print(f"   Searching Wikipedia for: {term}")
                content = self.search_wikipedia(term)
                
                if content:
                    print(f"   ✅ Found content ({len(content)} characters)")
                    search_terms_used.append(term)
                    
                    # Generate answer
                    answer = self.answer_question(question, content)
                    
                    # Simple check if answer seems complete
                    if len(answer) > 50 and "don't have enough information" not in answer.lower():
                        return answer, search_terms_used
                else:
                    print(f"   ❌ No content found")
                
                previous_attempts.append(term)
            
            # If we didn't find a good answer, continue to next iteration
            print(f"   No satisfactory answer found in iteration {iteration + 1}")
        
        # If we've exhausted all iterations
        return "I couldn't find a satisfactory answer to your question after searching Wikipedia with multiple terms. Please try rephrasing your question or asking about a different topic.", search_terms_used


def main():
    """Main entry point for the Wikipedia agent."""
    parser = argparse.ArgumentParser(
        description="Wikipedia Agent - Answer questions using Wikipedia and AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python wikipedia_agent.py "Who was the first person to walk on the moon?"
  python wikipedia_agent.py "What is the capital of France?"
  python wikipedia_agent.py --max-iterations 5 "How does photosynthesis work?"
        """
    )
    
    parser.add_argument(
        "question",
        help="The question to answer"
    )
    
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum number of search iterations (default: 3)"
    )
    
    parser.add_argument(
        "--api-key",
        help="Google Gemini API key (can also be set via GEMINI_API_KEY environment variable)"
    )
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = args.api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: Google Gemini API key is required.")
        print("   Set it via --api-key argument or GEMINI_API_KEY environment variable.")
        print("   You can get an API key from: https://makersuite.google.com/app/apikey")
        sys.exit(1)
    
    try:
        # Initialize agent
        agent = WikipediaAgent(api_key, args.max_iterations)
        
        # Process query
        answer, search_terms = agent.process_query(args.question)
        
        # Display results
        print("\n" + "="*60)
        print("📝 ANSWER:")
        print("="*60)
        print(answer)
        
        if search_terms:
            print(f"\n🔍 Search terms used: {', '.join(search_terms)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()