#!/usr/bin/env python3
"""
Demo script to show how the Wikipedia Agent would work.
This is for demonstration purposes when network access is limited.
"""

def demo_wikipedia_agent():
    """Demo showing the expected workflow of the Wikipedia Agent."""
    
    print("🔍 Wikipedia Agent Demo")
    print("=" * 50)
    
    # Simulated user question
    question = "Who was the first person to walk on the moon?"
    print(f"🤔 User Question: {question}")
    
    print(f"\n📍 Iteration 1/3")
    print("🔍 Generated search terms: Neil Armstrong, Apollo 11, Moon landing, First moonwalk, Lunar surface")
    print("   Searching Wikipedia for: Neil Armstrong")
    print("   ✅ Found content (2847 characters)")
    
    # Simulated answer
    answer = """Neil Armstrong was the first person to walk on the moon. He was an American astronaut and aeronautical engineer who became the first person to step onto the lunar surface on July 20, 1969, during the Apollo 11 mission. Armstrong famously said "That's one small step for man, one giant leap for mankind" as he stepped onto the Moon's surface. He was the mission commander for Apollo 11, which was launched from Kennedy Space Center on July 16, 1969, along with crew members Buzz Aldrin and Michael Collins."""
    
    print("\n" + "="*60)
    print("📝 ANSWER:")
    print("="*60)
    print(answer)
    print(f"\n🔍 Search terms used: Neil Armstrong")
    
    print("\n" + "="*60)
    print("✅ Demo completed successfully!")
    print("The actual agent would:")
    print("1. Use Google Gemini 2.0 Flash to generate search terms")
    print("2. Search Wikipedia for relevant content")
    print("3. Synthesize answers using the LLM")
    print("4. Iterate if needed for better results")

if __name__ == "__main__":
    demo_wikipedia_agent()