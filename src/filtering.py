import os
import google.generativeai as genai
from dotenv import load_dotenv
from dateparser.search import search_dates

load_dotenv()

def requires_memory_recall(prompt: str) -> bool:
    """
    Determines if a given prompt requires memory recall using the Gemini API.
    
    Args:
        prompt (str): The user's input prompt.
        
    Returns:
        bool: True if memory recall is required, False otherwise.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found in environment variables.")
        return False

    try:
        genai.configure(api_key=api_key)
        
        # Using gemini-3-flash-preview for maximum speed and up-to-date performance
        model = genai.GenerativeModel(
            model_name='gemini-3-flash-preview',
            system_instruction=(
                "You are a binary classifier. Your task is to determine if a user prompt "
                "requires retrieving personal memories, past interactions, or specific user-related context "
                "stored in a memory database. General knowledge questions or creative writing tasks do NOT require memory recall. "
                "Reply ONLY with 'YES' or 'NO'."
            )
        )
        
        response = model.generate_content(f"User Prompt: '{prompt}'")
        text = response.text.strip().upper()
        
        return "YES" in text # if 'Yes' is in text then it returns TRUE
        
    except Exception as e:
        print(f"Error calling Gemini API in filtering.py: {e}")
        return False


def extract_time_filter(prompt: str):
    """
    Extracts datetime references from a prompt using dateparser. This will ONLY be called if memory recall is flagged as TRUE. 
    
    Args:
        prompt (str): The user's input prompt.
        
    Returns:
        datetime or None: The parsed datetime if found, None otherwise.
    """
    settings = {
        'PREFER_DATES_FROM': 'past',  # Memory queries usually refer to the past
    }
    # search_dates scans the entire prompt for any date/time expressions
    results = search_dates(prompt, settings=settings)
    
    if results:
        # Returns the first match: (matched_text, datetime_object)
        matched_text, parsed_datetime = results[0]
        return parsed_datetime
    
    return None


if __name__ == "__main__":
    # Test queries
    test_queries = [
        # Should need memory recall + have time filter
        "What did I see yesterday?",
        "Show me what I was looking at 2 hours ago",
        "What was on my desk last week?",
        
        # Should need memory recall + NO time filter
        "What did I have for breakfast?",
        "Where did I put my keys?",
        "What was that book I was reading?",
        
        # Should NOT need memory recall (general knowledge)
        "What is the capital of France?",
        "How do I make pasta?",
        "What's 2 + 2?",
        "What is the weather like on next Monday?",
    ]
    
    print("=" * 70)
    print("FILTERING TEST")
    print("=" * 70)
    
    for query in test_queries:
        print(f"\nQuery: \"{query}\"")
        print("-" * 50)
        
        # Test memory recall filter
        needs_memory = requires_memory_recall(query)
        print(f"  Needs Memory Recall: {needs_memory}")
        
        # Only check time filter if memory recall is needed
        if needs_memory:
            time_filter = extract_time_filter(query)
            if time_filter:
                print(f"  Time Filter: {time_filter}")
            else:
                print(f"  Time Filter: None (no time reference)")
        else:
            print(f"  Time Filter: SKIPPED (no memory recall needed)")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
