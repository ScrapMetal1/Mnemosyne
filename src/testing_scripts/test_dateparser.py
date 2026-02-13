
from dateparser.search import search_dates

def test_dates():
    queries = [
        "What did I see yesterday?",
        "Show me what I was looking at 2 hours ago",
        "What was on my desk last week?",
        "What was on my desk in the previous week?",
        "What happened on Monday?"
    ]
    
    settings = {'PREFER_DATES_FROM': 'past'}
    
    for q in queries:
        results = search_dates(q, settings=settings)
        print(f"Query: '{q}' -> {results}")

if __name__ == "__main__":
    test_dates()
