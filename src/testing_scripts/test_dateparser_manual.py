
from dateparser.search import search_dates
import dateparser

def test_manual():
    print(f"parse('last week'): {dateparser.parse('last week')}")
    print(f"search_dates('last week'): {search_dates('last week')}")
    print(f"search_dates('1 week ago'): {search_dates('1 week ago')}")

if __name__ == "__main__":
    test_manual()
