
from datetime import datetime
import dateparser

def test_datetime_comp():
    # Simulate storage timestamp (naive)
    now = datetime.now()
    iso_str = now.isoformat()
    parsed_storage_time = datetime.fromisoformat(iso_str)
    
    print(f"Storage Time (naive): {parsed_storage_time} (tzinfo: {parsed_storage_time.tzinfo})")

    # Simulate dateparser (naive?)
    query = "yesterday"
    from dateparser.search import search_dates
    results = search_dates(query)
    if results:
        _, parsed_start_time = results[0]
        print(f"Filter Time (naive?): {parsed_start_time} (tzinfo: {parsed_start_time.tzinfo})")
        
        try:
            comparison = parsed_storage_time > parsed_start_time
            print(f"Comparison result: {comparison}")
            print("✅ Datetime comparison successful")
        except TypeError as e:
            print(f"❌ Datetime comparison FAILED: {e}")
    else:
        print("Could not parse date")

if __name__ == "__main__":
    test_datetime_comp()
