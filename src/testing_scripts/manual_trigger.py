import requests
import sys

def manual_trigger():
    url = "http://127.0.0.1:5000/api/memory/capture"
    print(f"📸 Triggering Memory Capture at {url}...")
    
    try:
        response = requests.post(url, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                print("\n✅ MEMORY CAPTURED!")
                print(f"📝 Description: {data.get('description')}")
                print(f"⚡ VLM Time:    {data.get('FastVLM describe time'):.2f}ms")
            else:
                print(f"\n❌ Error: {data.get('message')}")
        else:
            print(f"\n❌ Server Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"\n💥 Connection Failed: {e}")
        print("Is service.py running?")

if __name__ == "__main__":
    manual_trigger()
