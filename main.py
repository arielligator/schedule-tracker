# main.py
import sys
import threading
import time
import webbrowser

from streamlit.web import cli as stcli

URL = "http://localhost:8501"

def open_browser():
    # Give Streamlit a few seconds to start
    time.sleep(3)
    print(f"👉 [DEBUG] opening browser at {URL}")
    webbrowser.open(URL, new=2)

if __name__ == "__main__":
    print("👉 [DEBUG] wrapper starting…")

    # 1️⃣ Kick off only the browser-opener in a background thread
    threading.Thread(target=open_browser, daemon=True).start()

    # 2️⃣ Now invoke Streamlit **in the main thread**
    sys.argv = [
        "streamlit", "run", "schedule_tracker_gui.py",
        "--server.headless=false"
    ]
    try:
        sys.exit(stcli.main())
    except KeyboardInterrupt:
        # This will let you Ctrl+C to shut down cleanly
        print("👉 Received Ctrl+C — shutting down.")
        sys.exit(0)
