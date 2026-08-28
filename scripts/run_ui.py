import os
import sys

def main():
    print("Starting Ashva UI...")
    dashboard_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src", "ui", "dashboard.py")
    os.system(f"python -m streamlit run {dashboard_path}")

if __name__ == "__main__":
    main()
