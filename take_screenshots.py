"""
Screenshot Helper for Advanced Stock Predictor AI
This script helps you take screenshots of the application for documentation.
"""

import time
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def setup_driver():
    """Setup Chrome driver for screenshots"""
    chrome_options = Options()
    chrome_options.add_argument('--headless')  # Run in background
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--window-size=1920,1080')
    
    try:
        driver = webdriver.Chrome(options=chrome_options)
        return driver
    except Exception as e:
        print(f"Error setting up Chrome driver: {e}")
        print("Please install ChromeDriver or use manual screenshots")
        return None

def take_screenshots():
    """Take screenshots of the application"""
    driver = setup_driver()
    if not driver:
        return
    
    base_url = "http://localhost:8504"
    screenshot_dir = "screenshots"
    
    # Ensure screenshot directory exists
    os.makedirs(screenshot_dir, exist_ok=True)
    
    try:
        # Main dashboard
        print("Taking main dashboard screenshot...")
        driver.get(base_url)
        time.sleep(5)  # Wait for page to load
        driver.save_screenshot(f"{screenshot_dir}/main-dashboard.png")
        
        # Navigate through different sections
        # Note: This would need to be customized based on your actual UI elements
        print("Screenshots saved to screenshots/ directory")
        
    except Exception as e:
        print(f"Error taking screenshots: {e}")
    finally:
        driver.quit()

def manual_screenshot_guide():
    """Print guide for manual screenshots"""
    print("""
    📸 MANUAL SCREENSHOT GUIDE
    =========================
    
    Since automated screenshots require additional setup, here's how to take them manually:
    
    1. 🚀 Start your Streamlit app: `streamlit run main.py`
    2. 🌐 Open http://localhost:8504 in your browser
    3. 📱 Set browser to full screen (F11)
    4. 📸 Take screenshots of these sections:
    
    📊 MAIN DASHBOARD:
    - Full page view with metrics and stock selection
    - Save as: screenshots/main-dashboard.png
    
    📈 TECHNICAL ANALYSIS:
    - Candlestick chart with moving averages
    - RSI chart with signals
    - Save as: screenshots/technical-analysis.png
    
    🤖 AI PREDICTIONS:
    - Model comparison charts
    - Future predictions plot
    - Save as: screenshots/ai-predictions.png
    
    📊 INDICATORS:
    - RSI analysis panel
    - MACD charts
    - Save as: screenshots/indicators.png
    
    💡 Tips:
    - Use high resolution (1920x1080 or higher)
    - Ensure dark theme is active for consistency
    - Include sidebar and main content in shots
    - Take screenshots with actual stock data loaded
    """)

if __name__ == "__main__":
    print("📸 Screenshot Helper for Advanced Stock Predictor AI")
    print("=" * 50)
    
    # Try automated screenshots first
    print("Attempting automated screenshots...")
    try:
        take_screenshots()
    except ImportError:
        print("Selenium not installed. Showing manual guide...")
        manual_screenshot_guide()
    except Exception as e:
        print(f"Automated screenshots failed: {e}")
        print("Showing manual guide...")
        manual_screenshot_guide()
