import sys
import time
import random
import json
import argparse
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

class CaptchaBotSolver:
    def __init__(self, url, headless=False):
        self.url = url
        self.headless = headless
        self.movements = []
        self.start_time = None
        self.driver = None
        self.session_id = None

    def setup_driver(self):
        """Set up the Chrome WebDriver"""
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--window-size=1920,1080")
        
        # Initialize the WebDriver
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        self.driver.maximize_window()

    def get_timestamp(self):
        """Get current timestamp in seconds since script started"""
        return (time.time() - self.start_time)

    def record_movement(self, x, y):
        """Record a mouse movement"""
        self.movements.append({
            'x': x,
            'y': y,
            'timestamp': self.get_timestamp()
        })

    def generate_linear_path(self, start_x, start_y, end_x, end_y, steps):
        """Generate a linear path between two points"""
        path = []
        for i in range(steps + 1):
            ratio = i / steps
            x = start_x + (end_x - start_x) * ratio
            y = start_y + (end_y - start_y) * ratio
            path.append((x, y))
        return path

    def simulate_bot_movement(self, container):
        """Simulate bot-like mouse movements"""
        # Get container dimensions
        container_location = container.location
        container_size = container.size
        
        start_x = container_location['x']
        start_y = container_location['y']
        width = container_size['width']
        height = container_size['height']
        
        # Define waypoints for the bot path
        waypoints = [
            (start_x, start_y),  # Start at top-left
            (start_x + width, start_y),  # Top-right
            (start_x + width, start_y + height),  # Bottom-right
            (start_x, start_y + height),  # Bottom-left
            (start_x + width/2, start_y + height/2)  # Center (goal)
        ]
        
        # Generate points along straight lines between waypoints
        all_points = []
        points_per_segment = 20
        
        for i in range(len(waypoints) - 1):
            sx, sy = waypoints[i]
            ex, ey = waypoints[i + 1]
            
            path_segment = self.generate_linear_path(sx, sy, ex, ey, points_per_segment)
            all_points.extend(path_segment[1:])  # Avoid duplicating points
        
        # Execute movement with consistent timing (very bot-like)
        from selenium.webdriver.common.action_chains import ActionChains
        
        actions = ActionChains(self.driver)
        actions.move_to_element(container).perform()  # Move to the container first
        
        for x, y in all_points:
            actions.move_by_offset(x - actions.w3c_actions.pointer_action.source.pointer.x, 
                                   y - actions.w3c_actions.pointer_action.source.pointer.y).perform()
            self.record_movement(x, y)
            time.sleep(0.05)  # Constant time between movements
    
    def extract_session_id(self):
        """Extract the session ID from the page"""
        try:
            # This assumes the session ID is stored in a input field or data attribute
            # Adjust the selector based on your actual implementation
            session_element = self.driver.find_element(By.ID, "session-id")
            self.session_id = session_element.get_attribute("value")
            return True
        except Exception as e:
            print(f"Error extracting session ID: {e}")
            return False

    def check_captcha_result(self):
        """Check if the CAPTCHA was solved successfully"""
        try:
            # Wait for result to appear (adjust selector and timeout as needed)
            result_element = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".result-message"))
            )
            
            result_text = result_element.text
            return "success" in result_text.lower()
        except Exception as e:
            print(f"Error checking result: {e}")
            return False

    def submit_movements_directly(self):
        """Submit movements directly via JavaScript"""
        payload = {
            'session_id': self.session_id,
            'movements': self.movements,
            'completed': True,
            'success': True,
            'detailed': True
        }
        
        # Convert payload to JSON string
        payload_json = json.dumps(payload)
        
        # Execute JavaScript to send the request
        script = f"""
        fetch('/captcha/verify/', {{
            method: 'POST',
            headers: {{
                'Content-Type': 'application/json',
            }},
            body: '{payload_json}'
        }})
        .then(response => response.json())
        .then(data => {{
            console.log('CAPTCHA verification result:', data);
            // Create a result element to display the outcome
            const resultElement = document.createElement('div');
            resultElement.className = 'result-message';
            resultElement.textContent = data.is_human ? 'Success: Passed as human' : 'Failed: Detected as bot';
            document.body.appendChild(resultElement);
            return data;
        }});
        """
        
        self.driver.execute_script(script)
        
        # Wait for the result to appear
        time.sleep(3)

    def solve_captcha(self):
        """Main method to solve the CAPTCHA"""
        try:
            # Setup
            self.setup_driver()
            self.start_time = time.time()
            
            # Navigate to the CAPTCHA page
            print(f"Navigating to {self.url}")
            self.driver.get(self.url)
            
            # Wait for the game container to load
            game_container = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, "game-container"))
            )
            
            # Extract session ID
            if not self.extract_session_id():
                print("Failed to extract session ID")
                return False
            
            print(f"Session ID: {self.session_id}")
            
            # Simulate bot movement
            print("Simulating mouse movements...")
            self.simulate_bot_movement(game_container)
            
            # Submit movements directly via JavaScript
            print("Submitting movements...")
            self.submit_movements_directly()
            
            # Check result
            result = self.check_captcha_result()
            if result:
                print("Successfully passed the CAPTCHA test")
            else:
                print("Failed to pass the CAPTCHA test (detected as bot)")
            
            return result
            
        except Exception as e:
            print(f"Error solving CAPTCHA: {e}")
            return False
        finally:
            # Cleanup
            if self.driver:
                self.driver.quit()

def main():
    parser = argparse.ArgumentParser(description='Automated CAPTCHA solver script')
    parser.add_argument('url', help='URL of the CAPTCHA page')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode')
    args = parser.parse_args()
    
    solver = CaptchaBotSolver(args.url, args.headless)
    success = solver.solve_captcha()
    
    if success:
        print("SUCCESSFUL: The bot passed the CAPTCHA test")
        sys.exit(0)
    else:
        print("UNSUCCESSFUL: The bot failed the CAPTCHA test")
        sys.exit(1)

if __name__ == "__main__":
    main()