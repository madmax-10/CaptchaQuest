import requests
from bs4 import BeautifulSoup
from PIL import Image
import pytesseract
from io import BytesIO
from faker import Faker

fake = Faker()

# Configuration
BASE_URL = "http://127.0.0.1:5000"
NUM_ACCOUNTS = 20  # Set to 10-20
TESSERACT_CMD = r"/opt/homebrew/Cellar/tesseract/5.5.0"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

def extract_captcha_text(captcha_url, session):
    response = session.get(captcha_url)
    image = Image.open(BytesIO(response.content))
    image = image.convert('L')
    image = image.point(lambda x: 0 if x < 128 else 255, '1')
    text = pytesseract.image_to_string(image, config='--psm 6')
    return text.strip().upper()

def scrape_registration_page(session):
    response = session.get(f"{BASE_URL}/register")
    soup = BeautifulSoup(response.text, 'html.parser')
    form = soup.find('form')
    fields = {}
    for input_tag in form.find_all('input'):
        name = input_tag.get('name')
        if name:
            fields[name] = input_tag.get('value', '')
    return fields

def register_account(session, fields):
    captcha_url = f"{BASE_URL}/captcha/generate"
    captcha_text = extract_captcha_text(captcha_url, session)
    
    data = fields.copy()
    data['username'] = fake.user_name()
    data['password'] = fake.password(length=10)
    data['captcha'] = captcha_text
    
    print(f"Username: {data['username']}, Password: {data['password']}")
    
    response = session.post(f"{BASE_URL}/register", data=data)
    return "Registration successful" in response.text

def main():
    with requests.Session() as session:
        fields = scrape_registration_page(session)
        for _ in range(NUM_ACCOUNTS):
            success = register_account(session, fields)
            if success:
                print(f"Account {_ + 1} created successfully!")
            else:
                print(f"Account {_ + 1} failed.")

if __name__ == "__main__":
    main()