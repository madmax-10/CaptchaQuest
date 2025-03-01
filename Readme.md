# CaptchaQuest - A Web Security Learning Platform  
_Technical Documentation and Instruction Manual_  
## Overview  
CaptchaQuest is a cutting-edge, game-based CAPTCHA system designed to secure digital interactions against sophisticated bots. Unlike traditional text/audio CAPTCHAs, we leverage motion analysis, AI-driven pattern recognition, and real-time user behavior metrics to distinguish humans from bots. This project demonstrates:  
- **Dynamic Obstacle Course**: Drag a ball through a maze while AI assesses your movements.  
- **DeepSeek LLM Integration**: Analyzes cursor patterns to flag bot-like behavior.  
**Key Features**:  
- AI-powered CAPTCHA Verification: Combines motion tracking, pattern analysis, and bot detection using DeepSeek’s LLM.  
- Interactive Challenges: Solve CAPTCHA puzzles while avoiding obstacles to mimic real bot-detection systems.  
- Security Vulnerabilities: Practices common flaws like improper input validation, timing attacks, and weak authentication.  

---

## Team Members  
- Yugesh Bhattari  
- Ritiz Adhikari  

---

## Security Concepts Covered  
1. AI-powered CAPTCHA Bypass: Exploiting model biases and pattern recognition.  
2. Session Management Weaknesses: Cookie tampering, insecure storage.  

---

## Tools & Resources Used  
- **Backend**: Django (Python), SQLite, OpenRouter API (DeepSeek)  
- **Frontend**: HTML5 Canvas, CSS/JS, WebSocket for real-time updates.  
- **AI Tools**: DeepSeek LLMs for movement analysis.  
- **Deployment**: GitHub  

---

## Build & Run Instructions  
### Requirements  
- Python 3.8+  
- Git  
- SQLite (pre-installed in Python)  
### Setup  
1. **Clone the Repository**:  
   ```bash  
   git clone https://github.com/yourteam/captchaquest.git  
   cd captcha_project
   ```  
2. **Install Dependencies**:  
   ```bash  
   # Create a virtual environment  
   python -m venv venv  
   source venv/bin/activate  # On Windows: venv\Scripts\activate  

   # Install Python packages  
   pip install -r requirements.txt  
   ```  
3. **Database Initialization**:  
   ```bash  
   python manage.py migrate  
   python manage.py createsuperuser  # For admin access (optional)  
   ```  
4. **Configure API Keys**:  
   - Create an `api.txt` file (or use environment variables) for DeepSeek’s API key:  
     ```  
     OPENROUTER_API_KEY=your_openrouter_api_key  
     ```  
5. **Run the Server**:  
   ```bash  
   python manage.py runserver  
   ```  
   Visit **`http://localhost:8000/captcha`** to start the first challenge!  

---

## Learning Resources  
1. Understanding CAPTCHA Systems: Learn how traditional math/text-based CAPTCHAs are defeated.  
2. Motion Detection Tutorials: Study how to track canvas movements and analyze patterns.  
3. DeepLearning.AI Courses: Explore GPT models and ethical AI use in security.  

---

## Demo  
Watch the [Demonstration Video](https://youtu.be/SotYcXCqOPw) to see how to bypass the CAPTCHA and exploit vulnerabilities.  

---

## API  
- **External APIs**: OpenRouter (DeepSeek) for LLM-backed security analysis.  
API KEY DOCS: https://docs.google.com/document/d/1XrMZHxlOtLMjbaD2j5DCqnFnG_5DT4GzZkU6HER0xYA/edit?tab=t.0 

---

## Future Improvements  
In the next phase, we’ll evolve CaptchaQuest into a self-learning security platform. By harvesting real-time user interaction data (e.g., cursor speed, path efficiency), we’ll train a custom ML model to detect anomalies with pinpoint accuracy. This model will:  
- Adapt to Bot Tactics: Recognize and counter new evasion strategies.  
- Continuous Learning: Improve detection rates as more users interact with the system.  
- User-Centric Design: Maintain low friction for genuine users while blocking bots.  
We’re thrilled to pioneer a CAPTCHA system that evolves alongside threats, making online security both smarter and seamless.  

---

## Security Notice  
- Intentional Vulnerabilities: Do not use this app in production!  
- No Sensitive Data: Never input real credentials into the application.  
- Ethical Use Only: Respect all laws and target only permissioned systems.  

---

## Contact  
- GitHub: @madmax-10  
- Email: ritiztech@gmail.com yugeshbhattarai18@gmail.com  

---  
_All done!_