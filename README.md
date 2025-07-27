# 🐔 Sharko Chicken Voice Assistant - Syrian Voice Agent

An intelligent voice assistant system for Syrian restaurants using Syrian Damascene dialect with direct phone call support.

## 🌟 Key Features

- **🎙️ Direct phone calls** via SIP/LiveKit
- **🗣️ Authentic Syrian dialect** with natural TTS
- **🧠 Advanced AI** for understanding orders
- **📱 Interactive testing interface** with Streamlit
- **📊 Complete order management system**
- **💰 Free/low-cost solutions**

## 🏗️ System Architecture

```
📞 Phone call → LiveKit SIP → Google STT (Syrian) → 
GPT-4-mini (Syrian Context) → ElevenLabs TTS → 📞 Voice response
```

### Technologies Used:
- **LiveKit**: Real-time voice communication
- **Google Speech**: STT with Syrian dialect support
- **GPT-4-mini**: Arabic natural language processing
- **ElevenLabs**: TTS with natural Arabic voices
- **FastAPI**: Backend API for orders
- **Streamlit**: Testing and monitoring interface

## 🚀 Installation and Setup

### 1. System Requirements

```bash
# Python 3.9 or newer
python --version

# Git to download the project
git --version
```

### 2. Clone the Project

```bash
git clone <repository-url>
cd syrian-voice-agent
```

### 3. Create Virtual Environment

```bash
# Create environment
python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 4. Install Requirements

```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables

```bash
# Copy template file
cp .env.template .env

# Edit file and add API Keys
nano .env
```

## 🔑 Getting API Keys (Free)

### LiveKit (Real-time Communication)
1. Go to [LiveKit Cloud](https://cloud.livekit.io)
2. Create a free account
3. Create a new project
4. Get: `API Key`, `API Secret`, `WebSocket URL`

### Google Cloud Speech (STT)
1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Enable Speech-to-Text API
3. Create Service Account and download JSON credentials
4. **Free**: 60 minutes monthly

### OpenAI (GPT-4-mini)
1. Go to [OpenAI Platform](https://platform.openai.com)
2. Get API key
3. **Economical**: GPT-4-mini is 60x cheaper than GPT-4

### ElevenLabs (TTS)
1. Go to [ElevenLabs](https://elevenlabs.io)
2. Create a free account
3. **Free**: 10,000 characters monthly
4. You can clone an authentic Syrian voice

## 🎯 Running the System

### 1. Start Backend API

```bash
# In separate terminal
cd backend
python api.py

# Or using uvicorn
uvicorn backend_api:app --reload
or
uvicorn backend_api:app --reload --host 0.0.0.0 --port 8000
```

### 2. Start Voice Agent

```bash
# In separate terminal
python agent.py

# Or using LiveKit CLI
livekit-agents dev agent.py
python charco-voice-agent/agent.py dev
```

### 3. Start Testing Interface

```bash
# In third terminal
streamlit run ui.py

# Or
python -m streamlit run ui.py
```

## 📞 Phone and SIP Setup

### Option 1: LiveKit Cloud (Recommended)
```bash
# Connect phone number with LiveKit
# Supports direct SIP trunking
```

### Option 2: Twilio Integration
```python
# Add Twilio for phone calls
# In agent.py add:
from livekit.plugins import twilio

# Configure Twilio
twilio_config = {
    "account_sid": "your_twilio_sid",
    "auth_token": "your_twilio_token",
    "phone_number": "+1234567890"
}
```

### Option 3: Asterisk (Self-hosted)
```bash
# Install Asterisk
sudo apt-get install asterisk

# Configure SIP trunk
# Configuration files in /etc/asterisk/
```

## 🧪 Testing Methods

### 1. Text Testing (Streamlit)
- Open http://localhost:8501
- Type messages in Syrian dialect
- Monitor detected intents and responses

### 2. Voice Testing (Local)
```bash
# Run terminal mode for voice testing
python agent.py --mode terminal
```

### 3. Phone Call Testing
- Call the configured number
- Try different scenarios
- Monitor logs in the interface

## 📋 Suggested Testing Scenarios

### Scenario 1: Simple Order
```
Customer: "Hello, I want grilled chicken"
Expected: Welcome + confirmation + suggest additions
```

### Scenario 2: Complex Order
```
Customer: "I want two grilled chickens, fattoush salad, and lemon juice"
Expected: Confirm quantities + calculate total
```

### Scenario 3: Menu Inquiry
```
Customer: "What dishes do you have?"
Expected: Display menu with categories and prices
```

### Scenario 4: Order Modification
```
Customer: "No, I don't want chicken, I want shawarma"
Expected: Modify order + confirm change
```

## 🎨 Syrian Voice Customization

### Clone Authentic Syrian Voice:
1. Record voice samples (10-30 minutes)
2. Upload to ElevenLabs
3. Create voice clone
4. Use voice ID in TTS settings

### Improve Dialect:
```python
# In agent.py
SYRIAN_PHRASES = {
    "greeting": ["أهلاً وسهلاً", "هلا وغلا", "يا هلا بيك"],
    "confirmation": ["تمام", "حاضر", "ماشي"],
    "questions": ["شو بدك؟", "كيف بتحب؟", "شو رايك؟"]
}
```

## 📊 Performance Monitoring

### Important Metrics:
- **Response Time**: < 3 seconds
- **Speech Recognition Accuracy**: > 85%
- **Intent Detection**: > 90%
- **Call Success Rate**: > 95%

### Monitoring Tools:
```bash
# LiveKit logs
tail -f livekit.log

# Backend logs
tail -f backend.log

# Metrics in Streamlit
# Check "Statistics" tab
```

## 🚨 Troubleshooting

### Common Issues:

#### 1. LiveKit Connection Issue
```bash
# Verify credentials
echo $LIVEKIT_API_KEY
echo $LIVEKIT_URL

# Try ping
curl -I $LIVEKIT_URL
```

#### 2. Google STT Issue
```bash
# Verify credentials
gcloud auth application-default login
gcloud projects list
```

#### 3. ElevenLabs TTS Issue
```bash
# Check quota usage
curl -H "xi-api-key: $ELEVENLABS_API_KEY" \
  https://api.elevenlabs.io/v1/user
```

#### 4. Backend API Issue
```bash
# Verify API is running
curl http://localhost:8001/

# Check logs
tail -f backend.log
```

## 💡 Developing Additional Features

### Add New Menu Items:
```python
# In agent.py add to MENU_ITEMS
"Za'atar Manakish": {"name": "Za'atar Manakish", "price": 7, "category": "Pastries"}
```

### Add Other Languages:
```python
# Support Turkish and English
SUPPORTED_LANGUAGES = {
    "ar-SY": "Syrian",
    "tr-TR": "Turkish", 
    "en-US": "English"
}
```

### Payment System Integration:
```python
# Add Stripe or PayPal
from stripe import checkout

@function_tool
async def process_payment(amount: float, currency: str = "USD"):
    # Process payment
    pass
```

## 📈 Scaling and Deployment

### Cloud Deployment:
```bash
# Using Docker
docker build -t syrian-voice-agent .
docker run -p 8001:8001 syrian-voice-agent

# Using Railway/Heroku
git push railway main
```

### Performance Optimization:
- Use Redis for caching
- Add Load balancer
- Enable CDN for audio files

## 🤝 Contributing

We welcome contributions! Especially:
- Improving Syrian dialect
- Adding other Arabic dialects
- Improving speech recognition accuracy
- Adding new features

## 📞 Support and Help

For help or questions:
- **GitHub Issues**: For technical problems
- **Discord/Slack**: For direct discussion
- **Email**: For private inquiries

## 📄 License

MIT License - Can be used commercially and freely

---

**🎯 Important Note**: This system is designed for real commercial use. All technologies used are production-ready and suitable for real restaurants.

**💰 Estimated Operating Costs**:
- LiveKit: Free for limited use
- Google STT: ~$1.44/hour
- GPT-4-mini: ~$0.15/1K requests  
- ElevenLabs: ~$22/million characters

**📊 For medium-sized restaurant**: ~$50-100/month
