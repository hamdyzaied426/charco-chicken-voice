# 🐔 مساعد شاركو تشيكن الصوتي - Syrian Voice Agent

نظام مساعد صوتي ذكي للمطاعم السورية باستخدام اللهجة السورية الدمشقية مع دعم المكالمات الهاتفية المباشرة.

## 🌟 المميزات الرئيسية

- **🎙️ مكالمات هاتفية مباشرة** عبر SIP/LiveKit
- **🗣️ لهجة سورية أصيلة** مع TTS طبيعي
- **🧠 ذكاء اصطناعي متقدم** لفهم الطلبات
- **📱 واجهة اختبار تفاعلية** مع Streamlit
- **📊 نظام إدارة الطلبات** الكامل
- **💰 حلول مجانية/منخفضة التكلفة**

## 🏗️ معمارية النظام

```
📞 مكالمة هاتفية → LiveKit SIP → Google STT (Syrian) → 
GPT-4-mini (Syrian Context) → ElevenLabs TTS → 📞 استجابة صوتية
```

### التقنيات المستخدمة:
- **LiveKit**: Real-time voice communication
- **Google Speech**: STT بدعم اللهجة السورية 
- **GPT-4-mini**: معالجة الطبيعية للغة العربية
- **ElevenLabs**: TTS بأصوات عربية طبيعية
- **FastAPI**: Backend API للطلبات
- **Streamlit**: واجهة الاختبار والمراقبة

## 🚀 التثبيت والإعداد

### 1. متطلبات النظام

```bash
# Python 3.9 أو أحدث
python --version

# Git لتحميل المشروع
git --version
```

### 2. استنساخ المشروع

```bash
git clone <repository-url>
cd syrian-voice-agent
```

### 3. إنشاء البيئة الافتراضية

```bash
# إنشاء البيئة
python -m venv venv

# تفعيل البيئة
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 4. تثبيت المتطلبات

```bash
pip install -r requirements.txt
```

### 5. إعداد المتغيرات البيئية

```bash
# نسخ ملف القالب
cp .env.template .env

# تحرير الملف وإضافة API Keys
nano .env
```

## 🔑 الحصول على API Keys (المجانية)

### LiveKit (Real-time Communication)
1. اذهب إلى [LiveKit Cloud](https://cloud.livekit.io)
2. أنشئ حساب مجاني
3. أنشئ مشروع جديد
4. احصل على: `API Key`, `API Secret`, `WebSocket URL`

### Google Cloud Speech (STT)
1. اذهب إلى [Google Cloud Console](https://console.cloud.google.com)
2. فعّل Speech-to-Text API
3. أنشئ Service Account وحمّل JSON credentials
4. **مجاني**: 60 دقيقة شهرياً

### OpenAI (GPT-4-mini)
1. اذهب إلى [OpenAI Platform](https://platform.openai.com)
2. احصل على API key
3. **اقتصادي**: GPT-4-mini أرخص 60x من GPT-4

### ElevenLabs (TTS)
1. اذهب إلى [ElevenLabs](https://elevenlabs.io)
2. أنشئ حساب مجاني
3. **مجاني**: 10,000 حرف شهرياً
4. يمكنك استنساخ صوت سوري أصيل

## 🎯 تشغيل النظام

### 1. تشغيل Backend API

```bash
# في terminal منفصل
cd backend
python api.py

# أو باستخدام uvicorn
uvicorn api:app --host 0.0.0.0 --port 8001 --reload
```

### 2. تشغيل Voice Agent

```bash
# في terminal منفصل
python agent.py

# أو باستخدام LiveKit CLI
livekit-agents dev agent.py
```

### 3. تشغيل واجهة الاختبار

```bash
# في terminal ثالث
streamlit run ui.py

# أو
python -m streamlit run ui.py
```

## 📞 إعداد الهاتف والSIP

### خيار 1: LiveKit Cloud (موصى به)
```bash
# اربط رقم هاتف مع LiveKit
# يدعم SIP trunking مباشرة
```

### خيار 2: Twilio Integration
```python
# إضافة Twilio للمكالمات الهاتفية
# في agent.py أضف:
from livekit.plugins import twilio

# كونفيجورة Twilio
twilio_config = {
    "account_sid": "your_twilio_sid",
    "auth_token": "your_twilio_token",
    "phone_number": "+1234567890"
}
```

### خيار 3: Asterisk (Self-hosted)
```bash
# تثبيت Asterisk
sudo apt-get install asterisk

# كونفيجورة SIP trunk
# ملفات الإعداد في /etc/asterisk/
```

## 🧪 طرق الاختبار

### 1. اختبار نصي (Streamlit)
- افتح http://localhost:8501
- اكتب رسائل باللهجة السورية
- راقب النوايا المكتشفة والاستجابات

### 2. اختبار صوتي (محلي)
```bash
# تشغيل وضع terminal للاختبار الصوتي
python agent.py --mode terminal
```

### 3. اختبار المكالمات الهاتفية
- اتصل على الرقم المكونف
- جرب سيناريوهات مختلفة
- راقب السجلات في الواجهة

## 📋 سيناريوهات الاختبار المقترحة

### سيناريو 1: طلب بسيط
```
الزبون: "أهلا، بدي طلب فروج مشوي"
المتوقع: ترحيب + تأكيد + اقتراح إضافات
```

### سيناريو 2: طلب معقد
```
الزبون: "بدي تنين فروج مشوي وسلطة فتوش وعصير ليمون"
المتوقع: تأكيد الكميات + حساب المجموع
```

### سيناريو 3: استفسار عن القائمة
```
الزبون: "شو عندكم أطباق؟"
المتوقع: عرض القائمة بالفئات والأسعار
```

### سيناريو 4: تعديل الطلب
```
الزبون: "لأ، ما بدي الفروج، بدي شاورما"
المتوقع: تعديل الطلب + تأكيد التغيير
```

## 🎨 تخصيص الصوت السوري

### استنساخ صوت سوري أصيل:
1. سجل عينات صوتية (10-30 دقيقة)
2. ارفعها على ElevenLabs
3. انشئ voice clone
4. استخدم الـ voice ID في إعدادات TTS

### تحسين اللهجة:
```python
# في agent.py
SYRIAN_PHRASES = {
    "greeting": ["أهلاً وسهلاً", "هلا وغلا", "يا هلا بيك"],
    "confirmation": ["تمام", "حاضر", "ماشي"],
    "questions": ["شو بدك؟", "كيف بتحب؟", "شو رايك؟"]
}
```

## 📊 مراقبة الأداء

### مؤشرات مهمة:
- **Response Time**: < 3 ثواني
- **Speech Recognition Accuracy**: > 85%
- **Intent Detection**: > 90%
- **Call Success Rate**: > 95%

### أدوات المراقبة:
```bash
# سجلات LiveKit
tail -f livekit.log

# سجلات Backend
tail -f backend.log

# مؤشرات في Streamlit
# تفحص تبويب "الإحصائيات"
```

## 🚨 استكشاف الأخطاء

### مشاكل شائعة:

#### 1. مشكلة اتصال LiveKit
```bash
# تأكد من صحة credentials
echo $LIVEKIT_API_KEY
echo $LIVEKIT_URL

# جرب ping
curl -I $LIVEKIT_URL
```

#### 2. مشكلة Google STT
```bash
# تأكد من credentials
gcloud auth application-default login
gcloud projects list
```

#### 3. مشكلة ElevenLabs TTS
```bash
# تفحص استهلاك الحصة
curl -H "xi-api-key: $ELEVENLABS_API_KEY" \
  https://api.elevenlabs.io/v1/user
```

#### 4. مشكلة Backend API
```bash
# تأكد من تشغيل API
curl http://localhost:8001/

# تفحص السجلات
tail -f backend.log
```

## 💡 تطوير ميزات إضافية

### إضافة أصناف جديدة:
```python
# في agent.py أضف على MENU_ITEMS
"منقوشة زعتر": {"name": "منقوشة زعتر", "price": 7, "category": "معجنات"}
```

### إضافة لغات أخرى:
```python
# دعم التركية والإنجليزية
SUPPORTED_LANGUAGES = {
    "ar-SY": "سوري",
    "tr-TR": "تركي", 
    "en-US": "إنجليزي"
}
```

### تكامل مع أنظمة الدفع:
```python
# إضافة Stripe أو PayPal
from stripe import checkout

@function_tool
async def process_payment(amount: float, currency: str = "USD"):
    # معالجة الدفع
    pass
```

## 📈 التوسع والنشر

### النشر على السحابة:
```bash
# باستخدام Docker
docker build -t syrian-voice-agent .
docker run -p 8001:8001 syrian-voice-agent

# باستخدام Railway/Heroku
git push railway main
```

### تحسين الأداء:
- استخدم Redis للـ caching
- أضف Load balancer
- فعل CDN للملفات الصوتية

## 🤝 المساهمة

نرحب بالمساهمات! خاصة:
- تحسين اللهجة السورية
- إضافة لهجات عربية أخرى
- تحسين دقة التعرف على الصوت
- إضافة ميزات جديدة

## 📞 الدعم والمساعدة

للمساعدة أو الأسئلة:
- **GitHub Issues**: للمشاكل التقنية
- **Discord/Slack**: للنقاش المباشر
- **Email**: للاستفسارات الخاصة

## 📄 الترخيص

MIT License - يمكن استخدامه تجارياً ومجاناً

---

**🎯 ملاحظة مهمة**: هذا النظام مصمم للاستخدام التجاري الفعلي. جميع التقنيات المستخدمة production-ready ومناسبة للمطاعم الحقيقية.

**💰 تكلفة تشغيل تقديرية**:
- LiveKit: مجاني للاستخدام المحدود
- Google STT: ~$1.44/ساعة
- GPT-4-mini: ~$0.15/1K requests  
- ElevenLabs: ~$22/مليون حرف

**📊 للمطعم متوسط الحجم**: ~$50-100/شهر
