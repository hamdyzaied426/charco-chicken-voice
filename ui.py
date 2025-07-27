"""
Streamlit Testing Interface for Syrian Voice Agent
Allows testing voice interactions and monitoring conversations
Enhanced with Speech-to-Text recording functionality (Google STT Only)
"""

import streamlit as st
import asyncio
import json
import requests
import io
import tempfile
import os
from typing import Optional, Dict, Any
import base64
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from dotenv import load_dotenv
import speech_recognition as sr
from audio_recorder_streamlit import audio_recorder
import numpy as np
import wave
import openai
import requests

load_dotenv()

# Initialize OpenAI client
if os.getenv("OPENAI_API_KEY"):
    openai.api_key = os.getenv("OPENAI_API_KEY")

# ElevenLabs settings
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "jBpfuIE2acCO8z3wKNL1")

if "live_logs" not in st.session_state:
    st.session_state.live_logs = []

# Initialize STT session state
if "recorded_audio" not in st.session_state:
    st.session_state.recorded_audio = None
if "transcribed_text" not in st.session_state:
    st.session_state.transcribed_text = ""

# Streamlit receives logs here from agent.py
def handle_logs():
    from fastapi import FastAPI, Request
    from threading import Thread
    import uvicorn

    app = FastAPI()

    @app.post("/log")
    async def receive_log(req: Request):
        data = await req.json()
        if "role" in data and "content" in data:
            st.session_state.live_logs.append(data)
        return {"success": True}

    def run_server():
        uvicorn.run(app, host="0.0.0.0", port=8501)

    t = Thread(target=run_server, daemon=True)
    t.start()

handle_logs()

def generate_elevenlabs_speech(text: str) -> bytes:
    """Generate speech using ElevenLabs API"""
    try:
        if not ELEVENLABS_API_KEY:
            st.error("مفتاح ElevenLabs API غير متوفر")
            return None
        
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": ELEVENLABS_API_KEY
        }
        
        data = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5,
                "style": 0.0,
                "use_speaker_boost": True
            }
        }
        
        response = requests.post(url, json=data, headers=headers)
        
        if response.status_code == 200:
            return response.content
        else:
            st.error(f"خطأ ElevenLabs: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        st.error(f"خطأ في توليد الصوت: {str(e)}")
        return None

def play_elevenlabs_audio(text: str, message_id: str):
    """Play audio using ElevenLabs TTS"""
    try:
        with st.spinner("🔄 جاري توليد الصوت..."):
            audio_bytes = generate_elevenlabs_speech(text)
            
        if audio_bytes:
            st.success("✅ تم توليد الصوت!")
            st.audio(audio_bytes, format="audio/mpeg")
        else:
            st.error("❌ فشل في توليد الصوت")
            
    except Exception as e:
        st.error(f"خطأ في تشغيل الصوت: {str(e)}")

def transcribe_audio_google(audio_bytes: bytes) -> str:
    """Transcribe audio using Google Speech Recognition"""
    tmp_file_path = None
    try:
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 300
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 0.8
        recognizer.operation_timeout = None
        recognizer.phrase_threshold = 0.3
        recognizer.non_speaking_duration = 0.8
        
        # Create temporary file
        tmp_file_path = tempfile.mktemp(suffix=".wav")
        
        # Write audio bytes to file
        with open(tmp_file_path, "wb") as tmp_file:
            tmp_file.write(audio_bytes)
        
        # Read audio file
        with sr.AudioFile(tmp_file_path) as source:
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.record(source)
        
        # Try different Arabic language codes
        languages_to_try = ["ar-SA", "ar-EG", "ar-JO", "ar"]
        
        for lang in languages_to_try:
            try:
                text = recognizer.recognize_google(audio, language=lang)
                if text.strip():
                    return text.strip()
            except sr.UnknownValueError:
                continue
            except sr.RequestError:
                continue
        
        return ""
        
    except sr.UnknownValueError:
        st.warning("لم أتمكن من فهم الصوت - تأكد من وضوح النطق")
        return ""
    except sr.RequestError as e:
        st.error(f"خطأ في خدمة Google Speech Recognition: {str(e)}")
        return ""
    except Exception as e:
        st.error(f"خطأ في معالجة الصوت: {str(e)}")
        return ""
    finally:
        # Clean up temporary file safely
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                import time
                time.sleep(0.1)
                os.unlink(tmp_file_path)
            except (OSError, PermissionError):
                try:
                    time.sleep(0.5)
                    os.unlink(tmp_file_path)
                except:
                    pass

def process_recorded_audio(audio_bytes: bytes) -> str:
    """Process recorded audio and return transcribed text using Google STT"""
    if not audio_bytes:
        return ""
    
    return transcribe_audio_google(audio_bytes)

def get_gpt4_response(user_input: str, conversation_history: list = None) -> Dict[str, Any]:
    """Get response from GPT-4 mini acting as Syrian restaurant employee"""
    try:
        if not os.getenv("OPENAI_API_KEY"):
            return {
                "response_text": "عذراً، مفتاح OpenAI غير متوفر. تأكد من إعداد OPENAI_API_KEY في ملف .env",
                "intent": "error",
                "entities": [],
                "confidence": 0.0
            }
        
        # Build conversation context
        messages = [
            {
                "role": "system", 
                "content": """أنت أحمد، موظف سوري شغال في مطعم شاركو تشيكن في دمشق. 
                
خصائصك:
- تتكلم باللهجة السورية الطبيعية
- ودود ومرحب بالعملاء
- خبير بقائمة المطعم وتقدر تنصح العملاء
- تاخد الطلبات وتجمع معلومات التوصيل
- تتعامل مع العملاء كأنك موظف حقيقي في المطعم

قائمة المطعم:
🍗 المشاوي: فروج مشوي (25,000 ل.س), شاورما دجاج (18,000 ل.س), كباب (30,000 ل.س)
🥗 السلطات: فتوش (12,000 ل.س), تبولة (10,000 ل.س), سلطة يونانية (15,000 ل.س)
🍽️ المقبلات: حمص (8,000 ل.س), متبل (10,000 ل.س), لبنة (12,000 ل.س)
🥤 المشروبات: عصير طبيعي (7,000 ل.س), شاي (3,000 ل.س), قهوة (5,000 ل.س)

ملاحظات:
- وقت التوصيل: 20-30 دقيقة حسب المنطقة
- أسعار التوصيل: 5,000 ل.س داخل دمشق
- المطعم مفتوح من 10 صباحاً حتى 11 مساءً
- تقبل الدفع نقداً أو بالفيزا

تكلم طبيعي واسأل عن التفاصيل لما تحتاجها زي الاسم والعنوان والهاتف."""
            }
        ]
        
        # Add conversation history if available
        if conversation_history:
            for msg in conversation_history[-6:]:
                if msg.get("type") in ["user_text", "user_voice", "user"]:
                    messages.append({"role": "user", "content": msg["content"]})
                elif msg.get("type") == "agent":
                    messages.append({"role": "assistant", "content": msg["content"]})
        
        # Add current user input
        messages.append({"role": "user", "content": user_input})
        
        # Call GPT-4 mini
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=300,
            temperature=0.7,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Simple intent detection from response content
        intent = "unknown"
        if any(word in response_text.lower() for word in ["أهلا", "مرحبا", "أهلين"]):
            intent = "greeting"
        elif any(word in response_text.lower() for word in ["طلب", "تريد", "بدك"]):
            intent = "taking_order"
        elif any(word in response_text.lower() for word in ["اسم", "هاتف", "عنوان"]):
            intent = "collecting_info"
        elif any(word in response_text.lower() for word in ["قائمة", "منيو", "عندنا"]):
            intent = "menu_inquiry"
        elif any(word in response_text.lower() for word in ["خلاص", "تمام", "وصل"]):
            intent = "order_confirmation"
        
        # Extract mentioned menu items
        entities = []
        menu_items = ["فروج مشوي", "شاورما", "كباب", "فتوش", "تبولة", "حمص", "متبل", "لبنة"]
        for item in menu_items:
            if item in response_text:
                entities.append({"entity": "menu_item", "value": item})
        
        return {
            "response_text": response_text,
            "intent": intent,
            "entities": entities,
            "confidence": 0.95,
            "model": "gpt-4o-mini"
        }
        
    except Exception as e:
        return {
            "response_text": f"عذراً، صارت مشكلة في النظام: {str(e)}",
            "intent": "error",
            "entities": [],
            "confidence": 0.0
        }

def simulate_agent_response(user_input: str, use_gpt: bool = True) -> Dict[str, Any]:
    """Main agent response function - can use GPT-4 or simple rules"""
    
    if use_gpt and os.getenv("OPENAI_API_KEY"):
        return get_gpt4_response(user_input, st.session_state.conversation_history)
    else:
        # Fallback to simple rule-based responses
        intent = "unknown"
        entities = []
        
        if any(word in user_input.lower() for word in ["مرحبا", "أهلا", "هلا"]):
            intent = "greeting"
        elif any(word in user_input.lower() for word in ["فروج", "شاورما", "كباب"]):
            intent = "order_item"
            menu_items = ["فروج مشوي", "شاورما", "كباب", "فتوش", "تبولة", "حمص", "متبل"]
            for item in menu_items:
                if item in user_input:
                    entities.append({"entity": "menu_item", "value": item})
        elif any(word in user_input.lower() for word in ["قائمة", "منيو", "ايش عندكم"]):
            intent = "get_menu"
        elif any(word in user_input.lower() for word in ["خلاص", "كفاية", "بس هيك"]):
            intent = "finalize_order"
        elif any(word in user_input.lower() for word in ["شكرا", "مع السلامة"]):
            intent = "goodbye"
        
        responses = {
            "greeting": "أهلاً وسهلاً! مرحبا بيك في شاركو تشيكن، شو بدك تطلب اليوم؟",
            "order_item": f"تمام! حاضر. شو رايك نضيفلك {entities[0]['value'] if entities else 'طبق'} عالطلب؟",
            "get_menu": "عندنا مشاوي زي الفروج المشوي والشاورما، وسلطات زي الفتوش والتبولة، ومقبلات زي الحمص والمتبل. شو بتحب تطلب؟",
            "finalize_order": "تمام! ممكن رقم الهاتف للتوصيل؟ والطلب حيوصلك خلال 25 دقيقة.",
            "goodbye": "شكراً إلك! طلبك في الطريق، ونشوفك مرة تانية قريباً!",
            "unknown": "عذراً، ما فهمت شو بدك تطلب. ممكن تعيد كمان مرة؟"
        }
        
        return {
            "response_text": responses.get(intent, responses["unknown"]),
            "intent": intent,
            "entities": entities,
            "confidence": 0.85 if intent != "unknown" else 0.3,
            "model": "rule-based"
        }

def extract_order_info(conversation_history: list) -> Dict[str, Any]:
    """Extract order information from conversation history"""
    order_items = []
    customer_name = ""
    customer_phone = ""
    customer_address = ""
    
    # Menu items with prices
    menu_prices = {
        "فروج مشوي": 25000,
        "شاورما دجاج": 18000, 
        "شاورما": 18000,
        "كباب": 30000,
        "فتوش": 12000,
        "تبولة": 10000,
        "سلطة يونانية": 15000,
        "حمص": 8000,
        "متبل": 10000,
        "لبنة": 12000,
        "عصير طبيعي": 7000,
        "شاي": 3000,
        "قهوة": 5000
    }
    
    for msg in conversation_history:
        content = msg.get("content", "").lower()
        
        # Extract menu items
        for item, price in menu_prices.items():
            if item in content or any(word in content for word in item.split()):
                if not any(existing["name"] == item for existing in order_items):
                    order_items.append({
                        "name": item,
                        "price": price,
                        "quantity": 1
                    })
        
        # Extract customer info
        if any(word in content for word in ["اسمي", "أنا", "اسم"]):
            words = content.split()
            for i, word in enumerate(words):
                if word in ["اسمي", "أنا"] and i + 1 < len(words):
                    customer_name = words[i + 1]
                    break
        
        # Extract phone
        import re
        phone_match = re.search(r'(\d{10,})', content)
        if phone_match:
            customer_phone = phone_match.group(1)
        
        # Extract address
        if any(word in content for word in ["عنوان", "شارع", "منطقة", "حي"]):
            customer_address = content
    
    return {
        "items": order_items,
        "customer_name": customer_name or "زبون",
        "customer_phone": customer_phone or "غير محدد",
        "customer_address": customer_address or "غير محدد"
    }

def save_order_to_backend(order_info: Dict[str, Any]) -> bool:
    """Save order to backend API or fallback to JSON file"""
    try:
        if not order_info["items"]:
            return False
        
        total_price = sum(item["price"] * item["quantity"] for item in order_info["items"])
        
        order_data = {
            "order_id": f"ORD{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "customer_name": order_info["customer_name"],
            "customer_phone": order_info["customer_phone"],
            "customer_address": order_info["customer_address"],
            "items": order_info["items"],
            "total": total_price,
            "status": "pending",
            "eta": 25,
            "created_at": datetime.now().isoformat()
        }
        
        # Try backend API first
        try:
            response = call_backend_api("orders", method="POST", data=order_data)
            if response is not None:
                st.success("✅ تم حفظ الطلب في Backend API")
                return True
        except Exception as e:
            st.warning(f"⚠️ Backend API غير متاح: {str(e)}")
        
        # Fallback: Save to local JSON file
        orders_file = "orders.json"
        
        # Load existing orders
        existing_orders = []
        if os.path.exists(orders_file):
            try:
                with open(orders_file, "r", encoding="utf-8") as f:
                    existing_orders = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                existing_orders = []
        
        # Add new order
        existing_orders.append(order_data)
        
        # Save updated orders
        with open(orders_file, "w", encoding="utf-8") as f:
            json.dump(existing_orders, f, ensure_ascii=False, indent=2)
        
        st.info("💾 تم حفظ الطلب في الملف المحلي (Backend غير متاح)")
        return True
        
    except Exception as e:
        st.error(f"خطأ في حفظ الطلب: {str(e)}")
        return False

def get_local_orders() -> list:
    """Get orders from local JSON file"""
    orders_file = "orders.json"
    if os.path.exists(orders_file):
        try:
            with open(orders_file, "r", encoding="utf-8") as f:
                orders = json.load(f)
                return orders
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    return []

def get_local_stats() -> Dict[str, Any]:
    """Calculate stats from local orders"""
    try:
        orders = get_local_orders()
        
        if not orders:
            return {
                "total_orders": 0,
                "total_revenue": 0,
                "average_order_value": 0,
                "status_breakdown": {}
            }
        
        total_orders = len(orders)
        total_revenue = 0
        
        # Calculate total revenue safely
        for order in orders:
            order_total = order.get("total", 0)
            if isinstance(order_total, (int, float)):
                total_revenue += order_total
        
        average_order_value = total_revenue / total_orders if total_orders > 0 else 0
        
        # Status breakdown
        status_breakdown = {}
        for order in orders:
            status = order.get("status", "pending")
            status_breakdown[status] = status_breakdown.get(status, 0) + 1
        
        result = {
            "total_orders": total_orders,
            "total_revenue": total_revenue,
            "average_order_value": average_order_value,
            "status_breakdown": status_breakdown
        }
        
        return result
        
    except Exception as e:
        st.error(f"خطأ في حساب الإحصائيات المحلية: {str(e)}")
        return {
            "total_orders": 0,
            "total_revenue": 0,
            "average_order_value": 0,
            "status_breakdown": {}
        }

def check_order_completion(conversation_history: list) -> bool:
    """Check if order is complete and ready to save"""
    if not conversation_history:
        return False
    
    recent_messages = [msg.get("content", "").lower() for msg in conversation_history[-5:]]
    recent_text = " ".join(recent_messages)
    
    completion_indicators = [
        "تمام", "خلاص", "كفاية", "بس هيك", "هاي الطلبة", 
        "ممكن رقم الهاتف", "وين العنوان", "هاي البيانات"
    ]
    
    has_items = any(item in recent_text for item in [
        "فروج", "شاورما", "كباب", "فتوش", "تبولة", "حمص", "متبل"
    ])
    
    has_completion = any(indicator in recent_text for indicator in completion_indicators)
    
    return has_items and has_completion

def save_conversation_to_json():
    """Save conversation history to JSON file"""
    try:
        with open("conversation.json", "w", encoding="utf-8") as f:
            json.dump(st.session_state.conversation_history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Error saving conversation: {str(e)}")

def process_and_save_conversation():
    """Process conversation and save order if complete"""
    save_conversation_to_json()
    
    if check_order_completion(st.session_state.conversation_history):
        order_info = extract_order_info(st.session_state.conversation_history)
        
        if order_info["items"]:
            st.info("🍽️ تم اكتشاف طلب كامل، جاري الحفظ...")
            
            if save_order_to_backend(order_info):
                st.success("✅ تم حفظ الطلب بنجاح!")
                
                # Force refresh of stats by clearing ALL cache
                cache_keys_to_clear = ['stats_cache', 'orders_cache', 'stats_timestamp', 'stats_from_backend']
                for key in cache_keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]
                
                # Force immediate stats recalculation
                new_stats = get_local_stats()
                st.session_state.stats_cache = new_stats
                st.session_state.stats_from_backend = False
                st.session_state.stats_timestamp = datetime.now().timestamp()
                
                with st.expander("📋 ملخص الطلب"):
                    st.write(f"**الزبون:** {order_info['customer_name']}")
                    st.write(f"**الهاتف:** {order_info['customer_phone']}")
                    st.write(f"**العنوان:** {order_info['customer_address']}")
                    st.write("**الأصناف:**")
                    
                    total = 0
                    for item in order_info["items"]:
                        st.write(f"- {item['name']}: {item['price']:,} ل.س")
                        total += item["price"]
                    
                    st.write(f"**المجموع الكلي:** {total:,} ل.س")
                    
                # Show updated stats immediately
                st.balloons()
                st.success(f"🔄 تم تحديث الإحصائيات! الآن لديك {new_stats['total_orders']} طلب")
                st.info("📊 انتقل لتبويب الإحصائيات لرؤية التحديث الفوري")
                
            else:
                st.error("❌ فشل في حفظ الطلب - تحقق من الاتصال بالخادم")

# Page configuration
st.set_page_config(
    page_title="شاركو تشيكن - اختبار المساعد الصوتي",
    page_icon="🐔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.rtl-text {
    direction: rtl;
    text-align: right;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    color: #1f2937;
    font-weight: 500;
}

.agent-response {
    background-color: #f0f9ff;
    color: #1e40af;
    padding: 15px;
    border-radius: 12px;
    border-left: 5px solid #3b82f6;
    margin: 12px 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    font-weight: 500;
}

.user-input {
    background-color: #fef3c7;
    color: #92400e;
    padding: 15px;
    border-radius: 12px;
    border-left: 5px solid #f59e0b;
    margin: 12px 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    font-weight: 500;
}

.recorded-input {
    background-color: #f3e8ff;
    color: #7c3aed;
    padding: 15px;
    border-radius: 12px;
    border-left: 5px solid #8b5cf6;
    margin: 12px 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    font-weight: 500;
}

.intent-box {
    background-color: #e0f2fe;
    color: #0f4c75;
    padding: 10px;
    border-radius: 8px;
    margin: 8px 0;
    border: 1px solid #b3e5fc;
    font-weight: 500;
}

.status-connected {
    background-color: #dcfce7;
    color: #166534;
    padding: 8px 12px;
    border-radius: 6px;
    font-weight: 600;
}

.status-disconnected {
    background-color: #fee2e2;
    color: #dc2626;
    padding: 8px 12px;
    border-radius: 6px;
    font-weight: 600;
}

.metric-card {
    background-color: #ffffff;
    color: #374151;
    padding: 16px;
    border-radius: 8px;
    border: 2px solid #e5e7eb;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    text-align: center;
}

.system-status {
    background-color: #f9fafb;
    color: #374151;
    padding: 12px;
    border-radius: 8px;
    border: 1px solid #d1d5db;
    margin: 4px 0;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state variables"""
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'current_order' not in st.session_state:
        st.session_state.current_order = []
    if 'agent_responses' not in st.session_state:
        st.session_state.agent_responses = []

def call_backend_api(endpoint: str, method: str = "GET", data: Dict = None) -> Optional[Dict]:
    """Call backend API"""
    base_url = "http://localhost:8000"
    
    try:
        if method == "GET":
            response = requests.get(f"{base_url}/{endpoint}")
        elif method == "POST":
            response = requests.post(f"{base_url}/{endpoint}", json=data)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Connection Error: {str(e)}")
        return None

def main():
    init_session_state()

    # Load conversation from JSON
    if "conversation_loaded" not in st.session_state:
        if os.path.exists("conversation.json"):
            try:
                with open("conversation.json", "r", encoding="utf-8") as f:
                    st.session_state.conversation_history = json.load(f)
            except json.JSONDecodeError:
                st.session_state.conversation_history = []
        else:
            st.session_state.conversation_history = []
        st.session_state.conversation_loaded = True

    # Header
    st.title("🐔 شاركو تشيكن - اختبار المساعد الصوتي")
    st.markdown('<div class="rtl-text">نظام اختبار للمساعد الصوتي باللهجة السورية</div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ إعدادات النظام")

        # Agent Status
        st.subheader("حالة المساعد")
        agent_status = st.selectbox(
            "Connection Status:",
            ["متصل ✅", "غير متصل ❌", "في انتظار المكالمة 📞"]
        )

        if "متصل" in agent_status:
            st.markdown('<div class="status-connected">Connected Successfully</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-disconnected">Connection Failed</div>', unsafe_allow_html=True)

        # Voice Settings
        st.subheader("إعدادات الصوت")
        voice_model = st.selectbox(
            "TTS Model:",
            ["ElevenLabs - Sarah", "Google TTS - ar-SY", "Azure - Syrian"]
        )

        # STT Settings
        st.subheader("إعدادات التعرف على الصوت")
        st.info("🎤 Google Speech Recognition (Arabic)")
        
        # AI Model Settings
        st.subheader("إعدادات الذكاء الاصطناعي")
        use_gpt = st.checkbox("استخدام GPT-4 mini", value=True, help="للحصول على ردود ذكية ومتقدمة")
        auto_tts = st.checkbox("تشغيل الصوت تلقائياً", value=False, help="تشغيل رد ElevenLabs تلقائياً")
        
        if use_gpt:
            st.info("🤖 أحمد - موظف افتراضي سوري")
        else:
            st.info("📝 نظام الردود البسيط")
        
        audio_quality = st.selectbox(
            "Audio Quality:",
            ["High (44.1kHz)", "Medium (22kHz)", "Low (16kHz)"]
        )

        speech_rate = st.slider("Speech Rate", 0.5, 2.0, 1.0, 0.1)

        # API Settings
        st.subheader("إعدادات API")
        api_status = call_backend_api("")
        if api_status:
            st.markdown('<div class="status-connected">✅ Backend API Connected</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-disconnected">❌ Backend API Disconnected</div>', unsafe_allow_html=True)

        # Check Google STT status
        try:
            test_recognizer = sr.Recognizer()
            st.markdown('<div class="status-connected">✅ Google STT Ready</div>', unsafe_allow_html=True)
        except Exception as e:
            st.markdown('<div class="status-disconnected">❌ Google STT Error</div>', unsafe_allow_html=True)

        # Check OpenAI API key
        if os.getenv("OPENAI_API_KEY"):
            st.markdown('<div class="status-connected">✅ OpenAI API Key Found</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-disconnected">❌ OpenAI API Key Missing</div>', unsafe_allow_html=True)
            st.warning("⚠️ أضف OPENAI_API_KEY في ملف .env لاستخدام GPT-4")

        # Check ElevenLabs API key
        if ELEVENLABS_API_KEY:
            st.markdown('<div class="status-connected">✅ ElevenLabs API Key Found</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-disconnected">❌ ElevenLabs API Key Missing</div>', unsafe_allow_html=True)
            st.warning("⚠️ أضف ELEVENLABS_API_KEY في ملف .env للصوت")

        # Clear conversation
        if st.button("🗑️ مسح المحادثة"):
            st.session_state.conversation_history = []
            st.session_state.current_order = []
            st.session_state.transcribed_text = ""
            save_conversation_to_json()
            st.rerun()

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎙️ اختبار المحادثة", "📋 القائمة والطلبات", "📊 الإحصائيات", "🔧 المراقبة"])

    with tab1:
        st.header("Voice Conversation Testing")
    
        col1, col2 = st.columns([1, 1])
    
        with col1:
            st.subheader("🎤 Voice Input")
            
            st.markdown("**اضغط لبدء التسجيل:**")
            st.info("💡 تحدث بوضوح باللهجة السورية")
            
            audio_bytes = audio_recorder(
                text="🎙️ اضغط للتسجيل",
                recording_color="#e74c3c",
                neutral_color="#2c3e50",
                icon_name="microphone",
                icon_size="2x",
                pause_threshold=1.5,
                sample_rate=16000,
            )
            
            if audio_bytes:
                st.audio(audio_bytes, format="audio/wav")
                st.success(f"✅ تم التسجيل! الحجم: {len(audio_bytes)} بايت")
                
                col_a, col_b = st.columns([1, 1])
                
                with col_a:
                    if st.button("🎯 تحويل الصوت لنص", type="primary"):
                        with st.spinner("🔄 جاري التحويل باستخدام Google STT..."):
                            transcribed_text = process_recorded_audio(audio_bytes)
                            if transcribed_text:
                                st.session_state.transcribed_text = transcribed_text
                                st.success("✅ تم التحويل بنجاح!")
                                st.balloons()
                                st.rerun()
                            else:
                                st.error("❌ فشل في تحويل الصوت - حاول مرة أخرى")
                
                with col_b:
                    if st.button("🔄 تسجيل جديد"):
                        st.session_state.transcribed_text = ""
                        st.info("🎤 اضغط زرار التسجيل مرة أخرى")
                        st.rerun()
                
                if st.session_state.transcribed_text:
                    st.markdown(f"""
                    <div class="recorded-input rtl-text">
                        <strong>🎙️ النص المحول:</strong><br>
                        {st.session_state.transcribed_text}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button("📤 إرسال النص", type="secondary"):
                        user_input = st.session_state.transcribed_text
                        
                        with st.spinner("🤖 جاري الرد..."):
                            response_data = simulate_agent_response(user_input, use_gpt=use_gpt)
                        
                        st.session_state.conversation_history.append({
                            "timestamp": datetime.now().strftime("%H:%M:%S"),
                            "type": "user_voice",
                            "content": user_input,
                            "intent": response_data["intent"],
                            "entities": response_data["entities"],
                            "stt_provider": "Google Speech"
                        })
                        
                        st.session_state.conversation_history.append({
                            "timestamp": datetime.now().strftime("%H:%M:%S"),
                            "type": "agent",
                            "content": response_data["response_text"],
                            "confidence": response_data["confidence"],
                            "model": response_data.get("model", "unknown")
                        })
                        
                        process_and_save_conversation()
                        
                        if auto_tts and ELEVENLABS_API_KEY:
                            st.info("🔊 تشغيل تلقائي للصوت...")
                            play_elevenlabs_audio(response_data["response_text"], "auto_tts")
                        
                        st.session_state.transcribed_text = ""
                        st.success("📤 تم إرسال الرسالة!")
                        st.rerun()
            else:
                st.info("🎙️ لم يتم التسجيل بعد - اضغط زرار الميكروفون للبدء")
            
            st.markdown("---")
            
            st.subheader("📱 رفع ملف صوتي")
            uploaded_file = st.file_uploader(
                "ارفع ملف صوتي للاختبار",
                type=['wav', 'mp3', 'm4a', 'ogg'],
                help="يمكنك رفع ملف صوتي باللغة العربية"
            )
            
            if uploaded_file:
                st.audio(uploaded_file, format='audio/wav')
                if st.button("🎯 تحليل الملف المرفوع"):
                    with st.spinner("🔄 جاري معالجة الملف..."):
                        audio_bytes_uploaded = uploaded_file.read()
                        transcribed_text = process_recorded_audio(audio_bytes_uploaded)
                        if transcribed_text:
                            st.success(f"✅ النص المحول: {transcribed_text}")
                            if st.button("📝 استخدام هذا النص"):
                                st.session_state.transcribed_text = transcribed_text
                                st.rerun()
                        else:
                            st.error("❌ لم أتمكن من تحويل الملف المرفوع")
            
            st.markdown("---")
    
            st.subheader("⌨️ اختبار نصي")
            user_input = st.text_area(
                "اكتب رسالة باللهجة السورية:",
                placeholder="مثال: أهلا، بدي طلب فروج مشوي وسلطة فتوش",
                height=100
            )
    
            if st.button("📤 إرسال النص", type="primary"):
                if user_input.strip():
                    with st.spinner("🤖 جاري الرد..."):
                        response_data = simulate_agent_response(user_input, use_gpt=use_gpt)
    
                    st.session_state.conversation_history.append({
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "type": "user_text",
                        "content": user_input,
                        "intent": response_data["intent"],
                        "entities": response_data["entities"]
                    })
    
                    st.session_state.conversation_history.append({
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "type": "agent",
                        "content": response_data["response_text"],
                        "confidence": response_data["confidence"],
                        "model": response_data.get("model", "unknown")
                    })
                    
                    process_and_save_conversation()
                    
                    if auto_tts and ELEVENLABS_API_KEY:
                        st.info("🔊 تشغيل تلقائي للصوت...")
                        play_elevenlabs_audio(response_data["response_text"], "auto_tts_text")
                    
                    st.success("📤 تم إرسال الرسالة!")
                    st.rerun()
    
        with col2:
            st.subheader("🤖 رد المساعد")
    
            if st.session_state.conversation_history:
                for msg in st.session_state.conversation_history[-6:]:
                    if msg["type"] in ["user_text", "user_voice", "user"]:
                        msg_type = msg.get("type", "user")
                        icon = "🎙️" if msg_type == "user_voice" else "⌨️" if msg_type == "user_text" else "🗣️"
                        input_type = "صوت" if msg_type == "user_voice" else "نص" if msg_type == "user_text" else "إدخال"
                        css_class = "recorded-input" if msg_type == "user_voice" else "user-input"
                        
                        st.markdown(f"""
                        <div class="{css_class} rtl-text">
                            <strong>{icon} الزبون ({input_type} - {msg['timestamp']}):</strong><br>
                            {msg['content']}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if "intent" in msg:
                            stt_info = f" | STT: {msg.get('stt_provider', 'N/A')}" if msg_type == "user_voice" else ""
                            st.markdown(f"""
                            <div class="intent-box">
                                🎯 القصد المكتشف: {msg['intent']}<br>
                                📝 الكيانات: {', '.join([e['value'] for e in msg.get('entities', [])])}{stt_info}
                            </div>
                            """, unsafe_allow_html=True)
                    
                    else:  # agent
                        model_used = msg.get('model', 'unknown')
                        model_icon = "🤖" if model_used == "gpt-4o-mini" else "📝"
                        model_name = "أحمد (GPT-4)" if model_used == "gpt-4o-mini" else "نظام بسيط"
                        
                        st.markdown(f"""
                        <div class="agent-response rtl-text">
                            <strong>{model_icon} {model_name} ({msg['timestamp']}):</strong><br>
                            {msg['content']}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if st.button(f"🔊 تشغيل الرد (ElevenLabs)", key=f"play_{msg['timestamp']}"):
                            play_elevenlabs_audio(msg['content'], msg['timestamp'])
            else:
                st.info("ابدأ محادثة لترى الردود هنا")
            
            if st.session_state.conversation_history:
                st.markdown("---")
                st.subheader("🍽️ إدارة الطلب")
                
                col_order1, col_order2 = st.columns([1, 1])
                
                with col_order1:
                    if st.button("📋 معاينة الطلب الحالي"):
                        order_info = extract_order_info(st.session_state.conversation_history)
                        if order_info["items"]:
                            with st.expander("📋 تفاصيل الطلب", expanded=True):
                                st.write(f"**الزبون:** {order_info['customer_name']}")
                                st.write(f"**الهاتف:** {order_info['customer_phone']}")
                                st.write("**الأصناف:**")
                                
                                total = 0
                                for item in order_info["items"]:
                                    st.write(f"- {item['name']}: {item['price']:,} ل.س")
                                    total += item["price"]
                                
                                st.write(f"**المجموع الكلي:** {total:,} ل.س")
                        else:
                            st.info("لا توجد أصناف في الطلب بعد")
                
                with col_order2:
                    if st.button("✅ تأكيد وحفظ الطلب"):
                        order_info = extract_order_info(st.session_state.conversation_history)
                        if order_info["items"]:
                            if save_order_to_backend(order_info):
                                st.success("✅ تم حفظ الطلب بنجاح!")
                                st.balloons()
                            else:
                                st.error("❌ فشل في حفظ الطلب")
                        else:
                            st.warning("⚠️ لا يوجد طلب لحفظه")
    
            st.markdown("---")
            st.subheader("🔴 محادثة صوتية مباشرة")
    
            if st.session_state.get("live_logs"):
                for msg in st.session_state.live_logs[-10:]:
                    if msg["role"] == "user":
                        st.markdown(f"""
                        <div class="recorded-input rtl-text">
                            <strong>🧑 الزبون (مباشر):</strong><br>
                            {msg['content']}
                        </div>
                        """, unsafe_allow_html=True)
                    elif msg["role"] == "agent":
                        st.markdown(f"""
                        <div class="agent-response rtl-text">
                            <strong>🤖 المساعد (مباشر):</strong><br>
                            {msg['content']}
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("لا توجد محادثة صوتية حالياً. ابدأ مكالمة لـ LiveKit.")
    
            if st.button("🧹 مسح المحادثة المباشرة"):
                st.session_state.live_logs = []
                st.success("تم مسح المحادثة الصوتية.")

    
    with tab2:
        st.header("القائمة والطلبات")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📋 قائمة الطعام")
            menu_data = call_backend_api("menu")
            if menu_data:
                for category, items in menu_data["menu"].items():
                    st.markdown(f"### {category}")
                    for item in items:
                        st.markdown(f"**{item['name']}** - {item['price']} ليرة")
                        st.caption(item['description'])
                    st.markdown("---")
        
        with col2:
            st.subheader("🛒 الطلبات الأخيرة")
            
            if st.button("🔄 تحديث الطلبات"):
                st.rerun()
            
            # Try backend API first, then fallback to local file
            orders_data = call_backend_api("orders")
            orders_from_backend = True
            
            if not orders_data or not orders_data.get("orders"):
                # Fallback to local orders
                local_orders = get_local_orders()
                if local_orders:
                    orders_data = {"orders": local_orders}
                    orders_from_backend = False
            
            if orders_data and orders_data.get("orders"):
                source_info = "من Backend API" if orders_from_backend else "من الملف المحلي"
                st.success(f"✅ تم العثور على {len(orders_data['orders'])} طلب ({source_info})")
                
                for order in reversed(orders_data["orders"][-5:]):  # Show last 5 orders, newest first
                    order_id = order.get('order_id', 'غير محدد')
                    order_status = order.get('status', 'غير محدد')
                    order_date = order.get('created_at', '')[:19] if order.get('created_at') else 'غير محدد'
                    
                    with st.expander(f"طلب {order_id} - {order_status} ({order_date})"):
                        st.write(f"**الزبون:** {order['customer_name']}")
                        st.write(f"**الهاتف:** {order['customer_phone']}")
                        st.write(f"**العنوان:** {order.get('customer_address', 'غير محدد')}")
                        st.write(f"**المجموع:** {order['total']:,} ليرة")
                        st.write(f"**وقت التوصيل:** {order['eta']} دقيقة")
                        
                        if order.get('items'):
                            st.write("**الأصناف:**")
                            for item in order['items']:
                                st.write(f"- {item['name']}: {item['price']:,} ل.س")
            else:
                st.info("لا توجد طلبات بعد")
                if not call_backend_api(""):
                    st.warning("⚠️ Backend API غير متصل - سيتم حفظ الطلبات محلياً")
                    st.info("💡 الطلبات الجديدة ستحفظ في ملف orders.json")
    
    with tab3:
        st.header("📊 إحصائيات المطعم")
        
        col_refresh, col_auto = st.columns([1, 1])
        
        with col_refresh:
            if st.button("🔄 تحديث الإحصائيات"):
                # Clear cache to force refresh
                if 'stats_cache' in st.session_state:
                    del st.session_state.stats_cache
                if 'orders_cache' in st.session_state:
                    del st.session_state.orders_cache
                st.rerun()
        
        with col_auto:
            auto_refresh_stats = st.checkbox("🔄 تحديث تلقائي كل 10 ثواني", value=False)
            if auto_refresh_stats:
                import time
                time.sleep(0.1)  # Small delay
                st.rerun()
        
        # Cache stats to avoid recalculation
        if 'stats_cache' not in st.session_state or 'stats_timestamp' not in st.session_state or \
           (datetime.now().timestamp() - st.session_state.get('stats_timestamp', 0)) > 5:
            
            # Try backend API first, then fallback to local stats
            stats_data = call_backend_api("stats")
            stats_from_backend = True
            
            if not stats_data:
                # Fallback to local stats
                stats_data = get_local_stats()
                stats_from_backend = False
            
            # Cache the results
            st.session_state.stats_cache = stats_data
            st.session_state.stats_from_backend = stats_from_backend
            st.session_state.stats_timestamp = datetime.now().timestamp()
        else:
            # Use cached data
            stats_data = st.session_state.stats_cache
            stats_from_backend = st.session_state.stats_from_backend
        
        if stats_data and stats_data.get("total_orders", 0) > 0:
            source_info = "من Backend API" if stats_from_backend else "من الملف المحلي"
            st.info(f"📊 الإحصائيات محسوبة {source_info} | آخر تحديث: {datetime.now().strftime('%H:%M:%S')}")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("مجموع الطلبات", stats_data["total_orders"])
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("مجموع الإيرادات", f"{stats_data['total_revenue']:,} ليرة")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("متوسط قيمة الطلب", f"{stats_data['average_order_value']:,.0f} ليرة")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                if stats_data["total_orders"] > 0:
                    completion_rate = stats_data["status_breakdown"].get("delivered", 0) / stats_data["total_orders"] * 100
                    st.metric("معدل الإنجاز", f"{completion_rate:.1f}%")
                else:
                    st.metric("معدل الإنجاز", "0%")
                st.markdown('</div>', unsafe_allow_html=True)
            
            if stats_data["status_breakdown"]:
                st.subheader("توزيع حالة الطلبات")
                status_df = pd.DataFrame(
                    list(stats_data["status_breakdown"].items()),
                    columns=["الحالة", "العدد"]
                )
                fig = px.pie(status_df, values="العدد", names="الحالة", title="توزيع حالة الطلبات")
                st.plotly_chart(fig, use_container_width=True)
            
            # Real-time orders display
            st.subheader("📋 الطلبات الأخيرة (في الإحصائيات)")
            local_orders = get_local_orders()
            if local_orders:
                st.success(f"✅ {len(local_orders)} طلب محفوظ محلياً:")
                
                # Show last 3 orders in compact format
                for order in reversed(local_orders[-3:]):
                    with st.container():
                        col_order1, col_order2, col_order3 = st.columns([2, 1, 1])
                        
                        with col_order1:
                            st.write(f"**{order['customer_name']}** ({order['order_id']})")
                            items_text = ", ".join([item['name'] for item in order.get('items', [])])
                            st.caption(f"الأصناف: {items_text}")
                        
                        with col_order2:
                            st.metric("المبلغ", f"{order['total']:,} ل.س")
                        
                        with col_order3:
                            st.metric("الحالة", order['status'])
                        
                        st.divider()
                
        else:
            st.warning("⚠️ لا توجد طلبات لحساب الإحصائيات")
            
            if not stats_from_backend:
                st.info("💡 ابدأ بعمل طلبات لرؤية الإحصائيات")
                
                # Check if there are local orders file
                local_orders = get_local_orders()
                if local_orders:
                    st.warning(f"🔍 وجدت {len(local_orders)} طلب في الملف المحلي ولكن لم تظهر في الإحصائيات")
                    st.json(local_orders[-1] if local_orders else {})  # Debug: show last order
                else:
                    st.info("📂 لا يوجد ملف orders.json بعد")
            else:
                st.warning("تأكد من تشغيل Backend API على localhost:8000")
            
            # Show empty metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("مجموع الطلبات", 0)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("مجموع الإيرادات", "0 ليرة")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("متوسط قيمة الطلب", "0 ليرة")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("معدل الإنجاز", "0%")
                st.markdown('</div>', unsafe_allow_html=True)
            
            if st.button("🔍 اختبار الاتصال بالخادم"):
                test_result = call_backend_api("")
                if test_result:
                    st.success("✅ الخادم يعمل بشكل صحيح")
                else:
                    st.error("❌ فشل الاتصال بالخادم - سيتم استخدام التخزين المحلي")
            
            # Debug section
            with st.expander("🔧 تشخيص المشاكل"):
                st.write("**ملفات موجودة:**")
                if os.path.exists("orders.json"):
                    st.success("✅ orders.json موجود")
                    try:
                        with open("orders.json", "r", encoding="utf-8") as f:
                            orders_content = json.load(f)
                            st.write(f"**عدد الطلبات في الملف:** {len(orders_content)}")
                            
                            if orders_content:
                                st.write("**آخر طلب:**")
                                last_order = orders_content[-1]
                                st.json({
                                    "order_id": last_order.get("order_id", "غير محدد"),
                                    "customer_name": last_order.get("customer_name", "غير محدد"),
                                    "total": last_order.get("total", 0),
                                    "items_count": len(last_order.get("items", [])),
                                    "status": last_order.get("status", "غير محدد")
                                })
                                
                            # Manual stats calculation for debugging
                            manual_stats = get_local_stats()
                            st.write("**إحصائيات محسوبة يدوياً:**")
                            st.json(manual_stats)
                            
                            # Force update stats cache
                            if st.button("🔄 إعادة حساب الإحصائيات فوراً"):
                                # Clear all cache
                                for key in ['stats_cache', 'stats_timestamp', 'stats_from_backend']:
                                    if key in st.session_state:
                                        del st.session_state[key]
                                
                                # Recalculate
                                new_stats = get_local_stats()
                                st.session_state.stats_cache = new_stats
                                st.session_state.stats_from_backend = False
                                st.session_state.stats_timestamp = datetime.now().timestamp()
                                
                                st.success(f"✅ تم إعادة الحساب! الطلبات: {new_stats['total_orders']}")
                                st.rerun()
                                
                    except Exception as e:
                        st.error(f"خطأ في قراءة orders.json: {str(e)}")
                else:
                    st.error("❌ orders.json غير موجود")
                
                if os.path.exists("conversation.json"):
                    st.success("✅ conversation.json موجود")
                else:
                    st.info("conversation.json غير موجود")
                
                # Cache status
                st.write("**حالة الـ Cache:**")
                if 'stats_cache' in st.session_state:
                    st.write(f"Cache موجود: {st.session_state.stats_cache}")
                else:
                    st.write("Cache فارغ")
    
    with tab4:
        st.header("🔧 مراقبة المحادثات")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("سجل المحادثات")
            if st.session_state.conversation_history:
                conversation_df = pd.DataFrame(st.session_state.conversation_history)
                st.dataframe(conversation_df, use_container_width=True)
                
                if st.button("📥 تحميل سجل المحادثة"):
                    json_str = json.dumps(st.session_state.conversation_history, ensure_ascii=False, indent=2)
                    st.download_button(
                        label="💾 حفظ JSON",
                        data=json_str,
                        file_name=f"conversation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            else:
                st.info("لا توجد محادثات للعرض")
        
        with col2:
            st.subheader("إعدادات المراقبة")
            
            auto_refresh = st.checkbox("تحديث تلقائي", False)
            if auto_refresh:
                st.info("التحديث التلقائي مفعل")
            
            log_level = st.selectbox(
                "مستوى السجل:",
                ["INFO", "DEBUG", "WARNING", "ERROR"]
            )
            
            st.subheader("صحة النظام")
            system_health = {
                "LiveKit": "🟢 متصل",
                "Google STT": "🟢 متصل", 
                "GPT-4 mini": "🟢 متصل" if os.getenv("OPENAI_API_KEY") else "🔴 مفتاح API مفقود",
                "ElevenLabs TTS": "🟢 متصل" if ELEVENLABS_API_KEY else "🔴 مفتاح API مفقود",
                "Backend API": "🟢 متصل" if call_backend_api("") else "🔴 غير متصل"
            }
            
            for service, status in system_health.items():
                if "🟢" in status:
                    st.markdown(f'<div class="system-status"><strong>{service}:</strong> {status}</div>', unsafe_allow_html=True)
                elif "🟡" in status:
                    st.markdown(f'<div class="system-status" style="background-color: #fef3c7; color: #92400e;"><strong>{service}:</strong> {status}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="system-status" style="background-color: #fee2e2; color: #dc2626;"><strong>{service}:</strong> {status}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()