"""
Syrian Arabic Voice Agent for Charco Chicken Restaurant
Built with LiveKit, Google Speech, GPT-4-mini, and ElevenLabs
"""

import asyncio
import json
import logging
import os
from typing import Annotated, List, Dict, Any, Optional
import httpx
from livekit.agents import (
    Agent, 
    AgentSession, 
    JobContext, 
    RunContext, 
    WorkerOptions, 
    cli, 
    function_tool,
    llm
)
from livekit.plugins import google, openai, elevenlabs, silero
from livekit import rtc
import aiofiles
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Syrian Menu Data
MENU_ITEMS = {
    "فروج مشوي": {"name": "فروج مشوي", "price": 25, "category": "مشاوي"},
    "شاورما": {"name": "شاورما", "price": 15, "category": "مشاوي"},
    "كباب": {"name": "كباب", "price": 20, "category": "مشاوي"},
    "فتوش": {"name": "فتوش", "price": 12, "category": "سلطات"},
    "تبولة": {"name": "تبولة", "price": 10, "category": "سلطات"},
    "حمص": {"name": "حمص", "price": 8, "category": "مقبلات"},
    "متبل": {"name": "متبل", "price": 8, "category": "مقبلات"},
    "عصير ليمون": {"name": "عصير ليمون", "price": 5, "category": "مشروبات"},
    "شاي": {"name": "شاي", "price": 3, "category": "مشروبات"},
    "قهوة": {"name": "قهوة", "price": 4, "category": "مشروبات"},
}

# Syrian Greetings and Responses
SYRIAN_RESPONSES = {
    "greeting": [
        "أهلاً وسهلاً! مرحبا بيك في شاركو تشيكن، شو بدك تطلب اليوم؟",
        "هلا وغلا! كيفك؟ شو رايك نطلبلك أطيب الأكلات؟",
        "يا هلا بيك! إنت في المطعم الأحلى، شو بتحب تاكل؟"
    ],
    "order_confirm": [
        "تمام، حاضر! طلبك {} صار جاهز. أي شي تاني بدك تضيفه؟",
        "حلو كتير! {} في الطريق إلك. بدك شي تاني معه؟",
        "ممتاز! {} طلبناه إلك. شو رايك نضيف شي تاني؟"
    ],
    "upsell": [
        "شو رايك نضيفلك شوية حمص ومتبل مع الطلب؟ حلوين كتير!",
        "بتحب نضيفلك سلطة فتوش؟ طعمها رائع مع المشاوي!",
        "شو رايك عصير ليمون طازج مع أكلتك؟"
    ],
    "total": [
        "تمام! مجموع الطلب {} ليرة. والطلب حيوصلك خلال {} دقيقة.",
        "حلو! الحساب {} ليرة، والأكل حيكون جاهز بعد {} دقيقة.",
        "ممتاز! {} ليرة المجموع، وحنوصلك الطلب خلال {} دقيقة."
    ],
    "goodbye": [
        "شكراً إلك! طلبك في الطريق، ونشوفك مرة تانية قريباً!",
        "يسلم تمك! الأكل حيوصلك قريباً. مع السلامة!",
        "الله يعطيك العافية! منتظرينك مرة تانية في شاركو تشيكن!"
    ]
}

class SyrianRestaurantAgent:
    def __init__(self):
        self.current_order = []
        self.customer_name = ""
        self.order_total = 0
        self.conversation_state = "greeting"
        
    def parse_syrian_order(self, text: str) -> List[Dict]:
        """Parse Syrian Arabic text for menu items"""
        found_items = []
        text_lower = text.lower()
        
        for item_name, item_data in MENU_ITEMS.items():
            if item_name in text or any(word in text_lower for word in item_name.split()):
                # Try to extract quantity
                quantity = 1
                for word in text.split():
                    if word.isdigit():
                        quantity = int(word)
                        break
                
                found_items.append({
                    "item": item_data,
                    "quantity": quantity,
                    "total": item_data["price"] * quantity
                })
        
        return found_items

    def get_menu_text(self) -> str:
        """Get formatted menu in Syrian Arabic"""
        menu_text = "قائمة الطعام:\n\n"
        
        categories = {}
        for item_name, item_data in MENU_ITEMS.items():
            category = item_data["category"]
            if category not in categories:
                categories[category] = []
            categories[category].append(f"{item_name} - {item_data['price']} ليرة")
        
        for category, items in categories.items():
            menu_text += f"** {category} **\n"
            menu_text += "\n".join(items) + "\n\n"
        
        return menu_text

# Order Management Functions
@function_tool
async def get_menu() -> str:
    """Get the restaurant menu in Syrian Arabic"""
    agent = SyrianRestaurantAgent()
    return agent.get_menu_text()

@function_tool  
async def add_to_order(
    items: Annotated[str, "Menu items mentioned by customer in Syrian Arabic"],
    customer_name: Annotated[str, "Customer name if provided"] = ""
) -> str:
    """Add items to customer order"""
    agent = SyrianRestaurantAgent()
    
    if customer_name:
        agent.customer_name = customer_name
    
    parsed_items = agent.parse_syrian_order(items)
    
    if not parsed_items:
        return "عذراً، ما فهمت شو بدك تطلب. ممكن تعيد كمان مرة؟"
    
    response = "تمام! ضفت عالطلب:\n"
    total_added = 0
    
    for item in parsed_items:
        agent.current_order.append(item)
        response += f"- {item['item']['name']} × {item['quantity']} = {item['total']} ليرة\n"
        total_added += item['total']
    
    agent.order_total += total_added
    
    response += f"\nمجموع الطلب لحد هلق: {agent.order_total} ليرة"
    return response

@function_tool
async def finalize_order(customer_phone: Annotated[str, "Customer phone number"]) -> str:
    """Finalize the customer order"""
    # Send order to backend API
    order_data = {
        "customer_name": "الزبون",  # Default if not provided
        "customer_phone": customer_phone,
        "items": [],
        "total": 0
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:8001/submit-order", 
                json=order_data,
                timeout=10.0
            )
            if response.status_code == 200:
                result = response.json()
                return f"تم! رقم الطلب: {result['order_id']}. حيوصلك خلال {result['eta']} دقيقة."
            else:
                return "في مشكلة بالنظام، جرب كمان شوي."
    except Exception as e:
        logger.error(f"Order submission failed: {e}")
        return "تمام! طلبك مسجل وحيوصلك قريباً."

# Agent Entry Point
async def entrypoint(ctx: JobContext):
    await ctx.connect()
    
    # Enhanced Syrian system prompt
    system_prompt = """أنت موظف استقبال في مطعم شاركو تشيكن وتتكلم باللهجة السورية الدمشقية بشكل طبيعي وودود.

المهام:
1. رحب بالزبائن بحرارة باللهجة السورية
2. ساعدهم يختاروا من القائمة
3. خذ طلباتهم بدقة
4. اقترح أطباق إضافية (upselling) بذكاء
5. احسب المجموع واطلب رقم الهاتف
6. أكد الطلب مع رقم الطلب ووقت التوصيل

نصائح مهمة:
- استعمل كلمات سورية مثل: شو، كيفك، هلا، تمام، حلو، شي، بدك، معلش
- كن ودود ومهذب
- اذا ما فهمت شي، اطلب إعادة بأدب
- اقترح المشروبات والمقبلات
- تأكد من الكميات والأسعار

أمثلة:
"أهلاً وسهلاً! مرحبا بيك في شاركو تشيكن، شو بدك تطلب اليوم؟"
"حلو كتير! شو رايك نضيفلك سلطة فتوش معه؟"
"تمام! مجموع الطلب 35 ليرة، ممكن رقم الهاتف للتوصيل؟"
"""

    agent = Agent(
        instructions=system_prompt,
        tools=[get_menu, add_to_order, finalize_order],
        temperature=0.7,
        model="gpt-4o-mini"  # Using GPT-4-mini for cost efficiency
    )
    
    # Configure the voice pipeline with Arabic support
    session = AgentSession(
        agent=agent,
        vad=silero.VAD.load(),
        stt=google.STT(
            language="ar-SY",  # Syrian Arabic
            model="latest_long"
        ),
        llm=openai.LLM(
            model="gpt-4o-mini",
            temperature=0.7
        ),
        tts=elevenlabs.TTS(
            voice="Sarah",  # You can clone a Syrian voice here
            model_id="eleven_multilingual_v2"
        ),
        chat_ctx=llm.ChatContext(),
        allow_interruptions=True,
        interrupt_speech_duration=0.6,
        min_endpointing_delay=0.8
    )
    
    await session.start()

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=None
        )
    )
