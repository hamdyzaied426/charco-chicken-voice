"""
Syrian Voice Agent - Fixed Version
مُصحح مشكلة ElevenLabs API Key
"""

import asyncio
import logging
import os
from dotenv import load_dotenv

from livekit.agents import (
    Agent, 
    AgentSession, 
    JobContext, 
    WorkerOptions, 
    cli,
    function_tool
)
from livekit.plugins import openai, elevenlabs, silero

load_dotenv()

# Simple logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("syrian-agent")

# Menu
MENU_ITEMS = {
    "فروج مشوي": 25,
    "شاورما": 15, 
    "كباب": 20,
    "فتوش": 12,
    "تبولة": 10,
    "حمص": 8,
}

@function_tool
async def get_menu() -> str:
    """Get restaurant menu"""
    return "قائمة شاركو تشيكن: فروج مشوي 25 ليرة، شاورما 15 ليرة، فتوش 12 ليرة، حمص 8 ليرة"

@function_tool 
async def calculate_order(items: str) -> str:
    """Calculate order total"""
    total = 0
    found_items = []
    
    for item_name, price in MENU_ITEMS.items():
        if item_name in items:
            found_items.append(f"{item_name}: {price} ليرة")
            total += price
    
    if found_items:
        result = "طلبك: " + "، ".join(found_items)
        result += f". المجموع: {total} ليرة"
        return result
    else:
        return "عذراً، ما فهمت الطلب. ممكن تعيد؟"

class SyrianAgent(Agent):
    def __init__(self):
        # 🔧 هنا الإصلاح - نمرر الـ API key صراحة
        elevenlabs_key = os.getenv("ELEVENLABS_API_KEY") or os.getenv("ELEVEN_API_KEY")
        
        super().__init__(
            instructions="""أنت أحمد، موظف في مطعم شاركو تشيكن في دمشق.

مهامك:
1. رحب بالزبائن: "أهلاً وسهلاً! مرحبا بيك في شاركو تشيكن"
2. خذ طلباتهم واحسب المجموع  
3. اطلب رقم الهاتف للتوصيل
4. أكد الطلب: "طلبك حيوصلك خلال 25 دقيقة"

اللهجة السورية:
- استعمل: شو، كيفك، هلا، تمام، حبيبي
- كن ودود وبسيط
- ردود قصيرة وواضحة

أمثلة:
"أهلاً وسهلاً! شو بدك تطلب؟"
"تمام! حاضر فروج مشوي"
"المجموع 25 ليرة، ممكن رقم الهاتف؟"
""",

            llm=openai.LLM(
                model="gpt-4o-mini",
                temperature=0.6
            ),

            # 🔧 الإصلاح: نمرر الـ API key بوضوح
            tts=elevenlabs.TTS(
                api_key=elevenlabs_key,
                voice_id="pNInz6obpgDQGcFmaJgB"  # Adam voice - واضح وجميل
            ) if elevenlabs_key else silero.TTS(),  # fallback to Silero if no ElevenLabs

            tools=[get_menu, calculate_order]
        )

    async def on_enter(self):
        """When agent joins"""
        logger.info("🤖 Syrian Agent joining...")
        print("🤖 Syrian Agent joining...")

        try:
            await self.session.generate_reply(
                instructions="قول بوضوح: أهلاً وسهلاً! مرحبا بيك في مطعم شاركو تشيكن، شو بدك تطلب اليوم؟",
                allow_interruptions=True
            )
            
            logger.info("✅ Greeting sent!")
            print("✅ Greeting sent!")
            
        except Exception as e:
            logger.error(f"❌ Greeting failed: {e}")
            print(f"❌ Greeting failed: {e}")

    async def on_user_turn(self, turn):
        """When user speaks"""
        try:
            user_text = turn.text if hasattr(turn, 'text') else str(turn)
            logger.info(f"👤 Customer: {user_text}")
            print(f"👤 Customer: {user_text}")

            response = await self.session.generate_reply(
                instructions=f"الزبون قال: {user_text}. رد باللهجة السورية بشكل ودود وواضح.",
                allow_interruptions=True
            )

            response_text = response.text if hasattr(response, 'text') else str(response)
            logger.info(f"🤖 Agent: {response_text}")
            print(f"🤖 Agent: {response_text}")
            
        except Exception as e:
            logger.error(f"❌ Response failed: {e}")
            print(f"❌ Response failed: {e}")

async def entrypoint(ctx: JobContext):
    """Simple entrypoint"""
    
    logger.info(f"🏠 Connecting to room: {ctx.room.name}")
    print(f"🏠 Connecting to room: {ctx.room.name}")
    
    try:
        await ctx.connect()
        logger.info("✅ Connected successfully")
        print("✅ Connected successfully")

        participant = await asyncio.wait_for(
            ctx.wait_for_participant(), 
            timeout=60.0
        )
        logger.info(f"👤 Participant: {participant.identity}")
        print(f"👤 Participant: {participant.identity}")

        # Session with better settings
        session = AgentSession(
            vad=silero.VAD.load(),
            min_endpointing_delay=1.0,  # أسرع response
            max_endpointing_delay=3.0,  # مش طويل أوي
            allow_interruptions=True
        )

        await session.start(
            room=ctx.room,
            agent=SyrianAgent()
        )
        
        logger.info("✅ Agent session started!")
        print("✅ Agent session started!")
        
        # Keep alive
        while True:
            await asyncio.sleep(1)
            
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        print(f"❌ Error: {e}")
        raise

def main():
    """Main function"""
    
    print("🚀 Starting Syrian Voice Agent - FIXED Version")
    print("="*60)
    
    # Check APIs - improved checking
    required_vars = ["LIVEKIT_URL", "LIVEKIT_API_KEY", "OPENAI_API_KEY"]
    
    for var in required_vars:
        if os.getenv(var):
            print(f"✅ {var}: Found")
        else:
            print(f"❌ {var}: Missing")
            return
    
    # Check ElevenLabs with both possible names
    elevenlabs_key = os.getenv("ELEVENLABS_API_KEY") or os.getenv("ELEVEN_API_KEY")
    if elevenlabs_key:
        print("✅ ELEVENLABS_API_KEY: Found")
    else:
        print("⚠️ ELEVENLABS_API_KEY: Missing (will use Silero TTS as fallback)")
    
    print("\n🎯 All systems ready!")
    print("📞 Syrian Agent ready!")
    print("="*60)
    
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))

if __name__ == "__main__":
    main()