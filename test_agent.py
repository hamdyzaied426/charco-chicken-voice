import asyncio
import logging
from dotenv import load_dotenv
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.plugins import openai, elevenlabs, silero

load_dotenv()
logger = logging.getLogger("test-agent")

class SimpleAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""أنت موظف في مطعم شاركو تشيكن. 
            قول للزبون: أهلاً وسهلاً! مرحبا بيك في شاركو تشيكن، شو بدك تطلب اليوم؟
            تكلم باللهجة السورية وكن ودود.""",
            
            # Simplified TTS - no STT for now
            llm=openai.LLM(model="gpt-4o-mini", temperature=0.7),
            
            tts=elevenlabs.TTS(
                voice="Sarah",
                model_id="eleven_multilingual_v2"
            ),
        )
    
    async def on_enter(self):
        logger.info("🤖 Agent joined the call!")
        print("🤖 Agent joined the call!")
        
        # Send immediate greeting
        await self.session.generate_reply(
            instructions="قول مرحبا بالزبون بحرارة باللهجة السورية",
            allow_interruptions=True
        )
        
        logger.info("✅ Greeting sent!")
        print("✅ Greeting sent!")

async def entrypoint(ctx: JobContext):
    logger.info(f"🏠 Connecting to room: {ctx.room.name}")
    print(f"🏠 Connecting to room: {ctx.room.name}")
    
    await ctx.connect()
    
    participant = await ctx.wait_for_participant()
    logger.info(f"👤 Participant joined: {participant.identity}")
    print(f"👤 Participant joined: {participant.identity}")
    
    session = AgentSession(
        vad=silero.VAD.load(),
        min_endpointing_delay=1.0,
        max_endpointing_delay=3.0,
        allow_interruptions=True,
    )
    
    try:
        await session.start(ctx.room, SimpleAgent())
        logger.info("✅ Session started successfully!")
        print("✅ Session started successfully!")
        
    except Exception as e:
        logger.error(f"❌ Session failed: {e}")
        print(f"❌ Session failed: {e}")
        raise

if __name__ == "__main__":
    # Enable debug logging
    logging.basicConfig(level=logging.INFO)
    
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))