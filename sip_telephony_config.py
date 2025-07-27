"""
SIP and Telephony Configuration for Syrian Voice Agent
Enables real phone call support via multiple providers
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from livekit import rtc
from livekit.agents import JobContext, JobRequest
import os
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

class SIPTelephonyManager:
    """Manages SIP and telephony integration for voice agent"""
    
    def __init__(self):
        self.sip_config = self._load_sip_config()
        self.active_calls = {}
        
    def _load_sip_config(self) -> Dict[str, Any]:
        """Load SIP configuration from environment"""
        return {
            # LiveKit SIP configuration
            "livekit": {
                "url": os.getenv("LIVEKIT_URL"),
                "api_key": os.getenv("LIVEKIT_API_KEY"),
                "api_secret": os.getenv("LIVEKIT_API_SECRET"),
                "sip_trunk": os.getenv("LIVEKIT_SIP_TRUNK"),
                "phone_number": os.getenv("LIVEKIT_PHONE_NUMBER")
            },
            
            # Twilio SIP configuration  
            "twilio": {
                "account_sid": os.getenv("TWILIO_ACCOUNT_SID"),
                "auth_token": os.getenv("TWILIO_AUTH_TOKEN"),
                "phone_number": os.getenv("TWILIO_PHONE_NUMBER"),
                "webhook_url": os.getenv("TWILIO_WEBHOOK_URL")
            },
            
            # Vonage/Nexmo configuration
            "vonage": {
                "api_key": os.getenv("VONAGE_API_KEY"),
                "api_secret": os.getenv("VONAGE_API_SECRET"),
                "phone_number": os.getenv("VONAGE_PHONE_NUMBER")
            },
            
            # Custom SIP provider
            "custom_sip": {
                "server": os.getenv("SIP_SERVER"),
                "username": os.getenv("SIP_USERNAME"),
                "password": os.getenv("SIP_PASSWORD"),
                "port": int(os.getenv("SIP_PORT", "5060"))
            }
        }

class LiveKitSIPHandler:
    """LiveKit SIP integration - Primary choice for this project"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.room_name = "charco-chicken-calls"
        
    async def setup_sip_trunk(self):
        """Setup SIP trunk for incoming calls"""
        try:
            # LiveKit SIP configuration
            sip_config = {
                "trunk_id": "charco-chicken-trunk",
                "inbound_address": self.config["livekit"]["sip_trunk"],
                "phone_number": self.config["livekit"]["phone_number"],
                "room_config": {
                    "name": self.room_name,
                    "empty_timeout": 300,  # 5 minutes
                    "max_participants": 2   # Caller + Agent
                }
            }
            
            logger.info(f"SIP Trunk configured: {sip_config}")
            return sip_config
            
        except Exception as e:
            logger.error(f"SIP trunk setup failed: {e}")
            raise

    async def handle_incoming_call(self, call_context: Dict[str, Any]):
        """Handle incoming phone call"""
        caller_id = call_context.get("caller_id", "Unknown")
        logger.info(f"📞 Incoming call from: {caller_id}")
        
        # Create room for this call
        room_name = f"call-{caller_id}-{asyncio.get_event_loop().time()}"
        
        # Agent will join this room automatically
        return {
            "room_name": room_name,
            "caller_id": caller_id,
            "timestamp": asyncio.get_event_loop().time()
        }

class TwilioSIPHandler:
    """Twilio SIP integration - Alternative option"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    async def setup_webhook(self):
        """Setup Twilio webhook for call handling"""
        from twilio.rest import Client
        
        try:
            client = Client(
                self.config["twilio"]["account_sid"],
                self.config["twilio"]["auth_token"]
            )
            
            # Configure webhook URL
            webhook_config = {
                "voice_url": f"{self.config['twilio']['webhook_url']}/voice",
                "voice_method": "POST",
                "status_callback": f"{self.config['twilio']['webhook_url']}/status"
            }
            
            logger.info("Twilio webhook configured")
            return webhook_config
            
        except Exception as e:
            logger.error(f"Twilio setup failed: {e}")
            raise

    def generate_twiml_response(self, call_data: Dict[str, Any]) -> str:
        """Generate TwiML response to connect call to LiveKit"""
        
        # TwiML to bridge call to LiveKit room
        twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="alice" language="ar-SA">أهلاً وسهلاً في شاركو تشيكن</Say>
    <Connect>
        <Room>charco-chicken-{call_data.get('call_id', 'default')}</Room>
    </Connect>
</Response>"""
        
        return twiml

class AudioCodecManager:
    """Manages audio codecs for optimal quality"""
    
    SUPPORTED_CODECS = {
        "opus": {"quality": "high", "bandwidth": "low"},
        "g722": {"quality": "medium", "bandwidth": "medium"}, 
        "pcmu": {"quality": "low", "bandwidth": "high"},
        "pcma": {"quality": "low", "bandwidth": "high"}
    }
    
    @classmethod
    def get_optimal_codec(cls, connection_type: str = "mobile") -> str:
        """Get optimal codec based on connection type"""
        if connection_type == "mobile":
            return "opus"  # Best compression
        elif connection_type == "landline":
            return "g722"  # Good quality for landlines
        else:
            return "opus"  # Default

class CallQualityMonitor:
    """Monitors call quality and adjusts settings"""
    
    def __init__(self):
        self.quality_metrics = {
            "latency": [],
            "packet_loss": [],
            "jitter": [],
            "audio_quality": []
        }
    
    async def monitor_call_quality(self, room: rtc.Room):
        """Monitor ongoing call quality"""
        try:
            # Get connection stats
            stats = await room.engine.get_stats()
            
            for track_id, track_stats in stats.items():
                if hasattr(track_stats, 'remote_inbound_rtp'):
                    rtp_stats = track_stats.remote_inbound_rtp
                    
                    # Record metrics
                    self.quality_metrics["latency"].append(
                        getattr(rtp_stats, 'round_trip_time', 0)
                    )
                    self.quality_metrics["packet_loss"].append(
                        getattr(rtp_stats, 'packets_lost', 0)
                    )
                    self.quality_metrics["jitter"].append(
                        getattr(rtp_stats, 'jitter', 0)
                    )
            
            # Analyze and take action if needed
            await self._analyze_quality()
            
        except Exception as e:
            logger.error(f"Quality monitoring error: {e}")
    
    async def _analyze_quality(self):
        """Analyze quality metrics and adjust if needed"""
        if len(self.quality_metrics["latency"]) > 10:
            avg_latency = sum(self.quality_metrics["latency"][-10:]) / 10
            
            if avg_latency > 0.3:  # 300ms
                logger.warning(f"High latency detected: {avg_latency:.3f}s")
                # Could trigger codec change or other optimizations

# SIP Provider Factory
class SIPProviderFactory:
    """Factory to create SIP handlers based on configuration"""
    
    @staticmethod
    def create_handler(provider: str, config: Dict[str, Any]):
        """Create appropriate SIP handler"""
        
        if provider.lower() == "livekit":
            return LiveKitSIPHandler(config)
        elif provider.lower() == "twilio":
            return TwilioSIPHandler(config)
        else:
            raise ValueError(f"Unsupported SIP provider: {provider}")

# Main telephony integration function
async def setup_telephony_integration(ctx: JobContext):
    """Setup telephony integration for the voice agent"""
    
    # Initialize telephony manager
    telephony_manager = SIPTelephonyManager()
    
    # Choose primary provider (LiveKit recommended)
    primary_provider = "livekit"
    
    try:
        # Create SIP handler
        sip_handler = SIPProviderFactory.create_handler(
            primary_provider, 
            telephony_manager.sip_config
        )
        
        # Setup SIP trunk/webhook
        if primary_provider == "livekit":
            await sip_handler.setup_sip_trunk()
        elif primary_provider == "twilio":
            await sip_handler.setup_webhook()
        
        logger.info(f"✅ Telephony integration setup complete with {primary_provider}")
        
        return sip_handler
        
    except Exception as e:
        logger.error(f"❌ Telephony setup failed: {e}")
        # Fallback to text-only mode
        logger.info("🔄 Falling back to text-only mode")
        return None

# Usage example in main agent
"""
# In agent.py, modify the entrypoint:

async def entrypoint(ctx: JobContext):
    await ctx.connect()
    
    # Setup telephony first
    telephony_handler = await setup_telephony_integration(ctx)
    
    if telephony_handler:
        logger.info("📞 Phone calls enabled")
    else:
        logger.info("💬 Text-only mode")
    
    # Rest of agent setup...
    agent = Agent(...)
    session = AgentSession(...)
    
    await session.start()
"""

# Environment variables needed for .env file:
"""
# Add to .env file:

# LiveKit SIP Configuration
LIVEKIT_SIP_TRUNK=sip:your-trunk@livekit.cloud
LIVEKIT_PHONE_NUMBER=+1234567890

# Twilio Configuration (Alternative)
TWILIO_ACCOUNT_SID=your_twilio_sid
TWILIO_AUTH_TOKEN=your_twilio_token
TWILIO_PHONE_NUMBER=+1234567890
TWILIO_WEBHOOK_URL=https://your-server.com/webhooks

# Vonage Configuration (Alternative)
VONAGE_API_KEY=your_vonage_key
VONAGE_API_SECRET=your_vonage_secret
VONAGE_PHONE_NUMBER=+1234567890

# Custom SIP Provider
SIP_SERVER=sip.your-provider.com
SIP_USERNAME=your_username
SIP_PASSWORD=your_password
SIP_PORT=5060
"""
