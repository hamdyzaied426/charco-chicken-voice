"""
Quick debug script to check connection stability
"""

import os
import asyncio
import aiohttp
from dotenv import load_dotenv

load_dotenv()

async def test_stable_connection():
    """Test if LiveKit connection is stable"""
    
    url = os.getenv("LIVEKIT_URL")
    if not url:
        print("❌ No LIVEKIT_URL found")
        return False
    
    print(f"🔗 Testing stable connection to: {url}")
    
    # Test multiple connections
    success_count = 0
    total_tests = 5
    
    for i in range(total_tests):
        try:
            print(f"   Test {i+1}/{total_tests}...", end=" ")
            
            async with aiohttp.ClientSession() as session:
                # Quick HTTP test
                test_url = url.replace("wss://", "https://")
                async with session.get(test_url, timeout=5) as response:
                    if response.status in [200, 404]:
                        print("✅")
                        success_count += 1
                    else:
                        print(f"❌ ({response.status})")
                        
            await asyncio.sleep(1)  # Wait between tests
            
        except Exception as e:
            print(f"❌ ({str(e)[:30]}...)")
    
    success_rate = (success_count / total_tests) * 100
    print(f"\n📊 Success Rate: {success_rate:.1f}% ({success_count}/{total_tests})")
    
    if success_rate >= 80:
        print("✅ Connection is stable enough for voice calls")
        return True
    else:
        print("⚠️ Connection might be unstable")
        return False

async def main():
    print("🔍 Connection Stability Test")
    print("="*40)
    
    stable = await test_stable_connection()
    
    print("\n" + "="*40)
    if stable:
        print("🎉 Ready for voice agent testing!")
        print("💡 Run: python agent_stable.py dev")
    else:
        print("⚠️ Consider checking internet connection")
        print("💡 Still try: python agent_stable.py dev")

if __name__ == "__main__":
    asyncio.run(main())