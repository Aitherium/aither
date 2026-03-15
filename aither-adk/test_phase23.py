"""Phase 2-3: LLM routing + AitherNode validation"""
import asyncio, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

async def main():
    from adk import AitherAgent
    from adk.llm import LLMRouter

    # 1. Effort-based routing
    print("-- Effort-Based Model Routing --")
    router = LLMRouter()
    for effort in [1, 3, 5, 7, 9]:
        m = router.model_for_effort(effort)
        print(f"  Effort {effort} -> {m}")

    # 2. List models (via LLM provider)
    print("\n-- Available Models --")
    try:
        models = router.list_models()
        for m in models[:10]:
            print(f"  {m}")
    except Exception as e:
        print(f"  list_models error: {e}")

    # 3. Live agent chat
    print("\n-- Live Agent Chat (vLLM) --")
    agent = AitherAgent("aither", builtin_tools=False)
    try:
        response = await agent.chat("Say hello in exactly 5 words.")
        print(f"  Content: {response.content[:200]}")
        print(f"  Model: {getattr(response, 'model', '?')}")
        print(f"  Usage: {getattr(response, 'usage', '?')}")
        print("  PASS: Live chat works")
    except Exception as e:
        print(f"  FAIL: {e}")

    # 4. Streaming test
    print("\n-- Streaming Chat --")
    try:
        chunks = []
        async for chunk in agent.chat_stream("Count from 1 to 5."):
            chunks.append(chunk)
        print(f"  Received {len(chunks)} chunks")
        full = "".join(str(c) for c in chunks)
        print(f"  Content: {full[:200]}")
        print("  PASS: Streaming works")
    except Exception as e:
        print(f"  FAIL: {e}")

    # 5. Multi-identity test
    print("\n-- Multi-Identity Agent Creation --")
    for name in ["lyra", "demiurge", "saga"]:
        try:
            a = AitherAgent(name, builtin_tools=False)
            r = await a.chat("Introduce yourself in one sentence.")
            print(f"  {name}: {r.content[:100]}")
        except Exception as e:
            print(f"  {name} FAIL: {e}")

asyncio.run(main())
