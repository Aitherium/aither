class CostCalculator:
    # Vendor cost markup — stored for reference; markup is applied separately
    # in AitherACTA.compute_request_cost() via SERVICE_MARKUP.
    # These prices are what WE pay the vendor, per 1M tokens (USD).
    MARKUP = 2.5  # Our target markup over vendor cost

    # Actual vendor costs per 1M tokens (USD) — sourced March 2026.
    # Benchmarks: Claude Opus 4.6 = $5/$25, Claude Sonnet 4.6 = $3/$15,
    # GPT-5.4 = $2.50/$15, Gemini 3.1 Pro = $2/$12, Gemini 2.5 Flash = $0.30/$2.50
    PRICING = {
        # ── Google Gemini 3.x (current frontier) ─────────────────────────
        "gemini-3.1-pro":         {"input": 2.00,  "output": 12.00},
        "gemini-3-flash":         {"input": 0.50,  "output": 3.00},
        "gemini-3-pro":           {"input": 2.00,  "output": 12.00},
        # ── Google Gemini 2.5 (stable, production) ───────────────────────
        "gemini-2.5-pro":         {"input": 1.25,  "output": 10.00},
        "gemini-2.5-flash":       {"input": 0.30,  "output": 2.50},
        "gemini-2.5-flash-lite":  {"input": 0.10,  "output": 0.40},
        # ── Google Gemini 2.0 (deprecated June 2026) ─────────────────────
        "gemini-2.0-flash":       {"input": 0.10,  "output": 0.40},
        "gemini-2.0-flash-lite":  {"input": 0.075, "output": 0.30},
        # ── Anthropic Claude (current) ────────────────────────────────────
        "claude-opus-4.6":        {"input": 5.00,  "output": 25.00},
        "claude-opus-4.5":        {"input": 5.00,  "output": 25.00},
        "claude-sonnet-4.6":      {"input": 3.00,  "output": 15.00},
        "claude-sonnet-4.5":      {"input": 3.00,  "output": 15.00},
        "claude-haiku-4.5":       {"input": 1.00,  "output": 5.00},
        # ── OpenAI (current) ──────────────────────────────────────────────
        "gpt-5.4":                {"input": 2.50,  "output": 15.00},
        "gpt-5-mini":             {"input": 0.25,  "output": 2.00},
        "gpt-4.1":                {"input": 2.00,  "output": 8.00},
        "gpt-4.1-mini":           {"input": 0.40,  "output": 1.60},
        # ── DeepSeek (high value, cost-efficient) ────────────────────────
        "deepseek-v3.2":          {"input": 0.56,  "output": 1.68},
        "deepseek-r1":            {"input": 1.35,  "output": 5.40},
    }

    @staticmethod
    def calculate(model_name, input_tokens, output_tokens):
        # Normalize model name
        base_model = "gemini-2.0-flash" # Default
        for key in CostCalculator.PRICING:
            if key in model_name.lower():
                base_model = key
                break

        rates = CostCalculator.PRICING.get(base_model, CostCalculator.PRICING["gemini-2.0-flash"])

        input_cost = (input_tokens / 1_000_000) * rates["input"]
        output_cost = (output_tokens / 1_000_000) * rates["output"]

        return input_cost + output_cost
