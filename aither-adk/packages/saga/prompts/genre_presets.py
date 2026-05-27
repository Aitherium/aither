"""Genre presets for Saga — tone, setting, and starter prompts."""

GENRE_PRESETS = {
    "fantasy_epic": {
        "name": "Fantasy Epic",
        "tone": "balanced",
        "magic": "high",
        "technology": "medieval",
        "starter": (
            "In an age of legends, when dragons still soared above the mountain peaks "
            "and magic flowed like rivers through the land..."
        ),
    },
    "dark_fantasy": {
        "name": "Dark Fantasy",
        "tone": "grimdark",
        "magic": "moderate",
        "technology": "medieval",
        "starter": (
            "The sun has not risen in three years. In the endless twilight, humanity "
            "clings to survival in fortress-cities..."
        ),
    },
    "romance": {
        "name": "Romantic Fantasy",
        "tone": "balanced",
        "magic": "moderate",
        "technology": "renaissance",
        "starter": (
            "Across the ballroom floor, their eyes met for the first time — a moment "
            "that would change everything..."
        ),
    },
    "litrpg": {
        "name": "LitRPG Adventure",
        "tone": "balanced",
        "magic": "high",
        "technology": "medieval",
        "starter": (
            "[System Initialized] Welcome, Traveler. Your journey begins now. "
            "Current Level: 1. Class: Unassigned..."
        ),
    },
    "scifi": {
        "name": "Sci-Fi Adventure",
        "tone": "balanced",
        "magic": "none",
        "technology": "futuristic",
        "starter": (
            "The jump drive hummed to life. Through the viewport, the stars stretched "
            "into brilliant lines as the ship prepared for hyperspace..."
        ),
    },
    "cyberpunk": {
        "name": "Cyberpunk",
        "tone": "dark",
        "magic": "none",
        "technology": "futuristic",
        "starter": (
            "The neon signs of Night City reflected off rain-slicked streets. In the "
            "shadows, a deal was about to go very wrong..."
        ),
    },
    "mystery": {
        "name": "Mystery & Thriller",
        "tone": "dark",
        "magic": "low",
        "technology": "modern",
        "starter": (
            'The letter arrived at midnight. Three words, written in blood: '
            '"Remember what happened."'
        ),
    },
    "horror": {
        "name": "Horror",
        "tone": "grimdark",
        "magic": "low",
        "technology": "modern",
        "starter": (
            "The old house groaned in the wind. But there was no wind tonight. "
            "Something else was moving within the walls..."
        ),
    },
    "historical": {
        "name": "Historical Drama",
        "tone": "balanced",
        "magic": "none",
        "technology": "renaissance",
        "starter": (
            "The year was 1789. As revolution swept through France, a secret society "
            "gathered in the shadows of Notre-Dame..."
        ),
    },
    "isekai": {
        "name": "Isekai",
        "tone": "light",
        "magic": "epic",
        "technology": "medieval",
        "starter": (
            "One moment you were crossing the street. The next, you awoke in a meadow "
            "under two moons. A floating message appeared: [Welcome to Aetheria]..."
        ),
    },
}
