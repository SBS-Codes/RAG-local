"""
Knowledge Base Module
=====================

Provides sample educational text, metadata, and curated questions for testing
the Retrieval-Augmented Generation (RAG) pipeline.
"""

from pathlib import Path
from typing import List, Dict, Any

# Current file directory
BASE_DIR = Path(__file__).resolve().parent
TEXT_FILE_PATH = BASE_DIR / "knowledge_base.txt"

# Fallback default text if file is not found
DEFAULT_DOCUMENT = """The Exploration of Mars: The Red Planet, Water, and Robotic Missions

Mars is the fourth planet from the Sun and the second-smallest planet in the Solar System. It is frequently referred to as the "Red Planet" because the iron oxide (rust) prevalent on its surface gives it a distinct reddish appearance. Mars possesses a very thin atmosphere composed of approximately 95% carbon dioxide, 2.6% molecular nitrogen, 1.9% argon, and trace amounts of water vapor and oxygen. Due to this thin atmosphere, atmospheric pressure at the Martian surface averages less than 1% of Earth's pressure, making it impossible for liquid water to exist stably on the surface for extended periods.

Despite its dry conditions today, evidence gathered by space probes shows that ancient Mars was once much warmer and wetter. Massive outflow channels, dried lake basins, and river valley networks carved across the landscape indicate that liquid water once flowed freely billions of years ago. Today, water on Mars exists primarily in the form of ice. Vast ice deposits are concentrated in the permanent polar ice caps and trapped beneath the shallow mantle of Martian soil. Radar sounding instruments aboard orbiting spacecraft have also detected potential deposits of subglacial liquid water deep beneath the southern polar ice sheet.

Robotic explorers have revolutionized our understanding of Mars. In 2012, NASA landed the Curiosity rover in Gale Crater to investigate whether Mars could ever have supported microbial life. Curiosity discovered chemical evidence of ancient freshwater lakes with the essential elements required for life, including carbon, nitrogen, and oxygen. In February 2021, the Perseverance rover touched down in Jezero Crater, an ancient river delta. Perseverance is equipped to drill and collect Martian rock and soil cores for eventual return to Earth. Accompanying Perseverance was Ingenuity, a solar-powered helicopter drone that achieved the historic first powered, controlled aerodynamic flight on another planet.

The geography of Mars features some of the most extreme geological formations in the Solar System. Olympus Mons is an enormous shield volcano rising roughly 22 kilometers (nearly 72,000 feet) high, making it nearly three times taller than Mount Everest. Another prominent feature is Valles Marineris, an immense system of canyons stretching over 4,000 kilometers along the Martian equator, which is more than four times deeper and ten times longer than the Grand Canyon. Mars also has two small, irregularly shaped moons named Phobos and Deimos, which scientists believe are captured asteroids from the nearby asteroid belt.

Planning crewed expeditions to Mars presents severe engineering and biological challenges. Astronauts traveling to Mars face transit times between six and nine months each way, exposing them to high levels of cosmic radiation and prolonged microgravity that causes bone density loss. On the surface, astronauts must endure frigid temperatures averaging minus 60 degrees Celsius (-80 degrees Fahrenheit) and violent global dust storms that can shroud the entire planet for months at a time, completely blocking sunlight needed for solar arrays. Developing reliable life-support systems, radiation shielding, and in-situ resource utilization for producing fuel and water will be essential for human survival.
""".strip()


def load_knowledge_base(file_path: Path = TEXT_FILE_PATH) -> str:
    """Loads document text from a text file, or returns default text if file is missing."""
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return DEFAULT_DOCUMENT


# Curated questions to demonstrate retrieval precision
SAMPLE_QUESTIONS: List[Dict[str, Any]] = [
    {
        "question": "Why is Mars called the Red Planet, and what is its atmosphere made of?",
        "topic": "Atmosphere and Surface Color",
        "expected_concept": "Iron oxide (rust) and 95% carbon dioxide"
    },
    {
        "question": "Where does water exist on Mars today, and did it ever have liquid water?",
        "topic": "Water and Ice",
        "expected_concept": "Polar ice caps, shallow soil, ancient lake basins, and subglacial radar detections"
    },
    {
        "question": "What did the Curiosity and Perseverance rovers discover on Mars?",
        "topic": "Robotic Exploration",
        "expected_concept": "Curiosity found ancient freshwater lake evidence in Gale Crater; Perseverance landed in Jezero Crater with Ingenuity"
    },
    {
        "question": "What is Olympus Mons, and how does it compare to mountains on Earth?",
        "topic": "Martian Geography",
        "expected_concept": "Shield volcano ~22 km high, nearly 3 times taller than Mount Everest"
    },
    {
        "question": "What are the biggest challenges humans will face on a mission to Mars?",
        "topic": "Human Spaceflight Challenges",
        "expected_concept": "6-9 month transit radiation, bone density loss, -60°C temperature, and planet-wide dust storms"
    },
    {
        "question": "What is the diameter of Jupiter's Great Red Spot?",
        "topic": "Negative Test (Out of Context)",
        "expected_concept": "The model should state that this information is not found in the context (preventing hallucination)"
    }
]
