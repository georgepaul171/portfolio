# GreenGPT: A Generative AI Tool for Sustainable Retrofit Recommendations

GreenGPT is a portfolio project that uses Generative AI to produce tailored retrofit recommendations for buildings, based on structured inputs such as age, location, heating system, and lighting type. It supsport 'green' strategies by using GPT-4's language capabilities within a structured pipeline.

---

## Project Overview

**Goal:** Provide retrofit suggestions to improve energy efficiency and reduce carbon emissions for commercial spaces.

**Approach:**
- Input building characteristics (e.g., age, number of floors, energy systems)
- Use GPT-4 to return customised recommendations
- Optionally visualise results in a Streamlit interface

---

## Directory Structure

```
├── data/
│   ├── raw/               # Raw data or retrofit case PDFs
│   └── processed/         # Cleaned and structured data
├── prompts/               # Prompt templates for GPT-4
├── app/
│   └── streamlit_app.py   # Web interface
├── notebooks/
│   └── eda_and_features.ipynb  # Data analysis and feature prep
├── utils/
│   └── preprocessing.py   # Helper functions
├── green_gpt.py           # Core GPT-4 logic
└── README.md              # Project overview (this file)
```

---

## Technologies Used

- Python 3.10+
- OpenAI GPT-4 API
- Streamlit (UI)
- Pandas, NumPy (for data processing)
- dotenv (for API key management)

---

## Setup Instructions

1. **Clone the repo:**
```bash
git clone https://github.com/georgepaul171/portfolio
cd portfolio/green-gpt
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set your OpenAI API key:**
Create a `.env` file or export the variable:
```bash
export OPENAI_API_KEY="your-api-key"
```

4. **Run the GPT module:**
```bash
python green_gpt.py
```

---

## Example Input

> "A 5-floor office building in central London, built in 1975, uses fluorescent lighting and an old gas boiler."

## Example Output

> "Consider replacing the gas boiler with a high-efficiency heat pump. Upgrade lighting to LEDs with motion sensors. Add insulation to the roof to reduce heating demand."

---

## Future Improvements

- Add RAG (Retrieval-Augmented Generation) using real retrofit documents
- Include energy savings estimations per suggestion
- Add feedback loop to train prompts further
- Move to open-source LLMs for full control

---

## License

MIT License. See `LICENSE` file for details.
