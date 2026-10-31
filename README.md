# MindAI — Voice Agent

A minimalist, touch-driven voice agent powered by the OpenAI Agents SDK and Streamlit.

## Features
- **Centered 3D Iridescent Figure**: Translucent, responsive visual core.
- **Touch-To-Talk Interaction**: Tap the figure to listen, tap to answer.
- **Dynamic Backgrounds**: Floating aurora gradients, soundwave rings, and stardust particles.
- **API Key Security**: On-screen gateway or environment secret authentication.

## Local Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file with your OpenAI API key:
   ```bash
   OPENAI_API_KEY=your_key_here
   ```
5. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## Deploy to Streamlit Community Cloud
1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Select your repository, set the main file path to `app.py`.
4. In **Advanced Settings -> Secrets**, add:
   ```toml
   OPENAI_API_KEY = "sk-..."
   ```
5. Click **Deploy**!
