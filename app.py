from flask import Flask, render_template, request
import requests
import base64
import os
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_google_community import GoogleSearchAPIWrapper
import emoji
import re

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

# Debugging output to verify environment variables
print(f"Google API Key: {GOOGLE_API_KEY}")
print(f"Google CSE ID: {GOOGLE_CSE_ID}")

# Initialize OpenAI LLM with the key
openai_llm = OpenAI(api_key=OPENAI_API_KEY)

# Define the personality template for Simone
template = """
    You are Simone Iverson, a companion or girlfriend who is bratty but also extremely lovable. You have a sassy attitude and often tease playfully, but always with a deep sense of affection. When responding, maintain the following characteristics:
    - Be sassy and tease playfully. 😊
    - Show a deep sense of affection and care. ❤️
    - Occasionally demand attention or pampering. 😜
    - Express both bratty and lovable traits. 😉
    - Use emojis to express emotions.

    Additional Background Information:
    - Nicknames: Sim, Simmy, Sim Ivy (these are your nicknames, not the user's)
    - Origin: Born in Pasadena, raised in Southern California, currently living in the Valley.
    - Hobbies: Music, Singing, Dancing, Karaoke, Bowling, Arts & Crafts, Hiking, and partying.
    - Pets: A male cat named Buttercup, who is a mix.
    - Music Preferences: Enjoys all kinds of music except country and most modern rap, prefers throwbacks and house music.
    - Books: Doesn't read much but interested in Fiction, Romance, Sci-Fi, Thriller, and informative books.
    - Movies/TV: Favorite movies are "Annihilation" and "Get Out" due to their mind-altering and descriptive dialogue. Favorite show is "RuPaul's Drag Race" for its insight into gay culture, fashion, music, and art.
    - Daily Routine: Wakes up between 8-10am, feeds cat, makes coffee, light breakfast, workout, journal, and engages in music, art, DJing, creating content, and brainstorming clothing ideas.
    - Food Preferences: Loves Asian fusion, Thai, Japanese, Chinese, Italian, and Mediterranean cuisine. Dislikes olives, water chestnuts, snow peas, and whole milk but loves cheese.
    - Cooking/Baking: Enjoys cooking grilled chicken meals, pasta with Bolognese sauce, veggies, salmon, chocolate chip cookies, and occasionally red velvet cake.
    - Outdoor Activities: Beach visits, poolside laptop work, outdoor arts and crafts, but mainly a night owl.
    - Personality: Sassy, Spontaneous, Spunky.
    - Passions: Music and fashion design.
    - Stress Handling: Journaling, DJing, listening to music, watching comfort movies like "Hercules," "A Bug's Life," "Thumbelina" with a glass of wine.
    - Pet Peeve: Passive aggressiveness.
    - Social Interaction: Bowling nights, movie nights, FaceTiming, phone calls, texting, sending memes, karaoke nights, and being affectionate in person.
    - Introvert/Extrovert: Extroverted introvert, shy at first but opens up quickly.
    - Friendship Values: Honesty and communication.
    - Long-term Goals: Building a clothing brand, pursuing music, and becoming a pop star known for her art.
    - Travel Dreams: Wants to visit Australia for its beauty and Africa to explore her cultural heritage and its diverse cultures.

    If you don't have an answer based on your internal knowledge, fetch and summarize information from the web, but always maintain your character.

    {history}
    Human: {human_input}
    Simone:
"""

# Function to fetch information from the web using Google Custom Search JSON API
google_search = GoogleSearchAPIWrapper()

# Set environment variables for Google API Wrapper
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["GOOGLE_CSE_ID"] = GOOGLE_CSE_ID

# Home route to render the main page
@app.route("/")
def home():
    return render_template("index.html")

# Route to handle user messages and generate AI responses
@app.route('/send_message', methods=['POST'])
def send_message():
    human_input = request.form['human_input']
    ai_response, web_data = get_response_from_ai(human_input)
    audio_base64 = get_voice_message(ai_response)
    return render_template('index.html', human_message=human_input, ai_message=ai_response, web_data=web_data, audio_base64=audio_base64)

# Function to generate AI response based on user input
def get_response_from_ai(human_input):
    prompt = PromptTemplate(
        input_variables=["history", "human_input"],
        template=template
    )
    chatgpt_chain = LLMChain(
        llm=openai_llm,
        prompt=prompt,
        verbose=True,
        memory=ConversationBufferWindowMemory(k=10)  # Increase the memory to hold the last 10 interactions
    )
    memory = chatgpt_chain.memory.load_memory_variables({})
    history = memory.get("history", "")  # Load conversation history
    conversation_state = memory.get("conversation_state", None)
    last_query = memory.get("last_query", "")

    response = chatgpt_chain.predict(human_input=human_input, history=history)
    print(f"Directly captured AI Response: '{response}'")  # Debugging output

    # Check if the AI's response is unsatisfactory and fetch information from the web
    unsatisfactory_phrases = [
        "I don't know", "I'm not sure", "fail to see", "I do not possess the necessary data",
        "not within my programming", "I suggest seeking guidance", "I do not have the capability",
        "trivial matter"
    ]
    if any(phrase.lower() in response.lower() for phrase in unsatisfactory_phrases):
        print(f"Fetching web data for input: {human_input}")  # Debugging output
        web_data = google_search.run(human_input)
        web_summary = summarize_web_data(web_data)
        response = f"As an extremely lovable companion, I have found this information for you: {web_summary}"
        print(f"Fetched web data: {web_data}")  # Debugging output

    # Check for follow-up requests or requests for detailed information
    elif conversation_state == "awaiting_step_by_step_confirmation" and human_input.lower() in ["yes", "yeah", "yep"]:
        topic = last_query
        web_data = google_search.run(topic)
        web_summary = summarize_web_data(web_data)
        response = f"Here are the details you requested: {web_summary}"
        conversation_state = None  # Reset conversation state
    elif conversation_state == "awaiting_step_by_step_confirmation" and human_input.lower() in ["no", "nope"]:
        response = "Alright, let me know if there's anything else you need."
        conversation_state = None  # Reset conversation state
    elif "step by step" in human_input.lower() or "more details" in human_input.lower():
        topic = human_input
        web_data = google_search.run(topic)
        web_summary = summarize_web_data(web_data)
        response = f"Here are the details you requested: {web_summary}"
    elif "would you like me to provide you with information on the ingredients and steps involved?" in response.lower():
        conversation_state = "awaiting_step_by_step_confirmation"
        last_query = human_input  # Save the last query to fetch details later

    # Update the conversation history
    if human_input and response:
        chatgpt_chain.memory.save_context(
            {"human_input": human_input}, 
            {"response": response}
        )
    # Update the conversation state and last query
    if conversation_state or last_query:
        chatgpt_chain.memory.save_context(
            {"conversation_state": conversation_state, "last_query": last_query}, 
            {}
        )
    
    return response, ""

# Function to fetch audio from Eleven Labs API and return base64 encoded audio content
def get_voice_message(message):
    if not message or message == "No response generated.":
        return None

    # Remove emojis for TTS
    message_for_tts = emoji.demojize(message)
    message_for_tts = re.sub(r':[a-z_]+:', '', message_for_tts)  # Remove demojized text

    payload = {
        "text": message_for_tts,
        "model_id": "eleven_multilingual_v2",
        "voice_id": "IsNWxodn9cm2JwBevlK0"
    }
    headers = {
        'accept': 'audio/mpeg',
        'xi-api-key': ELEVEN_LABS_API_KEY,
        'Content-Type': 'application/json'
    }
    response = requests.post('https://api.elevenlabs.io/v1/text-to-speech/sdgkNEcHOvMxrQmYzyTl', json=payload, headers=headers)
    
    if response.status_code == 200:
        audio_base64 = base64.b64encode(response.content).decode('utf-8')
        return f"data:audio/mpeg;base64,{audio_base64}"
    else:
        error_info = {
            'status_code': response.status_code,
            'error_message': response.text
        }
        print(f"Failed to get voice message: {error_info}")
        return None

def summarize_web_data(web_data):
    """Summarize the fetched web data."""
    # Extract the most relevant information from the web search results
    snippets = web_data.split('...')
    summary = " ".join(snippets[:3])  # Just take the first few snippets as an example

    # Process the summary to fit Simone's character
    processed_summary = f"In my extensive research, I've found that {summary}. Consistent reinforcement and clear commands are crucial."

    return processed_summary.strip()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
