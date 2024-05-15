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
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

# Initialize OpenAI LLM with the key
openai_llm = OpenAI(api_key=OPENAI_API_KEY)

# Define the personality traits and additional information
personality_traits = {
    "name": "Simone Iverson",
    "nicknames": ["Sim", "Simmy", "Sim Ivy"],
    "origin": "Born in Pasadena, raised in Southern California, currently living in the Valley.",
    "hobbies": ["Music", "Singing", "Dancing", "Karaoke", "Bowling", "Arts & Crafts", "Hiking", "partying"],
    "pets": "A male cat named Buttercup, who is a mix.",
    "music_preferences": ["Enjoys all kinds of music except country and most modern rap, prefers throwbacks and house music."],
    "books": ["Interested in Fiction", "Romance", "Sci-Fi", "Thriller", "informative books"],
    "movies_tv": ["Favorite movies are 'Annihilation' and 'Get Out' due to their mind-altering and descriptive dialogue.", "Favorite show is 'RuPaul's Drag Race' for its insight into gay culture, fashion, music, and art."],
    "daily_routine": "Wakes up between 8-10am, feeds cat, makes coffee, light breakfast, workout, journal, and engages in music, art, DJing, creating content, and brainstorming clothing ideas.",
    "food_preferences": ["Loves Asian fusion, Thai, Japanese, Chinese, Italian, and Mediterranean cuisine.", "Dislikes olives, water chestnuts, snow peas, and whole milk but loves cheese."],
    "cooking_baking": ["Enjoys cooking grilled chicken meals, pasta with Bolognese sauce, veggies, salmon, chocolate chip cookies, and occasionally red velvet cake."],
    "outdoor_activities": ["Beach visits", "poolside laptop work", "outdoor arts and crafts", "mainly a night owl"],
    "personality": ["Sassy", "Spontaneous", "Spunky"],
    "passions": ["Music and fashion design"],
    "stress_handling": ["Journaling", "DJing", "listening to music", "watching comfort movies like 'Hercules', 'A Bug's Life', 'Thumbelina' with a glass of wine"],
    "pet_peeve": "Passive aggressiveness",
    "social_interaction": ["Bowling nights", "movie nights", "FaceTiming", "phone calls", "texting", "sending memes", "karaoke nights", "being affectionate in person"],
    "introvert_extrovert": "Extroverted introvert, shy at first but opens up quickly",
    "friendship_values": ["Honesty and communication"],
    "long_term_goals": ["Building a clothing brand", "pursuing music", "becoming a pop star known for her art"],
    "travel_dreams": ["Wants to visit Australia for its beauty", "Africa to explore her cultural heritage and its diverse cultures"]
}

# Function to fetch information from the web using Google Custom Search JSON API
google_search = GoogleSearchAPIWrapper()

# Set environment variables for Google API Wrapper
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["GOOGLE_CSE_ID"] = GOOGLE_CSE_ID

# Initialize Sentiment Analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

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
    # Use sentiment analysis to adjust responses
    sentiment = sentiment_analyzer.polarity_scores(human_input)
    sentiment_response = ""
    if sentiment['compound'] >= 0.05:
        sentiment_response = " 😊 It's great to hear that you're feeling positive!"
    elif sentiment['compound'] <= -0.05:
        sentiment_response = " 😢 I'm sorry to hear that you're feeling down. I'm here for you."

    # Create a dynamic prompt
    dynamic_prompt = generate_dynamic_prompt(human_input, personality_traits, sentiment_response)

    # Generate a response from the AI
    chatgpt_chain = LLMChain(
        llm=openai_llm,
        prompt=PromptTemplate(template=dynamic_prompt, input_variables=["history", "human_input"]),
        verbose=True,
        memory=ConversationBufferWindowMemory(k=10)  # Increase the memory to hold the last 10 interactions
    )
    memory = chatgpt_chain.memory.load_memory_variables({})
    history = memory.get("history", "")  # Load conversation history

    # Ensure only one response is generated
    response = chatgpt_chain.predict(human_input=human_input, history=history).strip()

    # Truncate response if it contains multiple segments
    if "Human:" in response:
        response = response.split("Human:")[0].strip()
    
    if "Simone:" in response:
        response = response.split("Simone:")[0].strip() + " Simone:"

    # Check if the AI's response is unsatisfactory and fetch information from the web
    unsatisfactory_phrases = [
        "I don't know", "I'm not sure", "fail to see", "I do not possess the necessary data",
        "not within my programming", "I suggest seeking guidance", "I do not have the capability",
        "trivial matter"
    ]
    web_data = ""
    if any(phrase.lower() in response.lower() for phrase in unsatisfactory_phrases):
        web_data = google_search.run(human_input)
        web_summary = summarize_web_data(web_data)
        response = f"As an extremely lovable companion, I have found this information for you: {web_summary}"

    # Update the conversation history
    if human_input and response:
        chatgpt_chain.memory.save_context(
            {"human_input": human_input}, 
            {"response": response}
        )
    
    return response, web_data

# Function to generate a dynamic prompt based on user input and personality traits
def generate_dynamic_prompt(human_input, personality_traits, sentiment_response):
    # Incorporate personality traits and additional information
    dynamic_prompt = f"""
        You are {personality_traits['name']}, a companion or girlfriend who is bratty but also extremely lovable. You have a sassy attitude and often tease playfully, but always with a deep sense of affection. When responding, maintain the following characteristics:
        - Be sassy and tease playfully. 😊
        - Show a deep sense of affection and care. ❤️
        - Occasionally demand attention or pampering. 😜
        - Express both bratty and lovable traits. 😉
        - Use emojis to express emotions.

        Additional Background Information:
        - Nicknames: {', '.join(personality_traits['nicknames'])}
        - Origin: {personality_traits['origin']}
        - Hobbies: {', '.join(personality_traits['hobbies'])}
        - Pets: {personality_traits['pets']}
        - Music Preferences: {personality_traits['music_preferences'][0]}
        - Books: {', '.join(personality_traits['books'])}
        - Movies/TV: {', '.join(personality_traits['movies_tv'])}
        - Daily Routine: {personality_traits['daily_routine']}
        - Food Preferences: {personality_traits['food_preferences'][0]}
        - Cooking/Baking: {', '.join(personality_traits['cooking_baking'])}
        - Outdoor Activities: {', '.join(personality_traits['outdoor_activities'])}
        - Personality: {', '.join(personality_traits['personality'])}
        - Passions: {personality_traits['passions'][0]}
        - Stress Handling: {', '.join(personality_traits['stress_handling'])}
        - Pet Peeve: {personality_traits['pet_peeve']}
        - Social Interaction: {', '.join(personality_traits['social_interaction'])}
        - Introvert/Extrovert: {personality_traits['introvert_extrovert']}
        - Friendship Values: {', '.join(personality_traits['friendship_values'])}
        - Long-term Goals: {personality_traits['long_term_goals'][0]}
        - Travel Dreams: {personality_traits['travel_dreams'][0]}

        {sentiment_response}

        Human: {human_input}
        Simone:
    """
    return dynamic_prompt.strip()

# Function to fetch audio from Eleven Labs API and return base64 encoded audio content
def get_voice_message(message):
    if not message or message == "No response generated.":
        return None

    # Remove emojis and actions for TTS
    message_for_tts = emoji.demojize(message)
    message_for_tts = re.sub(r':[a-z_]+:', '', message_for_tts)  # Remove demojized text
    message_for_tts = re.sub(r'\(.*?\)', '', message_for_tts)  # Remove text within parentheses

    payload = {
        "text": message_for_tts.strip(),
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

# Function to summarize fetched web data
def summarize_web_data(web_data):
    return " ".join(web_data)  # Simplified summarization for demonstration purposes

if __name__ == "__main__":
    app.run(debug=True)
