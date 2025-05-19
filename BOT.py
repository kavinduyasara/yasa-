import streamlit as st
from transformers import pipeline
import logging
import torch
import google.generativeai as genai
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, filename="app.log")
logger = logging.getLogger(__name__)

# Mock product database
PRODUCT_DB = {
    "laptop": [
        {"name": "Budget Gamer X", "price": 700, "features": "16GB RAM, GTX 1650, 512GB SSD"},
        {"name": "ProBook Ultra", "price": 1200, "features": "32GB RAM, RTX 3060, 1TB SSD"},
        {"name": "EcoLap", "price": 500, "features": "8GB RAM, Intel UHD, 256GB SSD"}
    ],
    "phone": [
        {"name": "SmartPhone A", "price": 300, "features": "4GB RAM, 64GB Storage"},
        {"name": "ProPhone Z", "price": 800, "features": "8GB RAM, 128GB Storage"}
    ]
}

# Initialize sentiment analysis pipeline
try:
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Using CPU for sentiment analysis.")
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        revision="714eb0f",
        device=0 if torch.cuda.is_available() else -1
    )
except Exception as e:
    logger.error(f"Failed to initialize sentiment analyzer: {e}")
    st.error("Error initializing sentiment analysis. Feedback features may be limited.")
    sentiment_analyzer = lambda x: [{"label": "UNKNOWN", "score": 0.0}]

# Initialize Gemini API with the provided key
try:
    gemini_api_key = "AIzaSyAXYXTHiNKJFgcn2jwnRtmme8F723Z6P6o"
    genai.configure(api_key=gemini_api_key)
    llm = genai.GenerativeModel("gemini-1.5-flash")
    logger.info("Successfully connected to Gemini API with provided key.")
except Exception as e:
    logger.error(f"Failed to initialize Gemini API: {e}")
    st.error(
        f"<div style='background-color: #ffebee; padding: 10px; border-radius: 5px; color: #c62828;'>"
        f"Error: Failed to connect to Gemini API. Please verify your API key at "
        f"<a href='https://aistudio.google.com/' target='_blank'>Google AI Studio</a> or check your network. "
        f"Details: {str(e)}</div>",
        unsafe_allow_html=True
    )
    st.stop()

# Chat history and session management
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "sidebar_state" not in st.session_state:
    st.session_state.sidebar_state = "expanded"
if "user_name" not in st.session_state:
    st.session_state.user_name = "Guest"
if "preferences" not in st.session_state:
    st.session_state.preferences = {"budget": 1000, "category": None}
if "previous_preferences" not in st.session_state:
    st.session_state.previous_preferences = []

# Function to interact with Gemini
def get_llm_response(user_input):
    history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.chat_history])
    prompt = f"You are a Smart Shopping Assistant. Greet {st.session_state.user_name} and use the conversation history to provide personalized product recommendations. History: {history}\nUser: {user_input}\nAssistant: "
    try:
        response = llm.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Error generating response with Gemini API: {e}")
        raise Exception(f"Failed to generate response: {str(e)}")

# Function to parse user query and extract preferences
def parse_query(query):
    try:
        query = query.lower()
        category = "laptop" if "laptop" in query else "phone" if "phone" in query else None
        budget = st.session_state.preferences["budget"] if "budget" not in query else (1000 if "budget" in query or "cheap" in query else 2000)
        preferences = {
            "gaming": "gaming" in query,
            "cheap": "budget" in query or "cheap" in query,
            "high-end": "high-end" in query or "premium" in query
        }
        return category, budget, preferences
    except Exception as e:
        logger.error(f"Error parsing query: {e}")
        return None, st.session_state.preferences["budget"], {}

# Function to recommend products
def recommend_products(category, budget, preferences):
    try:
        if not category or category not in PRODUCT_DB:
            return None, "Sorry, I don't have products for that category."
        
        recommendations = []
        for product in PRODUCT_DB[category]:
            if product["price"] <= budget:
                score = 1
                if preferences.get("gaming") and ("GTX" in product["features"] or "RTX" in product["features"]):
                    score += 2
                if preferences.get("cheap") and product["price"] < budget * 0.5:
                    score += 1
                if preferences.get("high-end") and product["price"] > budget * 0.75:
                    score += 1
                recommendations.append((product, score))
        
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:2], None
    except Exception as e:
        logger.error(f"Error recommending products: {e}")
        return None, "Error finding recommendations. Please try again."

# Function to analyze user feedback
def analyze_feedback(feedback):
    try:
        result = sentiment_analyzer(feedback)[0]
        return result["label"], result["score"]
    except Exception as e:
        logger.error(f"Error analyzing feedback: {e}")
        return "UNKNOWN", 0.0

# Streamlit app
def main():
    # Professional CSS with branding
    css = """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
        .main { 
            background: linear-gradient(135deg, #e3f2fd, #bbdefb); 
            color: #333; 
            font-family: 'Roboto', sans-serif; 
            padding: 20px; 
            min-height: 100vh; 
        }
        .header { 
            background: #1976d2; 
            color: white; 
            padding: 10px 20px; 
            border-radius: 10px; 
            margin-bottom: 20px; 
            display: flex; 
            justify-content: space-between; 
            align-items: center; 
        }
        .header .logo { font-size: 24px; font-weight: 700; }
        .stTextInput > div > input { 
            border-radius: 10px; 
            padding: 12px; 
            background-color: #f5f5f5; 
            color: #333; 
            border: 1px solid #ddd; 
            transition: all 0.3s ease; 
        }
        .stTextInput > div > input:focus { 
            border-color: #1976d2; 
            box-shadow: 0 0 5px rgba(25, 118, 210, 0.5); 
        }
        .stButton > button { 
            background: #1976d2; 
            color: white; 
            border-radius: 10px; 
            padding: 10px 20px; 
            border: none; 
            transition: all 0.3s ease; 
        }
        .stButton > button:hover { 
            background: #1565c0; 
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2); 
        }
        .chat-bubble { 
            border-radius: 15px; 
            padding: 12px 15px; 
            margin: 5px 0; 
            max-width: 70%; 
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); 
        }
        .user-bubble { 
            background: #bbdefb; 
            color: #333; 
            margin-left: auto; 
            border-bottom-right-radius: 0; 
        }
        .assistant-bubble { 
            background: #ffffff; 
            color: #333; 
            margin-right: auto; 
            border-bottom-left-radius: 0; 
        }
        .product-card { 
            background: #fff; 
            border-radius: 10px; 
            padding: 15px; 
            margin: 5px 0; 
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); 
            border-left: 4px solid #1976d2; 
        }
        .timestamp { 
            font-size: 0.7em; 
            color: #757575; 
            margin-top: 5px; 
        }
        .modal { 
            background: #fff; 
            border-radius: 10px; 
            padding: 20px; 
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); 
            max-width: 400px; 
            margin: 0 auto; 
        }
        .stMarkdown { color: #333; }
        .sidebar .sidebar-content { 
            background: #f5f5f5; 
            color: #333; 
        }
        .high-contrast { 
            filter: invert(90%) hue-rotate(180deg); 
        }
        @media (max-width: 600px) { 
            .chat-bubble, .product-card { max-width: 90%; } 
            .main { padding: 10px; } 
            .header { flex-direction: column; text-align: center; } 
        }
        </style>
    """
    st.markdown(css, unsafe_allow_html=True)

    # Header with logo placeholder
    st.markdown(
        "<div class='header'><div class='logo'>SmartShop AI</div><div><a href='mailto:support@smartshop.ai'>Contact Support</a></div></div>",
        unsafe_allow_html=True
    )

    # Sidebar with professional settings
    with st.sidebar:
        if st.button("☰ Menu"):
            st.session_state.sidebar_state = "expanded" if st.session_state.sidebar_state == "collapsed" else "collapsed"

        if st.session_state.sidebar_state == "expanded":
            st.title("⚙️ Settings")
            st.markdown("Personalize your experience!")

            # User profile
            st.session_state.user_name = st.text_input("Your Name", value=st.session_state.user_name, help="Enter your name for a personalized experience.")
            
            # Save preferences
            budget = st.number_input("Default Budget ($)", min_value=100, max_value=5000, value=st.session_state.preferences["budget"], step=100)
            if st.button("💾 Save Preferences"):
                # Store current preferences as previous before updating
                st.session_state.previous_preferences.append(st.session_state.preferences.copy())
                st.session_state.preferences = {"budget": budget, "category": st.session_state.preferences["category"]}
                st.success("Preferences saved!")
            
            # Revert to previous preferences
            if st.button("⏮️ Previous"):
                if st.session_state.previous_preferences:
                    st.session_state.preferences = st.session_state.previous_preferences.pop()
                    st.success("Reverted to previous preferences!")
                else:
                    st.warning("No previous preferences available.")

            # New Chat button
            if st.button("➕ New Chat"):
                st.session_state.chat_history = []
                st.session_state.messages = []
                st.success("Started a new chat!")

            # Clear history
            if st.button("🗑️ Clear History"):
                st.session_state.chat_history = []
                st.session_state.messages = []
                st.success("History cleared!")

            # Recent searches
            if st.session_state.messages:
                st.markdown("### 📋 Recent Searches")
                user_queries = [msg["content"] for msg in st.session_state.messages if msg["role"] == "user"]
                for i, query in enumerate(user_queries[:3], 1):
                    st.markdown(f"{i}. {query}")

            # Help center
            with st.expander("❓ Help Center"):
                st.markdown("""
                    **FAQs:**
                    - *How do I use this?* Enter a query like "budget gaming laptop".
                    - *What if I get an error?* Check your API key or network.
                    - *Need more help?* Contact us at support@smartshop.ai.

                    **Tips:**
                    - Be specific with preferences (e.g., "cheap phone with good storage").
                """)
                if st.button("Toggle Chatbot"):
                    st.write("Chatbot toggle feature coming soon!")

            # Accessibility
            if st.checkbox("High Contrast Mode"):
                st.markdown("<style>.main, .chat-bubble, .product-card, .modal { filter: invert(90%) hue-rotate(180deg); }</style>", unsafe_allow_html=True)

    # Main interface
    st.markdown("<h1 style='color: #1976d2;'>🛒 Smart Shopping Assistant</h1>", unsafe_allow_html=True)
    st.markdown(
        "Hello, {}! I'm your professional shopping assistant powered by Gemini 1.5 Flash. Let's find the perfect product! ✨".format(st.session_state.user_name)
    )

    # Display conversation history with avatars
    for message in st.session_state.messages:
        timestamp = datetime.now().strftime("%H:%M:%S")
        with st.container():
            if message["role"] == "user":
                st.markdown(
                    f"<div class='chat-bubble user-bubble'>{message['content']}</div>"
                    f"<div class='timestamp' style='text-align: right;'>{timestamp}</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div class='chat-bubble assistant-bubble'>{message['content']}</div>"
                    f"<div class='timestamp'>{timestamp}</div>",
                    unsafe_allow_html=True
                )

    # User input with form
    with st.form(key="user_input_form"):
        user_input = st.text_input(
            "What would you like to buy?",
            placeholder="e.g., budget gaming laptop",
            help="Enter a product category and preferences (e.g., budget, gaming, high-end)."
        )
        submit_button = st.form_submit_button("🔍 Search")

    if submit_button and user_input:
        logger.info(f"User input: {user_input}")
        if not any(msg["content"] == user_input for msg in st.session_state.messages):
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.container():
                timestamp = datetime.now().strftime("%H:%M:%S")
                st.markdown(
                    f"<div class='chat-bubble user-bubble'>{user_input}</div>"
                    f"<div class='timestamp' style='text-align: right;'>{timestamp}</div>",
                    unsafe_allow_html=True
                )

        with st.spinner("🤖 Processing your request..."):
            try:
                category, budget, preferences = parse_query(user_input)
                recommendations, error = recommend_products(category, budget, preferences)
                
                response = get_llm_response(user_input)
                if error:
                    response += f"\n{error}"
                elif recommendations:
                    response += "\n### Recommended Products:\n"
                    for product, score in recommendations:
                        st.markdown(
                            f"<div class='product-card'>"
                            f"<strong>{product['name']}</strong> (${product['price']})<br>"
                            f"{product['features']}<br>"
                            f"Match Score: {score} ✨"
                            f"<button style='margin-top: 10px; background: #1976d2; color: white; border: none; padding: 5px 10px; border-radius: 5px;' "
                            f"onclick='navigator.clipboard.writeText(\"{product['name']} - ${product['price']} - {product['features']}\");alert(\"Copied to clipboard!\");'>📋 Copy</button>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                else:
                    response += "\nNo products match your criteria. Please adjust your preferences! 😊"
                
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                with st.container():
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    st.markdown(
                        f"<div class='chat-bubble assistant-bubble'>{response}</div>"
                        f"<div class='timestamp'>{timestamp}</div>",
                        unsafe_allow_html=True
                    )
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                st.error(
                    f"<div style='background-color: #ffebee; padding: 10px; border-radius: 5px; color: #c62828;'>"
                    f"Error: Something went wrong. Please try again or contact support at <a href='mailto:support@smartshop.ai'>support@smartshop.ai</a>. "
                    f"Details: {str(e)}</div>",
                    unsafe_allow_html=True
                )

        # Feedback modal
        if st.button("➡️ Next Message"):
            with st.form(key="feedback_form"):
                st.markdown("<div class='modal' role='dialog' aria-label='Feedback Form'>", unsafe_allow_html=True)
                st.markdown("### Your Feedback Matters! 🌟")
                rating = st.number_input("Rate (1 to 5 stars)", min_value=1, max_value=5, value=3, format="%d", help="1 = Poor, 5 = Excellent")
                feedback = st.text_area("Share Your Thoughts:", placeholder="e.g., Great selection!", help="Your input helps us improve.", max_chars=500)
                if st.form_submit_button("Submit Feedback"):
                    if not feedback.strip():
                        st.error("Please provide some feedback.")
                    else:
                        sentiment, score = analyze_feedback(feedback)
                        st.success(f"Thank you for your feedback! Sentiment: **{sentiment}** (Confidence: {score:.2f})")
                        if sentiment == "NEGATIVE":
                            st.session_state.messages.append({"role": "assistant", "content": "Sorry you didn't like these! Can you specify what you're looking for?"})
                            st.session_state.chat_history.append({"role": "assistant", "content": "Sorry you didn't like these! Can you specify what you're looking for?"})
                            with st.container():
                                timestamp = datetime.now().strftime("%H:%M:%S")
                                st.markdown(
                                    f"<div class='chat-bubble assistant-bubble'>Sorry you didn't like these! Can you specify what you're looking for?</div>"
                                    f"<div class='timestamp'>{timestamp}</div>",
                                    unsafe_allow_html=True
                                )
                        st.session_state.messages.append({"role": "user", "content": f"Feedback: {feedback} (Rating: {rating} stars)"})
                st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    logger.info("Starting Streamlit app...")
    main()