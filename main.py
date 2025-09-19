from fastapi import FastAPI, Query, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import requests
import httpx
from pydantic import BaseModel
from dotenv import load_dotenv
import os, random, io, csv
import shutil
import uuid
from datetime import datetime
import zipfile
from rembg import remove
from PIL import Image
from tempfile import NamedTemporaryFile
import shutil
from random import randint
import pandas as pd
from pathlib import Path
from typing import List, Optional
import pillow_heif
from io import BytesIO
import base64

load_dotenv()

open_weather_key = os.getenv('OPEN_WEATHER_KEY')
generate_api_url = os.getenv('GENERATE_API_URL')
whatbeats_api_url = os.getenv('WHATBEATS_API_URL')
api_key = os.getenv('GENERATE_API_KEY')
generate_note_prompt = os.getenv('GENERATE_NOTE_PROMPT')
generate_image_api_url = os.getenv('GENERATE_IMAGE_API_URL')

# Ollama configuration
OLLAMA_BASE_URL = "http://host.docker.internal:11434"  # Default Ollama port
OLLAMA_DEFAULT_MODEL = "llama3.1:8b"  # Default model to use

# Chat history configuration
CHAT_HISTORY_DIR = "/app/chat_history"
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)

app = FastAPI()

with open('./data/wordle-words.txt', 'r') as file:
    words = file.readlines()
words = [word.strip() for word in words]

class MessageRequest(BaseModel):
    category: str
    
class LCDMessageRequest(BaseModel):
    message: str

class WeatherRequest(BaseModel):
    city: str

class GenerateRequest(BaseModel):
    message: str
    key: str
class whatBeatsRequest(BaseModel):
    key: str
    current_object: str
    player_input:str

class GenerateNote(BaseModel):
    student_name: str
    previous_note: str
    concepts: str
    key:str
    
class GenerateImage(BaseModel):
    prompt: str
    key: str

class WaitlistRequest(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phone: Optional[str] = None
    email: str

class ChatMessage(BaseModel):
    role: str  # "user", "assistant", "system"
    content: str

class OllamaChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    conversation_id: Optional[str] = None
    stream: Optional[bool] = False

class OllamaGenerateRequest(BaseModel):
    model: str
    prompt: str
    stream: Optional[bool] = False

class SimpleChatRequest(BaseModel):
    message: str
    stream: Optional[bool] = False

class SimpleGenerateRequest(BaseModel):
    prompt: str
    stream: Optional[bool] = False

class ConversationInfo(BaseModel):
    conversation_id: str
    created_at: str
    last_updated: str
    message_count: int
    title: str
    last_message_date: Optional[str] = None
    last_message_time: Optional[str] = None

def parse_json_from_string(string_with_json):
    start_index = string_with_json.find('{')
    if start_index == -1:
        raise ValueError("No JSON object found in the string")
    end_index = string_with_json.rfind('}') + 1
    if end_index == 0:
        raise ValueError("Invalid JSON format: No closing '}' found")
    json_string = string_with_json[start_index:end_index]
    parsed_json = json.loads(json_string)
    return parsed_json

# Chat history helper functions
def get_conversation_file_path(conversation_id: str) -> str:
    """Get the file path for a conversation"""
    return os.path.join(CHAT_HISTORY_DIR, f"{conversation_id}.json")

def save_conversation(conversation_id: str, messages: list, title: str = None):
    """Save conversation to JSON file"""
    conversation_data = {
        "conversation_id": conversation_id,
        "created_at": datetime.utcnow().isoformat() if not os.path.exists(get_conversation_file_path(conversation_id)) else None,
        "last_updated": datetime.utcnow().isoformat(),
        "title": title or f"Conversation {conversation_id[:8]}",
        "model": OLLAMA_DEFAULT_MODEL,
        "messages": messages
    }
    
    # If file exists, preserve creation date and title
    file_path = get_conversation_file_path(conversation_id)
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
            conversation_data["created_at"] = existing_data.get("created_at", conversation_data["created_at"])
            conversation_data["title"] = existing_data.get("title", conversation_data["title"])
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(conversation_data, f, indent=2, ensure_ascii=False)

def load_conversation(conversation_id: str) -> dict:
    """Load conversation from JSON file"""
    file_path = get_conversation_file_path(conversation_id)
    if not os.path.exists(file_path):
        return {"messages": [], "conversation_id": conversation_id}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def list_conversations() -> List[ConversationInfo]:
    """List all conversations"""
    conversations = []
    for filename in os.listdir(CHAT_HISTORY_DIR):
        if filename.endswith('.json'):
            conversation_id = filename[:-5]  # Remove .json extension
            try:
                with open(os.path.join(CHAT_HISTORY_DIR, filename), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    messages = data.get("messages", [])
                    
                    # Get last message timestamp info
                    last_message_date = None
                    last_message_time = None
                    if messages:
                        last_message = messages[-1]
                        last_message_date = last_message.get("date")
                        last_message_time = last_message.get("time")
                    
                    conversations.append(ConversationInfo(
                        conversation_id=conversation_id,
                        created_at=data.get("created_at", ""),
                        last_updated=data.get("last_updated", ""),
                        message_count=len(messages),
                        title=data.get("title", f"Conversation {conversation_id[:8]}"),
                        last_message_date=last_message_date,
                        last_message_time=last_message_time
                    ))
            except (json.JSONDecodeError, KeyError):
                continue
    
    # Sort by last_updated descending
    conversations.sort(key=lambda x: x.last_updated, reverse=True)
    return conversations

def generate_conversation_title(first_message: str) -> str:
    """Generate a title from the first message"""
    # Take first 50 characters and clean it up
    title = first_message[:50].strip()
    if len(first_message) > 50:
        title += "..."
    return title

def create_timestamped_message(role: str, content: str) -> dict:
    """Create a message with timestamp information"""
    now = datetime.utcnow()
    return {
        "role": role,
        "content": content,
        "timestamp": now.isoformat(),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S UTC")
    }


# Add CORS middleware
origins = [
    "http://localhost",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "https://coderlab.work",
    "https://elianrenteria.github.io",
    "https://elianrenteria.dev",
    "76.176.106.64",
    "https://modern-spaniel-locally.ngrok-free.app",
    "https://test.checkrx.com",
    "https://staging.checkrx.com",
    "https://app.checkrx.com",
    "https://checkrx.com"
]
#origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the states from the CSV file into a list
def load_states():
    states = []
    with open("states.csv", "r") as file:
        reader = csv.reader(file)
        for row in reader:
            states.append(row[0])  # Assuming each state is in the first column
    return states

# Initialize the list of states
states = load_states()

@app.get("/api/state")
async def pick_state():
    random_state = random.choice(states)
    return {"state": random_state}

@app.get("/api/wordle")
async def pick_word():
    global words
    random_word = random.choice(words)
    return {"word": random_word}

@app.get("/api/validWordleWord")
async def is_valid_word(word: str = Query(...)):
    global words
    if word.lower() in words:
        return {"isValid": True}
    return {"isValid": False}


@app.post("/api/trivia")
async def generate_question(request: MessageRequest):
    category = request.category
    message = "Generate me a random trivia question with the correct answer and 3 false answers, The Topic should be " + category + " and respond ONLY in json format as given here: {\"question\":\"\",\"answers\":[\"\",\"\", \"\", \"\"]} for the answers value it should be an array where the first index is the correct answer."
    response = requests.post(generate_api_url, json={"message": message, "key": api_key})
    return parse_json_from_string(response.json()["response"])

@app.post("/api/generate")
async def gernerate(request: GenerateRequest):
    if request.key == api_key:
        response = requests.post(generate_api_url, json={"message": request.message})
        return response.json()["response"]
    return {"error": "Invalid API Key"}


@app.post("/api/whatbeats")
async def what_beats(request: whatBeatsRequest):
    if request.key == api_key:
        m = f"""You are an AI for a game called "What Beats Rock?" The game works as follows:
- The player submits an object that they believe can "beat" the current object.
- You must determine if their input is valid or invalid based on basic logic and reasoning.
- If valid, accept it and provide a short sentence creative or logical explanation for why it wins.
- If invalid, reject it and explain why in a short sentence.
- The accepted input becomes the new object for the next round.

Current object: {request.current_object}  
Player's input: {request.player_input}  

Respond in this format:
- **Validity:** "Accepted" or "Rejected"  
- **Explanation:** A short, fun, or logical reason why it beats or fails."  

Make sure responses are engaging, fun, and logical!
"""
        response = requests.post(whatbeats_api_url, json={"message": m})
        return response.json()["response"]
    return {"error": "Invalid API Key"}


@app.post("/api/note")
async def generate_note(request: GenerateNote):
    message = f"{generate_note_prompt}\nname: {request.student_name}; \nprevious note: {request.previous_note}; \nconcepts: {request.concepts}"
    response = requests.post(generate_api_url, json={"message": message, "key": request.key})
    return response.json()["response"]

@app.get("/api/weather")
async def get_weather(city: str = Query(...)):
    try:
        geolocation_response = requests.get(f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={open_weather_key}")
        geolocation_data = geolocation_response.json()
        if not geolocation_data:
            return {"error": "geolocation api failed"}
        else:
            print(geolocation_data)

        lat = geolocation_data[0]["lat"]
        lon = geolocation_data[0]["lon"]

        weather_response = requests.get(f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={open_weather_key}")
        weather_response.raise_for_status()
        weather_data = weather_response.json()

        return {"weather": weather_data["main"], "wind": weather_data["wind"], "misc": weather_data["weather"]}

    except Exception as e:
        print(e)
        return {"error":  "request failed"}


@app.get("/api/fact")
async def get_fact():
    try:
        response = requests.get("https://uselessfacts.jsph.pl/api/v2/facts/random?language=en")
        return {"fact": response.json()["text"]}
    except():
        return {"Error": "facts api error"}

UPLOAD_DIR = "/app/shared_files"

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)


@app.post("/api/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    return {"filenames": [file.filename for file in files]}


@app.get("/api/download")
def download_all_files():
    zip_filename = "all_files.zip"
    zip_filepath = os.path.join(UPLOAD_DIR, zip_filename)

    with zipfile.ZipFile(zip_filepath, 'w') as zipf:
        for root, _, files in os.walk(UPLOAD_DIR):
            for file in files:
                if file != zip_filename:  # Avoid zipping the zip file itself
                    zipf.write(os.path.join(root, file), file)

    return FileResponse(zip_filepath, filename=zip_filename)


@app.post("/api/clear")
def clear_files():
    for root, dirs, files in os.walk(UPLOAD_DIR):
        for file in files:
            os.remove(os.path.join(root, file))
    return JSONResponse(content={"message": "All files have been deleted."})


@app.post("/api/remove-bg")
async def remove_bg(image: UploadFile = File(...)):
    contents = await image.read()
    input_image = Image.open(io.BytesIO(contents))

    # Remove the background
    output_image = remove(input_image)

    # Save the output image to a temporary file
    with NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        output_image.save(tmp, format="PNG")
        tmp_path = tmp.name

    # Return the image as a file response
    return FileResponse(tmp_path, media_type="image/png", filename="processed-image.png")

POKE_API_BASE_URL = "https://pokeapi.co/api/v2/pokemon/"

@app.get("/api/pokemon")
async def get_random_pokemon():
    variables = {
        "id": randint(1, 1025)
    }

    URL = "https://beta.pokeapi.co/graphql/v1beta"
    query = """
    query samplePokeAPIquery($id: Int!) {
      pokemon_v2_pokemon(where: {id: {_eq: $id}}) {
        name
        pokemon_v2_pokemonsprites {
          sprites(path: "front_default")
        }
      }
    }
    """
    response = requests.post(URL, json={"query": query, "variables": variables}).json()
    payload = {
        "name": response["data"]["pokemon_v2_pokemon"][0]["name"],
        "image": response["data"]["pokemon_v2_pokemon"][0]["pokemon_v2_pokemonsprites"][0]["sprites"]
    }
    return payload
    '''
    # Total number of Pokémon in the API (you could make this dynamic if necessary)
    max_pokemon_id = 1025  # As of Gen 9

    # Generate a random Pokémon ID
    random_id = random.randint(1, max_pokemon_id)

    # Fetch Pokémon data from PokéAPI
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{POKE_API_BASE_URL}{random_id}/")

    if response.status_code == 200:
        data = response.json()
        pokemon_name = data['name']
        pokemon_image = data['sprites']['front_default']

        # Return the Pokémon name and image
        return {
            "name": pokemon_name,
            "image": pokemon_image
        }
    else:
        return {"error": "Failed to fetch data from PokeAPI"}
    '''

LCDMessage = ""

@app.post("/api/set_message")
async def set_message(request: LCDMessageRequest):
    if len(request.message) > 16:
        return {"error": "Message is too long. Max 16 characters allowed."}
    else:
        global LCDMessage
        LCDMessage = request.message
        return {"message": LCDMessage}
    
@app.get("/api/get_message")
async def get_message():
    global LCDMessage
    return {"message": LCDMessage}


path = "./starbucks_drinks.csv"
data = pd.read_csv(path)

@app.get("/api/starbucks")
async def get_random_starbucks_drink():
    # Pick a new random row each time the endpoint is called
    random_row = data.sample(n=1)
    drink_data = json.loads(random_row.to_json(orient='records'))[0]
    return {"drink": drink_data, "image": image_data[drink_data["Beverage"]]}  # Use 'records' for better formatting

# Path to store the images
PHOTO_DIR = "./photos"
METADATA_FILE = os.path.join(PHOTO_DIR, "metadata.txt")

# Ensure the directory exists
os.makedirs(PHOTO_DIR, exist_ok=True)

# Helper function to read metadata
def read_metadata():
    metadata = {}
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r") as file:
            for line in file:
                parts = line.strip().split('|')
                if len(parts) == 3:
                    filename, note, date = parts
                    metadata[filename] = {"note": note, "date": date}
    return metadata

# Function to convert HEIC to JPEG
def convert_heic_to_jpeg(heic_path, jpeg_path):
    heif_file = pillow_heif.open_heif(heic_path)
    image = Image.frombytes(
        heif_file.mode, 
        heif_file.size, 
        heif_file.data
    )
    image.save(jpeg_path, "JPEG")

# Helper function to write metadata
def write_metadata(filename, note, date):
    with open(METADATA_FILE, "a") as file:
        file.write(f"{filename}|{note}|{date}\n")
        
def delete_image(filename):
    metadata = read_metadata()
    
    # Check if the image exists in metadata
    if filename not in metadata:
        raise HTTPException(status_code=404, detail="Image not found in metadata")

    # Remove the image file if it exists
    file_path = os.path.join(PHOTO_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)

    # Rewrite metadata.txt without the deleted entry
    with open(METADATA_FILE, "w") as file:
        for img, data in metadata.items():
            if img != filename:  # Keep other entries
                file.write(f"{img}|{data['note']}|{data['date']}\n")

    return {"message": "Image deleted successfully"}

# FastAPI Endpoint
@app.delete("/api/delete_image/{filename}")
async def delete_image_endpoint(filename: str):
    return delete_image(filename)


# Upload endpoint with HEIC conversion
@app.post("/api/upload_image")
async def upload_image(note: str = Form(...), date: str = Form(...), file: UploadFile = File(...)):
    original_filename = file.filename
    file_extension = original_filename.lower().split('.')[-1]

    # Define file path for saving
    file_path = os.path.join(PHOTO_DIR, original_filename)

    # Save the uploaded file
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Convert HEIC files
    if file_extension == "heic":
        jpeg_filename = original_filename.rsplit(".", 1)[0] + ".jpg"
        jpeg_path = os.path.join(PHOTO_DIR, jpeg_filename)

        try:
            convert_heic_to_jpeg(file_path, jpeg_path)
            os.remove(file_path)  # Remove original HEIC file after conversion
            final_filename = jpeg_filename
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"HEIC conversion failed: {str(e)}")
    else:
        final_filename = original_filename  # Keep original name for non-HEIC files

    # Store metadata
    write_metadata(final_filename, note, date)

    return {"filename": final_filename, "note": note, "upload_date": date}

# **New Endpoint: Get all images as Base64 with metadata**
@app.get("/api/images")
async def get_all_images():
    metadata = read_metadata()
    images = []

    for filename, data in metadata.items():
        file_path = os.path.join(PHOTO_DIR, filename)
        
        if os.path.exists(file_path):
            # Read image and encode as Base64
            with open(file_path, "rb") as img_file:
                base64_image = base64.b64encode(img_file.read()).decode("utf-8")

            images.append({
                "image": f"data:image/jpeg;base64,{base64_image}",  # Data URL format
                "note": data["note"],
                "date": data["date"],
                "filename": filename
            })

    return {"images": images}

# Serve image files individually
@app.get("/api/images/file/{image_name}")
async def get_image_file(image_name: str):
    file_path = os.path.join(PHOTO_DIR, image_name)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        raise HTTPException(status_code=404, detail="Image file not found")
    
@app.post("/api/waitlist")
async def add_to_waitlist(request: WaitlistRequest):
    # Define the static directory path
    static_dir = "/app/static"
    
    # Create the static directory if it doesn't exist
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
    
    # Define the waitlist file path
    waitlist_file = os.path.join(static_dir, "waitlist.txt")
    
    try:
        # Format the fields, using empty string for None values
        first_name = request.first_name or ""
        last_name = request.last_name or ""
        phone = request.phone or ""
        email = request.email
        
        # Create the line with fields separated by spaces
        line = f"{first_name} {last_name} {phone} {email}\n"
        
        # Append the formatted line to the file (creates file if it doesn't exist)
        with open(waitlist_file, "a") as file:
            file.write(line)
        
        return {
            "message": "Successfully added to waitlist", 
            "first_name": request.first_name,
            "last_name": request.last_name,
            "phone": request.phone,
            "email": request.email
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add to waitlist: {str(e)}")

@app.post("/api/generate_image")
async def generate_image(request: GenerateImage):
    if request.key != api_key:
        return {"error": "Invalid API Key"}
    image = requests.post(generate_image_api_url+"/generate", json={"prompt": request.prompt})
    response = requests.get(generate_image_api_url+image.json()["image_url"])
    return StreamingResponse(BytesIO(response.content), media_type="image/png")

@app.get("/api/ollama/models")
async def get_ollama_models():
    """Get list of available Ollama models"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(status_code=500, detail="Failed to fetch models from Ollama")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error connecting to Ollama: {str(e)}")

@app.post("/api/ollama/chat")
async def chat_with_ollama(request: OllamaChatRequest):
    """Advanced chat with Ollama models with persistent history"""
    try:
        # Generate conversation ID if not provided
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        # Convert request messages to our internal format with timestamps
        input_messages = []
        for msg in request.messages:
            input_messages.append(create_timestamped_message(msg.role, msg.content))
        
        # Load existing conversation if ID provided and messages aren't the full conversation
        if request.conversation_id:
            conversation_data = load_conversation(conversation_id)
            existing_messages = conversation_data.get("messages", [])
            
            # Only add new messages that aren't already in history
            for new_msg in input_messages:
                # Check if this message is already in history
                is_duplicate = False
                for existing_msg in existing_messages:
                    if (existing_msg.get("role") == new_msg["role"] and 
                        existing_msg.get("content") == new_msg["content"]):
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    existing_messages.append(new_msg)
            
            all_messages = existing_messages
        else:
            all_messages = input_messages
        
        # Prepare Ollama request (filter out timestamps for Ollama)
        ollama_messages = [{"role": msg["role"], "content": msg["content"]} for msg in all_messages]
        ollama_request = {
            "model": request.model,
            "messages": ollama_messages,
            "stream": request.stream or False
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=ollama_request
            )
            
            if response.status_code == 200:
                if request.stream:
                    # Handle streaming response - for advanced chat, we'll collect the full response
                    # before saving to history, but still stream to client
                    full_response = ""
                    
                    async def generate_stream():
                        nonlocal full_response
                        async for line in response.aiter_lines():
                            if line.strip():
                                try:
                                    chunk = json.loads(line)
                                    # Accumulate the response content
                                    if chunk.get("message", {}).get("content"):
                                        full_response += chunk["message"]["content"]
                                    
                                    # Forward the chunk as-is for streaming
                                    yield f"data: {json.dumps(chunk)}\n\n"
                                    
                                    if chunk.get("done", False):
                                        # Save conversation after streaming is complete
                                        assistant_message = create_timestamped_message("assistant", full_response)
                                        all_messages.append(assistant_message)
                                        
                                        # Generate title for new conversations
                                        title = None
                                        if not request.conversation_id and len(all_messages) >= 2:
                                            first_user_msg = next((msg for msg in all_messages if msg["role"] == "user"), None)
                                            if first_user_msg:
                                                title = generate_conversation_title(first_user_msg["content"])
                                        
                                        # Save updated conversation
                                        save_conversation(conversation_id, all_messages, title)
                                        break
                                except json.JSONDecodeError:
                                    continue
                    
                    return StreamingResponse(
                        generate_stream(),
                        media_type="text/plain",
                        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
                    )
                else:
                    # Handle non-streaming response
                    result = response.json()
                    assistant_response = result.get("message", {}).get("content", "")
                    
                    # Add assistant response to history with timestamp
                    assistant_message = create_timestamped_message("assistant", assistant_response)
                    all_messages.append(assistant_message)
                    
                    # Generate title for new conversations
                    title = None
                    if not request.conversation_id and len(all_messages) >= 2:
                        # Use the first user message for title
                        first_user_msg = next((msg for msg in all_messages if msg["role"] == "user"), None)
                        if first_user_msg:
                            title = generate_conversation_title(first_user_msg["content"])
                    
                    # Save updated conversation
                    save_conversation(conversation_id, all_messages, title)
                    
                    # Return the original Ollama response plus our metadata
                    response_data = result.copy()
                    response_data.update({
                        "conversation_id": conversation_id,
                        "message_count": len(all_messages),
                        "timestamp": assistant_message["timestamp"],
                        "date": assistant_message["date"],
                        "time": assistant_message["time"]
                    })
                    
                    return response_data
            else:
                raise HTTPException(status_code=500, detail="Failed to get response from Ollama")
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with Ollama: {str(e)}")

@app.post("/api/ollama/generate")
async def generate_with_ollama(request: OllamaGenerateRequest):
    """Generate text with Ollama models"""
    try:
        ollama_request = {
            "model": request.model,
            "prompt": request.prompt,
            "stream": request.stream or False
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=ollama_request
            )
            
            if response.status_code == 200:
                if request.stream:
                    # Handle streaming response
                    async def generate_stream():
                        async for line in response.aiter_lines():
                            if line.strip():
                                try:
                                    chunk = json.loads(line)
                                    # Forward the chunk as-is for streaming
                                    yield f"data: {json.dumps(chunk)}\n\n"
                                    if chunk.get("done", False):
                                        break
                                except json.JSONDecodeError:
                                    continue
                    
                    return StreamingResponse(
                        generate_stream(),
                        media_type="text/plain",
                        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
                    )
                else:
                    # Handle non-streaming response
                    return response.json()
            else:
                raise HTTPException(status_code=500, detail="Failed to generate response from Ollama")
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with Ollama: {str(e)}")

@app.post("/api/chat")
async def simple_chat(request: SimpleChatRequest):
    """Simple chat endpoint using default model (llama3.1:8b) - no history"""
    try:
        ollama_request = {
            "model": OLLAMA_DEFAULT_MODEL,
            "messages": [{"role": "user", "content": request.message}],
            "stream": request.stream or False
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=ollama_request
            )
            
            if response.status_code == 200:
                if request.stream:
                    # Handle streaming response
                    async def generate_stream():
                        async for line in response.aiter_lines():
                            if line.strip():
                                try:
                                    chunk = json.loads(line)
                                    # Forward the chunk as-is for streaming
                                    yield f"data: {json.dumps(chunk)}\n\n"
                                    if chunk.get("done", False):
                                        break
                                except json.JSONDecodeError:
                                    continue
                    
                    return StreamingResponse(
                        generate_stream(),
                        media_type="text/plain",
                        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
                    )
                else:
                    # Handle non-streaming response
                    result = response.json()
                    return {
                        "model": OLLAMA_DEFAULT_MODEL,
                        "response": result.get("message", {}).get("content", ""),
                        "done": result.get("done", True)
                    }
            else:
                raise HTTPException(status_code=500, detail="Failed to get response from Ollama")
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with Ollama: {str(e)}")

@app.post("/api/generate_text")
async def simple_generate(request: SimpleGenerateRequest):
    """Simple text generation endpoint using default model (llama3.1:8b)"""
    try:
        ollama_request = {
            "model": OLLAMA_DEFAULT_MODEL,
            "prompt": request.prompt,
            "stream": request.stream or False
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=ollama_request
            )
            
            if response.status_code == 200:
                if request.stream:
                    # Handle streaming response
                    async def generate_stream():
                        async for line in response.aiter_lines():
                            if line.strip():
                                try:
                                    chunk = json.loads(line)
                                    # Forward the chunk as-is for streaming
                                    yield f"data: {json.dumps(chunk)}\n\n"
                                    if chunk.get("done", False):
                                        break
                                except json.JSONDecodeError:
                                    continue
                    
                    return StreamingResponse(
                        generate_stream(),
                        media_type="text/plain",
                        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
                    )
                else:
                    # Handle non-streaming response
                    result = response.json()
                    return {
                        "model": OLLAMA_DEFAULT_MODEL,
                        "response": result.get("response", ""),
                        "done": result.get("done", True)
                    }
            else:
                raise HTTPException(status_code=500, detail="Failed to generate response from Ollama")
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with Ollama: {str(e)}")

# Conversation Management Endpoints
@app.get("/api/conversations")
async def get_conversations():
    """Get list of all conversations"""
    try:
        conversations = list_conversations()
        return {"conversations": conversations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing conversations: {str(e)}")

@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get a specific conversation by ID"""
    try:
        conversation_data = load_conversation(conversation_id)
        if not conversation_data.get("messages"):
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conversation_data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading conversation: {str(e)}")

@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a specific conversation"""
    try:
        file_path = get_conversation_file_path(conversation_id)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        os.remove(file_path)
        return {"message": f"Conversation {conversation_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting conversation: {str(e)}")

@app.delete("/api/conversations")
async def clear_all_conversations():
    """Clear all conversation history"""
    try:
        deleted_count = 0
        for filename in os.listdir(CHAT_HISTORY_DIR):
            if filename.endswith('.json'):
                os.remove(os.path.join(CHAT_HISTORY_DIR, filename))
                deleted_count += 1
        return {"message": f"Deleted {deleted_count} conversations successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing conversations: {str(e)}")

@app.put("/api/conversations/{conversation_id}/title")
async def update_conversation_title(conversation_id: str, title: str = Query(...)):
    """Update conversation title"""
    try:
        conversation_data = load_conversation(conversation_id)
        if not conversation_data.get("messages"):
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        save_conversation(conversation_id, conversation_data["messages"], title)
        return {"message": "Title updated successfully", "title": title}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating title: {str(e)}")
