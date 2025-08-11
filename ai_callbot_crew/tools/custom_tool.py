import re
import speech_recognition as sr
import json
from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field

# Import Coqui TTS and its necessary components
from TTS.api import TTS

# INPUT SCHEMAS FOR TOOLS

class STTToolInput(BaseModel):
    """Input schema for the Speech-to-Text tool, now listening to the microphone."""    
    pass

class DataExtractorToolInput(BaseModel):
    """Input schema for the Data Extraction tool."""
    text_to_extract_from: str = Field(..., description="The text from which to extract data.")

class TTSToolInput(BaseModel):
    """Input schema for the Text-to-Speech tool."""
    text_to_speak: str = Field(..., description="The text to be converted into speech.")
    output_path: str = Field(..., description="The file path to save the generated audio.")


# CUSTOM TOOL IMPLEMENTATIONS

class STTTool(BaseTool):
    name: str = "Speech-to-Text Tool"
    description: str = (
        "A tool that transcribes live audio from the microphone into text. "
        "Useful for converting a user's spoken words into a format the LLM can understand."
    )
    args_schema: Type[BaseModel] = STTToolInput

    def _run(self) -> str:
        """
        Transcribes live audio from the microphone to text.
        
        Returns:
            The transcribed text as a string.
        """
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening...")
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source)
        
        try:
            # You can switch to the open-source CMU Sphinx engine by using:
            # text = r.recognize_sphinx(audio)
            text = r.recognize_google(audio)
            print(f"Heard: {text}")
            return text
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError as e:
            return f"Could not request results from service; {e}"
        except Exception as e:
            return f"An unexpected error occurred during transcription: {e}"

class DataExtractorTool(BaseTool):
    name: str = "Data Extractor Tool"
    description: str = (
        "A tool that extracts specific data from a given text. "
        "Useful for pulling out key information like names, phone numbers, and email addresses."
    )
    args_schema: Type[BaseModel] = DataExtractorToolInput

    def _run(self, text_to_extract_from: str) -> str:
        """
        Extracts specific data from the given text using regex and returns it as a JSON string.
        """
        extracted_data = {}
        appointment_keywords = ["appointment", "schedule", "book", "meeting", "time to meet"]
        
        phone_match = re.search(r'(\d{3}[-.\s]??\d{3}[-.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-.\s]??\d{4}|\d{10})', text_to_extract_from)
        email_match = re.search(r'[\w.+-]+@[\w-]+\.[\w.-]+', text_to_extract_from)
        name_match = re.search(r'(?:my name is|I am|this is)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)', text_to_extract_from, re.IGNORECASE)        
        appointment_found = any(keyword in text_to_extract_from.lower() for keyword in appointment_keywords)

        if phone_match:
            extracted_data['phone_number'] = phone_match.group(0)
        
        if email_match:
            extracted_data['email'] = email_match.group(0)

        if name_match:
            extracted_data['name'] = name_match.group(1)

        extracted_data['appointment'] = "yes" if appointment_found else "no"

        return json.dumps(extracted_data)

class TTSTool(BaseTool):
    name: str = "Text-to-Speech Tool"
    description: str = (
        "A tool that converts text into an audio file using Coqui TTS. "
        "Useful for generating a spoken response to the user."
    )
    args_schema: Type[BaseModel] = TTSToolInput

    def __init__(self):
        super().__init__()
        # Initialize Coqui TTS with a specific model
        # You MUST download a model and provide the correct paths here
        self.tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)

    def _run(self, text_to_speak: str, output_path: str) -> str:
        """
        Converts the given text to speech and saves it as an audio file.
        """
        try:
            self.tts.tts_to_file(text=text_to_speak, file_path=output_path)
            return f"Successfully generated speech for text and saved to '{output_path}'"
        except Exception as e:
            return f"An error occurred during TTS conversion: {e}"
