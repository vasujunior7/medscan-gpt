# if you dont use pipenv uncomment the following:
from dotenv import load_dotenv
load_dotenv()

#VoiceBot UI with Gradio
import os
import gradio as gr
import logging

from brain_of_the_doctor import encode_image, analyze_image_with_query
from voice_of_the_patient import record_audio, transcribe_with_groq
from voice_of_the_doctor import text_to_speech_with_gtts, text_to_speech_with_elevenlabs

# Verify API keys are loaded
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not found in environment variables")
if not os.getenv("ELEVEN_LABS_API_KEY"):
    raise ValueError("ELEVEN_LABS_API_KEY not found in environment variables")

system_prompt="""You have to act as a professional doctor, i know you are not but this is for learning purpose. 
            What's in this image?. Do you find anything wrong with it medically? 
            If you make a differential, suggest some remedies for them. Donot add any numbers or special characters in 
            your response. Your response should be in one long paragraph. Also always answer as if you are answering to a real person.
            Donot say 'In the image I see' but say 'With what I see, I think you have ....'
            Dont respond as an AI model in markdown, your answer should mimic that of an actual doctor not an AI bot, 
            Keep your answer concise (max 2 sentences). No preamble, start your answer right away please"""


def process_inputs(audio_filepath, image_filepath):
    try:
        if not os.path.exists(audio_filepath):
            return "No audio file provided", "Please record audio first", None

        speech_to_text_output = transcribe_with_groq(
            GROQ_API_KEY=os.environ.get("GROQ_API_KEY"), 
            audio_filepath=audio_filepath,
            stt_model="whisper-large-v3"
        )

        # Handle the image input
        if image_filepath and os.path.exists(image_filepath):
            doctor_response = analyze_image_with_query(
                query=system_prompt+speech_to_text_output, 
                encoded_image=encode_image(image_filepath), 
                model="mixtral-8x7b-32768"
            )
        else:
            doctor_response = "No image provided for me to analyze"

        output_filepath = "doctor_response.mp3"
        voice_of_doctor = text_to_speech_with_elevenlabs(
            input_text=doctor_response, 
            output_filepath=output_filepath
        )

        return speech_to_text_output, doctor_response, output_filepath
    except Exception as e:
        logging.error(f"Error in process_inputs: {e}")
        return f"Error: {str(e)}", "Error occurred", None


# Create the interface
iface = gr.Interface(
    fn=process_inputs,
    inputs=[
        gr.Audio(sources=["microphone"], type="filepath", label="Record your voice"),
        gr.Image(type="filepath", label="Upload an image")
    ],
    outputs=[
        gr.Textbox(label="Speech to Text"),
        gr.Textbox(label="Doctor's Response"),
        gr.Audio(label="Doctor's Voice Response")
    ],
    title="AI Doctor with Vision and Voice",
    description="Record your voice and upload an image for medical analysis"
)

if __name__ == "__main__":
    iface.launch(debug=True, share=False)

#http://127.0.0.1:7860