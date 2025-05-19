import functions_framework
from google.cloud import texttospeech
from google.cloud import storage
import os
import uuid

# Initialize clients
tts_client = texttospeech.TextToSpeechClient()
storage_client = storage.Client()

# GCS_BUCKET_NAME = os.environ.get("TTS_AUDIO_BUCKET_NAME")
GCS_BUCKET_NAME = "uplas-tts-audio-bucket" # Replace with your actual bucket name

@functions_framework.http
def generate_tts_audio(request):
    if request.method != 'POST':
        return 'Only POST requests are accepted', 405

    request_json = request.get_json(silent=True)
    if not request_json:
        return 'Invalid JSON payload', 400

    text_to_speak = request_json.get('text')
    voice_name_short = request_json.get('voice', 'alloy') # e.g., 'alloy', 'echo'
    language_code = request_json.get('language', 'en-US')
    # context_course_id = request_json.get('context_course_id') # For logging/analytics
    # context_topic_id = request_json.get('context_topic_id')

    if not text_to_speak:
        return 'Missing "text" parameter', 400

    # Map frontend voice names to Google Cloud TTS voice parameters
    # This is a simplified mapping. Refer to Google Cloud TTS documentation for full voice list.
    # https://cloud.google.com/text-to-speech/docs/voices
    # For "character" voices, you might need to select specific voice names that fit the persona.
    # The names 'alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer' are from OpenAI's TTS.
    # Google's voices have names like 'en-US-Wavenet-D' or 'en-GB-News-K'.
    # We'll need a mapping or adjust frontend to send Google-compatible names.
    # For example, let's assume frontend 'alloy' maps to a standard neutral Google voice.

    voice_params = {
        "alloy": {"language_code": language_code, "name": f"{language_code.split('-')[0]}-Standard-A" if language_code.startswith('en') else f"{language_code}-Standard-A"}, # Default, adjust as needed
        "echo": {"language_code": language_code, "name": f"{language_code.split('-')[0]}-Wavenet-D" if language_code.startswith('en') else f"{language_code}-Wavenet-A"}, # More expressive Wavenet
        "fable": {"language_code": language_code, "name": f"{language_code.split('-')[0]}-Wavenet-F" if language_code.startswith('en') else f"{language_code}-Wavenet-C"}, # Storyteller-like Wavenet
        "onyx": {"language_code": language_code, "name": f"{language_code.split('-')[0]}-Wavenet-B" if language_code.startswith('en') else f"{language_code}-Wavenet-B"}, # Deeper Wavenet
        "nova": {"language_code": language_code, "name": f"{language_code.split('-')[0]}-Wavenet-C" if language_code.startswith('en') else f"{language_code}-Wavenet-D"}, # Bright Wavenet
        "shimmer": {"language_code": language_code, "name": f"{language_code.split('-')[0]}-News-K" if language_code.startswith('en') else f"{language_code}-Standard-D"}, # Warm/News Wavenet
    }

    selected_voice_config = voice_params.get(voice_name_short.lower())
    if not selected_voice_config: # Fallback if voice name is not in our map
        actual_language_code = language_code
        actual_voice_name = f"{language_code}-Standard-A" # A generic standard voice
    else:
        actual_language_code = selected_voice_config["language_code"]
        actual_voice_name = selected_voice_config["name"]

    # Ensure compatibility with available voices for the language
    # This part might need more robust error handling or listing available voices.
    # For now, we assume the constructed name is valid or falls back gracefully.
    # A better approach: query available voices for the language_code if the specific name isn't found.

    synthesis_input = texttospeech.SynthesisInput(text=text_to_speak)
    voice = texttospeech.VoiceSelectionParams(
        language_code=actual_language_code,
        name=actual_voice_name 
        # ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL # or specific based on character
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    try:
        response = tts_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
    except Exception as e:
        print(f"TTS API Error: {e}")
        # Attempt a more generic voice if the specific one failed
        try:
            generic_voice_name = f"{language_code.split('-')[0]}-Standard-A" # A very standard voice
            if actual_voice_name == generic_voice_name: # Already tried generic, so fail
                raise e 
            print(f"TTS failed with {actual_voice_name}, trying generic voice {generic_voice_name}")
            voice.name = generic_voice_name
            response = tts_client.synthesize_speech(
                input=synthesis_input, voice=voice, audio_config=audio_config
            )
        except Exception as final_e:
             print(f"TTS API Error (even with generic voice): {final_e}")
             return f"Error generating speech: {final_e}", 500


    # Save to GCS
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    blob_name = f"tts_audio/{uuid.uuid4()}.mp3"
    blob = bucket.blob(blob_name)

    try:
        blob.upload_from_string(response.audio_content, content_type='audio/mpeg')
        # Make the blob publicly readable (or use signed URLs for better security)
        blob.make_public()
        public_url = blob.public_url
        print(f"Audio uploaded to: {public_url}")
        return {"audio_url": public_url}, 200
    except Exception as e:
        print(f"GCS Upload Error: {e}")
        return f"Error saving audio file: {e}", 500

# The Django backend's /api/tts/ view will call this Cloud Function.
