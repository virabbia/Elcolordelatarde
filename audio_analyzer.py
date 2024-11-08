import os
import whisper
import re
import datetime
import pandas as pd
import gradio as gr

# Step 1: Transcribe Audio to Text using Whisper (without saving the transcript)
def transcribe_audio(audio_path):
    try:
        print(f"Starting transcription for: {audio_path}")
        model = whisper.load_model("base")
        result = model.transcribe(audio_path, language="Spanish")
        text = result["text"]
        segments = result["segments"]

        return text, segments
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None, None

# Step 2: Find repeated patterns (phrases or words)
def find_repeated_patterns(text, segments):
    if text is None or segments is None:
        print("No text or segments available for analysis.")
        return [], []
    
    flags = []
    
    try:
        # Find consecutive word repetitions (e.g., "Hola, hola")
        print("Searching for consecutive word repetitions...")
        matches = re.finditer(r'\b(\w+)\b(?:\s*[.,;:]*\s+)\1\b', text, re.IGNORECASE)
        
        filtered_repeats = []

        for match in matches:
            repeated_word = match.group(1)
            start_pos = match.start()
            
            # Extract a longer phrase containing the repeated word for more context
            words = text.split()
            index = text[:start_pos].count(" ")
            phrase = " ".join(words[max(0, index - 5):min(len(words), index + 10)])
            
            # Find the corresponding segment
            for segment in segments:
                segment_start = segment['start'] * len(text) / segments[-1]['end']
                segment_end = segment['end'] * len(text) / segments[-1]['end']
                if segment_start <= start_pos <= segment_end:
                    filtered_repeats.append((repeated_word, segment['start'], phrase))
                    break

        # Log the results
        if filtered_repeats:
            for repeated_word, start_time, phrase in filtered_repeats:
                log_entry = f"{str(datetime.timedelta(seconds=int(start_time)))} - Palabra repetida: '{repeated_word}' en: '{phrase}'."
                flags.append(log_entry)
                print(log_entry)
        else:
            print("No consecutive word repetitions found.")

        return flags, []
    except Exception as e:
        print(f"Error during pattern analysis: {e}")
        return [], []

# Main function to analyze audio
def analyze_audio(audio_path):
    # Transcribe the audio
    print("Starting analysis...")
    text, segments = transcribe_audio(audio_path)
    
    # Find repeated patterns in the text
    print("Finding repeated patterns in the transcribed text...")
    flags, timestamps = find_repeated_patterns(text, segments)
    
    # Return both flags and timestamps
    return flags, timestamps

# Gradio interface for local testing
def analyze_audio_ui(audio_path):
    # Transcribe and analyze the audio
    flags, timestamps = analyze_audio(audio_path)
    
    # Create a DataFrame from the timestamps for better visualization
    if flags:
        df = pd.DataFrame([flag.split(" - ", 1) for flag in flags], columns=["Tiempo", "Mensaje"])
    else:
        df = pd.DataFrame([("No se encontraron patrones repetidos.")], columns=["Tiempo", "Mensaje"])
    
    return df

# Main function for Gradio interface
def main():
    interface = gr.Interface(
        fn=analyze_audio_ui,
        inputs=gr.Audio(type="filepath", label="Sube un archivo de audio para analizar"),
        outputs=gr.Dataframe(label="Resultados del análisis"),
        title="Analizador de Repeticiones de Audio",
        description="Sube un archivo de audio para detectar palabras o frases repetidas consecutivamente.",
        allow_flagging="never",
        live=False,  # Enable re-upload after clearing previous results
        theme="default",  # Use a built-in Gradio theme
        css="""
            .input_interface {
                width: 25%;
                height: 150px;
                overflow: hidden;
            }
            .output_interface {
                width: 75%;
            }
            .gradio-input {
                transform: scale(0.75);  /* Make the dropbox smaller */
                transform-origin: top left;
            }
        """
    )
    interface.launch()

if __name__ == "__main__":
    main()
