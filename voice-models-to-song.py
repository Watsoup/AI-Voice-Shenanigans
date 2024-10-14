import argparse
import os
import logging
import torch
import numpy as np
import librosa
from scipy.signal import resample
from pydub import AudioSegment
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from resemblyzer import VoiceEncoder, preprocess_wav

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model(model_path = "./XTTS-v2/"):
    config = XttsConfig()
    config.load_json(os.path.join(model_path, "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=model_path)
    model.cuda()
    return model, config

def load_speaker_embeddings(input_path):
    logging.info(f"Loading speaker embeddings from {input_path}")
    embeddings = torch.load(input_path)
    return embeddings['gpt_cond_latent'], embeddings['speaker_embedding']

def extract_vocals(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    return y_harmonic, sr

def convert_voice(model, original_vocals, sr, gpt_cond_latent, speaker_embedding, chunk_size=8192, overlap=1024):
    converted_vocals = []
    for i in range(0, len(original_vocals), chunk_size - overlap):
        chunk = original_vocals[i:i+chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        
        # Convert the chunk to mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=chunk, sr=sr)
        
        # Use the model to convert the voice
        converted_chunk = model.voice_conversion(
            mel_spec,
            speaker_embedding,
            gpt_cond_latent
        )
        
        converted_vocals.append(converted_chunk)
    
    # Concatenate and handle overlaps
    converted_vocals = np.concatenate(converted_vocals)
    return converted_vocals[:len(original_vocals)]

def create_cover(model, config, gpt_cond_latent, speaker_embedding, original_path, instrumental_path, output_path):
    # Extract vocals from the original song
    original_vocals, sr = extract_vocals(original_path)
    
    # Convert the vocals using the AI model
    converted_vocals = convert_voice(model, original_vocals, sr, gpt_cond_latent, speaker_embedding)
    
    # Load the instrumental
    instrumental = AudioSegment.from_file(instrumental_path)
    
    # Convert the converted vocals to an AudioSegment
    converted_vocals_segment = AudioSegment(
        converted_vocals.tobytes(),
        frame_rate=sr,
        sample_width=converted_vocals.dtype.itemsize,
        channels=1
    )
    
    # Ensure the converted vocals are the same length as the instrumental
    if len(converted_vocals_segment) > len(instrumental):
        converted_vocals_segment = converted_vocals_segment[:len(instrumental)]
    elif len(converted_vocals_segment) < len(instrumental):
        converted_vocals_segment = converted_vocals_segment + AudioSegment.silent(duration=len(instrumental) - len(converted_vocals_segment))
    
    # Mix the converted vocals with the instrumental
    final_audio = instrumental.overlay(converted_vocals_segment)
    
    # Export the final audio
    final_audio.export(output_path, format="mp3")
    logging.info(f"AI cover saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="AI Voice Conversion Cover Generator")
    parser.add_argument("--model_path", type=str, required=False, help="Path to the XTTS-v2 model directory", default="./XTTS-v2/")
    parser.add_argument("--voice_model", type=str, required=True, help="Path to the saved voice model")
    parser.add_argument("--original", type=str, required=True, help="Path to the original song MP3")
    parser.add_argument("--instrumental", type=str, required=True, help="Path to the instrumental MP3")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output MP3")
    
    args = parser.parse_args()

    try:
        logging.info("Loading TTS model...")
        model, config = load_model(args.model_path)

        logging.info("Loading voice model...")
        gpt_cond_latent, speaker_embedding = load_speaker_embeddings(args.voice_model)

        logging.info("Creating AI cover...")
        create_cover(model, config, gpt_cond_latent, speaker_embedding, args.original, args.instrumental, args.output)

        logging.info("Done!")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()