import argparse
import os
import logging
import torch
import numpy as np
import librosa
# from scipy.signal import resample
from pydub import AudioSegment
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model(model_path="./XTTS-v2/"):
    config = XttsConfig()
    config.load_json(os.path.join(model_path, "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=model_path)
    if torch.cuda.is_available():
        model.cuda()
    else:
        logging.warning("CUDA is not available. Running on CPU.")
    return model, config

def load_speaker_embeddings(input_path):
    logging.info(f"Loading speaker embeddings from {input_path}")
    embeddings = torch.load(input_path, map_location=torch.device('cpu'))
    return embeddings['gpt_cond_latent'], embeddings['speaker_embedding']

def extract_vocals(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    y_harmonic, _ = librosa.effects.hpss(y)
    return y_harmonic, sr

def convert_voice(model, original_vocals, sr, gpt_cond_latent, speaker_embedding, chunk_size=8192, overlap=1024):
    converted_vocals = []
    for i in range(0, len(original_vocals), chunk_size - overlap):
        chunk = original_vocals[i:i+chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        
        mel_spec = librosa.feature.melspectrogram(y=chunk, sr=sr)
        
        converted_chunk = model.voice_conversion(
            mel_spec,
            speaker_embedding,
            gpt_cond_latent
        )
        
        converted_vocals.append(converted_chunk)
    
    converted_vocals = np.concatenate(converted_vocals)
    return converted_vocals[:len(original_vocals)]

def create_cover(model, config, gpt_cond_latent, speaker_embedding, original_path, instrumental_path, output_path):
    original_vocals, sr = extract_vocals(original_path)
    
    converted_vocals = convert_voice(model, original_vocals, sr, gpt_cond_latent, speaker_embedding)
    
    instrumental = AudioSegment.from_file(instrumental_path)
    
    converted_vocals_segment = AudioSegment(
        converted_vocals.tobytes(),
        frame_rate=sr,
        sample_width=converted_vocals.dtype.itemsize,
        channels=1
    )
    
    if len(converted_vocals_segment) > len(instrumental):
        converted_vocals_segment = converted_vocals_segment[:len(instrumental)]
    elif len(converted_vocals_segment) < len(instrumental):
        converted_vocals_segment = converted_vocals_segment + AudioSegment.silent(duration=len(instrumental) - len(converted_vocals_segment))
    
    final_audio = instrumental.overlay(converted_vocals_segment)
    
    final_audio.export(output_path, format="mp3")
    logging.info(f"AI cover saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="AI Voice Conversion Cover Generator")
    parser.add_argument("--model_path", type=str, default="./XTTS-v2/", help="Path to the XTTS-v2 model directory")
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