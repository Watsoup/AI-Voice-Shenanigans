import argparse
import os
import logging
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from scipy.io.wavfile import write

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model(model_path = "./XTTS-v2/"):
    config = XttsConfig()
    config.load_json(os.path.join(model_path, "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=model_path)
    model.cuda()
    return model, config

def get_audio_files(directory):
    audio_extensions = ('.wav', '.mp3', '.flac', '.ogg')
    audio_files = [os.path.join(directory, f) for f in os.listdir(directory) 
                   if f.lower().endswith(audio_extensions)]
    logging.info(f"Found {len(audio_files)} audio file(s) in {directory}")
    return audio_files

def synthesize_speech(model, config, text, reference_audios, language="fr"):
    logging.info(f"Synthesizing speech with {len(reference_audios)} reference audio(s)")
    outputs = model.synthesize(
        text,
        config,
        speaker_wav=reference_audios,
        gpt_cond_len=3,
        language=language,
    )
    return outputs['wav']

def save_audio(audio, output_path, sample_rate=24000):
    write(output_path, sample_rate, audio)
    logging.info(f"Audio saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Text-to-Speech using XTTS-v2 model")
    parser.add_argument("--input_dir", "-d", type=str, required=True, help="Directory containing reference audio file(s)")
    parser.add_argument("--text", "-t", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--language", "-l", type=str, default="fr", help="Language code (e.g., 'fr' for French)")
    
    args = parser.parse_args()

    try:
        logging.info("Loading model...")
        model, config = load_model()

        logging.info("Finding reference audio files...")
        reference_audios = get_audio_files(args.input_dir)
        if not reference_audios:
            raise ValueError(f"No audio files found in {args.input_dir}")

        logging.info("Synthesizing speech...")
        audio = synthesize_speech(model, config, args.text, reference_audios, args.language)

        output_filename = os.path.basename(args.input_dir) + "_output.wav"
        output_path = os.path.join(args.input_dir, output_filename)
        logging.info(f"Saving audio to {output_path}...")
        save_audio(audio, output_path)

        logging.info("Done!")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()