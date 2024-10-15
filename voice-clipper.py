import sys
from pydub import AudioSegment
from pydub.silence import split_on_silence
from argparse import ArgumentParser

def remove_silence(input_file, output_file, silence_thresh=-50, min_silence_len=500, keep_silence=500):
    # Load the audio file
    audio = AudioSegment.from_file(input_file)

    # Split the audio file into chunks based on silence
    chunks = split_on_silence(audio, 
                              min_silence_len=min_silence_len, 
                              silence_thresh=silence_thresh, 
                              keep_silence=keep_silence)

    # Combine the chunks back together
    combined = AudioSegment.empty()
    for chunk in chunks:
        combined += chunk

    # Export the combined audio file
    combined.export(output_file, format="wav")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("input_file", help="The input audio file")
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    input_file = args.input_file
    output_file = "output" + input_file
    remove_silence(input_file, output_file)