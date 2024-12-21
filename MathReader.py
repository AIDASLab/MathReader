import os
import re
import time
from transformers import T5ForConditionalGeneration, T5Tokenizer
import pandas as pd
from nemo.collections.tts.models import VitsModel

import pandas as pd
import soundfile as sf
import torch
from datasets import load_dataset
from pydub import AudioSegment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = "Hyeonsieun/GTtoNT_addmoretoken_ver2"
tokenizer = T5Tokenizer.from_pretrained(path)
model = T5ForConditionalGeneration.from_pretrained(path)
model.to(device)
model.eval()
audio_generator = VitsModel.from_pretrained("tts_en_lj_vits")


latex_pattern = r"(\\\(.+?\\\)|\\\[.+?\\\])"

def T5_inference(text, model, tokenizer):
    input_text = f"translate the LaTeX equation to a text pronouncing the formula: {text}"
    inputs = tokenizer.encode(
        input_text,
        return_tensors='pt',
        max_length=325,
        padding='max_length',
        truncation=True
    ).to(device)

    # Get correct sentence ids.
    corrected_ids = model.generate(
        inputs,
        max_length=325,
        num_beams=5, # `num_beams=1` indicated temperature sampling.
        early_stopping=True
    )

    # Decode.
    corrected_sentence = tokenizer.decode(
        corrected_ids[0],
        skip_special_tokens=False
    )
    return corrected_sentence

def postprocessing_T5(GT_raw):
    start_index = GT_raw.find("<pad>") + len("<pad>")
    end_index = GT_raw.find("</s>")
    GT_result = GT_raw[start_index:end_index]
    return GT_result


def extract_text_and_latex(mmd_file_path):
    
    with open(mmd_file_path, "r", encoding="utf-8") as file:
        content = file.read()

    
    sentence_list = re.split(latex_pattern, content)
    return sentence_list



def process_latex_in_list(sentence_list):
    
    replaced_list = []

    
    for sentence in sentence_list:
        
        if re.match(latex_pattern, sentence):
            sentence_input = '$'+sentence[2:-2]+'$'
            spoken_english = T5_inference(sentence_input, model, tokenizer)
            spoken_english = postprocessing_T5(spoken_english)
            replaced_list.append(spoken_english)
        
        else:
            replaced_list.append(sentence)
    return replaced_list

def split_sentence_into_chunks(sentence, chunk_size=10):
    
    words = sentence.split()
    
    
    chunks = [words[i:i + chunk_size] for i in range(0, len(words), chunk_size)]
    
    
    chunked_sentences = [' '.join(chunk) for chunk in chunks]
    
    return chunked_sentences



print("-----------OCR start---------")
OCR_start = time.time()
os.system(
    'nougat "/home/work/sieun/icassp2025/1.pdf" -o "test" ' # Write the path of the PDF file you want to perform OCR on.
)
OCR_end = time.time()
print("-----------OCR end---------")

OCR_time = OCR_end - OCR_start
print(f"OCR_time : {OCR_time}")

###############################################

print("-----------extract latex start---------")
extract_latex_start = time.time()
LaTeXs = []
mmd_foler_path = "test"
mmd_files = os.listdir(mmd_foler_path)
mmd_files = [os.path.join(mmd_foler_path, file) for file in mmd_files]


LaTeXs = []
for file in mmd_files:
    LaTeXs = extract_text_and_latex(file)
extract_latex_end = time.time()
print("-----------extract latex end---------")
extract_latex_time = extract_latex_end - extract_latex_start
print(f"extract_latex_time : {extract_latex_time}")


###############################################

print("-----------LaTeX translate start---------")
LaTeX_translate_start = time.time()
spoken_english_list = process_latex_in_list(LaTeXs)
LaTeX_translate_end = time.time()
LaTeX_translate_time = LaTeX_translate_end - LaTeX_translate_start
print("-----------LaTeX translate end---------")
print(f"LaTeX translate time : {LaTeX_translate_time}")

###############################################

# Load VITS

print("-----------TTS start---------")

TTS_start = time.time()
spoken_english_whole_sentence = ' '.join(spoken_english_list)
spoken_english_whole_sentence = spoken_english_whole_sentence.replace('\n\n', '   ').replace('*', ' ').replace('#', ' ').replace('_', '')
print(spoken_english_whole_sentence)
spoken_english_seperated_list = split_sentence_into_chunks(spoken_english_whole_sentence)


for i, sentence in enumerate(spoken_english_seperated_list):
    with torch.no_grad():
        parsed = audio_generator.parse(sentence)
        audio = audio_generator.convert_text_to_waveform(tokens=parsed)
    # Save the audio to disk in a file called speech.wav
    if isinstance(audio, torch.Tensor):
        audio = audio.to("cpu").numpy()
    sf.write(f"./test_audio/{i}.wav", audio.T, 22050, format="WAV")



folder_path = 'test_audio'


combined_audio = AudioSegment.empty()


wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]


def extract_number(file_name):
    
    match = re.search(r'(\d+)', file_name)
    return int(match.group()) if match else float('inf') 


wav_files.sort(key=extract_number)


for file_name in wav_files:
    file_path = os.path.join(folder_path, file_name)
    
    
    audio = AudioSegment.from_wav(file_path)
    
    
    combined_audio += audio


output_path = 'result_audio.wav'


combined_audio.export(output_path, format='wav')

TTS_end = time.time()
print("-----------TTS end---------")
TTS_time = TTS_end - TTS_start
print(f"TTS_time : {TTS_time}")



total_time = TTS_time + LaTeX_translate_time + extract_latex_time + OCR_time
print("-----------end-----------")
print(f"OCR_time : {OCR_time}")
print(f"GT2NT_time : {LaTeX_translate_time}")
print(f"TTS_time : {TTS_time}")
print(f"total: {total_time}")

