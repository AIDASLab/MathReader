# MathReader: Text-to-Speech for Mathematical Documents

This is the official repository for MathReader, an advanced TTS document reader for academic mathematical documents. 

Demo page: [https://hyeonsieun.github.io/MathReader_demo/](https://hyeonsieun.github.io/MathReader_demo/)

### This page is for submission to ICASSP 2025.

The experimental code and test dataset developed for our research can be found here.

---

## How to use MathReader

1. Install Nougat and NVIDIA NeMo and transformers library in your development environment.
   - [Nougat github](https://github.com/facebookresearch/nougat)
   - [NVIDIA NeMo](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/tts/models.html#vits)
   - You can also set up the environment using the following code through the **mathreader_environment.yml** file

     : `conda env create -f ./mathreader_environment.yml`
2. Create a folder named 'test_audio' in the same location as MathReader.py.

3. Modify line 102 in MathReader.py (Write the path of the PDF file you want to perform OCR on.).

4. Run `python MathReader.py` in the terminal.


