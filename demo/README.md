# MathSpeech: Leveraging Small LMs for Accurate Conversion in Mathematical Speech-to-Formula

## Abstract
In various academic and professional settings, such as mathematics lectures or research presentations, it is often necessary to convey mathematical expressions orally. However, reading mathematical expressions aloud without accompanying visuals can significantly hinder comprehension, especially for those who are hearing-impaired or rely on subtitles due to language barriers. For instance, when a presenter reads Euler's Formula, current Automatic Speech Recognition (ASR) models often produce a verbose and error-prone textual description (e.g., e to the power of i x equals cosine of x plus i $\textit{side}$ of x), instead of the concise LaTeX format (i.e., $e^{ix} = \cos(x) + i\sin(x)$), which hampers clear understanding and communication. To address this issue, we introduce MathSpeech, a novel pipeline that integrates ASR models with small Language Models (sLMs) to correct errors in mathematical expressions and accurately convert spoken expressions into structured LaTeX representations. Evaluated on a new dataset derived from lecture recordings, MathSpeech demonstrates LaTeX generation capabilities comparable to leading commercial Large Language Models (LLMs), while leveraging fine-tuned small language models of only 120M parameters.
Specifically, in terms of CER, BLEU, and ROUGE scores for LaTeX translation, MathSpeech demonstrated significantly superior capabilities compared to GPT-4o. We observed a decrease in CER from 0.390 to 0.298, and higher ROUGE/BLEU scores compared to GPT-4o.

### This page is for anonymous submission for AAAI 2025.

Here, you can find the benchmark dataset, experimental code, and fine-tuned model checkpoints for MathSpeech, which we have developed for our research.

---

## Benchmart Dataset
The MathSpeech benchmark dataset is available on huggingface🤗.

- [MathSpeech in huggingface🤗 dataset](https://huggingface.co/datasets/1anonymous1/MathSpeech)


#### Dataset statistics
<table border="1" style="border-collapse: collapse; width: 50%;">
    <thead>
        <tr>
            <th style="text-align: left;">The number of files</th>
            <td>1,101</td>
        </tr>
    </thead>
    <thead>
        <tr>
            <th style="text-align: left;">Total Duration</th>
            <td>5583.2 seconds</td>
        </tr>
    </thead>
    <tbody>
        <tr>
            <th style="text-align: left;">Average Duration per file</th>
            <td>5.07 seconds</td>
        </tr>
        <tr>
            <th style="text-align: left;">The number of speakers</th>
            <td>10</td>
        </tr>
        <tr>
            <th style="text-align: left;">The number of men</th>
            <td>8</td>
        </tr>
        <tr>
            <th style="text-align: left;">The number of women</th>
            <td>2</td>
        </tr>
        <tr>
            <th style="text-align: left;">source</th>
            <td><a href="https://www.youtube.com/@mitocw" target="_blank">[MIT OpenCourseWare]</td>
        </tr>
    </tbody>
</table>



#### WERs of various ASR models on the Mathspeech benchmark
<table style="width:100%; border-collapse: collapse;">
  <thead>
    <tr>
      <th></th>
      <th>Models</th>
      <th>Params</th>
      <th>WER(%) (Leaderboard)</th>
      <th>WER(%) (Formula)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="4">OpenAI</td>
      <td>Whisper-base</td>
      <td>74M</td>
      <td>10.3</td>
      <td>34.7</td>
    </tr>
    <tr>
      <td>Whisper-small</td>
      <td>244M</td>
      <td>8.59</td>
      <td>29.5</td>
    </tr>
    <tr>
      <td>Whisper-largeV2</td>
      <td>1550M</td>
      <td>7.83</td>
      <td>31.0</td>
    </tr>
    <tr>
      <td>Whisper-largeV3</td>
      <td>1550M</td>
      <td>7.44</td>
      <td>33.3</td>
    </tr>
    <tr>
      <td>NVIDIA</td>
      <td>Canary-1B</td>
      <td>1B</td>
      <td>6.5</td>
      <td>35.2</td>
    </tr>
  </tbody>
</table>

##### The WER for Leaderboard was from the [HuggingFace Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard), while the WER for Formula was measured using our MathSpeech Benchmark.


---
## Dependencies
- PyTorch==2.3.1
- transformers==4.44.0

## Experiment codes
You can download the MathSpeech checkpoint from the following [link](https://drive.google.com/file/d/1y2SoovDDRSB36paRHUE995wysvZ-1nch/view?usp=sharing).








