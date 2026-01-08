
import { AutoProcessor, AutoModelForAudioFrameClassification, env } from '@huggingface/transformers';
import wavefile from 'wavefile';
import fs from 'fs';

// Skip local model checks to speed up
env.allowLocalModels = false;

async function run() {
    console.log('Loading model...');
    const model_id = 'onnx-community/pyannote-segmentation-3.0';

    try {
        const processor = await AutoProcessor.from_pretrained(model_id);
        const model = await AutoModelForAudioFrameClassification.from_pretrained(model_id);

        console.log('Model loaded. Reading audio...');
        // Download a sample wav if needed or use one if exists. 
        // Using a public sample URL for "Baby Elephant Walk" (usually safe) or similar 
        // But transformers.js often takes a Float32Array. 

        // Let's use a known sample URL that transformers.js utility might handle or fetch manually.
        const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav';
        const buffer = await (await fetch(url)).arrayBuffer();

        // Read .wav using wavefile
        const wav = new wavefile.WaveFile(new Uint8Array(buffer));
        const samples = wav.getSamples();
        // wavefile returns Uint8Array or Int16Array etc. need Float32
        // Mono check
        const data = wav.fmt.numChannels === 2 ? samples[0] : samples;

        // Convert to float32
        const float32Data = new Float32Array(data.length);
        const factor = 32768.0; // Assuming 16-bit
        for (let i = 0; i < data.length; i++) {
            float32Data[i] = data[i] / factor;
        }

        console.log(`Audio loaded: ${float32Data.length} samples`);

        console.log('Running inference...');
        const inputs = await processor(float32Data);
        const { logits } = await model(inputs);

        console.log('Logits shape:', logits.dims);

    } catch (e) {
        console.error('Error:', e);
    }
}

run();
