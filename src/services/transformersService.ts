
import { AutoProcessor, AutoModelForAudioFrameClassification, env } from '@huggingface/transformers';
import { WaveFile } from 'wavefile';

// Use local cache for models to avoid redownloading
// env.cacheDir = './.cache'; // Optional: defaults to standard cache
// Disable local models checking if you want to ensure it fetches from hub or cache
env.allowLocalModels = false;

const MODEL_ID = 'onnx-community/pyannote-segmentation-3.0';

// Singleton for efficiency
let processor: any = null;
let model: any = null;

async function loadModel() {
    if (!processor || !model) {
        console.log('Loading Pyannote model locally...');
        processor = await AutoProcessor.from_pretrained(MODEL_ID);
        model = await AutoModelForAudioFrameClassification.from_pretrained(MODEL_ID);
        console.log('Model loaded.');
    }
    return { processor, model };
}

export interface SpeakerSegment {
    speaker: string;
    start: number;
    end: number;
}

export async function diarizeAudio(audioBuffer: Buffer): Promise<SpeakerSegment[]> {
    try {
        console.log(`Starting local diarization. Buffer size: ${audioBuffer.length}`);

        // 1. Prepare Audio (WAV -> Float32Array)
        // Assume input is a valid WAV file.
        // 1. Prepare Audio (WAV -> Float32Array @ 16kHz)
        const wav = new WaveFile(audioBuffer);

        // Force 16kHz sample rate (Pyannote expects 16k)
        wav.toSampleRate(16000);

        // Extract samples
        const rawSamples = wav.getSamples();
        const fmt = wav.fmt as any;
        const numChannels = fmt.numChannels;
        const bitDepth = fmt.bitsPerSample;

        // Decode to Float32 Mono
        let channelData: Float32Array;

        if (numChannels === 2 && Array.isArray(rawSamples)) {
            // Stereo -> Mono (Take Left Channel)
            const left = rawSamples[0] || rawSamples;
            channelData = new Float32Array(left.length);
            // normalize based on bit depth
            const factor = bitDepth === 32 ? 2147483648.0 : (bitDepth === 8 ? 128.0 : 32768.0);

            for (let i = 0; i < left.length; i++) {
                channelData[i] = left[i] / factor;
            }
        } else {
            // Mono
            const data = rawSamples as unknown as any;
            channelData = new Float32Array(data.length);
            const factor = bitDepth === 32 ? 2147483648.0 : (bitDepth === 8 ? 128.0 : 32768.0);
            for (let i = 0; i < data.length; i++) {
                channelData[i] = data[i] / factor;
            }
        }

        const sampleRate = 16000; // Enforced


        // 2. Inference
        const { processor, model } = await loadModel();

        // Processor handles resampling to 16000Hz automatically if we pass raw audio + sampling_rate?
        // transformers.js processor() accepts `raw_speech` as Float32Array.
        // It *expects* 16000Hz usually. If our file is 44.1k, we MUST resample.
        // transformers.js processor does NOT always do resampling in JS yet. output usually warns "mismatched sample rate".
        // For safety: Let's assume input is close enough or use a resampler if needed.
        // But for now, let's pass it.
        const inputs = await processor(channelData);

        const { logits } = await model(inputs);
        // logits: [1, frames, 7]

        // 3. Decode
        const frames = logits.dims[1];
        const rawData = logits.data; // Float32Array flat
        const numClasses = 7;

        // Calculate time per frame
        // Total duration of audio / frames
        // We can estimate duration from original sample rate
        const totalDuration = channelData.length / sampleRate;
        const secondsPerFrame = totalDuration / frames;

        const segments: SpeakerSegment[] = [];
        let currentSpeaker: string | null = null;
        let startFrame = 0;

        // Powerset Mapping
        // 0: Silence
        // 1: A
        // 2: B
        // 3: C
        // 4: A+B
        // 5: A+C
        // 6: B+C

        const getSpeakerLabel = (classIdx: number): string | null => {
            if (classIdx === 0) return null; // Silence
            if (classIdx === 1) return "SPEAKER_00";
            if (classIdx === 2) return "SPEAKER_01";
            if (classIdx === 3) return "SPEAKER_02";
            if (classIdx === 4) return "SPEAKER_00"; // Treat overlap as dominant A? Or both? Let's just pick one for simplicity or emit overlap?
            if (classIdx === 5) return "SPEAKER_00";
            if (classIdx === 6) return "SPEAKER_01";
            return null;
        };

        for (let i = 0; i < frames; i++) {
            // Find argmax for this frame
            let maxVal = -Infinity;
            let maxIdx = 0;
            for (let c = 0; c < numClasses; c++) {
                const val = rawData[i * numClasses + c];
                if (val > maxVal) {
                    maxVal = val;
                    maxIdx = c;
                }
            }

            const speaker = getSpeakerLabel(maxIdx);

            if (speaker !== currentSpeaker) {
                // Close previous segment
                if (currentSpeaker) {
                    segments.push({
                        speaker: currentSpeaker,
                        start: startFrame * secondsPerFrame * 1000, // ms
                        end: i * secondsPerFrame * 1000 // ms
                    });
                }
                // Start new
                currentSpeaker = speaker;
                startFrame = i;
            }
        }

        // Close final
        if (currentSpeaker) {
            segments.push({
                speaker: currentSpeaker,
                start: startFrame * secondsPerFrame * 1000,
                end: frames * secondsPerFrame * 1000
            });
        }

        console.log(`Diarization complete. Found ${segments.length} segments.`);
        if (segments.length > 0) {
            console.log('--- RAW SEGMENTS (First 10) ---');
            console.log(JSON.stringify(segments.slice(0, 10), null, 2));
            console.log('-------------------------------');
        }
        return segments;

    } catch (error) {
        console.error("Transformers Diarization Error:", error);
        return []; // Graceful fallback
    }
}
