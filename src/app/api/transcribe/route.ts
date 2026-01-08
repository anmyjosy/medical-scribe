import { NextRequest, NextResponse } from 'next/server';
import { AssemblyAI } from 'assemblyai';
import { Storage } from '@google-cloud/storage';
import { v2 as SpeechV2 } from '@google-cloud/speech';
import { SpeechClient } from '@google-cloud/speech'; // Keep V1 just in case, but we use V2 for Malayalam
import path from 'path';
import { mergeDiarization } from '@/utils/diarizationMerge'; // Still needed? Maybe not if using Groq only.
import { refineSpeakerLabels, diarizeWithGroq } from '@/utils/groqUtils';

const assemblyClient = new AssemblyAI({
    apiKey: process.env.ASSEMBLYAI_API_KEY || ''
});

// Initialize Google Cloud Clients
const keyFilename = path.join(process.cwd(), 'google-key.json');
const storage = new Storage({ keyFilename });
const googleClient = new SpeechClient({ keyFilename });
const googleClientV2 = new SpeechV2.SpeechClient({
    keyFilename,
    apiEndpoint: 'us-speech.googleapis.com' // Required for 'us' multi-region (Chirp 3)
});
const BUCKET_NAME = 'medscribe-temp-uploads';

export async function POST(request: NextRequest) {
    console.log('Processing transcription request. Language check...');

    try {
        const formData = await request.formData();
        const file = (formData.get('audio') || formData.get('file')) as File;
        const language = (formData.get('language') as string) || 'ml-IN';

        if (!file) {
            return NextResponse.json({ error: 'No file uploaded' }, { status: 400 });
        }

        const buffer = Buffer.from(await file.arrayBuffer());
        const mimeType = file.type || 'audio/wav';

        console.log(`Language Identity: ${language}`);

        // ---------------------------------------------------------
        // HYBRID: GOOGLE CLOUD STT (V1) + PYANNOTE (HUGGING FACE)
        // ---------------------------------------------------------
        if (language.toLowerCase() === 'malayalam' || language.toLowerCase() === 'ml-in') {
            console.log('Using Hybrid Pipeline (Google V1 + Pyannote via HF) for Malayalam...');

            // 1. Upload to GCS
            const filename = `audio-${Date.now()}-${Math.random().toString(36).substring(7)}${mimeType.includes('mp3') ? '.mp3' : mimeType.includes('webm') ? '.webm' : '.wav'}`;
            const bucket = storage.bucket(BUCKET_NAME);
            const fileUpload = bucket.file(filename);

            console.log(`Uploading to GCS: gs://${BUCKET_NAME}/${filename}`);
            try {
                await fileUpload.save(buffer, { contentType: mimeType });
            } catch (err) {
                try { await bucket.create({ location: 'asia-south1' }); await fileUpload.save(buffer, { contentType: mimeType }); } catch (e) { }
            }
            console.log(`File uploaded to gs://${BUCKET_NAME}/${filename}`);
            const gcsUri = `gs://${BUCKET_NAME}/${filename}`;


            // 2. EXECUTION
            console.log('Starting Google STT Transcription...');

            // Task A: Google V2 Transcription (Chirp 3 - Batch Mode)
            const googleResponse = await (async () => {
                const parent = `projects/premium-pulsar-483503-k0/locations/us`; // 'us' multi-region
                const recognizerId = 'medscribe-malayalam-chirp-3-us-batch';
                const recognizerName = `${parent}/recognizers/${recognizerId}`;

                // 1. Get or Create Recognizer
                try {
                    await googleClientV2.getRecognizer({ name: recognizerName });
                    console.log('V2 Recognizer found:', recognizerName);
                } catch (error: any) {
                    if (error.code === 5 || (error.message && error.message.includes('not found'))) {
                        console.log('Creating V2 Recognizer (Chirp 3 Batch)...');
                        const [operation] = await googleClientV2.createRecognizer({
                            parent,
                            recognizerId,
                            recognizer: {
                                defaultRecognitionConfig: {
                                    features: {
                                        enableWordTimeOffsets: true,
                                    },
                                },
                                model: 'chirp_3',
                                languageCodes: ['ml-IN'],
                            }
                        });
                        console.log('Waiting for Recognizer creation...');
                        await operation.promise();
                        console.log('Recognizer created.');
                    } else {
                        throw error;
                    }
                }

                // 2. Transcribe using BatchRecognize
                console.log('Starting BatchRecognize...');
                const [operation] = await googleClientV2.batchRecognize({
                    recognizer: recognizerName,
                    config: {
                        features: { enableWordTimeOffsets: true },
                        autoDecodingConfig: {},
                    },
                    files: [{ uri: gcsUri }],
                    recognitionOutputConfig: {
                        inlineResponseConfig: {}, // Get results directly in response
                    }
                });

                console.log('Waiting for BatchRecognize operation...');
                const [response] = await operation.promise();

                // Parse Batch results
                const fileResult = response.results ? response.results[gcsUri] : null;

                if (!fileResult || !fileResult.transcript) {
                    console.log('No transcript in Batch response.');
                    return { results: [] };
                }

                return fileResult.transcript;
            })();

            // Task B Deleted (Pyannote)

            console.log('Google STT Completed.');

            // DEBUG: Print full Google response
            console.log('--- GOOGLE STT V2 RAW RESPONSE ---');
            // console.log(JSON.stringify(googleResponse, null, 2)); // reduce log noise
            console.log(`Results count: ${googleResponse.results?.length}`);
            console.log('-------------------------------');

            // Clean up
            try { await fileUpload.delete(); } catch (e) { }

            if (!googleResponse.results || googleResponse.results.length === 0) {
                // Try to fallback or check if Batch is needed? 
                // For now return empty.
                console.log('No results from V2 API.');
                return NextResponse.json({ text: '', utterances: [] });
            }

            // 3. Process & Merge
            // Extract all words from Google V2
            // V2 'WordInfo' uses 'startOffset' and 'endOffset' instead of 'startTime/endTime'
            const rawWords = googleResponse.results
                .flatMap((result: any) => result.alternatives?.[0]?.words || []);

            // Helper for parsing Duration (seconds/nanos) from V1 or V2
            const getMs = (t: any) => {
                if (!t) return 0;
                // V2 might use '3s' string or { seconds: "3", nanos: 0 }
                if (typeof t === 'string' && t.endsWith('s')) return parseFloat(t) * 1000;

                const s = t.seconds ? (typeof t.seconds === 'number' ? t.seconds : parseInt(String(t.seconds))) : 0;
                const n = t.nanos ? (typeof t.nanos === 'number' ? t.nanos : parseInt(String(t.nanos))) : 0;
                return s * 1000 + n / 1000000;
            };

            // Ensure words are sorted by time (handle startOffset vs startTime)
            rawWords.sort((a: any, b: any) => {
                const tA = a.startOffset || a.startTime;
                const tB = b.startOffset || b.startTime;
                return getMs(tA) - getMs(tB);
            });

            // Merge logic
            // Map V2 words to format expected by merger if necessary
            const normalizedWords = rawWords.map((w: any) => ({
                word: w.word,
                startTime: w.startOffset || w.startTime,
                endTime: w.endOffset || w.endTime
            }));

            // -------------------------------------------------------------------------
            // 4. LLM-Based Diarization (Text Splitting & Alignment)
            // -------------------------------------------------------------------------
            // User requested Groq to identify split points.
            // We bypass the acoustic diarization merge (which might be wrong) and use Groq directly on words.

            console.log('Using LLM-Based Diarization (Groq) for Malayalam...');
            const utterances = await diarizeWithGroq(normalizedWords);

            // If Groq fails or returns one block, we might fall back to acoustic?
            // But diarizeWithGroq has its own fallback to one block.

            // NOTE: We do NOT use mergeDiarization or key refineSpeakerLabels here 
            // because diarizeWithGroq essentially does both (splits and labels).

            const transcriptText = googleResponse.results
                .map((r: any) => r.alternatives?.[0]?.transcript).join('\n');

            return NextResponse.json({ text: transcriptText, utterances });

        } else {

            // ---------------------------------------------------------
            // ASSEMBLYAI - FOR ENGLISH / OTHERS
            // ---------------------------------------------------------
            console.log('Using AssemblyAI for transcription...');
            const base64Audio = buffer.toString('base64');
            const dataUrl = `data:${mimeType};base64,${base64Audio}`;

            const transcript = await assemblyClient.transcripts.transcribe({
                audio: dataUrl,
                speaker_labels: true,
                language_detection: true,
            });

            if (transcript.status === 'error') {
                throw new Error(`Transcription failed: ${transcript.error}`);
            }

            const utterances = (transcript.utterances || []).map(u => ({
                speaker: u.speaker,
                text: u.text,
                start: u.start,
                end: u.end
            }));

            return NextResponse.json({
                text: transcript.text || '',
                utterances
            });
        }

    } catch (error: any) {
        console.error('Transcription error:', error);
        return NextResponse.json({ error: error.message || 'Failed to transcribe' }, { status: 500 });
    }
}
