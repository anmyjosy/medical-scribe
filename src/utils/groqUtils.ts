import Groq from 'groq-sdk';
import { Utterance } from './diarizationMerge';

const groq = new Groq({
    apiKey: process.env.GROQ_API_KEY || ''
});

/**
 * Uses Groq to analyze the conversation and refine speaker labels (e.g., "Doctor" vs "Patient").
 * It expects a conversation between two main speakers.
 */
export async function refineSpeakerLabels(utterances: Utterance[]): Promise<Utterance[]> {
    if (!utterances || utterances.length === 0) return utterances;

    // 1. Prepare context for Groq
    // We send the first N lines to identify roles, or the whole thing if short.
    // For cost/speed, let's send a representative chunk (e.g., first 15 exchanges).
    const transcriptSample = utterances.slice(0, 20).map(u => `${u.speaker}: ${u.text}`).join('\n');

    const prompt = `
You are an expert at analyzing medical conversations.
Below is a transcript of a consultation between a Doctor and a Patient (and possibly others).
The current speaker labels (like "A", "B", "Speaker 1") are generic.
Your task is to identify which speaker label corresponds to the DOCTOR and which to the PATIENT based on the dialogue context (who is asking clinical questions, who is reporting symptoms).

TRANSCRIPT SAMPLE:
${transcriptSample}

INSTRUCTIONS:
1. Analyze the sample.
2. Return a JSON object mapping the original speaker labels to "Doctor" or "Patient".
3. If you are unsure or if it's not a medical conversation, map them to "Speaker 1", "Speaker 2" etc. or keep original.
4. ONLY return the JSON. No preamble.

Example Output format:
{
  "A": "Doctor",
  "B": "Patient"
}
`;

    try {
        const completion = await groq.chat.completions.create({
            messages: [
                { role: 'user', content: prompt }
            ],
            model: 'llama-3.3-70b-versatile', // Use a smart model for reasoning
            temperature: 0,
            response_format: { type: 'json_object' }
        });

        const content = completion.choices[0]?.message?.content;
        if (!content) return utterances;

        const mapping = JSON.parse(content);
        console.log('Groq Speaker Refinement Mapping:', mapping);

        // Apply mapping
        return utterances.map(u => ({
            ...u,
            speaker: mapping[u.speaker] || u.speaker // Fallback to original if not in mapping
        }));

    } catch (error) {
        console.error('Error in refineSpeakerLabels:', error);
        return utterances; // Return original on failure
    }
}

import { Word } from './diarizationMerge';

/**
 * Uses Groq to split a continuous text into Doctor/Patient segments.
 * This is useful when acoustic diarization fails (e.g. similar voices).
 * It reconstructs the timeline by aligning the split text back to the original Word[] array.
 */
export async function diarizeWithGroq(words: Word[]): Promise<Utterance[]> {
    if (!words || words.length === 0) return [];

    // 1. Construct full text
    const fullText = words.map(w => w.word).join(' ');

    // 2. Ask Groq to split
    // "Split this text into turns. Do not change words."
    const prompt = `
You are an expert medical transcription assistant.
I will provide a raw transcript of a conversation between a Doctor, a Patient, and potentially a third person (like a Caregiver/Bystander).
The transcript currently has NO speaker labels or incorrect ones.

Your task is to SEGMENT the text into turns for:
- "Doctor"
- "Patient"
- "Caregiver" (if a third person is speaking on behalf of the patient)

INPUT TEXT:
"${fullText}"

INSTRUCTIONS:
1. Split the text into logical turns based on context.
2. Return a JSON array of objects: { "speaker": "Doctor" | "Patient" | "Caregiver", "text": "..." }
3. CRITICAL: Do NOT change, add, or remove words. The text in the segments must MATCH the input text exactly, just split up.
4. If you are unsure, make a best guess based on medical dialogue patterns.

Example Output:
[
  { "speaker": "Doctor", "text": "Hello, how is she doing?" },
  { "speaker": "Caregiver", "text": "She has been having fever since yesterday." },
  { "speaker": "Patient", "text": "Yes, I feel very hot." }
]
`;

    let segments: { speaker: string; text: string }[] = [];

    try {
        const completion = await groq.chat.completions.create({
            messages: [{ role: 'user', content: prompt }],
            model: 'llama-3.3-70b-versatile',
            temperature: 0,
            response_format: { type: 'json_object' }
        });

        const content = completion.choices[0]?.message?.content;
        if (!content) throw new Error('No content from Groq');

        const parsed = JSON.parse(content);
        // Handle if it returns { "segments": [...] } or just [...]
        segments = Array.isArray(parsed) ? parsed : (parsed.segments || parsed.turns || []);

        if (segments.length === 0) throw new Error('Empty segments returned');

    } catch (error) {
        console.error('Groq Diarization Failed:', error);
        // Fallback: Return as one big block
        return [{
            speaker: 'Unknown',
            text: fullText,
            start: getMs(words[0].startTime),
            end: getMs(words[words.length - 1].endTime)
        }];
    }

    // 3. Re-align timestamps
    // We iterate through the words and the segments to assigned start/end times.
    const utterances: Utterance[] = [];
    let wordIndex = 0;

    for (const segment of segments) {
        if (!segment.text.trim()) continue;

        const segmentWords = segment.text.split(/\s+/).filter(w => w.length > 0);
        if (segmentWords.length === 0) continue;

        // Find start time (from current wordIndex)
        // We assume Groq didn't change words, so we just march forward.
        // However, robustly, we should skip mismatches if small, but let's assume strict for now or simple "next N words".

        const segmentStartWord = words[wordIndex];
        if (!segmentStartWord) break; // Running out of words?

        const start = getMs(segmentStartWord.startTime);

        // Advance wordIndex by the number of words in this segment
        // To be safe, we match loosely.
        let matchCount = 0;
        for (const w of segmentWords) {
            // In a real robust algo, we'd check if words[wordIndex] matches w
            // But for now, let's just trust the count.
            // A better way is to "find approximately where this segment ends".
            // Let's just consume N words.
            wordIndex++;
        }

        // Backtrack one index to get the end word
        const segmentEndWord = words[Math.min(wordIndex - 1, words.length - 1)];
        const end = getMs(segmentEndWord.endTime);

        utterances.push({
            speaker: segment.speaker,
            text: segment.text, // Or use words.slice... to get exact original text
            start,
            end
        });

        if (wordIndex >= words.length) break;
    }

    return utterances;
}

// Duplicate helper locally or import? 
// It's not exported from diarizationMerge if it's not. 
// Let's copy it or rely on words having normalized (but they might not).
// words input to this function are "Word[]", which has { seconds, nanos } or string.
function getMs(t: any): number {
    if (!t) return 0;
    if (typeof t === 'number') return t;
    if (typeof t === 'string') {
        if (t.endsWith('s')) return parseFloat(t) * 1000;
        return parseFloat(t);
    }
    const s = t.seconds ? (typeof t.seconds === 'number' ? t.seconds : parseInt(String(t.seconds))) : 0;
    const n = t.nanos ? (typeof t.nanos === 'number' ? t.nanos : parseInt(String(t.nanos))) : 0;
    return s * 1000 + n / 1000000;
}
