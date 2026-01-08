export interface Word {
    word: string;
    startTime: { seconds: string | number; nanos?: number } | string | number;
    endTime: { seconds: string | number; nanos?: number } | string | number;
    speakerTag?: number; // Old Google tag
}

interface SpeakerSegment {
    speaker: string;
    start: number; // ms
    end: number;   // ms
}

export interface Utterance {
    speaker: string;
    text: string;
    start: number;
    end: number;
}

// Helper to normalize time to MS
function getMs(t: any): number {
    if (!t) return 0;
    if (typeof t === 'number') return t; // Already ms? Assume no, usually Google returns Objects. 
    // If it's a raw number from Google, it might be weird. But let's assume our parsers handle it.

    if (typeof t === 'string') {
        if (t.endsWith('s')) return parseFloat(t) * 1000;
        return parseFloat(t); // Assume MS if no suffix? Or Seconds? Google V1 usually objects.
    }

    const s = t.seconds ? (typeof t.seconds === 'number' ? t.seconds : parseInt(String(t.seconds))) : 0;
    const n = t.nanos ? (typeof t.nanos === 'number' ? t.nanos : parseInt(String(t.nanos))) : 0;
    return s * 1000 + n / 1000000;
}

export function mergeDiarization(words: Word[], segments: SpeakerSegment[]): Utterance[] {
    if (!segments || segments.length === 0) {
        // Fallback: If no diarization, chunk by text or return one block?
        // Let's Just return one block 
        // Or if words rely on Google speakerTag (which is 0), use that.
        // But for this Hybrid, we prefer segments.
        return [{
            speaker: 'A',
            text: words.map(w => w.word).join(' '),
            start: getMs(words[0]?.startTime),
            end: getMs(words[words.length - 1]?.endTime)
        }];
    }

    // Assign each word to a speaker
    // Strategy: Find which segment covers the MIDPOINT of the word.
    const labeledWords = words.map(word => {
        const start = getMs(word.startTime);
        const end = getMs(word.endTime);
        const midpoint = (start + end) / 2;

        // Find match
        const match = segments.find(seg => midpoint >= seg.start && midpoint <= seg.end);

        // If no match (word outside segments), find closest?
        // Simple fallback: Unknown or Nearest.
        let speaker = match ? match.speaker : 'UNKNOWN';

        // Clean up Pyannote labels (SPEAKER_00 -> Speaker A)
        // Normalize: SPEAKER_00 -> A, SPEAKER_01 -> B
        // Pattern: SPEAKER_(\d+)
        const matchLabel = speaker.match(/SPEAKER_(\d+)/);
        if (matchLabel) {
            const num = parseInt(matchLabel[1]);
            speaker = String.fromCharCode(65 + num); // 0->A
        }

        return { word: word.word, start, end, speaker };
    });

    // Resolve UNKNOWNs using Forward and Backward filling

    // 1. Forward pass: Propagate known speakers to subsequent UNKNOWNs (fills gaps in middle/end)
    for (let i = 1; i < labeledWords.length; i++) {
        if (labeledWords[i].speaker === 'UNKNOWN' && labeledWords[i - 1].speaker !== 'UNKNOWN') {
            labeledWords[i].speaker = labeledWords[i - 1].speaker;
        }
    }

    // 2. Backward pass: Propagate known speakers to preceding UNKNOWNs (fills gaps at start)
    for (let i = labeledWords.length - 2; i >= 0; i--) {
        if (labeledWords[i].speaker === 'UNKNOWN' && labeledWords[i + 1].speaker !== 'UNKNOWN') {
            labeledWords[i].speaker = labeledWords[i + 1].speaker;
        }
    }

    // 3. Final Fallback: If segments are still UNKNOWN (e.g. entire file has no diarization segments), default to 'A'
    for (let i = 0; i < labeledWords.length; i++) {
        if (labeledWords[i].speaker === 'UNKNOWN') {
            labeledWords[i].speaker = 'A';
        }
    }

    // Collapse into Utterances
    const utterances: Utterance[] = [];
    if (labeledWords.length === 0) return [];

    let currentSpeaker = labeledWords[0].speaker;
    let currentWords: string[] = [labeledWords[0].word];
    let start = labeledWords[0].start;
    let end = labeledWords[0].end;

    for (let i = 1; i < labeledWords.length; i++) {
        const lw = labeledWords[i];
        if (lw.speaker === currentSpeaker) {
            currentWords.push(lw.word);
            end = lw.end;
        } else {
            // Close group
            utterances.push({
                speaker: currentSpeaker,
                text: currentWords.join(' '),
                start,
                end
            });
            // Start new
            currentSpeaker = lw.speaker;
            currentWords = [lw.word];
            start = lw.start;
            end = lw.end;
        }
    }
    // Final push
    utterances.push({
        speaker: currentSpeaker,
        text: currentWords.join(' '),
        start,
        end
    });

    return utterances;
}
