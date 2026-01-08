import { SOAPNote, TranscriptUtterance } from '@/types';

const API_BASE_URL = '';

/**
 * Transcribe audio with speaker diarization via API
 */
export const transcribeWithDiarization = async (
    audioBlob: Blob,
    language: string = 'English'
): Promise<{ text: string; utterances: TranscriptUtterance[] }> => {
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.webm');
    formData.append('language', language);

    const response = await fetch(`${API_BASE_URL}/api/transcribe`, {
        method: 'POST',
        body: formData
    });

    if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Transcription failed');
    }

    return await response.json();
};

/**
 * Generate SOAP note from transcript using API
 */
export const generateSOAPNote = async (
    transcriptText: string,
    utterances: TranscriptUtterance[]
): Promise<SOAPNote> => {
    const response = await fetch(`${API_BASE_URL}/api/generate-soap`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            text: transcriptText,
            utterances
        })
    });

    if (!response.ok) {
        throw new Error('SOAP note generation failed');
    }

    return await response.json();
};

/**
 * Main function to process consultation audio and generate SOAP note
 */
export const processConsultation = async (
    audioBlob: Blob,
    language: string = 'English'
): Promise<{ soapNote: SOAPNote; utterances: TranscriptUtterance[]; fullText: string }> => {
    // Step 1: Transcribe with diarization
    const { text, utterances } = await transcribeWithDiarization(audioBlob, language);

    // Step 2: Generate SOAP note from transcript
    let soapNote: SOAPNote;

    if (!text || text.trim().length === 0) {
        console.warn('Transcription text is empty. Skipping SOAP generation.');
        soapNote = {
            subjective: {
                chiefComplaint: 'No speech detected',
                historyOfPresentIllness: 'Audio recording did not contain clear speech.'
            },
            objective: {
                vitals: {
                    temperature: 'Not recorded',
                    bloodPressure: 'Not recorded',
                    pulse: 'Not recorded',
                    respiratoryRate: 'Not recorded'
                },
                appearance: ['Not documented']
            },
            assessment: 'No diagnosis possible',
            plan: 'Please retry recording with clearer speech.'
        };
    } else {
        soapNote = await generateSOAPNote(text, utterances);
    }

    return {
        soapNote,
        utterances,
        fullText: text
    };
};
