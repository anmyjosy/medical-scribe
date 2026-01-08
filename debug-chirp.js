const { v2 } = require('@google-cloud/speech');
const path = require('path');

const keyFilename = path.join(__dirname, 'google-key.json');
const client = new v2.SpeechClient({
    keyFilename,
    apiEndpoint: 'us-central1-speech.googleapis.com'
});

const PROJECT_ID = 'premium-pulsar-483503-k0';
const location = 'us-central1';
const recognizerName = `projects/${PROJECT_ID}/locations/${location}/recognizers/_`;

const config = {
    autoDecodingConfig: {},
    languageCodes: ['ml-IN'],
    model: 'chirp',
    features: {
        enableWordTimeOffsets: true,
        diarizationConfig: {
            minSpeakerCount: 2,
            maxSpeakerCount: 2,
        },
    },
};

const gcsUri = 'gs://medscribe-temp-uploads/audio-1767787446094-24gwgf.wav'; // Use a dummy or existing file if available, or just check config validation

const request = {
    recognizer: recognizerName,
    config: config,
    files: [{ uri: gcsUri }],
    recognitionOutputConfig: {
        inlineResponseConfig: {}
    }
};

async function run() {
    console.log('Testing Chirp Config...');
    try {
        const [operation] = await client.batchRecognize(request);
        console.log('Operation started');
        const [response] = await operation.promise();
        console.log('Success:', response);
    } catch (error) {
        console.error('Error Details:');
        console.log(JSON.stringify(error, null, 2));
        if (error.statusDetails) {
            console.log('Status Details:', JSON.stringify(error.statusDetails, null, 2));
        }
    }
}

run();
