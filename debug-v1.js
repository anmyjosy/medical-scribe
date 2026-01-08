const { SpeechClient } = require('@google-cloud/speech');
const path = require('path');

async function testV1() {
    try {
        console.log('Testing V1 Client Instantiation...');
        const keyFilename = path.join(__dirname, 'google-key.json');
        const client = new SpeechClient({ keyFilename });
        console.log('Client created successfully.');

        console.log('Project ID:', await client.getProjectId());

        const gcsUri = 'gs://medscribe-temp-uploads/audio-1767787446094-24gwgf.wav';
        const config = {
            languageCode: 'ml-IN',
            enableWordTimeOffsets: true,
            diarizationConfig: {
                enableSpeakerDiarization: true,
                minSpeakerCount: 2,
                maxSpeakerCount: 2,
            },
        };
        const request = {
            audio: { uri: gcsUri },
            config: config
        };

        console.log('Sending V1 Request...');
        const [operation] = await client.longRunningRecognize(request);
        console.log('Operation started:', operation.name);

    } catch (error) {
        console.error('V1 Error:', error);
    }
}

testV1();
