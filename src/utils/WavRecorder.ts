export class WavRecorder {
    private audioContext: AudioContext | null = null;
    private mediaStreamSource: MediaStreamAudioSourceNode | null = null;
    private scriptProcessor: ScriptProcessorNode | null = null;
    private audioData: Float32Array[] = [];
    private stream: MediaStream | null = null;
    private sampleRate: number = 16000;

    async start() {
        this.stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                noiseSuppression: true,
                echoCancellation: true,
                autoGainControl: true,
                channelCount: 1,
                // Request 16kHz if possible to match model, but we handle fallback
                sampleRate: this.sampleRate
            }
        });
        this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: this.sampleRate });

        // Fallback if sampleRate constraint isn't supported
        this.sampleRate = this.audioContext.sampleRate;

        this.mediaStreamSource = this.audioContext.createMediaStreamSource(this.stream);
        // Buffer size 4096, 1 input channel, 1 output channel
        this.scriptProcessor = this.audioContext.createScriptProcessor(4096, 1, 1);

        this.audioData = [];

        this.scriptProcessor.onaudioprocess = (e) => {
            const inputData = e.inputBuffer.getChannelData(0);
            // Clone data because input buffer is reused
            this.audioData.push(new Float32Array(inputData));
        };

        this.mediaStreamSource.connect(this.scriptProcessor);
        this.scriptProcessor.connect(this.audioContext.destination);
    }

    async stop(): Promise<Blob> {
        if (this.mediaStreamSource) this.mediaStreamSource.disconnect();
        if (this.scriptProcessor) {
            this.scriptProcessor.disconnect();
            this.scriptProcessor.onaudioprocess = null;
        }
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
        }
        if (this.audioContext) {
            await this.audioContext.close();
        }

        return this.encodeWAV(this.audioData, this.sampleRate);
    }

    private encodeWAV(samples: Float32Array[], sampleRate: number): Blob {
        const bufferLength = samples.reduce((acc, curr) => acc + curr.length, 0);
        const data = new Float32Array(bufferLength);
        let offset = 0;
        for (const chunk of samples) {
            data.set(chunk, offset);
            offset += chunk.length;
        }

        const buffer = new ArrayBuffer(44 + data.length * 2);
        const view = new DataView(buffer);

        // RIFF chunk descriptor
        this.writeString(view, 0, 'RIFF');
        view.setUint32(4, 36 + data.length * 2, true);
        this.writeString(view, 8, 'WAVE');

        // fmt sub-chunk
        this.writeString(view, 12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true); // PCM
        view.setUint16(22, 1, true); // Mono
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * 2, true);
        view.setUint16(32, 2, true);
        view.setUint16(34, 16, true); // 16-bit

        // data sub-chunk
        this.writeString(view, 36, 'data');
        view.setUint32(40, data.length * 2, true);

        // Write PCM samples
        let p = 44;
        for (let i = 0; i < data.length; i++) {
            let s = Math.max(-1, Math.min(1, data[i]));
            s = s < 0 ? s * 0x8000 : s * 0x7FFF;
            view.setInt16(p, s, true);
            p += 2;
        }

        return new Blob([view], { type: 'audio/wav' });
    }

    private writeString(view: DataView, offset: number, string: string) {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    }
}
