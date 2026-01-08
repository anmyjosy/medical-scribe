
import React, { useState, useRef, useEffect } from 'react';
import { Loader2 } from 'lucide-react';
import { MedicalNote, SOAPNote, TranscriptUtterance, PrefilledPatientData } from '@/types';
import { processConsultation } from '@/services/assemblyaiService';
import { generateKeyInsights } from '@/services/groqService';
import { WavRecorder } from '@/utils/WavRecorder';

interface ScribeSessionProps {
  prefilledData?: PrefilledPatientData;
  onCancel: () => void;
  onSave: (note: MedicalNote, audioBlob?: Blob) => void;
}

type SessionState = 'INIT' | 'CHOICE' | 'RECORDING' | 'CONFIRMATION' | 'PROCESSING' | 'RESULT';

const ScribeSession: React.FC<ScribeSessionProps> = ({ prefilledData, onCancel, onSave }) => {
  const [sessionState, setSessionState] = useState<SessionState>(prefilledData ? 'CHOICE' : 'INIT');
  const [patientName, setPatientName] = useState(prefilledData?.name || '');
  const [patientAge, setPatientAge] = useState(prefilledData?.age || '');
  const [patientGender, setPatientGender] = useState(prefilledData?.gender || 'Unknown');
  const [patientWeight, setPatientWeight] = useState(prefilledData?.weight || '');
  const [patientHeight, setPatientHeight] = useState(prefilledData?.height || '');
  const [patientBloodGroup, setPatientBloodGroup] = useState(prefilledData?.bloodGroup || '');
  const [patientNationality, setPatientNationality] = useState(prefilledData?.nationality || '');
  const [patientLanguage, setPatientLanguage] = useState(prefilledData?.language || 'English');
  const [recordingTime, setRecordingTime] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [generatedNote, setGeneratedNote] = useState<SOAPNote | null>(null);
  const [utterances, setUtterances] = useState<TranscriptUtterance[]>([]);
  const [fullTranscript, setFullTranscript] = useState<string>('');
  const [capturedAudio, setCapturedAudio] = useState<Blob | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [keyInsights, setKeyInsights] = useState<string[]>([]);
  const [isMuted, setIsMuted] = useState(false);
  const [isSaving, setIsSaving] = useState(false);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioStreamRef = useRef<MediaStream | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<any>(null);

  useEffect(() => {
    console.log('ScribeSession mounted with prefilledData:', prefilledData);
    if (prefilledData) {
      console.log('Setting session state to CHOICE. Language:', prefilledData.language);
      setSessionState('CHOICE');
      setPatientName(prefilledData.name);
      setPatientAge(prefilledData.age);
      setPatientGender(prefilledData.gender || 'Unknown');
      setPatientWeight(prefilledData.weight || '');
      setPatientHeight(prefilledData.height || '');
      setPatientBloodGroup(prefilledData.bloodGroup || '');
      setPatientNationality(prefilledData.nationality || '');
      setPatientLanguage(prefilledData.language || 'English');
    }
  }, [prefilledData]);

  useEffect(() => {
    if (sessionState === 'RECORDING') {
      timerRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);
    } else {
      if (timerRef.current) clearInterval(timerRef.current);
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [sessionState]);

  const wavRecorderRef = useRef<WavRecorder | null>(null);

  const startRecording = async () => {
    try {
      const recorder = new WavRecorder();
      await recorder.start(); // This requests mic access
      wavRecorderRef.current = recorder;

      setSessionState('RECORDING');
      setError(null);
    } catch (err) {
      console.error('Mic Error:', err);
      setError('Microphone access denied or error.');
    }
  };

  const stopRecording = async () => {
    if (wavRecorderRef.current && sessionState === 'RECORDING') {
      const audioBlob = await wavRecorderRef.current.stop();
      wavRecorderRef.current = null;

      setCapturedAudio(audioBlob);
      setAudioUrl(URL.createObjectURL(audioBlob));
      setSessionState('CONFIRMATION');
      setIsMuted(false);
    }
  };

  const toggleMute = () => {
    if (audioStreamRef.current) {
      audioStreamRef.current.getAudioTracks().forEach(track => {
        track.enabled = !track.enabled;
      });
      setIsMuted(!isMuted);
    }
  };

  const handleProcessAudio = async (blob: Blob, autoSave = false) => {
    setSessionState('PROCESSING');
    try {
      const result = await processConsultation(blob, patientLanguage);
      setGeneratedNote(result.soapNote);
      setUtterances(result.utterances);
      setFullTranscript(result.fullText);

      let finalInsights: string[] = [];
      try {
        finalInsights = await generateKeyInsights(result.fullText);
        setKeyInsights(finalInsights);
      } catch (iErr: any) {
        console.error('Insights generation error:', iErr);
        setKeyInsights([`Client-side Service Error: ${iErr.message || 'Check terminal'}`]);
      } finally {
        if (autoSave) {
          // AUTO SAVE LOGIC
          await onSave({
            id: Math.random().toString(36).substr(2, 9),
            patientId: patientName,
            date: new Date().toISOString(),
            type: 'Consultation',
            content: result.soapNote!,
            rawTranscript: result.fullText,
            utterances: result.utterances,
            summary: (result.soapNote as any).summary || result.soapNote!.assessment,
            keyInsights: finalInsights,
            // Pass extended patient info
            age: parseInt(patientAge),
            weight: patientWeight,
            height: patientHeight,
            bloodGroup: patientBloodGroup,
            gender: patientGender,
            nationality: patientNationality,
            language: patientLanguage
          } as any, blob);
        } else {
          setSessionState('RESULT');
        }
      }

    } catch (err) {
      console.error('Processing error:', err);
      setError('Cognitive synthesis interrupted.');
      setSessionState('INIT'); // Go back to start on error
    }
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    if (!patientName.trim()) {
      setError('Please provide a patient identity before initializing data upload.');
      return;
    }
    const file = event.target.files?.[0];
    if (!file) return;

    if (!file.type.startsWith('audio/')) {
      setError('Please upload an audio file.');
      return;
    }

    setCapturedAudio(file);
    setAudioUrl(URL.createObjectURL(file));
    setCapturedAudio(file);
    setAudioUrl(URL.createObjectURL(file));
    setSessionState('CONFIRMATION');
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  // 1. INITIALIZATION SCREEN
  if (sessionState === 'INIT') {
    return (
      <div className="min-h-screen bg-[#FDFDFB] flex flex-col items-center justify-center p-6 relative overflow-hidden">
        <button onClick={onCancel} className="absolute top-8 right-8 w-8 h-8 bg-black text-white rounded-full flex items-center justify-center hover:bg-slate-800 transition-all z-20 shadow-lg shadow-black/20">
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>
        </button>

        <div className="w-full max-w-xl space-y-8 animate-in fade-in slide-in-from-bottom-8 duration-1000">
          <div className="text-center">
            <span className="text-[10px] font-black uppercase tracking-[0.6em] text-blue-600 block mb-4">Identity Initialization</span>
            <h1 className="text-4xl font-extrabold tracking-tighter">Enter Patient Name</h1>
          </div>

          <div className="space-y-6">
            <div className="space-y-2">
              <label className="text-[8px] font-black uppercase tracking-widest text-black/30 ml-4">Patient Name</label>
              <input
                type="text"
                autoFocus
                value={patientName}
                onChange={(e) => setPatientName(e.target.value)}
                placeholder="e.g. Johnathan Doe"
                className="w-full text-2xl font-bold tracking-tight bg-white border-2 border-black/5 rounded-[24px] px-8 py-4 text-center focus:outline-none focus:border-blue-600 focus:shadow-2xl focus:shadow-blue-600/5 placeholder:text-black/5 transition-all"
              />
            </div>

            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
              <div className="space-y-1">
                <label className="text-[7px] font-black uppercase tracking-widest text-black/30 ml-2">Age</label>
                <input
                  type="number"
                  value={patientAge}
                  onChange={(e) => setPatientAge(e.target.value)}
                  placeholder="Yrs"
                  className="w-full bg-white border border-black/5 rounded-xl px-4 py-2 text-xs font-bold focus:border-blue-600 outline-none transition-all"
                />
              </div>
              <div className="space-y-1">
                <label className="text-[7px] font-black uppercase tracking-widest text-black/30 ml-2">Gender</label>
                <select
                  value={patientGender}
                  onChange={(e) => setPatientGender(e.target.value)}
                  className="w-full bg-white border border-black/5 rounded-xl px-4 py-2 text-xs font-bold focus:border-blue-600 outline-none transition-all appearance-none"
                >
                  <option value="">Select</option>
                  <option value="Male">Male</option>
                  <option value="Female">Female</option>
                  <option value="Other">Other</option>
                </select>
              </div>
              <div className="space-y-1">
                <label className="text-[7px] font-black uppercase tracking-widest text-black/30 ml-2">Weight</label>
                <input
                  type="text"
                  value={patientWeight}
                  onChange={(e) => setPatientWeight(e.target.value)}
                  placeholder="kg"
                  className="w-full bg-white border border-black/5 rounded-xl px-4 py-2 text-xs font-bold focus:border-blue-600 outline-none transition-all"
                />
              </div>
              <div className="space-y-1">
                <label className="text-[7px] font-black uppercase tracking-widest text-black/30 ml-2">Height</label>
                <input
                  type="text"
                  value={patientHeight}
                  onChange={(e) => setPatientHeight(e.target.value)}
                  placeholder="cm"
                  className="w-full bg-white border border-black/5 rounded-xl px-4 py-2 text-xs font-bold focus:border-blue-600 outline-none transition-all"
                />
              </div>
              <div className="space-y-1">
                <label className="text-[7px] font-black uppercase tracking-widest text-black/30 ml-2">Blood</label>
                <select
                  value={patientBloodGroup}
                  onChange={(e) => setPatientBloodGroup(e.target.value)}
                  className="w-full bg-white border border-black/5 rounded-xl px-4 py-2 text-xs font-bold focus:border-blue-600 outline-none transition-all appearance-none"
                >
                  <option value="">Select</option>
                  {['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-'].map(bg => (
                    <option key={bg} value={bg}>{bg}</option>
                  ))}
                </select>
              </div>
              <div className="space-y-1">
                <label className="text-[7px] font-black uppercase tracking-widest text-black/30 ml-2">Nationality</label>
                <input
                  type="text"
                  value={patientNationality}
                  onChange={(e) => setPatientNationality(e.target.value)}
                  placeholder="Country"
                  className="w-full bg-white border border-black/5 rounded-xl px-4 py-2 text-xs font-bold focus:border-blue-600 outline-none transition-all"
                />
              </div>
              <div className="space-y-1 col-span-2 md:col-span-1">
                <label className="text-[7px] font-black uppercase tracking-widest text-black/30 ml-2">Language</label>
                <select
                  value={patientLanguage}
                  onChange={(e) => setPatientLanguage(e.target.value)}
                  className="w-full bg-white border border-black/5 rounded-xl px-4 py-2 text-xs font-bold focus:border-blue-600 outline-none transition-all appearance-none text-blue-600"
                >
                  <option value="English">English</option>
                  <option value="Malayalam">Malayalam</option>
                  <option value="Hindi">Hindi</option>
                  <option value="Arabic">Arabic</option>
                </select>
              </div>
            </div>
          </div>

          {error && <p className="text-center text-red-500 font-bold uppercase text-[8px] tracking-widest">{error}</p>}
        </div>

        <div className="flex flex-col items-center gap-6">
          <button
            disabled={!patientName.trim()}
            onClick={() => setSessionState('CHOICE')}
            className={`px-10 py-3 rounded-xl text-[9px] font-black uppercase tracking-[0.4em] transition-all ${patientName.trim() ? 'bg-black text-white hover:scale-105 shadow-xl shadow-black/10' : 'bg-black/5 text-black/20 cursor-not-allowed'}`}
          >
            Continue to Capture
          </button>
        </div>
      </div>
    );
  }

  // 1.5 CHOICE SCREEN
  if (sessionState === 'CHOICE') {
    return (
      <div className="min-h-screen bg-[#FDFDFB] flex flex-col items-center justify-center p-6 relative overflow-hidden">
        <button onClick={onCancel} className="absolute top-8 right-8 w-8 h-8 bg-black text-white rounded-full flex items-center justify-center hover:bg-slate-800 transition-all z-20 shadow-lg shadow-black/20">
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>
        </button>

        <div className="w-full max-w-xl space-y-8 animate-in fade-in slide-in-from-bottom-8 duration-600">
          <div className="text-center space-y-10">
            <div className="space-y-4">
              <span className="text-[10px] font-black uppercase tracking-[0.6em] text-blue-600 block">System Ready</span>
              <h1 className="text-4xl font-extrabold tracking-tighter">Capture Session for {patientName}</h1>
              <div className="flex flex-center justify-center gap-3 flex-wrap">
                <div className="px-3 py-1 bg-black/5 rounded-full text-[10px] font-bold text-black/40">{patientGender || 'Unknown'} • {patientAge} Yrs</div>
                <div className="px-3 py-1 bg-black/5 rounded-full text-[10px] font-bold text-black/40">{patientBloodGroup || 'No Blood Type'}</div>
                <div className="px-3 py-1 bg-black/5 rounded-full text-[10px] font-bold text-black/40">{patientNationality || 'Unknown Nationality'}</div>
                <div className={`px-3 py-1 rounded-full text-[10px] font-bold ${patientLanguage && patientLanguage !== 'English' ? 'bg-blue-100 text-blue-700 border border-blue-200' : 'bg-black/5 text-black/40'}`}>Language: {patientLanguage}</div>
                <div className="px-3 py-1 bg-black/5 rounded-full text-[10px] font-bold text-black/40">{patientWeight}kg • {patientHeight}cm</div>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-4">
              <button
                onClick={startRecording}
                className="flex flex-col items-center justify-center p-8 bg-black text-white rounded-[32px] hover:scale-105 transition-all shadow-2xl shadow-black/20 group"
              >
                <div className="w-12 h-12 rounded-full bg-white/10 flex items-center justify-center mb-4 group-hover:bg-red-500 transition-colors">
                  <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" /></svg>
                </div>
                <span className="text-[10px] font-black uppercase tracking-[0.4em]">Start Recording</span>
              </button>

              <label className="flex flex-col items-center justify-center p-8 bg-white border-2 border-dashed border-black/5 rounded-[32px] hover:border-blue-600 hover:bg-blue-50/30 transition-all cursor-pointer group">
                <div className="w-12 h-12 rounded-full bg-blue-600/5 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                  <svg className="w-6 h-6 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" /></svg>
                </div>
                <span className="text-[10px] font-black uppercase tracking-[0.4em] text-blue-600">Upload Audio</span>
                <input type="file" accept="audio/*" onChange={handleFileUpload} className="hidden" />
              </label>
            </div>


          </div>
        </div>
      </div>
    );
  }

  // 2. RECORDING SCREEN
  if (sessionState === 'RECORDING') {
    return (
      <div className="min-h-screen bg-[#FDFDFB] flex flex-col items-center justify-center p-6 relative">
        <div className="absolute top-8 left-8 flex items-center gap-4">
          <div className="px-4 py-2 bg-black/5 rounded-full text-[10px] font-bold tracking-tight text-black/40">
            Scanning: <span className="text-black">{patientName}</span>
          </div>
        </div>

        <div className="w-full max-w-2xl text-center flex flex-col items-center gap-6">
          {/* Stop Button (Top) */}
          <div className="relative">
            <div className={`absolute -inset-8 bg-red-600/5 rounded-full ${isMuted ? '' : 'soft-pulse'}`}></div>
            <button
              onClick={stopRecording}
              className="w-32 h-32 rounded-full flex flex-col items-center justify-center transition-all bg-white border border-black/5 shadow-2xl hover:scale-105 z-10"
            >
              <div className="space-y-2 flex flex-col items-center">
                <div className="w-4 h-4 bg-red-500 rounded-md animate-pulse"></div>
                <span className="text-[8px] font-black uppercase tracking-[0.4em] text-red-500">Stop Sequence</span>
              </div>
            </button>
          </div>

          {/* Time & Status (Middle) */}
          <div className="space-y-2">
            <h2 className="text-[72px] font-extrabold tracking-tighter leading-none mono-data animate-in fade-in duration-500">
              {formatTime(recordingTime)}
            </h2>
            <div className="flex flex-col items-center gap-1 h-6">
              <span className={`text-[9px] font-black uppercase tracking-[0.6em] ${isMuted ? 'text-red-500' : 'text-blue-600'}`}>
                {isMuted ? 'Microphone Muted' : 'Ambient Analysis Active'}
              </span>
            </div>
          </div>

          {/* Mute Button (Bottom) */}
          <button
            onClick={toggleMute}
            className={`w-12 h-12 rounded-full flex items-center justify-center transition-all border ${isMuted ? 'bg-red-500 border-red-500 text-white shadow-lg shadow-red-500/20' : 'bg-white border-black/5 text-black/40 hover:bg-black/5'}`}
            title={isMuted ? "Unmute Microphone" : "Mute Microphone"}
          >
            {isMuted ? (
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 3l18 18" /></svg>
            ) : (
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" /></svg>
            )}
          </button>
        </div>
      </div>
    );
  }

  // 3. CONFIRMATION SCREEN
  if (sessionState === 'CONFIRMATION') {
    return (
      <div className="min-h-screen bg-[#FDFDFB] flex flex-col items-center justify-center p-6">
        <div className="w-full max-w-md space-y-8 animate-in fade-in zoom-in-95 duration-500">
          <div className="text-center space-y-4">
            <span className="text-[10px] font-black uppercase tracking-[0.6em] text-blue-600 block">Verification</span>
            <h2 className="text-3xl font-extrabold tracking-tight">Review Session Audio</h2>
            <p className="text-sm font-medium text-black/40">Ensure audio clarity before processing.</p>
          </div>

          {audioUrl && (
            <div className="p-6 bg-white rounded-[24px] border border-black/5 shadow-xl shadow-black/5">
              <audio controls src={audioUrl} className="w-full" />
            </div>
          )}

          <div className="grid grid-cols-2 gap-4 pt-4">
            <button
              onClick={() => {
                setCapturedAudio(null);
                setAudioUrl(null);
                setSessionState('CHOICE');
              }}
              className="px-6 py-4 bg-white border border-black/5 rounded-2xl text-[10px] font-black uppercase tracking-[0.2em] text-black/40 hover:text-red-500 hover:bg-red-50 hover:border-red-500/20 transition-all"
            >
              Discard
            </button>
            <button
              onClick={() => {
                if (capturedAudio) {
                  // Pass true for autoSave to skip result screen as requested
                  handleProcessAudio(capturedAudio, true);
                }
              }}
              className="px-6 py-4 bg-black text-white rounded-2xl text-[10px] font-black uppercase tracking-[0.2em] shadow-xl shadow-black/20 hover:scale-[1.02] active:scale-[0.98] transition-all flex items-center justify-center gap-2"
            >
              Proceed <Loader2 size={12} className={isSaving ? "animate-spin" : "hidden"} />
            </button>
          </div>
        </div>
      </div>
    );
  }

  // 4. PROCESSING SCREEN
  if (sessionState === 'PROCESSING') {
    return (
      <div className="min-h-screen bg-[#FDFDFB] flex flex-col items-center justify-center p-6">
        <div className="text-center space-y-10">
          <div className="relative w-20 h-20 mx-auto flex items-center justify-center">
            <div className="absolute inset-0 border-2 border-black/5 rounded-full"></div>
            <div className="absolute inset-0 border-t-2 border-blue-600 rounded-full animate-spin"></div>
            <div className="w-3 h-3 bg-blue-600 rounded-full animate-pulse"></div>
          </div>
          <div className="space-y-3">
            <h3 className="text-3xl font-extrabold tracking-tighter">Cognitive Synthesis...</h3>
            <p className="text-[9px] font-black uppercase tracking-[0.4em] text-black/20">Structuring Clinical Schema for {patientName}</p>
          </div>
        </div>
      </div>
    );
  }

  // 4. RESULT / VERIFICATION SCREEN
  if (sessionState === 'RESULT' && generatedNote) {
    return (
      <div className="min-h-screen bg-[#FDFDFB] flex flex-col p-6 md:p-12 animate-in fade-in duration-700">
        <div className="max-w-7xl mx-auto w-full space-y-10">
          <header className="flex flex-col md:flex-row justify-between items-start gap-4 border-b border-black/5 pb-4">
            <div className="flex-1 w-full">
              <span className="text-[8px] font-black uppercase tracking-[0.5em] text-blue-600 mb-1 block">Verification sequence</span>
              <h1 className="text-2xl font-extrabold tracking-tighter text-black">{patientName}</h1>
              <p className="text-[8px] font-bold text-black/20 uppercase tracking-[0.2em] mt-1">Consultation Archive Request • {new Date().toLocaleDateString()}</p>
            </div>

            <div className="flex gap-4">
              <button onClick={onCancel} className="text-[9px] font-black uppercase tracking-widest px-6 py-3 bg-black/5 rounded-xl hover:bg-red-50 hover:text-red-500 transition-all">Discard</button>
              <button
                disabled={isSaving}
                onClick={async () => {
                  setIsSaving(true);
                  // Artificial delay to ensure user sees the loading state
                  await new Promise(resolve => setTimeout(resolve, 2000));

                  await onSave({
                    id: Math.random().toString(36).substr(2, 9),
                    patientId: patientName,
                    date: new Date().toISOString(),
                    type: 'Consultation',
                    content: generatedNote,
                    rawTranscript: fullTranscript,
                    utterances: utterances,
                    summary: (generatedNote as any).summary || generatedNote.assessment,
                    keyInsights: keyInsights,
                    // Pass extended patient info for saving
                    ...({
                      age: parseInt(patientAge),
                      weight: patientWeight,
                      height: patientHeight,
                      bloodGroup: patientBloodGroup,
                      gender: patientGender,
                      nationality: patientNationality,
                      language: patientLanguage
                    } as any)
                  }, capturedAudio || undefined);
                  setIsSaving(false);
                }}
                className={`btn-nordic px-8 py-3 text-[9px] font-black uppercase tracking-[0.3em] flex items-center gap-2 ${isSaving ? 'opacity-70 cursor-wait' : ''}`}
              >
                {isSaving && <Loader2 size={12} className="animate-spin" />}
                {isSaving ? 'Archiving...' : 'Archive to Ledger'}
              </button>
            </div>
          </header>

          {/* AUDIO PREVIEW - NEW FEATURE */}
          {
            audioUrl && (
              <div className="p-4 bg-white rounded-[20px] border border-black/5 shadow-sm space-y-2">
                <div className="flex items-center gap-2">
                  <div className="w-1.5 h-1.5 bg-blue-600 rounded-full animate-pulse"></div>
                  <span className="text-[8px] font-black uppercase tracking-widest text-black/40">Verify Session Audio</span>
                </div>
                <audio controls src={audioUrl} className="w-full h-8" />
              </div>
            )
          }

          {/* Key Insights */}
          {
            keyInsights.length > 0 && (
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <div className="w-0.5 h-3 bg-orange-500 rounded-full"></div>
                  <h3 className="text-[10px] font-black uppercase tracking-[0.3em] text-orange-600">Key Insights</h3>
                </div>
                <div className="bg-orange-50/50 p-4 rounded-[20px] border border-orange-100/50 space-y-1">
                  {keyInsights.map((insight, idx) => (
                    <div key={idx} className="flex gap-2 items-start">
                      <span className="text-orange-400 font-bold text-xs mt-0.5">•</span>
                      <p className="text-xs font-medium text-black/80">{insight}</p>
                    </div>
                  ))}
                </div>
              </div>
            )
          }

          {/* Transcript View - Requested Feature */}
          {
            utterances.length > 0 && (
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <div className="w-0.5 h-3 bg-black rounded-full"></div>
                  <h3 className="text-[10px] font-black uppercase tracking-[0.3em] text-black/50">Session Transcript</h3>
                </div>
                <div className="bg-white p-6 rounded-[24px] border border-black/5 max-h-[300px] overflow-y-auto space-y-4 custom-scrollbar shadow-sm">
                  {utterances.map((u, i) => (
                    <div key={i} className="flex gap-4 group hover:bg-black/[0.02] p-2 rounded-xl transition-colors">
                      <span className={`text-[9px] font-black uppercase tracking-widest w-20 shrink-0 pt-0.5 truncate ${u.speaker === 'A' ? 'text-black' : 'text-black/40'}`}>
                        {`Speaker ${u.speaker}`}
                      </span>
                      <p className="text-xs font-medium text-black/70 leading-relaxed group-hover:text-black transition-colors">{u.text}</p>
                    </div>
                  ))}
                </div>
              </div>
            )
          }

          {/* Record Controls */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-10">
            <EditField title="Subjective" value={generatedNote.subjective.historyOfPresentIllness} label="Presenting Context" />
            <EditField title="Assessment" value={generatedNote.assessment} label="Clinical Logic" highlight />
            <EditField title="Action Plan" value={generatedNote.plan} label="Orders & Schedule" />
          </div>
        </div >
      </div >
    );
  }

  return null;
};

const EditField = ({ title, value, highlight, label }: { title: string; value: string; highlight?: boolean; label: string }) => (
  <div className="space-y-2">
    <div className="flex items-center gap-2">
      <div className="w-0.5 h-2 bg-black"></div>
      <label className="text-[8px] font-black uppercase tracking-widest text-black/30">{title}</label>
    </div>
    <div
      contentEditable
      className={`p-4 rounded-[24px] focus:outline-none focus:ring-2 focus:ring-blue-600/10 transition-all font-medium leading-relaxed text-xs border ${highlight ? 'bg-blue-600 text-white border-blue-600 italic shadow-xl shadow-blue-600/10' : 'bg-white text-black/60 border-black/5'}`}
      dangerouslySetInnerHTML={{ __html: value }}
    />
    <p className="text-[7px] font-black uppercase tracking-widest text-black/10 ml-4">{label}</p>
  </div>
);

export default ScribeSession;
