import React, { useState, useRef, useEffect } from 'react';
import { Plus, Search, Calendar, Clock, ChevronRight, Mic, Play, Pause, Square, FileText, ChevronDown, Trash2, Edit2, LogOut, Menu, X, ExternalLink, Sparkles, AlertCircle, Upload, File, MoreVertical, Check, Loader2, Download, Printer, Activity, Heart, Wind, Info, Volume2, Settings, User, Bell, RefreshCcw, BrainCircuit, ChevronLeft, Stethoscope } from 'lucide-react';
import Sidebar from './Sidebar';
import AudioPlayer from './AudioPlayer';
import { PrescriptionPad } from './PrescriptionPad';
import { uploadPatientFile, getConsultationFiles, deletePatientFile } from '@/services/fileUploadService';
import { User as UserType, Patient, MedicalNote, View, PatientFile, PrefilledPatientData } from '@/types';
import { askGroq } from '@/services/groqService';

// --- Types ---
interface ChatMessage {
  id: string;
  role: 'user' | 'ai';
  text: string;
}

interface DashboardProps {
  user: UserType;
  patients: Patient[];
  onLogout: () => void;
  onStartScribe: (data?: PrefilledPatientData) => void;
  onDeleteConsultation: (patientId: string, consultationId: string) => void;
  onDeletePatient: (patientId: string) => void;
  onUpdatePatient: (patientId: string, updates: { name?: string; age?: number; gender?: string; weight?: string; height?: string; bloodGroup?: string; nationality?: string; language?: string }) => void;
  onUpdateConsultation: (consultationId: string, updates: Partial<MedicalNote>) => void;
  onFileUpload?: (patientId: string, file: File) => Promise<void>;
  onFileDelete?: (patientId: string, fileName: string) => Promise<void>;
  defaultPatientId?: string | null;
  onClearDefaultPatient?: () => void;
}

const TabType = {
  SUMMARY: 'SUMMARY',
  TRANSCRIPT: 'TRANSCRIPT',
  INSIGHTS: 'INSIGHTS',
  UPLOADS: 'UPLOADS',
  PRESCRIPTION: 'PRESCRIPTION'
};

// --- Sub-components (Moved up for scope visibility) ---

const InsightCard = ({ title, items, color }: any) => {
  const isPurple = color === 'purple';
  return (
    <div className={`p - 6 border rounded - 3xl ${isPurple ? 'bg-[#faf8ff] border-purple-100' : 'bg-rose-50/30 border-rose-100'} `}>
      <h5 className={`text - [10px] font - black uppercase tracking - [0.2em] mb - 4 ${isPurple ? 'text-[#552483]' : 'text-rose-600'} `}>
        {title}
      </h5>
      <ul className="space-y-3">
        {items.map((item: string, i: number) => (
          <li key={i} className="flex items-center gap-3 text-[11px] font-bold text-slate-600">
            <div className={`w - 1.5 h - 1.5 rounded - full ${isPurple ? 'bg-[#552483]' : 'bg-rose-400'} `} />
            {item}
          </li>
        ))}
      </ul>
    </div>
  );
}

const ActionBtn = ({ icon, label, color = "text-slate-400", onClick }: any) => (
  <button onClick={onClick} className={`px-3 py-2 sm:px-4 sm:py-2.5 rounded-xl flex items-center gap-2 sm:gap-2.5 hover:bg-slate-50 transition-all ${color} group`}>
    <span className="opacity-70 group-hover:opacity-100 transition-opacity">{icon}</span>
    <span className="text-[10px] font-bold font-sans tracking-widest uppercase">{label}</span>
  </button>
);

const Tab = ({ label, active, onClick }: { label: string, active: boolean, onClick: () => void }) => (
  <button
    onClick={onClick}
    className={`py-6 relative text-[10px] sm:text-xs font-bold font-sans tracking-[0.1em] md:tracking-[0.2em] transition-all shrink-0 whitespace-nowrap ${active ? 'text-black' : 'text-slate-400 hover:text-slate-600'}`}
  >
    {label}
    {active && (
      <div className="absolute bottom-0 left-0 w-full h-[3px] bg-black rounded-full shadow-[0_-4px_10px_rgba(0,0,0,0.2)]" />
    )}
  </button>
);

// --- Reuse existing Modal ---
interface NewPatientModalProps {
  initialName?: string;
  onClose: () => void;
  onStart: (data: PrefilledPatientData) => void;
}

const NewPatientModal: React.FC<NewPatientModalProps> = ({ initialName, onClose, onStart }) => {
  const [name, setName] = useState(initialName || '');
  const [age, setAge] = useState('');
  const [weight, setWeight] = useState('');
  const [height, setHeight] = useState('');
  const [bloodGroup, setBloodGroup] = useState('');
  const [gender, setGender] = useState('Unknown');
  const [nationality, setNationality] = useState('');
  const [language, setLanguage] = useState('English');

  const handleStart = (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) return;
    onStart({ name, age, weight, height, bloodGroup, gender, nationality, language });
  };

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-black/20 backdrop-blur-sm animate-in fade-in duration-300" onClick={onClose}></div>
      <div className="relative w-full max-w-lg bg-white rounded-[24px] shadow-2xl shadow-black/10 p-6 animate-in zoom-in-95 slide-in-from-bottom-10 duration-500">
        <div className="flex justify-between items-start mb-4">
          <div>
            <span className="text-[8px] font-black uppercase tracking-[0.4em] text-black mb-1 block">New Session</span>
            <h2 className="text-xl font-extrabold tracking-tighter text-slate-800">Patient Intake</h2>
          </div>
          <button onClick={onClose} className="w-8 h-8 rounded-full bg-slate-50 flex items-center justify-center hover:bg-slate-100 transition-colors text-slate-400">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>
          </button>
        </div>

        <form onSubmit={handleStart} className="space-y-3">
          <div className="space-y-1">
            <label className="text-[7px] font-black uppercase tracking-widest text-slate-400 ml-4">Full Name</label>
            <input
              type="text"
              required
              autoFocus
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Start typing name..."
              className="w-full bg-slate-50 border border-slate-100 rounded-xl px-4 py-3 text-sm font-bold text-slate-700 focus:border-black focus:bg-white outline-none transition-all placeholder:text-slate-300"
            />
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1">
              <label className="text-[7px] font-black uppercase tracking-widest text-slate-400 ml-4">Age</label>
              <input
                type="number"
                value={age}
                onChange={(e) => setAge(e.target.value)}
                placeholder="Years"
                className="w-full bg-slate-50 border border-slate-100 rounded-xl px-4 py-3 text-xs font-bold text-slate-700 focus:border-black focus:bg-white outline-none transition-all"
              />
            </div>
            <div className="space-y-1">
              <label className="text-[7px] font-black uppercase tracking-widest text-slate-400 ml-4">Gender</label>
              <select
                value={gender}
                onChange={(e) => setGender(e.target.value)}
                className="w-full bg-slate-50 border border-slate-100 rounded-xl px-4 py-3 text-xs font-bold text-slate-700 focus:border-black focus:bg-white outline-none transition-all appearance-none"
              >
                <option value="">Select</option>
                <option value="Male">Male</option>
                <option value="Female">Female</option>
                <option value="Other">Other</option>
              </select>
            </div>
            <div className="space-y-1">
              <label className="text-[7px] font-black uppercase tracking-widest text-slate-400 ml-4">Nationality</label>
              <input
                type="text"
                value={nationality}
                onChange={(e) => setNationality(e.target.value)}
                placeholder="Country"
                className="w-full bg-slate-50 border border-slate-100 rounded-xl px-4 py-3 text-xs font-bold text-slate-700 focus:border-black focus:bg-white outline-none transition-all"
              />
            </div>
            <div className="space-y-1">
              <label className="text-[7px] font-black uppercase tracking-widest text-slate-400 ml-4">Blood Group</label>
              <select
                value={bloodGroup}
                onChange={(e) => setBloodGroup(e.target.value)}
                className="w-full bg-slate-50 border border-slate-100 rounded-xl px-4 py-3 text-xs font-bold text-slate-700 focus:border-black focus:bg-white outline-none transition-all appearance-none"
              >
                <option value="">Select</option>
                {['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-'].map(bg => (
                  <option key={bg} value={bg}>{bg}</option>
                ))}
              </select>
            </div>

            <div className="space-y-1 col-span-2">
              <label className="text-[7px] font-black uppercase tracking-widest text-slate-400 ml-4">Consultation Language</label>
              <select
                value={language}
                onChange={(e) => setLanguage(e.target.value)}
                className="w-full bg-slate-50 border border-slate-100 rounded-xl px-4 py-3 text-xs font-bold text-slate-700 focus:border-black focus:bg-white outline-none transition-all appearance-none"
              >
                <option value="English">English</option>
                <option value="Malayalam">Malayalam</option>
                <option value="Hindi">Hindi</option>
                <option value="Arabic">Arabic</option>
              </select>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1">
              <label className="text-[7px] font-black uppercase tracking-widest text-slate-400 ml-4">Weight</label>
              <input
                type="text"
                value={weight}
                onChange={(e) => setWeight(e.target.value)}
                placeholder="kg"
                className="w-full bg-slate-50 border border-slate-100 rounded-xl px-4 py-3 text-xs font-bold text-slate-700 focus:border-black focus:bg-white outline-none transition-all"
              />
            </div>
            <div className="space-y-1">
              <label className="text-[7px] font-black uppercase tracking-widest text-slate-400 ml-4">Height</label>
              <input
                type="text"
                value={height}
                onChange={(e) => setHeight(e.target.value)}
                placeholder="cm"
                className="w-full bg-slate-50 border border-slate-100 rounded-xl px-4 py-3 text-xs font-bold text-slate-700 focus:border-black focus:bg-white outline-none transition-all"
              />
            </div>
          </div>

          <div className="pt-2">
            <button
              type="submit"
              className="w-full py-3 bg-black text-white rounded-xl text-[10px] font-black uppercase tracking-[0.3em] shadow-xl shadow-black/10 hover:bg-slate-800 transition-all"
            >
              Start Recording
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

// --- Edit Patient Modal ---
interface EditPatientModalProps {
  patient: Patient;
  onClose: () => void;
  onSave: (updates: { name?: string; age?: number; gender?: string; weight?: string; height?: string; bloodGroup?: string; nationality?: string; language?: string }) => void;
}

const EditPatientModal: React.FC<EditPatientModalProps> = ({ patient, onClose, onSave }) => {
  const [name, setName] = useState(patient.name || '');
  const [age, setAge] = useState(patient.age?.toString() || '');
  const [weight, setWeight] = useState(patient.weight || '');
  const [height, setHeight] = useState(patient.height || '');
  const [bloodGroup, setBloodGroup] = useState(patient.bloodGroup || '');
  const [gender, setGender] = useState(patient.gender || 'Unknown');
  const [nationality, setNationality] = useState(patient.nationality || '');
  const [language, setLanguage] = useState(patient.language || 'English');

  const handleSave = (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) return;
    onSave({ name, age: parseInt(age) || 0, weight, height, bloodGroup, gender, nationality, language });
    onClose();
  };

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-black/20 backdrop-blur-sm animate-in fade-in duration-300" onClick={onClose}></div>
      <div className="relative w-full max-w-lg bg-white rounded-[24px] shadow-2xl p-6 animate-in zoom-in-95 slide-in-from-bottom-10 duration-500">
        <div className="flex justify-between items-start mb-4">
          <div>
            <span className="text-[8px] font-black uppercase tracking-[0.4em] text-black mb-1 block">Edit Details</span>
            <h2 className="text-xl font-extrabold tracking-tighter text-slate-800">Patient Information</h2>
          </div>
          <button onClick={onClose} className="w-8 h-8 rounded-full bg-slate-50 flex items-center justify-center hover:bg-slate-100 transition-colors text-slate-400">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>
          </button>
        </div>

        <form onSubmit={handleSave} className="space-y-3">
          <div className="space-y-1">
            <label className="text-[7px] font-black uppercase tracking-widest text-slate-400 ml-4">Full Name</label>
            <input type="text" required value={name} onChange={(e) => setName(e.target.value)}
              className="w-full bg-slate-50 border border-slate-100 rounded-xl px-4 py-3 text-sm font-bold text-slate-700 focus:border-black focus:bg-white outline-none transition-all" />
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1">
              <label className="text-[7px] font-black uppercase tracking-widest text-slate-400 ml-4">Age</label>
              <input type="number" value={age} onChange={(e) => setAge(e.target.value)} placeholder="Years"
                className="w-full bg-slate-50 border border-slate-100 rounded-xl px-4 py-3 text-xs font-bold text-slate-700 focus:border-black focus:bg-white outline-none transition-all" />
            </div>
            <div className="space-y-1">
              <label className="text-[7px] font-black uppercase tracking-widest text-slate-400 ml-4">Gender</label>
              <select value={gender} onChange={(e) => setGender(e.target.value)}
                className="w-full bg-slate-50 border border-slate-100 rounded-xl px-4 py-3 text-xs font-bold text-slate-700 focus:border-black focus:bg-white outline-none transition-all appearance-none">
                <option value="">Select</option>
                <option value="Male">Male</option>
                <option value="Female">Female</option>
                <option value="Other">Other</option>
              </select>
            </div>
            <div className="space-y-1">
              <label className="text-[7px] font-black uppercase tracking-widest text-slate-400 ml-4">Nationality</label>
              <input
                type="text"
                value={nationality}
                onChange={(e) => setNationality(e.target.value)}
                placeholder="Country"
                className="w-full bg-slate-50 border border-slate-100 rounded-xl px-4 py-3 text-xs font-bold text-slate-700 focus:border-black focus:bg-white outline-none transition-all"
              />

            </div>
            <div className="space-y-1">
              <label className="text-[7px] font-black uppercase tracking-widest text-slate-400 ml-4">Blood Group</label>
              <select value={bloodGroup} onChange={(e) => setBloodGroup(e.target.value)}
                className="w-full bg-slate-50 border border-slate-100 rounded-xl px-4 py-3 text-xs font-bold text-slate-700 focus:border-black focus:bg-white outline-none transition-all appearance-none">
                <option value="">Select</option>
                {['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-'].map(bg => (<option key={bg} value={bg}>{bg}</option>))}
              </select>
            </div>
            <div className="space-y-1 col-span-2">
              <label className="text-[7px] font-black uppercase tracking-widest text-slate-400 ml-4">Consultation Language</label>
              <select
                value={language}
                onChange={(e) => setLanguage(e.target.value)}
                className="w-full bg-slate-50 border border-slate-100 rounded-xl px-4 py-3 text-xs font-bold text-slate-700 focus:border-black focus:bg-white outline-none transition-all appearance-none"
              >
                <option value="English">English</option>
                <option value="Malayalam">Malayalam</option>
                <option value="Hindi">Hindi</option>
                <option value="Arabic">Arabic</option>
              </select>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1">
              <label className="text-[7px] font-black uppercase tracking-widest text-slate-400 ml-4">Weight</label>
              <input type="text" value={weight} onChange={(e) => setWeight(e.target.value)} placeholder="kg"
                className="w-full bg-slate-50 border border-slate-100 rounded-xl px-4 py-3 text-xs font-bold text-slate-700 focus:border-black focus:bg-white outline-none transition-all" />
            </div>
            <div className="space-y-1">
              <label className="text-[7px] font-black uppercase tracking-widest text-slate-400 ml-4">Height</label>
              <input type="text" value={height} onChange={(e) => setHeight(e.target.value)} placeholder="cm"
                className="w-full bg-slate-50 border border-slate-100 rounded-xl px-4 py-3 text-xs font-bold text-slate-700 focus:border-black focus:bg-white outline-none transition-all" />
            </div>
          </div>

          <div className="pt-2">
            <button type="submit"
              className="w-full py-3 bg-black text-white rounded-xl text-[10px] font-black uppercase tracking-[0.3em] shadow-xl shadow-black/10 hover:bg-slate-800 transition-all">
              Save Changes
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

// --- Main Component ---
const Dashboard: React.FC<DashboardProps> = ({ user, patients, onLogout, onStartScribe, onDeleteConsultation, onDeletePatient, onUpdatePatient, onUpdateConsultation, onFileUpload, onFileDelete, defaultPatientId, onClearDefaultPatient }) => {
  const [selectedPatient, setSelectedPatient] = useState<Patient | null>(null);
  const [selectedConsultation, setSelectedConsultation] = useState<MedicalNote | null>(null);
  const [activeTab, setActiveTab] = useState(TabType.SUMMARY);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isEditModalOpen, setIsEditModalOpen] = useState(false);
  const [translationLanguage, setTranslationLanguage] = useState<'Original' | 'Hindi' | 'Arabic' | 'Malayalam' | 'English'>('Original');
  const [isTranslateOpen, setIsTranslateOpen] = useState(false);
  const [translatedData, setTranslatedData] = useState<any>(null);
  const [isTranslating, setIsTranslating] = useState(false);
  const [isRAGMode, setIsRAGMode] = useState(false);

  // Sync selectedPatient with latest data from patients prop
  useEffect(() => {
    if (selectedPatient) {
      const updatedPatient = patients.find(p => p.id === selectedPatient.id);
      // Only update if the object reference has actually changed to avoid loops
      if (updatedPatient && updatedPatient !== selectedPatient) {
        console.log('Syncing selectedPatient with latest data:', updatedPatient.name);
        setSelectedPatient(updatedPatient);
      }
    }
  }, [patients, selectedPatient?.id]); // Depend on ID, not object, to find match
  const [isAISearching, setIsAISearching] = useState(false);

  // Generation State
  const [generationResult, setGenerationResult] = useState<{ type: string; content: string } | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [aiQuestion, setAIQuestion] = useState('');
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [seekTime, setSeekTime] = useState<number | null>(null);
  const [isAIChatOpen, setIsAIChatOpen] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [consultationFiles, setConsultationFiles] = useState<PatientFile[]>([]);
  const [isUploadingFile, setIsUploadingFile] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (isAIChatOpen || chatHistory.length > 0) {
      chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [isAIChatOpen, chatHistory]);

  // Auto-select most recent consultation
  useEffect(() => {
    if (selectedPatient && selectedPatient.consultations?.length > 0 && !selectedConsultation) {
      setSelectedConsultation(selectedPatient.consultations[0]);
      setSeekTime(null);
    }
  }, [selectedPatient]);

  // Consuming Default Patient ID (Auto-Navigation)
  useEffect(() => {
    if (defaultPatientId && patients.length > 0) {
      const match = patients.find(p => p.id === defaultPatientId);
      if (match) {
        if (!selectedPatient || selectedPatient.id !== match.id) {
          setSelectedPatient(match);
          window.history.pushState({ view: 'detail', patientId: match.id }, '', '');
        }
        onClearDefaultPatient?.();
      }
    }
  }, [defaultPatientId, patients]);

  // Sync selectedPatient with fresh data from patients prop
  useEffect(() => {
    if (selectedPatient) {
      const freshPatient = patients.find(p => p.id === selectedPatient.id);
      if (freshPatient) {
        // Update to fresh patient data
        setSelectedPatient(freshPatient);
        // If selected consultation no longer exists, clear it
        if (selectedConsultation) {
          const stillExists = freshPatient.consultations.find(c => c.id === selectedConsultation.id);
          if (!stillExists) {
            // Select the most recent consultation if any exist
            if (freshPatient.consultations.length > 0) {
              setSelectedConsultation(freshPatient.consultations[0]);
            } else {
              setSelectedConsultation(null);
            }
          }
        }
      } else {
        // Patient was deleted entirely
        setSelectedPatient(null);
        setSelectedConsultation(null);
      }
    }
  }, [patients]);

  // Handle Translation
  // Handle Translation
  useEffect(() => {
    const translateContent = async () => {
      // Optimize: If target language matches source, skip translation
      const sourceLanguage = selectedPatient?.language || 'English';

      if (translationLanguage === 'Original' || !selectedConsultation) {
        setTranslatedData(null);
        return;
      }

      setIsTranslating(true);
      try {
        const { translateText } = await import('../services/groqService');
        const content = selectedConsultation.content;
        const utterances = selectedConsultation.utterances || [];
        const insights = selectedConsultation.keyInsights || [];

        // 1. Translate Clinical Summary (Subjective, Assessment, Plan)
        // Check if we have a direct 'summary' field (new format) or 'subjective' (legacy)
        const textToTranslateSummary = content.summary || content.subjective?.historyOfPresentIllness || 'No summary available';

        const [summary, assessment, plan] = await Promise.all([
          translateText(textToTranslateSummary, translationLanguage),
          translateText(content.assessment, translationLanguage),
          translateText(content.plan, translationLanguage)
        ]);

        // 2. Translate Insights (Batch if small, or join)
        // Using delimiter method: Join with " ||| " -> Translate -> Split
        const insightsText = insights.join(' ||| ');
        const translatedInsightsStr = await translateText(insightsText, translationLanguage);
        const translatedInsights = translatedInsightsStr.split(' ||| ').map(s => s.trim());

        // 3. Translate Transcript (Utterances)
        // Optimization: If the target language matches the specific patient language (source audio), 
        // DO NOT translate the transcript (keep original).
        // Only translate transcript if languages differ (e.g. English summary -> Malayalam input is fine, but Malayalam audio -> Malayalam target should not change).
        let translatedUtterances = [...utterances];

        if (translationLanguage !== sourceLanguage && utterances.length > 0) {
          // This can be large, so we might need to chunk it. For now, let's try one big batch with delimiters.
          // If > 20 utterances, maybe take top 20 or chunk. Let's try top 30 for performance assurance.
          const transcriptText = utterances.slice(0, 30).map(u => u.text).join(' ||| ');

          if (transcriptText) {
            const translatedTranscriptStr = await translateText(transcriptText, translationLanguage);
            const translatedTexts = translatedTranscriptStr.split(' ||| ');
            translatedUtterances = utterances.map((u, i) => ({
              ...u,
              text: i < 30 ? (translatedTexts[i]?.trim() || u.text) : u.text
            }));
          }
        }

        // 4. Translate Prescription (if available)
        let translatedPrescription = selectedConsultation.prescription;
        if (selectedConsultation.prescription) {
          const prez = selectedConsultation.prescription;
          const notesPromise = translateText(prez.notes, translationLanguage);

          // Translate medications in parallel
          const medicationPromises = (prez.medications || []).map(async (med: any) => ({
            ...med,
            instructions: await translateText(med.instructions, translationLanguage),
            frequency: await translateText(med.frequency, translationLanguage),
          }));

          const [notesTr, medsTr] = await Promise.all([notesPromise, Promise.all(medicationPromises)]);

          translatedPrescription = {
            ...prez,
            notes: notesTr,
            medications: medsTr
          };
        }

        setTranslatedData({
          summary: summary, // Use the unified summary field
          subjective: { ...content.subjective, historyOfPresentIllness: summary }, // Fallback update just in case
          assessment: assessment,
          plan: plan,
          objective: content.objective,
          keyInsights: translatedInsights,
          utterances: translatedUtterances,
          prescription: translatedPrescription
        });

      } catch (error) {
        console.error('Translation failed:', error);
      } finally {
        setIsTranslating(false);
      }
    };

    translateContent();
  }, [translationLanguage, selectedConsultation]);


  // Handle Browser History (Back Button)
  useEffect(() => {
    const handlePopState = (event: PopStateEvent) => {
      if (selectedPatient) {
        // If we have a patient selected and user presses back, clear selection
        setSelectedPatient(null);
        setSelectedConsultation(null);
      }
    };

    window.addEventListener('popstate', handlePopState);
    return () => window.removeEventListener('popstate', handlePopState);
  }, [selectedPatient]);

  const handlePatientClick = (patient: Patient) => {
    // Push new history state when opening patient
    window.history.pushState({ type: 'patient', id: patient.id }, '', '');
    setSelectedPatient(patient);
    setActiveTab(TabType.SUMMARY);
    setActiveTab(TabType.SUMMARY);
    setChatHistory([]);
    setAIQuestion('');
    setSeekTime(null);
  };

  const handleBackToRegistry = () => {
    // Navigate back in history - this triggers popstate which handles the state update
    window.history.back();
  };

  const handleGenerate = async (type: 'Clinical Summary' | 'CPT Code' | 'ICD-10 Code') => {
    if (!selectedConsultation) return;
    setIsGenerating(true);
    setGenerationResult(null);

    let prompt = "";
    // Use utterances if available, otherwise fallback to raw transcript or summary
    const content = JSON.stringify(selectedConsultation.utterances || selectedConsultation.content);

    if (type === 'Clinical Summary') {
      prompt = `Generate a concise clinical summary for the following medical consultation. Focus on key medical events and decisions. Do NOT use markdown formatting. \n\nContext: ${content}`;
    } else if (type === 'CPT Code') {
      prompt = `Suggest the most appropriate CPT code(s) for this consultation. Provide ONLY the code and valid description. Do NOT use markdown. \n\nContext: ${content}`;
    } else if (type === 'ICD-10 Code') {
      prompt = `Suggest the most appropriate ICD-10 code(s) for the diagnosis. Provide ONLY the code and valid description. Do NOT use markdown. \n\nContext: ${content}`;
    }

    const result = await askGroq(prompt);
    setGenerationResult({ type, content: result });
    setIsGenerating(false);
  };

  const handleAskAI = async () => {
    if (!aiQuestion.trim() || !selectedConsultation) return;

    // Optimistic UI Update
    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      text: aiQuestion
    };

    setChatHistory(prev => [...prev, userMessage]);
    setAIQuestion('');
    setIsAISearching(true); // Using existing loading state for UI spinner if desired
    setIsGenerating(true);  // Using new loading state

    try {
      let context = '';

      if (isRAGMode && selectedPatient) {
        // --- RAG MODE: Full History ---

        // 1. Summarize Past Consultations
        // Filter out the *current* consultation so we don't duplicate it in "History"
        // Sort by date ascending (Oldest -> Newest)
        const pastConsultations = selectedPatient.consultations
          .filter(c => c.id !== selectedConsultation.id)
          .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());

        const historyContext = pastConsultations.map((c, index) => {
          // Format date unambiguously
          const dateStr = new Date(c.date).toLocaleDateString('en-GB', {
            day: '2-digit', month: 'long', year: 'numeric', hour: '2-digit', minute: '2-digit'
          });

          return `[Past Consultation #${index + 1}]
Date: ${dateStr} | Type: ${c.type}
Diagnosis: ${c.content.assessment}
Summary: ${c.content.summary || c.content.subjective?.historyOfPresentIllness || 'N/A'}
Plan: ${c.content.plan}
-------------------`;
        }).join('\n');

        // 2. Fetch Global File Context
        const { getPatientFileContext } = await import('../services/fileUploadService');
        const fileContext = await getPatientFileContext(selectedPatient.id);

        context = `
You are an advanced medical AI assistant.

=== PATIENT ===
Name: ${selectedPatient.name} | Age: ${selectedPatient.age} | Gender: ${selectedPatient.gender}

=== PAST MEDICAL HISTORY (Do NOT confuse with Current) ===
${historyContext.length > 0 ? historyContext : 'No past consultations.'}

=== UPLOADED PATIENT FILES ===
${fileContext}

=== CURRENT ACTIVE SESSION ===
Date: ${new Date(selectedConsultation.date).toLocaleDateString('en-GB')}
Content: ${JSON.stringify(selectedConsultation.content)}

INSTRUCTIONS:
1. You are answering questions about the patient's history.
2. VERIFY SPECIFIC RECORDS before claiming a symptom exists.
3. If Consultation #1 says "No chest pain" and #3 says "Chest pain", DO NOT say "Consistent chest pain". Say "Chest pain appeared in #3 but was absent in #1".
4. When listing history, use the format: "Date: Event".
5. Be extremely precise with chronology.
`;
      } else {
        // --- STANDARD MODE ---
        const { getPatientFileContext } = await import('../services/fileUploadService');
        const fileContext = await getPatientFileContext(selectedPatient?.id || '', selectedConsultation?.id);

        context = `
You are a medical assistant helping with the CURRENT consultation only.

Patient: ${selectedPatient?.name}
Current Consultation:
${JSON.stringify(selectedConsultation?.utterances || selectedConsultation?.content)}

Attached Files:
${fileContext}
`;
      }

      const { askGroq } = await import('../services/groqService'); // Ensure import if not at top, or rely on top lvl
      const response = await askGroq(userMessage.text, context);

      const aiMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'ai',
        text: response
      };
      setChatHistory(prev => [...prev, aiMessage]);

    } catch (error) {
      console.error('AI Chat Error:', error);
      setChatHistory(prev => [...prev, {
        id: Date.now().toString(),
        role: 'ai',
        text: "I'm sorry, I encountered an error analyzing the medical data."
      }]);
    } finally {
      setIsGenerating(false);
      setIsAISearching(false);
    }
  };

  // Fetch files when consultation changes
  useEffect(() => {
    const fetchFiles = async () => {
      if (selectedConsultation && selectedPatient) {
        setConsultationFiles([]); // Clear previous
        try {
          const files = await getConsultationFiles(selectedPatient.id, selectedConsultation.id);
          setConsultationFiles(files);
        } catch (error) {
          console.error('Failed to fetch files:', error);
        }
      }
    };

    fetchFiles();
  }, [selectedConsultation, selectedPatient]);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !selectedPatient || !selectedConsultation) return;

    setIsUploadingFile(true);
    try {
      const uploadedFile = await uploadPatientFile(selectedPatient.id, file, selectedConsultation.id);
      setConsultationFiles(prev => [uploadedFile, ...prev]);

      // Reset input
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    } catch (error: any) {
      console.error('File upload error:', error);
      alert(`Failed to upload file: ${error.message}`);
    } finally {
      setIsUploadingFile(false);
    }
  };

  const handleFileDelete = async (fileName: string) => {
    if (!selectedPatient || !selectedConsultation || !window.confirm('Are you sure you want to delete this file?')) return;

    try {
      await deletePatientFile(selectedPatient.id, fileName, selectedConsultation.id);
      setConsultationFiles(prev => prev.filter(f => f.fileName !== fileName));
    } catch (error: any) {
      console.error('Delete error:', error);
      alert(`Failed to delete file: ${error.message}`);
    }
  };

  const handleDelete = () => {
    if (selectedPatient && selectedConsultation) {
      if (window.confirm('Are you sure you want to delete this consultation record? This action cannot be undone.')) {
        onDeleteConsultation(selectedPatient.id, selectedConsultation.id);
        // Clear current selection - the data will refresh and useEffect will handle auto-selection
        setSelectedConsultation(null);
        setSeekTime(null);
      }
    }
  };

  const handleDeletePatientClick = () => {
    if (selectedPatient) {
      if (window.confirm(`Are you sure you want to delete patient "${selectedPatient.name}" and ALL their consultation records? This action cannot be undone.`)) {
        onDeletePatient(selectedPatient.id);
        setSelectedPatient(null);
        setSelectedConsultation(null);
      }
    }
  };

  const handleDeletePatientFromRegistry = (e: React.MouseEvent, patient: Patient) => {
    e.stopPropagation();
    if (window.confirm(`Are you sure you want to delete patient "${patient.name}" and ALL their consultation records? This action cannot be undone.`)) {
      onDeletePatient(patient.id);
    }
  };

  // --- Registry View ---
  if (!selectedPatient) {
    return (
      <div className="min-h-screen bg-[#FDFDFB] px-8 py-4 flex flex-col">
        <header className="flex justify-between items-center mb-6">
          <div className="flex items-center gap-4">
            <div className="w-10 h-10 rounded-2xl bg-black flex items-center justify-center text-white font-bold text-xs shadow-lg shadow-black/10">
              {user.name.charAt(0)}
            </div>
            <div>
              <p className="text-xs font-bold text-slate-800 leading-none">{user.name}</p>
              <p className="text-[9px] font-bold text-slate-500 uppercase tracking-wider mt-1">{user.specialty}</p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <button
              onClick={onLogout}
              className="flex items-center text-[10px] font-bold text-slate-400 hover:text-rose-500 transition-colors uppercase tracking-widest"
            >
              Exit <LogOut size={14} className="ml-2" />
            </button>
            <button
              onClick={() => setIsModalOpen(true)}
              className="hidden md:flex bg-black text-white px-6 py-3 rounded-2xl text-[10px] font-black uppercase tracking-widest hover:bg-slate-800 transition-all shadow-xl shadow-black/10"
            >
              New Session
            </button>
          </div>
        </header>

        <div className="max-w-7xl mx-auto w-full flex-1 flex flex-col">
          <div className="mb-6">
            <h1 className="text-3xl font-black text-slate-800 tracking-tight mb-2">Patient Registry</h1>
            <p className="text-slate-400 font-medium text-sm mb-4">Select a patient to view consultation history</p>

            {/* Search Bar */}
            <div className="relative max-w-md">
              <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400" size={18} />
              <input
                type="text"
                placeholder="Search patients by name..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-12 pr-4 py-3 bg-white border border-slate-200 rounded-2xl text-sm font-medium text-slate-700 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-black/5 focus:border-slate-400 transition-all shadow-sm"
              />
              {searchQuery && (
                <button
                  onClick={() => setSearchQuery('')}
                  className="absolute right-3 top-1/2 -translate-y-1/2 p-1 hover:bg-slate-100 rounded-lg transition-colors"
                >
                  <X size={16} className="text-slate-400" />
                </button>
              )}
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6 max-w-7xl">
            {patients
              .filter(patient =>
                patient.name.toLowerCase().includes(searchQuery.toLowerCase())
              )
              .map(patient => (
                <div
                  key={patient.id}
                  onClick={() => handlePatientClick(patient)}
                  className="bg-white p-3 rounded-[2rem] border border-slate-100 shadow-sm hover:shadow-xl hover:shadow-slate-200/50 hover:border-slate-200 transition-all cursor-pointer group"
                >
                  <div className="flex justify-between items-start mb-3">
                    <div className="w-12 h-12 rounded-2xl bg-slate-50 flex items-center justify-center text-black font-bold text-lg group-hover:bg-black group-hover:text-white transition-colors">
                      {patient.name.charAt(0)}
                    </div>
                    <div className="flex flex-col items-end gap-2">
                      <span className="bg-slate-50 text-black px-3 py-1 rounded-full text-[9px] font-black">{patient.consultations?.length || 0} FILES</span>
                      <button
                        onClick={(e) => handleDeletePatientFromRegistry(e, patient)}
                        className="w-8 h-8 flex items-center justify-center text-slate-300 hover:text-red-500 hover:bg-red-50 rounded-full transition-all"
                        title="Delete Patient"
                      >
                        <Trash2 size={14} />
                      </button>
                    </div>
                  </div>
                  <h3 className="text-lg font-extrabold text-slate-800 mb-1">{patient.name}</h3>
                  <p className="text-xs font-bold text-slate-400 mb-2">{patient.gender}, {patient.age} Years</p>
                  <div className="flex items-center gap-2 pt-3 border-t border-slate-50">
                    <span className="w-2 h-2 rounded-full bg-emerald-400"></span>
                    <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Last seen: {patient.lastSeen}</span>
                  </div>
                </div>
              ))}

            {/* No Results Message */}
            {searchQuery && patients.filter(p => p.name.toLowerCase().includes(searchQuery.toLowerCase())).length === 0 && (
              <div className="col-span-full flex flex-col items-center justify-center py-16 text-center">
                <div className="w-16 h-16 rounded-full bg-slate-100 flex items-center justify-center mb-4">
                  <Search size={28} className="text-slate-300" />
                </div>
                <h3 className="text-lg font-bold text-slate-600 mb-2">No patients found</h3>
                <p className="text-sm text-slate-400 mb-4">No patients match "{searchQuery}"</p>
                <button
                  onClick={() => setSearchQuery('')}
                  className="text-xs font-bold text-black hover:underline"
                >
                  Clear search
                </button>
              </div>
            )}

            <button
              onClick={() => setIsModalOpen(true)}
              className="hidden md:flex bg-slate-50 border-2 border-dashed border-slate-200 rounded-[2rem] flex-col items-center justify-center text-center p-6 hover:bg-slate-100 hover:border-black/10 transition-all group min-h-[240px] max-w-xs mx-auto md:mx-0"
            >
              <div className="w-12 h-12 rounded-full bg-white flex items-center justify-center mb-4 shadow-sm group-hover:scale-110 transition-transform text-black">
                <ExternalLink size={20} />
              </div>
              <span className="text-[10px] font-black uppercase tracking-widest text-black/40 group-hover:text-black transition-colors">Register New Patient</span>
            </button>
          </div>
        </div>

        {/* Mobile FAB for New Session on Registry */}
        <button
          onClick={() => setIsModalOpen(true)}
          className="md:hidden fixed bottom-6 right-6 w-14 h-14 bg-black text-white rounded-full flex items-center justify-center shadow-2xl shadow-black/30 z-50 active:scale-90 transition-transform"
        >
          <Plus size={24} />
        </button>

        {isModalOpen && (
          <NewPatientModal
            onClose={() => setIsModalOpen(false)}
            onStart={(data) => {
              setIsModalOpen(false);
              onStartScribe(data);
            }}
          />
        )}
      </div>
    );
  }

  // --- Detail View (Reference Implementation) ---
  return (
    <div className="flex min-h-screen bg-[#FDFDFB] flex-col md:flex-row">
      <Sidebar
        patient={selectedPatient}
        onBack={handleBackToRegistry}
        activeRecordId={selectedConsultation?.id}
        onSelectRecord={(rec) => {
          setSelectedConsultation(rec);
          setIsSidebarOpen(false); // Close sidebar on mobile selection
        }}
        onDeletePatient={handleDeletePatientClick}
        onEditPatient={() => setIsEditModalOpen(true)}

        isOpen={isSidebarOpen}
        onClose={() => setIsSidebarOpen(false)}
        onStartSession={() => {
          setIsSidebarOpen(false);
          if (selectedPatient) {
            console.log('SIDEBAR START SESSION. Patient:', selectedPatient);
            console.log('SIDEBAR Language:', selectedPatient.language);
            onStartScribe({
              ...selectedPatient,
              age: selectedPatient.age?.toString() || '',
              bloodGroup: selectedPatient.bloodGroup || '',
              weight: selectedPatient.weight || '',
              height: selectedPatient.height || '',
              gender: selectedPatient.gender || 'Unknown',
              nationality: selectedPatient.nationality || '',
              language: selectedPatient.language || 'English'
            });
          } else {
            setIsModalOpen(true);
          }
        }}
      />

      <main className="flex-1 flex flex-col h-screen overflow-hidden relative w-full">
        {/* Header */}
        <header className="h-16 bg-white/80 backdrop-blur-md border-b border-slate-50 px-4 md:px-8 flex items-center justify-between shrink-0 sticky top-0 z-10 w-full">
          <div className="flex items-center gap-4">
            <button
              className="md:hidden -ml-2 flex items-center gap-2"
              onClick={() => setIsSidebarOpen(true)}
            >
              <div className="w-8 h-8 rounded-full bg-slate-900 text-white flex items-center justify-center shadow-md">
                <Menu size={14} />
              </div>
              <span className="text-lg font-black text-slate-800 tracking-tight">{selectedPatient?.name}</span>
            </button>
            {/* Doctor Profile removed as requested */}
          </div>

          <div className="flex items-center gap-6">
            <div className="flex items-center gap-4 text-slate-300">
              {/* Desktop Exit Button */}
              <button
                onClick={onLogout}
                className="hidden md:flex items-center text-[10px] font-bold text-slate-400 hover:text-rose-500 transition-colors uppercase tracking-widest ml-2"
              >
                <span className="mr-2">Exit</span>
                <LogOut size={14} />
              </button>

              {/* Mobile Back Button */}
              <button
                onClick={handleBackToRegistry}
                className="md:hidden flex items-center text-[10px] font-bold text-slate-400 hover:text-black transition-colors uppercase tracking-widest ml-2"
              >
                <span className="mr-2">Back</span>
                <LogOut size={14} className="rotate-180" />
              </button>
            </div>
          </div>
        </header>

        {/* Content Area */}
        <div className="flex-1 overflow-y-auto p-4 md:p-8 custom-scrollbar">
          <div className="max-w-4xl mx-auto w-full">
            {selectedConsultation ? (
              <>
                {/* Top Row: Badge & Translation */}
                <div className="flex items-center justify-between mb-2">
                  <div className="px-2 py-0.5 bg-black/5 text-black text-[9px] font-extrabold rounded-md tracking-widest uppercase flex items-center gap-2">
                    E-Health Record
                  </div>


                  {/* Large Translation Loader Overlay */}
                  {isTranslating && (
                    <div className="fixed inset-0 z-[60] bg-white/80 backdrop-blur-sm flex flex-col items-center justify-center animate-in fade-in duration-300">
                      <div className="flex flex-col items-center gap-6">
                        <div className="relative">
                          <div className="w-16 h-16 border-4 border-slate-100 rounded-full"></div>
                          <div className="w-16 h-16 border-4 border-blue-600 rounded-full border-t-transparent animate-spin absolute top-0 left-0"></div>
                          <Sparkles className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 text-blue-600 animate-pulse" size={24} />
                        </div>
                        <div className="text-center space-y-2">
                          <h3 className="text-xl font-black text-slate-800 tracking-tight">Translating Record...</h3>
                          <p className="text-xs font-bold text-slate-400 uppercase tracking-widest">Converting to {translationLanguage}</p>
                        </div>
                      </div>
                    </div>
                  )}

                  <div className="flex items-center gap-3">
                    {/* Original Button */}
                    <button
                      onClick={() => setTranslationLanguage('Original')}
                      className={`text-[9px] font-bold uppercase tracking-widest transition-colors ${translationLanguage === 'Original' ? 'text-black' : 'text-slate-300 hover:text-slate-500'
                        }`}
                    >
                      Original
                    </button>

                    <div className="w-px h-3 bg-slate-200" />

                    {/* Translate Toggle */}
                    <div className="relative">
                      <button
                        onClick={() => setIsTranslateOpen(!isTranslateOpen)}
                        className={`flex items-center gap-1 text-[9px] font-bold uppercase tracking-widest transition-colors ${translationLanguage !== 'Original' ? 'text-black' : 'text-slate-300 hover:text-slate-500'
                          }`}
                      >
                        {translationLanguage !== 'Original' ? translationLanguage : 'Translate'}
                        <ChevronDown size={10} className={`text-slate-300 transition-transform ${isTranslateOpen ? 'rotate-180' : ''}`} />
                      </button>

                      {/* Dropdown Menu */}
                      {isTranslateOpen && (
                        <>
                          <div
                            className="fixed inset-0 z-10"
                            onClick={() => setIsTranslateOpen(false)}
                          />
                          <div className="absolute top-full right-0 mt-2 bg-white rounded-xl shadow-xl border border-slate-100 p-2 min-w-[120px] z-20 flex flex-col gap-1 animate-in zoom-in-95 duration-200">
                            {['Hindi', 'Arabic', 'Malayalam', 'English'].map((lang) => (
                              <button
                                key={lang}
                                onClick={() => {
                                  setTranslationLanguage(lang as any);
                                  setIsTranslateOpen(false);
                                }}
                                className={`text-left px-3 py-2 rounded-lg text-[10px] font-bold uppercase tracking-wider hover:bg-slate-50 transition-colors ${translationLanguage === lang ? 'bg-black text-white hover:bg-slate-800' : 'text-slate-600'
                                  }`}
                              >
                                {lang}
                              </button>
                            ))}
                          </div>
                        </>
                      )}
                    </div>
                  </div>
                </div>

                {/* Action Bar */}
                <div className="flex flex-col sm:flex-row sm:items-end justify-between mb-6 md:mb-8 gap-4">
                  <div>
                    <h1 className="text-2xl md:text-3xl font-black text-slate-800 tracking-tight leading-none mb-3">Session Archive</h1>
                    <div className="flex items-center gap-3">
                      <div className="flex items-center gap-1.5 text-slate-400">
                        <Clock size={12} />
                        <span className="text-[11px] font-bold uppercase tracking-tighter">
                          {new Date(selectedConsultation.date).toLocaleDateString(undefined, {
                            weekday: 'long',
                            year: 'numeric',
                            month: 'long',
                            day: 'numeric',
                            hour: '2-digit',
                            minute: '2-digit'
                          })}
                        </span>
                      </div>
                    </div>
                  </div>

                  <div className="flex flex-nowrap gap-2 sm:gap-3 w-full sm:w-auto items-stretch bg-white p-1 sm:p-1.5 rounded-[1.25rem] shadow-sm border border-slate-50">
                    {/* Audio Player */}
                    {selectedConsultation.audioUrl && (
                      <div className="flex-1 min-w-0 sm:min-w-[200px] pr-2 mr-1">
                        <AudioPlayer audioUrl={selectedConsultation.audioUrl} seekTime={seekTime} compact />
                      </div>
                    )}

                    {/* Actions */}
                    <div className="flex items-center gap-1 flex-none">
                      <ActionBtn
                        icon={<Sparkles size={14} />}
                        label={<><span className="sm:hidden">AI</span><span className="hidden sm:inline">ASK AI</span></>}
                        color={isAIChatOpen ? "text-black bg-black/5" : "text-black"}
                        onClick={() => setIsAIChatOpen(true)}
                      />
                      <button
                        onClick={handleDelete}
                        className="px-3 py-2 sm:px-4 sm:py-2.5 rounded-xl flex items-center gap-2 sm:gap-2.5 hover:bg-rose-50 transition-all text-rose-400 hover:text-rose-600 group shrink-0"
                      >
                        <span className="opacity-70 group-hover:opacity-100 transition-opacity"><Trash2 size={14} /></span>
                        <span className="text-[10px] font-black tracking-widest uppercase hidden sm:inline">Delete</span>
                      </button>
                    </div>
                  </div>
                </div>

                {/* Tabs & Content */}
                <div className="bg-white rounded-[2.5rem] border border-slate-50 shadow-[0_20px_50px_-15px_rgba(0,0,0,0.05)] flex flex-col min-h-[500px] overflow-hidden">
                  <div className="px-4 md:px-8 border-b border-slate-50 flex items-center gap-3 md:gap-10 bg-slate-50/50 overflow-x-auto scrollbar-hide">
                    <Tab label="SUMMARY" active={activeTab === TabType.SUMMARY} onClick={() => setActiveTab(TabType.SUMMARY)} />
                    <Tab label="TRANSCRIPT" active={activeTab === TabType.TRANSCRIPT} onClick={() => setActiveTab(TabType.TRANSCRIPT)} />
                    <Tab label="INSIGHTS" active={activeTab === TabType.INSIGHTS} onClick={() => setActiveTab(TabType.INSIGHTS)} />
                    <Tab label="PRESCRIPTION" active={activeTab === TabType.PRESCRIPTION} onClick={() => setActiveTab(TabType.PRESCRIPTION)} />
                    <Tab label="UPLOADS" active={activeTab === TabType.UPLOADS} onClick={() => setActiveTab(TabType.UPLOADS)} />
                  </div>

                  {/* Mobile FAB for New Session */}
                  <button
                    onClick={() => {
                      onStartScribe({
                        ...selectedPatient!,
                        age: selectedPatient!.age?.toString() || '',
                        bloodGroup: selectedPatient!.bloodGroup || '',
                        weight: selectedPatient!.weight || '',
                        height: selectedPatient!.height || '',
                        gender: selectedPatient!.gender || 'Unknown',
                        nationality: selectedPatient!.nationality || '',
                        language: selectedPatient!.language || 'English'
                      });
                    }}
                    className="md:hidden fixed bottom-6 right-6 w-14 h-14 bg-black text-white rounded-full flex items-center justify-center shadow-2xl shadow-black/30 z-50 active:scale-90 transition-transform"
                  >
                    <Plus size={24} />
                  </button>

                  <div className="p-5 md:p-10 flex-1">


                    {activeTab === TabType.TRANSCRIPT && (
                      <div className="flex flex-col h-[500px]">
                        {/* Conversation */}
                        <div className="flex-1 overflow-y-auto custom-scrollbar space-y-1 pr-2">
                          {(translatedData?.utterances || selectedConsultation.utterances)?.map((utterance: any, i: number) => {
                            const currentLanguage = selectedPatient?.language || 'English';
                            let speakerLabel = utterance.speaker;

                            // Auto-label A as Doctor for non-Malayalam languages
                            if (currentLanguage !== 'Malayalam' && utterance.speaker === 'A') {
                              speakerLabel = 'Doctor';
                            } else {
                              // Fallback for Malayalam or other speakers
                              speakerLabel = utterance.speaker.length === 1 ? `Speaker ${utterance.speaker}` : utterance.speaker;
                            }
                            const formatTime = (ms: number) => {
                              const seconds = Math.floor(ms / 1000);
                              const mins = Math.floor(seconds / 60);
                              const secs = seconds % 60;
                              return `${mins}:${secs.toString().padStart(2, '0')}`;
                            };

                            return (
                              <div
                                key={i}
                                onClick={() => setSeekTime(utterance.start / 1000)}
                                className={`flex gap-3 p-3 rounded-2xl cursor-pointer transition-all group hover:bg-slate-50`}
                              >
                                {/* Avatar */}
                                <div className={`w-8 h-8 rounded-xl flex items-center justify-center text-[10px] font-black shrink-0 ${speakerLabel === 'Doctor' ? 'bg-black text-white' : 'bg-slate-100 text-slate-500'}`}>
                                  {speakerLabel === 'Doctor' ? 'DR' : utterance.speaker.charAt(0).toUpperCase()}
                                </div>

                                {/* Content */}
                                <div className="flex-1 min-w-0">
                                  <div className="flex items-center gap-2 mb-1">
                                    <span
                                      className={`text-[10px] font-bold uppercase tracking-wider hover:underline cursor-pointer text-slate-400`}
                                      onClick={(e) => {
                                        e.stopPropagation();
                                        const newName = window.prompt("Rename speaker:", utterance.speaker);
                                        if (newName && newName !== utterance.speaker) {
                                          const updatedUtterances = selectedConsultation.utterances?.map(u =>
                                            u.speaker === utterance.speaker ? { ...u, speaker: newName } : u
                                          );

                                          // Update local state first for immediate feedback
                                          if (selectedConsultation && updatedUtterances) {
                                            const updatedConsultation = { ...selectedConsultation, utterances: updatedUtterances };
                                            setSelectedConsultation(updatedConsultation);
                                            // Then persist
                                            onUpdateConsultation(selectedConsultation.id, { utterances: updatedUtterances });
                                          }
                                        }
                                      }}
                                      title={"Click to rename speaker"}
                                    >
                                      {speakerLabel}
                                    </span>
                                    <span className="text-[9px] text-slate-300 font-medium">
                                      {formatTime(utterance.start)}
                                    </span>
                                  </div>
                                  <p className={`text-sm leading-relaxed text-slate-500`}>
                                    {utterance.text}
                                  </p>
                                </div>

                                {/* Play indicator on hover */}
                                <div className="opacity-0 group-hover:opacity-100 transition-opacity shrink-0 self-center">
                                  <div className="w-6 h-6 bg-black rounded-lg flex items-center justify-center">
                                    <Play size={10} className="text-white ml-0.5" />
                                  </div>
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    )}

                    {activeTab === TabType.INSIGHTS && (
                      <div className="space-y-8 animate-in zoom-in-95 duration-300">
                        {/* Key Insights */}
                        <div>
                          <h4 className="text-[10px] font-bold text-amber-600 uppercase tracking-widest mb-4">Key Insights</h4>
                          <div className="bg-amber-50/50 rounded-[2.5rem] p-6 border border-amber-100 relative overflow-hidden">


                            <div className="space-y-4 relative z-10">
                              {(translatedData?.keyInsights || selectedConsultation.keyInsights || [
                                "Patient shows good adherence to the prescribed medication regimen.",
                                "Blood pressure levels have stabilized compared to the previous visit.",
                                "Recommended to increase daily physical activity to 30 minutes.",
                                "Follow-up scheduled in 2 weeks to monitor progress."
                              ]).map((insight: string, i: number) => (
                                <div key={i} className="flex gap-3 items-start">
                                  <div className="w-1.5 h-1.5 rounded-full bg-black mt-1.5 shrink-0 opacity-50" />
                                  <p className="text-sm font-medium text-slate-600 leading-relaxed">{insight}</p>
                                </div>
                              ))}
                            </div>
                          </div>
                        </div>
                      </div>
                    )}

                    {activeTab === TabType.PRESCRIPTION && (
                      <div className="animate-in zoom-in-95 duration-300 h-full flex flex-col">
                        {/* Generate Button (if empty) */}
                        {(!selectedConsultation.prescription || !selectedConsultation.prescription.medications?.length) && (
                          <div className="mb-6 flex justify-center">
                            <button
                              onClick={async () => {
                                if (!selectedConsultation) return;
                                setIsGenerating(true);
                                try {
                                  const { generatePrescription } = await import('../services/groqService');
                                  // Use rawTranscript or summary
                                  // Construct formatted transcript from utterances if available
                                  let text = selectedConsultation.rawTranscript || '';
                                  if (!text && selectedConsultation.utterances) {
                                    text = selectedConsultation.utterances
                                      .map(u => `Speaker ${u.speaker}: ${u.text}`)
                                      .join('\n');
                                  }
                                  // Fallback if still empty
                                  if (!text) text = JSON.stringify(selectedConsultation.utterances) || '';

                                  if (text.length < 50) {
                                    setIsGenerating(false);
                                    return;
                                  }

                                  const { medications, notes } = await generatePrescription(text);

                                  if (!medications || medications.length === 0) {
                                  }

                                  const newPrescription = {
                                    id: Date.now().toString(),
                                    date: new Date().toISOString().split('T')[0],
                                    medications,
                                    notes
                                  };

                                  // Update local state
                                  const updatedConsultation = { ...selectedConsultation, prescription: newPrescription };
                                  onUpdateConsultation(selectedConsultation.id, updatedConsultation);
                                  setSelectedConsultation(updatedConsultation);

                                } catch (e) {
                                  console.error(e);
                                  alert('Failed to generate prescription');
                                } finally {
                                  setIsGenerating(false);
                                }
                              }}
                              className="flex items-center gap-2 px-6 py-3 bg-black text-white rounded-2xl text-[10px] font-black uppercase tracking-widest hover:bg-slate-800 shadow-xl"
                              disabled={isGenerating}
                            >
                              {isGenerating ? <Loader2 className="animate-spin" size={14} /> : <Sparkles size={14} />}
                              {isGenerating ? 'Generating...' : 'Auto-Generate Prescription'}
                            </button>
                          </div>
                        )}

                        <div className="flex-1 min-h-[600px] border border-slate-100 rounded-xl overflow-hidden">
                          <PrescriptionPad
                            patient={selectedPatient!}
                            medications={(translatedData?.prescription?.medications || selectedConsultation.prescription?.medications) || []}
                            notes={(translatedData?.prescription?.notes || selectedConsultation.prescription?.notes) || ''}
                            doctorName={user.name}
                            doctorSpecialty={user.specialty}
                            onSave={(meds, notes) => {
                              if (!selectedConsultation) return;
                              const updatedRx = {
                                id: selectedConsultation.prescription?.id || Date.now().toString(),
                                date: new Date().toISOString().split('T')[0],
                                medications: meds,
                                notes: notes
                              };
                              const updatedConsultation = { ...selectedConsultation, prescription: updatedRx };
                              onUpdateConsultation(selectedConsultation.id, updatedConsultation);
                              setSelectedConsultation(updatedConsultation);
                              alert('Prescription saved!');
                            }}
                          />
                        </div>
                      </div>
                    )}

                    {activeTab === TabType.UPLOADS && (
                      <div className="animate-in zoom-in-95 duration-300">
                        {/* Upload Button */}
                        <div className="mb-6">
                          <input
                            ref={fileInputRef}
                            type="file"
                            onChange={handleFileUpload}
                            className="hidden"
                            accept=".pdf,.doc,.docx,.png,.jpg,.jpeg,.txt"
                            disabled={isUploadingFile}
                          />
                          <button
                            onClick={() => fileInputRef.current?.click()}
                            disabled={isUploadingFile}
                            className="w-full md:w-auto px-6 py-3 rounded-2xl bg-black text-white font-bold text-xs uppercase tracking-widest hover:bg-slate-800 transition-all shadow-xl shadow-black/10 flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
                          >
                            {isUploadingFile ? <Loader2 size={14} className="animate-spin" /> : <Upload size={14} />}
                            {isUploadingFile ? 'Uploading...' : 'Upload File'}
                          </button>
                        </div>

                        {consultationFiles && consultationFiles.length > 0 ? (
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            {consultationFiles.map((file, idx) => {
                              const fileExtension = file.fileName.split('.').pop()?.toLowerCase() || '';
                              const isImage = ['jpg', 'jpeg', 'png', 'gif', 'webp'].includes(fileExtension);
                              const isPDF = fileExtension === 'pdf';

                              return (
                                <div key={idx} className="bg-white rounded-2xl p-4 border border-slate-100 hover:border-purple-200 transition-all group relative">
                                  {/* Delete Button */}
                                  <button
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      handleFileDelete(file.fileName);
                                    }}
                                    className="absolute top-2 right-2 p-1.5 bg-rose-50 rounded-lg text-rose-400 opacity-0 group-hover:opacity-100 transition-opacity hover:bg-rose-100"
                                    title="Delete File"
                                  >
                                    <Trash2 size={12} />
                                  </button>

                                  <div className="flex items-start gap-3">
                                    <div className="w-10 h-10 rounded-xl bg-purple-50 flex items-center justify-center shrink-0 text-purple-500">
                                      {isImage ? (
                                        <div className="relative w-full h-full rounded-xl overflow-hidden">
                                          <img src={file.fileUrl} alt={file.fileName} className="w-full h-full object-cover" />
                                        </div>
                                      ) : (
                                        <File size={18} />
                                      )}
                                    </div>
                                    <div className="flex-1 min-w-0">
                                      <p className="text-sm font-bold text-slate-800 truncate mb-1">{file.fileName}</p>
                                      <div className="flex items-center gap-2">
                                        <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">{(file.fileSize / 1024).toFixed(1)} KB</span>
                                        <span className="text-[10px] font-bold text-slate-300">•</span>
                                        <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">{new Date(file.uploadedAt).toLocaleDateString()}</span>
                                      </div>
                                    </div>
                                  </div>
                                  <div className="mt-4 flex items-center gap-3">
                                    <a
                                      href={file.fileUrl}
                                      target="_blank"
                                      rel="noopener noreferrer"
                                      className="flex-1 py-2 rounded-xl bg-slate-50 text-slate-600 font-bold text-[10px] uppercase tracking-wider flex items-center justify-center gap-2 hover:bg-slate-100 transition-colors"
                                    >
                                      <ExternalLink size={12} />
                                      View
                                    </a>
                                    <a
                                      href={file.fileUrl}
                                      download={file.fileName}
                                      className="py-2 px-3 rounded-xl bg-slate-50 text-slate-600 hover:text-black hover:bg-slate-100 transition-colors"
                                      title="Download"
                                    >
                                      <Download size={14} />
                                    </a>
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                        ) : (
                          <div className="bg-purple-50/30 rounded-2xl p-8 border border-purple-100 text-center">
                            <div className="w-16 h-16 rounded-full bg-purple-100 flex items-center justify-center mx-auto mb-4">
                              <File size={24} className="text-purple-400" />
                            </div>
                            <p className="text-sm font-bold text-slate-600 mb-2">No files uploaded yet</p>
                            <p className="text-xs text-slate-400">Use the "Upload File" button in the sidebar to add patient documents</p>
                          </div>
                        )}
                      </div>
                    )}

                    {activeTab === TabType.SUMMARY && (
                      <div className="animate-in fade-in zoom-in-95 duration-500 pb-10">
                        <div className="bg-white rounded-[2.5rem] border border-slate-100 shadow-xl shadow-slate-100/50 overflow-hidden">
                          <div className="p-8 space-y-8">
                            {/* Generation Actions */}


                            {/* Summary / Subjective */}
                            <section>
                              <h4 className="text-[10px] font-bold text-rose-400 uppercase tracking-widest mb-3">Summary</h4>
                              <p className="text-slate-600 font-medium text-sm leading-relaxed">
                                {translatedData?.summary || selectedConsultation.content.summary || selectedConsultation.content.subjective?.historyOfPresentIllness}
                              </p>
                            </section>

                            {/* Objective - Vitals */}
                            <section>
                              <h4 className="text-[10px] font-bold text-rose-400 uppercase tracking-widest mb-4">Objective</h4>

                              <div className="mb-6">
                                <p className="text-xs font-bold text-slate-500 mb-3">Vital Signs</p>
                                <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                                  {[
                                    { label: 'Temperature', value: selectedConsultation.content.objective?.vitals?.temperature || 'Not recorded', icon: <Activity size={12} /> },
                                    { label: 'Pulse', value: selectedConsultation.content.objective?.vitals?.pulse || 'Not recorded', icon: <Heart size={12} /> },
                                    { label: 'BP', value: selectedConsultation.content.objective?.vitals?.bloodPressure || 'Not recorded', icon: <Activity size={12} /> },
                                    { label: 'Respiratory', value: selectedConsultation.content.objective?.vitals?.respiratoryRate || 'Not recorded', icon: <Wind size={12} /> },
                                  ].map((vital, i) => (
                                    <div key={i} className="bg-slate-50 rounded-2xl p-4 border border-slate-100">
                                      <div className="text-[10px] font-bold text-slate-400 mb-1 flex items-center gap-1">
                                        {vital.label}
                                      </div>
                                      <div className="text-lg font-black text-slate-800">{vital.value}</div>
                                    </div>
                                  ))}
                                </div>
                              </div>

                              <div className="mb-6">
                                <p className="text-xs font-bold text-slate-500 mb-3">Appearance:</p>
                                <div className="flex flex-wrap gap-2">
                                  {(selectedConsultation.content.objective?.appearance || ['Not noted']).map((tag, i) => (
                                    <span key={i} className="px-3 py-1.5 bg-rose-50 rounded-full text-[10px] font-bold text-rose-400 border border-rose-100 uppercase tracking-wider">
                                      {tag}
                                    </span>
                                  ))}
                                </div>
                              </div>

                            </section>

                            {/* Assessment */}
                            <section>
                              <h4 className="text-[10px] font-bold text-rose-400 uppercase tracking-widest mb-3">Assessment</h4>
                              <div className="bg-slate-50 rounded-3xl p-6 border border-slate-100">
                                <p className="text-xs font-bold text-slate-500 mb-3">Likely Diagnosis -</p>
                                <div className="flex flex-wrap gap-2 items-center">
                                  <span className="text-sm font-medium text-slate-600 mr-2">Suspect</span>
                                  <span className="px-4 py-2 bg-emerald-50 text-emerald-600 rounded-xl text-xs font-black border border-emerald-100 shadow-sm">
                                    {(translatedData?.assessment || selectedConsultation.content.assessment).split(/[,.]/)[0] || "Viral Upper Respiratory Infection"}
                                  </span>
                                </div>
                                <p className="mt-4 text-xs font-medium text-slate-500 italic">
                                  {translatedData?.assessment || selectedConsultation.content.assessment}
                                </p>
                              </div>
                            </section>

                            {/* Plan */}
                            <section>
                              <h4 className="text-[10px] font-bold text-rose-400 uppercase tracking-widest mb-4">Plan</h4>

                              <div className="mb-6">
                              </div>

                              <div>
                                <p className="text-xs font-bold text-slate-500 mb-3">Medication -</p>
                                <div className="flex flex-wrap gap-3">
                                  {(translatedData?.plan || selectedConsultation.content.plan).split('\n').slice(0, 3).map((med: string, i: number) => (
                                    <div key={i} className="px-4 py-3 bg-blue-50 text-blue-600 rounded-2xl text-xs font-bold border border-blue-100 shadow-sm flex items-center gap-2">
                                      {med.replace(/^- /, '')}
                                      <Info size={12} className="opacity-50" />
                                    </div>
                                  ))}
                                </div>
                              </div>
                            </section>

                            {/* Generation Actions (Relocated) */}
                            <div className="pt-6 border-t border-slate-50 relative mt-4">
                              <div className="flex flex-col md:flex-row items-center justify-between gap-4 mb-4">
                                <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest flex items-center gap-2">
                                  {isGenerating ? <Loader2 size={12} className="animate-spin text-[#552483]" /> : <Sparkles size={12} />}
                                  AI Assistance
                                </span>
                                <div className="flex flex-wrap gap-2">
                                  <button
                                    disabled={isGenerating}
                                    onClick={() => handleGenerate('Clinical Summary')}
                                    className="px-4 py-2 bg-slate-50 hover:bg-slate-100 text-slate-600 rounded-xl text-[10px] font-bold uppercase tracking-wider transition-colors disabled:opacity-50 hover:bg-black hover:text-white"
                                  >
                                    Clinical Summary
                                  </button>
                                  <button
                                    disabled={isGenerating}
                                    onClick={() => handleGenerate('CPT Code')}
                                    className="px-4 py-2 bg-slate-50 hover:bg-slate-100 text-slate-600 rounded-xl text-[10px] font-bold uppercase tracking-wider transition-colors disabled:opacity-50 hover:bg-black hover:text-white"
                                  >
                                    CPT Code
                                  </button>
                                  <button
                                    disabled={isGenerating}
                                    onClick={() => handleGenerate('ICD-10 Code')}
                                    className="px-4 py-2 bg-slate-50 hover:bg-slate-100 text-slate-600 rounded-xl text-[10px] font-bold uppercase tracking-wider transition-colors disabled:opacity-50 hover:bg-black hover:text-white"
                                  >
                                    ICD-10 Code
                                  </button>
                                </div>
                              </div>

                              {/* Generated Content Result */}
                              {generationResult && (
                                <div className="animate-in fade-in slide-in-from-top-4 duration-500">
                                  <div className="bg-[#faf8ff] rounded-2xl p-6 border border-purple-100 relative">
                                    <button
                                      onClick={() => setGenerationResult(null)}
                                      className="absolute top-4 right-4 text-slate-400 hover:text-slate-600"
                                    >
                                      <X size={14} />
                                    </button>
                                    <h4 className="text-[10px] font-bold text-[#552483] uppercase tracking-widest mb-3">
                                      Generated {generationResult.type}
                                    </h4>
                                    <p className="text-sm font-medium text-slate-700 leading-relaxed whitespace-pre-wrap">
                                      {generationResult.content}
                                    </p>
                                  </div>
                                </div>
                              )}
                            </div>


                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </>
            ) : (
              <div className="h-full flex flex-col items-center justify-center opacity-30">
                <div className="w-24 h-24 bg-slate-100 rounded-full flex items-center justify-center mb-6 text-slate-300">
                  <FileText size={40} />
                </div>
                <p className="text-[#552483] font-black uppercase tracking-widest text-xs">Select a Session Record</p>
              </div>
            )}
          </div>
        </div>
      </main >

      {isModalOpen && (
        <NewPatientModal
          initialName={selectedPatient?.name}
          onClose={() => setIsModalOpen(false)}
          onStart={(data) => {
            setIsModalOpen(false);
            onStartScribe(data);
          }}
        />
      )}

      {/* AI Chat Drawer */}
      {
        isAIChatOpen && (
          <>
            <div
              className="fixed inset-0 bg-black/20 backdrop-blur-sm z-40 transition-opacity"
              onClick={() => setIsAIChatOpen(false)}
            />
            <div className="fixed top-0 right-0 h-full w-full md:w-[400px] bg-white shadow-2xl z-50 border-l border-slate-100 p-4 md:p-6 flex flex-col animate-in slide-in-from-right duration-300">
              <div className="flex items-center justify-between mb-8">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 bg-black rounded-2xl flex items-center justify-center text-white shadow-sm shadow-black/20">
                    <Sparkles size={20} />
                  </div>
                  <div>
                    <h3 className="font-bold text-slate-800">AI Assistant</h3>
                  </div>
                </div>
                <button
                  onClick={() => setIsAIChatOpen(false)}
                  className="p-2 hover:bg-slate-50 rounded-xl text-slate-400 hover:text-slate-600 transition-colors"
                >
                  <X size={20} />
                </button>
              </div>

              {/* RAG Mode Toggle */}
              <div className="mb-6 bg-slate-50 p-3 rounded-xl border border-slate-100 flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className={`p-2 rounded-lg ${isRAGMode ? 'bg-[#552483] text-white' : 'bg-white text-slate-400 border border-slate-200'}`}>
                    <BrainCircuit size={16} />
                  </div>
                  <div>
                    <p className={`text-xs font-bold uppercase tracking-wider ${isRAGMode ? 'text-[#552483]' : 'text-slate-500'}`}>
                      {isRAGMode ? 'Full Medical Memory' : 'Current Session Only'}
                    </p>
                    <p className="text-[10px] text-slate-400 font-medium">
                      {isRAGMode ? 'Analyzing past history & files' : 'Focused on this consultation'}
                    </p>
                  </div>
                </div>
                <button
                  onClick={() => setIsRAGMode(!isRAGMode)}
                  className={`w-10 h-6 rounded-full transition-colors relative ${isRAGMode ? 'bg-[#552483]' : 'bg-slate-200'}`}
                >
                  <div className={`absolute top-1 left-1 w-4 h-4 bg-white rounded-full shadow-sm transition-transform ${isRAGMode ? 'translate-x-4' : 'translate-x-0'}`} />
                </button>
              </div>

              <div className="flex-1 overflow-y-auto custom-scrollbar -mx-2 px-2 space-y-4 mb-6">
                <div className="bg-slate-50 p-4 rounded-2xl rounded-tl-none border border-slate-100 text-sm text-slate-600 leading-relaxed shadow-sm">
                  Hello! I've analyzed the consultation with {selectedPatient?.name || 'the patient'}. You can ask me about symptoms, diagnosis, or treatment plan.
                </div>
                {chatHistory.map(msg => (
                  <div key={msg.id} className={`p-4 rounded-2xl text-sm leading-relaxed shadow-sm animate-in fade-in slide-in-from-bottom-2 duration-300 ${msg.role === 'user' ? 'bg-black text-white rounded-tr-none ml-8' : 'bg-slate-50 text-slate-600 rounded-tl-none mr-8 border border-slate-100'}`}>
                    {msg.text}
                  </div>
                ))}

                {isAISearching && (
                  <div className="bg-slate-50 p-4 rounded-2xl rounded-tl-none border border-slate-100 text-sm text-slate-600 leading-relaxed shadow-sm w-fit">
                    <div className="flex gap-1">
                      <div className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                      <div className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                      <div className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                    </div>
                  </div>
                )}
                <div ref={chatEndRef} />
              </div>

              <div className="relative">
                <input
                  type="text"
                  value={aiQuestion}
                  onChange={(e) => setAIQuestion(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleAskAI()}
                  placeholder="Ask a question..."
                  className="w-full px-5 py-4 bg-white border border-slate-200 rounded-2xl text-sm text-slate-700 placeholder:text-slate-400 outline-none focus:ring-2 focus:ring-black/5 focus:border-slate-400 transition-all pr-12 shadow-sm"
                />
                <button
                  onClick={handleAskAI}
                  disabled={isAISearching || !aiQuestion.trim()}
                  className="absolute right-2 top-2 p-2 bg-black text-white rounded-xl shadow-md shadow-slate-200 hover:bg-slate-800 disabled:opacity-50 disabled:shadow-none transition-all"
                >
                  {isAISearching ? <RefreshCcw size={16} className="animate-spin" /> : <ChevronRight size={18} />}
                </button>
              </div>
            </div>
          </>
        )
      }

      {/* Floating Action Button (Detail View) - New Session */}
      {
        selectedPatient && !isSidebarOpen && (
          <button
            onClick={() => {
              console.log('FAB CLICKED. Patient:', selectedPatient);
              console.log('FAB Language Value:', selectedPatient.language);
              onStartScribe({
                ...selectedPatient, // Spread all properties first
                age: selectedPatient.age?.toString() || '',
                bloodGroup: selectedPatient.bloodGroup || '',
                weight: selectedPatient.weight || '',
                height: selectedPatient.height || '',
                gender: selectedPatient.gender || 'Unknown',
                nationality: selectedPatient.nationality || '',
                language: selectedPatient.language || 'English' // Explicitly overwrite/set language
              });
            }}
            className="fixed bottom-6 right-6 w-14 h-14 bg-black text-white rounded-full flex items-center justify-center shadow-2xl shadow-black/30 z-40 hover:scale-105 active:scale-95 transition-all md:hidden"
          >
            <Plus size={24} />
          </button>
        )
      }

      {/* Edit Patient Modal */}
      {
        isEditModalOpen && selectedPatient && (
          <EditPatientModal
            patient={selectedPatient}
            onClose={() => setIsEditModalOpen(false)}
            onSave={(updates) => {
              onUpdatePatient(selectedPatient.id, updates);
              setIsEditModalOpen(false);
            }}
          />
        )
      }
    </div >
  );
};


export default Dashboard;
