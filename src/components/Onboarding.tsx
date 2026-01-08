import React, { useState } from 'react';
import { User, Check, Building2, Globe, Stethoscope } from 'lucide-react';

interface OnboardingProps {
    initialName?: string;
    initialEmail?: string;
    onComplete: (data: { name: string; specialty: string; country: string }) => void;
    onExit: () => void;
}

const SPECIALTIES = [
    'General Practice',
    'Cardiology',
    'Dermatology',
    'Neurology',
    'Pediatrics',
    'Psychiatry',
    'Orthopedics',
    'Internal Medicine',
    'Oncology'
];

const COUNTRIES = [
    'India',
    'United States',
    'United Kingdom',
    'Canada',
    'Australia',
    'Singapore',
    'United Arab Emirates'
];

const Onboarding: React.FC<OnboardingProps> = ({ initialName = '', initialEmail = '', onComplete, onExit }) => {
    const [name, setName] = useState(initialName);
    const [specialty, setSpecialty] = useState('');
    const [country, setCountry] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [touched, setTouched] = useState({ name: false, specialty: false, country: false });

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setTouched({ name: true, specialty: true, country: true });

        if (!name || !specialty || !country) return;

        // Enforce Dr. prefix
        let finalName = name.trim();
        // Check if starts with Dr. (case insensitive)
        if (!finalName.toLowerCase().startsWith('dr.')) {
            // If user typed "Dr John", handle that too
            if (finalName.toLowerCase().startsWith('dr ')) {
                finalName = `Dr. ${finalName.substring(3)}`;
            } else {
                finalName = `Dr. ${finalName}`;
            }
        }

        setIsLoading(true);
        // Simulate a small delay for better UX
        setTimeout(() => {
            onComplete({ name: finalName, specialty, country });
            setIsLoading(false);
        }, 800);
    };

    return (
        <div className="min-h-screen bg-[#fcfbff] flex items-center justify-center p-4">
            <div className="bg-white w-full max-w-[420px] rounded-[2rem] p-6 md:p-8 shadow-[0_20px_60px_-15px_rgba(0,0,0,0.05)] border border-slate-100 relative overflow-hidden">
                {/* Decorative Background Elements */}
                <div className="absolute top-0 right-0 w-48 h-48 bg-slate-50 rounded-full blur-3xl -mr-24 -mt-24 pointer-events-none opacity-50" />
                <div className="absolute bottom-0 left-0 w-48 h-48 bg-slate-50 rounded-full blur-3xl -ml-24 -mb-24 pointer-events-none opacity-50" />

                {/* Exit / Sign Out Button */}
                <button
                    onClick={onExit}
                    className="absolute top-4 right-4 text-[9px] font-bold uppercase tracking-widest text-slate-400 hover:text-red-500 transition-colors z-10 flex items-center gap-1"
                >
                    Sign Out
                    <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" /></svg>
                </button>

                <div className="relative">
                    <div className="mb-6 text-center">
                        <div className="w-12 h-12 bg-black rounded-xl flex items-center justify-center text-white mx-auto mb-4 shadow-lg shadow-black/20 transform rotate-3">
                            <span className="text-xl font-black tracking-tighter">M</span>
                        </div>
                        <h1 className="text-2xl font-black text-slate-800 tracking-tight mb-2">Welcome, Doctor</h1>
                        <p className="text-slate-500 font-medium text-xs">Let's complete your professional profile.</p>
                    </div>

                    <form onSubmit={handleSubmit} className="space-y-4">
                        {/* Name Field */}
                        <div className="space-y-2">
                            <label className="text-[11px] font-bold text-slate-700 uppercase tracking-widest pl-1">Doctor Name</label>
                            <div className="relative group">
                                <div className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400 group-focus-within:text-black transition-colors">
                                    <User size={18} />
                                </div>
                                <input
                                    type="text"
                                    value={name}
                                    onChange={(e) => setName(e.target.value)}
                                    placeholder="e.g. Dr. John Doe"
                                    className={`w-full bg-slate-50 border-2 rounded-xl py-2.5 pl-10 pr-4 text-xs font-bold text-slate-800 placeholder:text-slate-400 outline-none transition-all ${touched.name && !name ? 'border-red-100 bg-red-50 focus:border-red-200' : 'border-slate-100 focus:border-black/10 focus:bg-white focus:shadow-lg focus:shadow-black/5'}`}
                                />
                            </div>
                            {touched.name && !name && <p className="text-[9px] font-bold text-red-500 pl-1 uppercase tracking-wider mt-1">Name is required</p>}
                        </div>

                        {/* Specialization Field */}
                        <div className="space-y-2">
                            <label className="text-[11px] font-bold text-slate-700 uppercase tracking-widest pl-1">Specialization</label>
                            <div className="relative group">
                                <div className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400 group-focus-within:text-black transition-colors">
                                    <Stethoscope size={18} />
                                </div>
                                <select
                                    value={specialty}
                                    onChange={(e) => setSpecialty(e.target.value)}
                                    className={`w-full bg-slate-50 border-2 rounded-xl py-2.5 pl-10 pr-8 text-xs font-bold text-slate-800 outline-none appearance-none cursor-pointer transition-all ${touched.specialty && !specialty ? 'border-red-100 bg-red-50 focus:border-red-200' : 'border-slate-100 focus:border-black/10 focus:bg-white focus:shadow-lg focus:shadow-black/5'} ${!specialty && 'text-slate-400'}`}
                                >
                                    <option value="" disabled>Select specialization</option>
                                    {SPECIALTIES.map(s => <option key={s} value={s}>{s}</option>)}
                                </select>
                                <div className="absolute right-4 top-1/2 -translate-y-1/2 pointer-events-none text-slate-400">
                                    <svg width="12" height="12" viewBox="0 0 12 12" fill="none" xmlns="http://www.w3.org/2000/svg">
                                        <path d="M2.5 4.5L6 8L9.5 4.5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                                    </svg>
                                </div>
                            </div>
                            {touched.specialty && !specialty && <p className="text-[10px] font-bold text-red-500 pl-1 uppercase tracking-wider">Specialization is required</p>}
                        </div>

                        {/* Country Field */}
                        <div className="space-y-2">
                            <label className="text-[11px] font-bold text-slate-700 uppercase tracking-widest pl-1">Country</label>
                            <div className="relative group">
                                <div className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400 group-focus-within:text-black transition-colors">
                                    <Globe size={18} />
                                </div>
                                <select
                                    value={country}
                                    onChange={(e) => setCountry(e.target.value)}
                                    className={`w-full bg-slate-50 border-2 rounded-xl py-2.5 pl-10 pr-8 text-xs font-bold text-slate-800 outline-none appearance-none cursor-pointer transition-all ${touched.country && !country ? 'border-red-100 bg-red-50 focus:border-red-200' : 'border-slate-100 focus:border-black/10 focus:bg-white focus:shadow-lg focus:shadow-black/5'} ${!country && 'text-slate-400'}`}
                                >
                                    <option value="" disabled>Select country</option>
                                    {COUNTRIES.map(c => <option key={c} value={c}>{c}</option>)}
                                </select>
                                <div className="absolute right-4 top-1/2 -translate-y-1/2 pointer-events-none text-slate-400">
                                    <svg width="12" height="12" viewBox="0 0 12 12" fill="none" xmlns="http://www.w3.org/2000/svg">
                                        <path d="M2.5 4.5L6 8L9.5 4.5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                                    </svg>
                                </div>
                            </div>
                            {touched.country && !country && <p className="text-[10px] font-bold text-red-500 pl-1 uppercase tracking-wider">Country is required</p>}
                        </div>

                        <div className="pt-4">
                            <button
                                type="submit"
                                disabled={isLoading}
                                className="w-full bg-black text-white py-3 rounded-xl font-black text-[10px] uppercase tracking-[0.2em] hover:bg-slate-800 transition-all active:scale-[0.98] shadow-xl shadow-black/20 flex items-center justify-center gap-2 group disabled:opacity-70 disabled:cursor-not-allowed mt-2"
                            >
                                {isLoading ? (
                                    <div className="w-3 h-3 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                                ) : (
                                    <>
                                        Complete
                                        <Check size={14} className="group-hover:scale-110 transition-transform" />
                                    </>
                                )}
                            </button>
                        </div>
                    </form>

                    {initialEmail && (
                        <div className="mt-6 pt-4 border-t border-slate-50 text-center">
                            <p className="text-[9px] font-bold text-slate-400 uppercase tracking-widest">
                                <span className="text-black">{initialEmail}</span>
                            </p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default Onboarding;
