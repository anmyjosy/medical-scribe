import React, { useState, useEffect, useRef } from 'react';
import { Patient, Medication } from '@/types';
import { Plus, Trash2, Printer, Download, Save, Stethoscope, FileText } from 'lucide-react';

interface PrescriptionPadProps {
    patient: Patient;
    medications: Medication[];
    notes: string;
    doctorName: string;
    doctorSpecialty: string;
    onSave: (meds: Medication[], notes: string) => void;
}

export const PrescriptionPad: React.FC<PrescriptionPadProps> = ({
    patient,
    medications: initialMeds,
    notes: initialNotes,
    doctorName,
    doctorSpecialty,
    onSave
}) => {
    const [meds, setMeds] = useState<Medication[]>(initialMeds);
    const [notes, setNotes] = useState(initialNotes);
    const [date, setDate] = useState(new Date().toISOString().split('T')[0]);

    useEffect(() => {
        setMeds(initialMeds);
    }, [initialMeds]);

    useEffect(() => {
        setNotes(initialNotes);
    }, [initialNotes]);

    const lastMedRef = useRef<HTMLDivElement>(null);
    const prevMedsLength = useRef(meds.length);

    useEffect(() => {
        if (meds.length > prevMedsLength.current) {
            lastMedRef.current?.scrollIntoView({ behavior: 'smooth' });
        }
        prevMedsLength.current = meds.length;
    }, [meds.length]);

    const handleAddMed = () => {
        setMeds([...meds, { name: '', dosage: '', frequency: '', duration: '', instructions: '' }]);
    };

    const handleRemoveMed = (index: number) => {
        const newMeds = [...meds];
        newMeds.splice(index, 1);
        setMeds(newMeds);
    };

    const handleChange = (index: number, field: keyof Medication, value: string) => {
        const newMeds = [...meds];
        newMeds[index] = { ...newMeds[index], [field]: value };
        setMeds(newMeds);
    };

    const handlePrint = () => {
        window.print();
    };

    return (
        <div className="flex flex-col h-full bg-slate-50/50">
            {/* Toolbar */}
            <div className="flex flex-col md:flex-row justify-between items-stretch md:items-center bg-white p-3 md:p-4 border-b border-slate-100 gap-3 md:gap-0 print:hidden">
                <div className="grid grid-cols-2 gap-2 md:flex md:gap-2">
                    <button onClick={handleAddMed} className="flex items-center justify-center gap-2 h-10 px-4 bg-slate-100 rounded-xl text-xs font-bold text-slate-700 hover:bg-slate-200 transition-colors">
                        <Plus size={16} /> Add Drug
                    </button>
                    <button onClick={() => onSave(meds, notes)} className="flex items-center justify-center gap-2 h-10 px-4 bg-emerald-50 text-emerald-600 rounded-xl text-xs font-bold hover:bg-emerald-100 transition-colors">
                        <Save size={16} /> Save
                    </button>
                </div>
                <button onClick={handlePrint} className="flex items-center justify-center gap-2 h-10 px-6 bg-blue-600 text-white rounded-xl text-xs font-bold hover:bg-blue-700 shadow-lg shadow-blue-200 transition-colors">
                    <Printer size={16} /> Print / PDF
                </button>
            </div>

            {/* Prescription Paper (A4 Aspect Ratio approx) */}
            <div className="flex-1 overflow-y-auto p-4 md:p-8 custom-scrollbar print:p-0 print:overflow-visible">
                <div className="w-full md:max-w-[210mm] min-h-[500px] md:min-h-[297mm] mx-auto bg-white shadow-2xl print:shadow-none print:w-full relative flex flex-col">

                    {/* Header */}
                    <div className="bg-sky-50 p-6 md:p-8 border-b-4 border-sky-100 print:bg-sky-50 print-color-exact">
                        <div className="flex justify-between items-start">
                            <div className="max-w-[70%]">
                                <h1 className="text-2xl md:text-3xl font-black text-sky-900 tracking-tight leading-tight">{doctorName}</h1>
                                <p className="text-sky-600 font-bold uppercase tracking-widest text-[10px] md:text-xs mt-1 md:mt-2 leading-snug">{doctorSpecialty}</p>
                                <p className="text-sky-400 text-[9px] md:text-[10px] mt-1 md:mt-2 font-medium">LICURE: 12548-20</p>
                            </div>
                            <div className="text-sky-800">
                                <Stethoscope className="w-8 h-8 md:w-12 md:h-12" strokeWidth={1.5} />
                            </div>
                        </div>
                    </div>

                    {/* Patient Info */}
                    <div className="px-6 md:px-8 pt-4 pb-1">
                        <div className="flex flex-wrap gap-y-4 text-sm text-slate-600 border-b border-slate-100 pb-2">
                            <div className="w-full md:w-1/2 md:pr-4">
                                <span className="font-bold text-slate-400 text-[10px] md:text-xs uppercase tracking-wider block mb-1">Patient Name</span>
                                <span className="font-bold text-slate-900 border-b border-slate-200 block pb-1 w-full text-base md:text-sm">{patient.name}</span>
                            </div>
                            <div className="w-1/2 md:w-1/4 pr-2 md:px-2">
                                <span className="font-bold text-slate-400 text-[10px] md:text-xs uppercase tracking-wider block mb-1">Age / Sex</span>
                                <span className="font-bold text-slate-900 border-b border-slate-200 block pb-1 w-full">{patient.age || '-'} / {patient.gender || '-'}</span>
                            </div>
                            <div className="w-1/2 md:w-1/4 pl-2">
                                <span className="font-bold text-slate-400 text-[10px] md:text-xs uppercase tracking-wider block mb-1">Date</span>
                                <input
                                    type="date"
                                    value={date}
                                    onChange={(e) => setDate(e.target.value)}
                                    className="font-bold text-slate-900 border-b border-slate-200 block pb-1 w-full bg-transparent outline-none p-0"
                                />
                            </div>

                        </div>
                    </div>

                    {/* Rx Body */}
                    <div className="px-6 md:px-8 py-4 flex-1">
                        <div className="mb-3 md:mb-4">
                            <h2 className="text-3xl md:text-4xl font-black text-sky-600 font-serif italic">Rx</h2>
                        </div>

                        <div className="space-y-6">
                            {meds.map((med, idx) => (
                                <div key={idx} ref={idx === meds.length - 1 ? lastMedRef : null} className="group relative pl-4 border-l-2 border-slate-100 hover:border-sky-200 transition-colors">
                                    <button
                                        onClick={() => handleRemoveMed(idx)}
                                        className="absolute -left-[26px] top-0 text-slate-300 hover:text-rose-500 opacity-0 group-hover:opacity-100 print:hidden"
                                    >
                                        <Trash2 size={16} />
                                    </button>

                                    <div className="grid grid-cols-12 gap-2 md:gap-4 items-baseline">
                                        <div className="col-span-8 md:col-span-8">
                                            <input
                                                type="text"
                                                value={med.name}
                                                onChange={(e) => handleChange(idx, 'name', e.target.value)}
                                                placeholder="Medication Name"
                                                autoFocus={idx === meds.length - 1 && med.name === ''}
                                                className="w-full text-lg font-bold text-slate-900 placeholder:text-slate-300 bg-transparent outline-none border-b border-transparent focus:border-slate-200 transition-all printing-no-border"
                                            />
                                        </div>
                                        <div className="col-span-4 md:col-span-4 text-right">
                                            <input
                                                type="text"
                                                value={med.dosage}
                                                onChange={(e) => handleChange(idx, 'dosage', e.target.value)}
                                                placeholder="Dosage"
                                                className="w-full text-right text-sm font-bold text-slate-500 placeholder:text-slate-300 bg-transparent outline-none border-b border-transparent focus:border-slate-200 transition-all printing-no-border"
                                            />
                                        </div>
                                        <div className="col-span-12 flex flex-wrap md:flex-nowrap gap-2 md:gap-4 text-xs font-medium text-slate-500 pl-1">
                                            <input
                                                type="text"
                                                value={med.frequency}
                                                onChange={(e) => handleChange(idx, 'frequency', e.target.value)}
                                                placeholder="Freq"
                                                className="w-1/3 md:w-1/4 bg-transparent outline-none border-b border-transparent focus:border-slate-200 transition-all printing-no-border"
                                            />
                                            <span>•</span>
                                            <input
                                                type="text"
                                                value={med.duration}
                                                onChange={(e) => handleChange(idx, 'duration', e.target.value)}
                                                placeholder="Dur"
                                                className="w-1/4 md:w-1/4 bg-transparent outline-none border-b border-transparent focus:border-slate-200 transition-all printing-no-border"
                                            />
                                            <span>•</span>
                                            <input
                                                type="text"
                                                value={med.instructions}
                                                onChange={(e) => handleChange(idx, 'instructions', e.target.value)}
                                                placeholder="Instructions"
                                                className="flex-1 bg-transparent outline-none border-b border-transparent focus:border-slate-200 transition-all printing-no-border"
                                            />
                                        </div>
                                    </div>
                                </div>
                            ))}

                            {meds.length === 0 && (
                                <div className="text-center py-10 border-2 border-dashed border-slate-100 rounded-xl">
                                    <p className="text-slate-400 text-sm">No medications added</p>
                                </div>
                            )}
                        </div>

                        {/* Notes Area */}
                        <div className="mt-12 pt-8 border-t border-slate-100">
                            <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-2">Instructions / Notes</h3>
                            <textarea
                                value={notes}
                                onChange={(e) => setNotes(e.target.value)}
                                placeholder="Additional notes for patient..."
                                className="w-full min-h-[100px] text-sm text-slate-600 bg-transparent outline-none resize-none printing-no-border"
                            />
                        </div>
                    </div>

                    {/* Footer */}
                    <div className="bg-sky-50 p-8 border-t-4 border-sky-100 print:bg-sky-50 print-color-exact mt-auto">
                        <div className="flex justify-between items-end">
                            <div className="text-[10px] text-sky-800 font-medium space-y-1">
                                <p className="flex items-center gap-2">HOSPITAL NAME</p>
                                <p>55 47 79 94 15 • email_here@email.com</p>
                                <p>Address Here Number 123 • www.webpage.com</p>
                            </div>
                            <div className="w-48 text-center">
                                <div className="border-b border-sky-900 mb-2"></div>
                                <p className="text-[10px] font-bold text-sky-900 uppercase tracking-widest">Signature</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <style>
                {`
                    @media print {
                        @page { margin: 0; size: auto; }
                        body * { visibility: hidden; }
                        .print\\:overflow-visible, .print\\:overflow-visible * { visibility: visible; }
                        .print\\:overflow-visible { position: absolute; left: 0; top: 0; width: 100%; height: 100%; margin: 0; padding: 0; background: white; }
                        .print\\:hidden { display: none !important; }
                        .print-color-exact { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
                        .printing-no-border { border-bottom: none !important; }
                        /* Hide placeholders when printing */
                        input::placeholder, textarea::placeholder { color: transparent; }
                    }
                `}
            </style>
        </div>
    );
};
