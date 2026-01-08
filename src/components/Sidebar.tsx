import React, { useState } from 'react';
import { User, Activity, Ruler, Weight, ChevronLeft, Plus, Trash2, Pencil, Upload, Loader2 } from 'lucide-react';
import { Patient, MedicalNote } from '../types';

interface SidebarProps {
    patient: Patient;
    onBack: () => void;
    onStartSession: () => void;
    activeRecordId?: string;
    onSelectRecord: (record: MedicalNote) => void;
    onDeletePatient?: () => void;
    onEditPatient?: () => void;

    isOpen?: boolean;
    onClose?: () => void;
}


const Sidebar: React.FC<SidebarProps> = ({ patient, onBack, onStartSession, activeRecordId, onSelectRecord, onDeletePatient, onEditPatient, isOpen = false, onClose }) => {

    return (
        <>
            {/* Mobile Overlay */}
            {isOpen && (
                <div
                    className="fixed inset-0 bg-black/20 backdrop-blur-sm z-40 md:hidden animate-in fade-in duration-300"
                    onClick={onClose}
                />
            )}

            <aside className={`
                w-[85vw] md:w-64 border-r border-slate-100 bg-white h-screen flex flex-col 
                fixed top-0 left-0 z-50 md:sticky md:top-0 md:transform-none md:transition-none
                transition-transform duration-300 ease-in-out shadow-2xl md:shadow-sm
                ${isOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'}
            `}>
                <div className="p-4 border-b border-slate-50">
                    <div className="mb-6">
                        {/* Desktop Back to Registry */}
                        <button
                            onClick={onBack}
                            className="hidden md:flex items-center text-[10px] font-bold text-slate-400 hover:text-black transition-colors group uppercase tracking-widest"
                        >
                            <ChevronLeft size={12} className="mr-1 group-hover:-translate-x-0.5 transition-transform" />
                            REGISTRY
                        </button>

                        {/* Mobile Close Sidebar */}
                        <button
                            onClick={onClose}
                            className="flex md:hidden items-center text-xs font-bold text-slate-400 hover:text-black transition-colors group uppercase tracking-widest"
                        >
                            <ChevronLeft size={12} className="mr-1 group-hover:-translate-x-0.5 transition-transform" />
                            BACK
                        </button>
                    </div>

                    <div className="flex flex-col items-center text-center">
                        <div className="w-12 h-12 bg-slate-50 rounded-[1.25rem] flex items-center justify-center text-black mb-3 shadow-[inset_0_2px_4px_rgba(0,0,0,0.05)] border border-slate-100">
                            <User size={20} strokeWidth={1.5} />
                        </div>
                        <h2 className="text-xl md:text-base font-bold text-slate-800 tracking-tight">{patient.name}</h2>
                        <div className="flex items-center gap-2 mt-0.5 justify-center">
                            <p className="text-slate-400 text-sm md:text-[10px] font-medium">{patient.gender || 'Unknown'} • {patient.age || '--'} Years</p>
                            {patient.language && (
                                <span className="px-1.5 py-0.5 bg-blue-50 text-blue-600 rounded text-[8px] font-bold uppercase tracking-wider">
                                    {patient.language}
                                </span>
                            )}
                        </div>
                        <div className="mt-3 flex items-center gap-2">
                            <div className="px-2 py-0.5 bg-slate-50 border border-slate-100 rounded-md text-xs md:text-[9px] font-bold text-slate-500 tracking-wider">
                                REF: {patient.id.split('-')[0].toUpperCase()}
                            </div>
                            {onEditPatient && (
                                <button onClick={onEditPatient} className="p-1.5 bg-slate-50 border border-slate-100 rounded-md hover:bg-slate-100 transition-colors" title="Edit Details">
                                    <Pencil size={10} className="text-slate-500" />
                                </button>
                            )}
                        </div>
                    </div>
                </div>

                <div className="p-4 space-y-5 flex-1 overflow-y-auto custom-scrollbar">
                    <div>
                        <div className="space-y-2">
                            <MetricCard icon={<Activity size={14} />} label="Nationality" value={patient.nationality} color="text-rose-500" bgColor="bg-rose-50/50" />
                            <MetricCard icon={<Ruler size={14} />} label="Height" value={patient.height} color="text-black" bgColor="bg-slate-50" />
                            <MetricCard icon={<Weight size={14} />} label="Weight" value={patient.weight} color="text-emerald-500" bgColor="bg-emerald-50/50" />
                        </div>
                    </div>

                    <div>
                        <div className="flex items-center justify-between mb-3">
                            <h3 className="text-xs md:text-[9px] font-bold text-slate-400 uppercase tracking-[0.2em]">Medical History</h3>
                        </div>
                        <div className="space-y-3 relative before:absolute before:left-[6px] before:top-2 before:bottom-2 before:w-[1px] before:bg-slate-100">
                            {patient.consultations && patient.consultations.length > 0 ? (
                                patient.consultations.map((record) => (
                                    <div key={record.id} onClick={() => onSelectRecord(record)} className="cursor-pointer">
                                        <TimelineItem
                                            date={record.date}
                                            title={
                                                record.content.title ||
                                                record.content.summary ||
                                                record.content.subjective?.chiefComplaint ||
                                                record.content.assessment ||
                                                'Consultation Record'
                                            }
                                            active={activeRecordId === record.id}
                                        />
                                    </div>
                                ))
                            ) : (
                                <p className="text-[9px] text-slate-400 pl-6">No previous records.</p>
                            )}
                        </div>
                    </div>
                </div>

                <div className="p-4 bg-slate-50 border-t border-slate-100 space-y-2">
                    <button
                        onClick={onStartSession}
                        className="hidden md:flex w-full bg-black text-white py-3 rounded-xl font-bold text-[10px] items-center justify-center hover:bg-slate-800 transition-all active:scale-[0.98] shadow-lg shadow-black/10 uppercase tracking-widest"
                    >
                        <Plus size={14} className="mr-2" />
                        Start Session
                    </button>
                    {/* Delete button moved to Dashboard Registry */}
                </div>
            </aside>
        </>
    );
};

const MetricCard = ({ icon, label, value, color, bgColor }: any) => (
    <div className={`p-3 rounded-xl flex items-center justify-between border border-transparent hover:border-slate-100 transition-all ${bgColor}`}>
        <div className="flex items-center">
            <div className={`${color} mr-2 opacity-80`}>{icon}</div>
            <span className="text-sm md:text-[10px] font-semibold text-slate-500 tracking-tight">{label}</span>
        </div>
        <span className={`text-sm md:text-[10px] font-bold ${color}`}>{value || '--'}</span>
    </div>
);

const TimelineItem = ({ date, title, active = false }: any) => {
    let displayDate = date;
    try {
        // Try parsing as ISO or standard date
        const d = new Date(date);
        if (!isNaN(d.getTime())) {
            displayDate = d.toLocaleDateString('en-US', {
                month: 'short',
                day: 'numeric',
                year: 'numeric'
            });
        }
    } catch (e) {
        displayDate = date.split('•')[0];
    }

    return (
        <div className="flex items-start gap-3 relative group hover:opacity-80 transition-opacity">
            <div className={`mt-1 w-[13px] h-[13px] rounded-full shrink-0 border-2 border-white shadow-sm z-[1] ${active ? 'bg-black' : 'bg-slate-200 group-hover:bg-slate-300'}`} />
            <div className="min-w-0 flex-1">
                <p className={`text-sm md:text-[10px] font-bold leading-tight line-clamp-2 ${active ? 'text-slate-800' : 'text-slate-400 group-hover:text-slate-600'}`}>{title}</p>
                <p className="text-xs md:text-[9px] font-medium text-slate-400/80 mt-0.5">{displayDate}</p>
            </div>
        </div>
    );
};

export default Sidebar;
