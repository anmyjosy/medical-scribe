'use client';

import React, { useState, useEffect } from 'react';
import { View, User, AuthState, MedicalNote, Patient, PrefilledPatientData } from '@/types';
import Landing from '@/components/Landing';
import Auth from '@/components/Auth';
import Onboarding from '@/components/Onboarding';
import Dashboard from '@/components/Dashboard';
import ScribeSession from '@/components/ScribeSession';
import { supabase } from '@/lib/supabaseClient';
import { uploadPatientFile, deletePatientFile, getPatientFiles } from '@/services/fileUploadService';

export default function Home() {
    const [currentView, setCurrentView] = useState<View>(View.LANDING);
    const viewRef = React.useRef(currentView);

    // Sync ref with state
    useEffect(() => {
        viewRef.current = currentView;
    }, [currentView]);

    // Handle Browser History
    useEffect(() => {
        const handlePopState = (event: PopStateEvent) => {
            // If we are in Scribe mode and go back, return to Dashboard
            if (currentView === View.SCRIBE) {
                setPreselectedPatientData(null);
                setCurrentView(View.DASHBOARD);
            }
            // Add more view handling if needed (e.g. Auth -> Landing)
        };

        window.addEventListener('popstate', handlePopState);
        return () => window.removeEventListener('popstate', handlePopState);
    }, [currentView]);

    const [authState, setAuthState] = useState<AuthState>({
        user: null,
        isAuthenticated: false
    });
    const [patients, setPatients] = useState<Patient[]>([]);
    const [preselectedPatientData, setPreselectedPatientData] = useState<PrefilledPatientData | null>(null);
    const [defaultPatientId, setDefaultPatientId] = useState<string | null>(null);

    // Load Patients from Supabase
    const fetchPatients = async () => {
        const { data: { user } } = await supabase.auth.getUser();
        if (!user) return;

        console.log('Fetching patients for user:', user.id);

        // Fetch patients and their consultations
        const { data, error } = await supabase
            .from('patients')
            .select('*, consultations(*)')
            .eq('user_id', user.id)
            .order('last_seen', { ascending: false });

        if (error) {
            console.error('Error fetching patients:', error);
        } else {
            const formattedPatients = await Promise.all((data as any[]).map(async p => {
                // Fetch files for each patient
                const files = await getPatientFiles(p.id);

                return {
                    ...p,
                    lastSeen: p.last_seen,
                    bloodGroup: p.blood_group,
                    language: p.language,
                    files: files,
                    consultations: (p.consultations || []).map((c: any) => ({
                        ...c,
                        keyInsights: c.key_insights,
                        rawTranscript: c.raw_transcript,
                        audioUrl: c.audio_url,
                        prescription: typeof c.prescription === 'string' ? JSON.parse(c.prescription) : c.prescription,
                    })).sort((a: MedicalNote, b: MedicalNote) => {
                        return new Date(b.date).getTime() - new Date(a.date).getTime();
                    })
                };
            }));
            setPatients(formattedPatients);
        }
    };

    useEffect(() => {
        // Check active session & Load Data
        supabase.auth.getSession().then(({ data: { session } }) => {
            if (session) {
                const user: User = {
                    id: session.user.id,
                    email: session.user.email || '',
                    name: session.user.user_metadata.name || session.user.email?.split('@')[0] || 'Clinician',
                    specialty: session.user.user_metadata.specialty,
                    country: session.user.user_metadata.country
                };
                setAuthState({ user, isAuthenticated: true });

                // Check for onboarding
                if (!user.specialty || !user.country || user.name === 'Clinician') {
                    setCurrentView(View.ONBOARDING);
                } else {
                    setCurrentView(View.DASHBOARD);
                    fetchPatients();
                }
            }
        });

        // Listen for auth changes
        const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
            if (session) {
                const user: User = {
                    id: session.user.id,
                    email: session.user.email || '',
                    name: session.user.user_metadata.name || session.user.email?.split('@')[0] || 'Clinician',
                    specialty: session.user.user_metadata.specialty,
                    country: session.user.user_metadata.country
                };
                setAuthState({ user, isAuthenticated: true });

                // Use ref here to avoid stale closure
                if (viewRef.current === View.AUTH || viewRef.current === View.LANDING || viewRef.current === View.ONBOARDING) {
                    if (!user.specialty || !user.country || user.name === 'Clinician') {
                        setCurrentView(View.ONBOARDING);
                    } else {
                        setCurrentView(View.DASHBOARD);
                        fetchPatients();
                    }
                }
            } else {
                setAuthState({ user: null, isAuthenticated: false });
                setCurrentView(View.LANDING);
                setPatients([]); // Clear data
            }
        });

        return () => subscription.unsubscribe();
    }, []);

    const handleLogout = async () => {
        await supabase.auth.signOut();
    };

    const handleOnboardingComplete = async (data: { name: string; specialty: string; country: string }) => {
        try {
            if (!authState.user) return;

            const { error } = await supabase.auth.updateUser({
                data: {
                    name: data.name,
                    specialty: data.specialty,
                    country: data.country
                }
            });

            if (error) {
                console.error('Error updating profile:', error);
                alert('Failed to update profile. Please try again.');
                return;
            }

            // Also save to public.doctors table
            const { error: dbError } = await supabase
                .from('doctors')
                .upsert({
                    id: authState.user.id,
                    email: authState.user.email,
                    name: data.name,
                    specialty: data.specialty,
                    country: data.country
                }, { onConflict: 'id' });

            if (dbError) {
                console.error('Error saving doctor profile to DB:', dbError);
                // We don't block here, as auth update succeeded
            }

            // Update local state
            setAuthState(prev => ({
                ...prev,
                user: prev.user ? { ...prev.user, ...data } : null
            }));

            setCurrentView(View.DASHBOARD);
            fetchPatients();

        } catch (error) {
            console.error('Onboarding Error:', error);
        }
    };

    const handleSaveNote = async (note: MedicalNote, audioBlob?: Blob) => {
        try {
            const { data: { user } } = await supabase.auth.getUser();
            if (!user) {
                alert('Error: No active user session found. Please login again.');
                return;
            }

            // 1. Upload Audio if exists
            let audioUrl = '';
            if (audioBlob) {
                const fileName = `${user.id}/${Date.now()}.webm`;
                const { error: uploadError } = await supabase.storage
                    .from('consultation-audio')
                    .upload(fileName, audioBlob);

                if (uploadError) {
                    console.error('Audio upload failed:', uploadError);
                    alert(`Warning: Audio upload failed. ${uploadError.message}`);
                } else {
                    const { data } = await supabase.storage.from('consultation-audio').createSignedUrl(fileName, 3600 * 24 * 365); // 1 year link
                    if (data?.signedUrl) audioUrl = data.signedUrl;
                }
            }

            // 2. Find or Create Patient
            let patientId = '';
            const { data: existingPatients, error: searchError } = await supabase
                .from('patients')
                .select('*')
                .eq('name', note.patientId)
                .eq('user_id', user.id)
                .limit(1);

            if (searchError) {
                console.error('Search error', searchError);
                alert(`Database Error (Searching Patient): ${searchError.message}`);
                return;
            }

            if (existingPatients && existingPatients.length > 0) {
                patientId = existingPatients[0].id;
                // Update existing patient with any new info provided
                await supabase.from('patients').update({
                    last_seen: new Date(note.date).toLocaleDateString(),
                    age: (note as any).age || existingPatients[0].age,
                    gender: (note as any).gender || existingPatients[0].gender,
                    weight: (note as any).weight || existingPatients[0].weight,
                    height: (note as any).height || existingPatients[0].height,
                    blood_group: (note as any).bloodGroup || existingPatients[0].blood_group,
                    nationality: (note as any).nationality || existingPatients[0].nationality,
                    language: (note as any).language || existingPatients[0].language
                }).eq('id', patientId);
            } else {
                const { data: newPatient, error: createError } = await supabase
                    .from('patients')
                    .insert({
                        user_id: user.id,
                        name: note.patientId,
                        age: (note as any).age || 0,
                        gender: (note as any).gender || 'Unknown',
                        weight: (note as any).weight || '',
                        height: (note as any).height || '',
                        blood_group: (note as any).bloodGroup || '',
                        nationality: (note as any).nationality || '',
                        language: (note as any).language || 'English',
                        last_seen: new Date(note.date).toLocaleDateString()
                    })
                    .select()
                    .single();

                if (createError || !newPatient) {
                    console.error('Error creating patient:', createError);
                    alert(`Error creating patient record: ${createError?.message || 'Unknown error'}`);
                    return;
                }
                patientId = newPatient.id;
            }

            // 3. Insert Consultation
            const { error: insertError } = await supabase
                .from('consultations')
                .insert({
                    patient_id: patientId,
                    date: note.date,
                    type: note.type,
                    content: note.content,
                    raw_transcript: note.rawTranscript,
                    utterances: note.utterances,
                    summary: note.summary,
                    key_insights: note.keyInsights,
                    audio_url: audioUrl
                });

            if (insertError) {
                console.error('Error saving consultation:', insertError);
                alert(`Error saving consultation: ${insertError.message}`);
            } else {
                await fetchPatients();
                setDefaultPatientId(patientId); // Set the target patient to open
                setCurrentView(View.DASHBOARD);
            }
        } catch (err: any) {
            console.error('Unexpected error:', err);
            alert(`Unexpected error saving note: ${err.message}`);
        }
    };

    const handleDeleteConsultation = async (patientId: string, consultationId: string) => {
        const { error } = await supabase
            .from('consultations')
            .delete()
            .eq('id', consultationId);

        if (error) {
            console.error('Error deleting consultation:', error);
            alert('Failed to delete consultation: ' + error.message);
        } else {
            await fetchPatients();
        }
    };

    const handleUpdateConsultation = async (consultationId: string, updates: Partial<MedicalNote>) => {
        const dbUpdates: any = {};
        if (updates.content) dbUpdates.content = updates.content;
        if (updates.utterances) dbUpdates.utterances = updates.utterances;
        if (updates.prescription) dbUpdates.prescription = updates.prescription;

        const { error } = await supabase
            .from('consultations')
            .update(dbUpdates)
            .eq('id', consultationId);

        if (error) {
            console.error('Error updating consultation:', error);
            alert('Failed to update consultation: ' + error.message);
        } else {
            await fetchPatients();
        }
    };

    const handleDeletePatient = async (patientId: string) => {
        const { error: consultError } = await supabase
            .from('consultations')
            .delete()
            .eq('patient_id', patientId);

        if (consultError) {
            console.error('Error deleting patient consultations:', consultError);
            alert('Failed to delete patient consultations: ' + consultError.message);
            return;
        }

        const { error: patientError } = await supabase
            .from('patients')
            .delete()
            .eq('id', patientId);

        if (patientError) {
            console.error('Error deleting patient:', patientError);
            alert('Failed to delete patient: ' + patientError.message);
        } else {
            await fetchPatients();
        }
    };

    const handleUpdatePatient = async (patientId: string, updates: { name?: string; age?: number; gender?: string; weight?: string; height?: string; bloodGroup?: string; nationality?: string; language?: string }) => {
        try {
            const { error } = await supabase
                .from('patients')
                .update({
                    name: updates.name,
                    age: updates.age,
                    gender: updates.gender,
                    weight: updates.weight,
                    height: updates.height,
                    blood_group: updates.bloodGroup,
                    nationality: updates.nationality,
                    language: updates.language || 'English'
                })
                .eq('id', patientId);

            if (error) throw error;
            fetchPatients();
        } catch (error: any) {
            console.error('Error updating patient:', error);
            alert(`Failed to update patient: ${error.message}`);
        }
    };

    const handleFileUpload = async (patientId: string, file: File) => {
        try {
            await uploadPatientFile(patientId, file);
            await fetchPatients();
        } catch (error: any) {
            console.error('File upload error:', error);
            throw error;
        }
    };

    const handleFileDelete = async (patientId: string, fileName: string) => {
        try {
            await deletePatientFile(patientId, fileName);
            await fetchPatients();
        } catch (error: any) {
            console.error('File delete error:', error);
            throw error;
        }
    };

    return (
        <div className="min-h-screen flex flex-col font-sans" >
            {currentView === View.LANDING && (
                <Landing onGetStarted={() => setCurrentView(View.AUTH)} />
            )}

            {currentView === View.AUTH && (
                <Auth onAuthSuccess={() => { }} onBack={() => setCurrentView(View.LANDING)} />
            )}
            {
                currentView === View.ONBOARDING && (
                    <Onboarding
                        initialName={authState.user?.name === 'Clinician' ? '' : authState.user?.name}
                        initialEmail={authState.user?.email}
                        onComplete={handleOnboardingComplete}
                        onExit={handleLogout}
                    />
                )
            }
            {
                currentView === View.DASHBOARD && authState.user && (
                    <Dashboard
                        user={authState.user}
                        patients={patients}
                        onLogout={handleLogout}
                        onStartScribe={(data) => {
                            setPreselectedPatientData(data || null);
                            if ((data as any)?.id) {
                                setDefaultPatientId((data as any).id);
                            }
                            window.history.pushState({ view: 'scribe' }, '', '');
                            setCurrentView(View.SCRIBE);
                        }}
                        onDeleteConsultation={handleDeleteConsultation}
                        onDeletePatient={handleDeletePatient}
                        onUpdatePatient={handleUpdatePatient}
                        onUpdateConsultation={handleUpdateConsultation}
                        onFileUpload={handleFileUpload}
                        onFileDelete={handleFileDelete}
                        defaultPatientId={defaultPatientId}
                        onClearDefaultPatient={() => setDefaultPatientId(null)}
                    />
                )
            }

            {
                currentView === View.SCRIBE && authState.user && (
                    <ScribeSession
                        prefilledData={preselectedPatientData || undefined}
                        onCancel={() => {
                            window.history.back();
                        }}
                        onSave={async (note, blob) => {
                            setPreselectedPatientData(null);
                            await handleSaveNote(note, blob);
                        }}
                    />
                )
            }
        </div>
    );
}
