#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 10:35:52 2024

@author: brandon
"""


"""
Code for a 2 component, single cell bursting model with flexible feedback functions. Implementation is an exact Gillespie scheme with option for delayed transcription.
"""
import numpy as np
from scipy.signal import find_peaks
from scipy.integrate import solve_ivp
import warnings


def hill_function(x, KD, n):
    """decreasing hill function"""
    return 1 / (1 + (x / KD) ** n)


def get_time_to_next_reaction(rate, random_number, p):
    if rate > 0:
        delta_t = np.clip(1 / rate * np.log(1 / random_number), a_min=0, a_max=p.max_time_to_next_reaction)
    else:
        delta_t = p.max_time_to_next_reaction
        
    return delta_t


class Params:
    def __init__(self):
        # time in minutes
        self.transcription_rate_0 = 1.0
        self.transcription_rate_1 = 0.0
        self.translation_rate = 1.0
        self.mrna_decay_rate = 0.23
        self.protein_decay_rate = 0.23
        self.initial_state = 0
        self.initial_mrna = 0.0
        self.initial_protein = 0.0
        self.Tmax = 180
        self.dt = 0.01
        self.number_of_random_numbers_to_pregenerate = 1e5
        self.max_time_to_next_reaction = 5
        self.k_on0 = 0.0
        self.k_off0 = 0.0
        self.k_on1 = 0.03
        self.k_off1 = 0.1
        self.KD_transcription_rate = 10.0
        self.KD_k_on = 10.0
        self.KD_k_off = 10.0
        self.n = 2.0
        self.delay = 0.0
        
        
def simulate_multiple_copies(p):
    """simulate one cell with multiple copies of the her1 promoter and gene, 
    which are all influenced by a communal pool of Her1 proteins"""
    if p is None:
        p = Params()
    #else:
        # check that the user-provided parameters have appropriate values
        #p = check_params(p)
                
        
    """vector describing the state of the full system. row=time, columns = (gene state 1, ..., gene state C, num. mrna, num. proteins """
    X = np.concatenate((np.array(p.initial_state), np.array([p.initial_mrna]), np.array([p.initial_protein])))
    if len(X.shape) == 1:
        X = np.expand_dims(X, axis=0)

    t = 0
    tvec = np.array([t])
    # now have two random numbers per update step
    lots_of_random_numbers = np.random.rand(int(p.number_of_random_numbers_to_pregenerate), 2)

    random_number_counter = 0
    delayed_transcription_reactions = []
    while t < p.Tmax:
        if random_number_counter < p.number_of_random_numbers_to_pregenerate:
            these_random_numbers = lots_of_random_numbers[random_number_counter]
        else:
            print("warning: ran out of pregenerated random numbers, generating new ones")
            lots_of_random_numbers = np.random.rand(int(p.number_of_random_numbers_to_pregenerate), 2)
            random_number_counter = 0
            these_random_numbers = lots_of_random_numbers[random_number_counter]

        X, tvec, t, delayed_transcription_reactions = update_multiple_copies(X, p, tvec, t, these_random_numbers, delayed_transcription_reactions)
        random_number_counter += 1

    return X, tvec, p


def update_multiple_copies(X, p, tvec, t, these_random_numbers, delayed_transcription_reactions):
    # get current values of the simulation
    current_X = X[-1]
    current_gene_states = current_X[:-2]
    current_mrna = current_X[-2]
    current_protein = current_X[-1]

    n_genes = len(current_gene_states)

    """compute time to next switching event"""
    # compute the switching rate ("propensity", in Gillespie's language)
    off_propensities = current_gene_states * (p.k_off0 + p.k_off1 * (current_protein / p.KD_k_off) ** p.n * hill_function(current_protein, p.KD_k_off, p.n))
    on_propensities = (1 - current_gene_states) *  (p.k_on0 + p.k_on1 * hill_function(current_protein, p.KD_k_on, p.n))
    transcription_propensity = np.sum(current_gene_states) * (p.transcription_rate_0 + p.transcription_rate_1 * hill_function(current_protein, p.KD_transcription_rate, p.n))
    translation_propensity = p.translation_rate * current_mrna
    mrna_decay_propensity = p.mrna_decay_rate * current_mrna
    protein_decay_propensity = p.protein_decay_rate * current_protein

    all_rates = np.concatenate((off_propensities, 
                                on_propensities, 
                                np.array([transcription_propensity]), 
                                np.array([translation_propensity]), 
                                np.array([mrna_decay_propensity]), 
                                np.array([protein_decay_propensity])))
    
    total_rate = np.sum(all_rates)
    
    # compute time to next reaction
    delta_t = get_time_to_next_reaction(total_rate, these_random_numbers[0], p)
    
    """compute which reaction happens"""
    reaction_id = np.where(np.cumsum(np.array(all_rates)) / total_rate > these_random_numbers[1])[0][0]
    
    """check for delayed reactions"""
    if len(delayed_transcription_reactions) > 0:
        #print('delay!')
        """delayed_transcription_reactions is a list of times when the transcription reaction should complete and reaction corresponding to the particular gene. 
        the list of times should be monotonically increasing"""
        delayed_production_time, delayed_reaction_id = delayed_transcription_reactions[0]
        if t + delta_t > delayed_production_time:
            delta_t = delayed_production_time - t
            reaction_id = 2 * n_genes + 4
            delayed_transcription_reactions = delayed_transcription_reactions[1:]
    
    """execute the reaction"""
    if reaction_id < n_genes:
        # switch off
        current_gene_states[reaction_id] = 0
    elif reaction_id < 2 * n_genes:
        # switch on
        current_gene_states[reaction_id - n_genes] = 1
    elif reaction_id == 2 * n_genes:
        # initiate transcription reaction
        delayed_transcription_reactions.append([t + delta_t + p.delay, reaction_id])
    elif reaction_id == 2 * n_genes + 1:
        # translate
        current_protein += 1
    elif reaction_id == 2* n_genes + 2:
        # mrna decay
        current_mrna -= 1
    elif reaction_id == 2 * n_genes + 3:
        # protein decay
        current_protein -= 1
    elif reaction_id == 2 * n_genes + 4:
        # complete delayed transcription
        current_mrna += 1


    X_out = np.concatenate((current_gene_states, np.array([current_mrna]), np.array([current_protein])))
    X = np.concatenate((X, np.expand_dims(X_out, axis=0)), axis=0)
    # update current time
    t += delta_t
    tvec = np.concatenate((tvec, np.array([t])))

    return X, tvec, t, delayed_transcription_reactions


def check_params(p):
    """run some checks on the Params class to make sure it's self consistent. adapt if needed"""
    # if the user passes an array for initial_state, indicating they want to simulate multiple copies,
    # but transcription_weights is not an array, turn it into a repeated array.
    if isinstance(p.initial_state, np.ndarray):
        if not isinstance(p.transcription_weights, np.ndarray):
            p.transcription_weights = p.transcription_weights * np.ones(len(p.initial_state))
    
    return p

            
def peak_intervals(trace, tvec=None, prominence=0.25):
    trace = trace / np.max(trace)
    peaks, _ = find_peaks(trace, prominence=prominence)
    
    if tvec is not None:
        intervals = np.diff(tvec[peaks])
    else:
        intervals = np.diff(peaks)

    return intervals    
    

def sim_ms2(state, tvec, production_rate, w, delta_t, sigma, detection_threshold, conversion_factor=1.0):
    """given a sequence of promoter state values (0,1), simulate a realistic MS2 trace. Adapted from Nick Lammers' cpHMM code.
    The main idea is to include a memory kernel that adds the contributions of all polymerases that have been on the gene 
    since the last observation time.
    
    can pass a vector of production_rate, representing, for instance, repression by the protein for amp reg.
    
    note that for the zebrafish her1 reporter, elongation is fast enough and the sequence short enough that the memory effect is neglibile. 
    Thus, we simplify by ignoring the fluorescence weighting by position along the sequence."""
    if not isinstance(production_rate, np.ndarray):
        production_rate = production_rate * np.ones_like(state)
    uniform_times = np.arange(np.min(tvec), np.max(tvec), delta_t)
    ms2 = np.zeros_like(uniform_times)
    
    for i in range(1, len(ms2)):
        # collect the set of previous times that could have contributed to the fluorescence signal at the current time
        t_end = uniform_times[i]
        t_start = np.max((np.min(tvec), t_end - w * delta_t ))
        start_ind = np.where(tvec <= t_start)[0][-1]
        end_ind = np.where(tvec <= t_end)[0][-1]
        times_window = tvec[start_ind:end_ind] 
        state_window = state[start_ind:end_ind]
        dt_window = np.diff(times_window)
        production_rate_window = production_rate[start_ind:end_ind]
        
        # sum the contributions of all the previous times
        ms2[i] = np.sum(production_rate_window[1:] * state_window[1:] * dt_window)
        
    
    #ms2 = np.interp(np.arange(np.min(tvec), np.max(tvec), delta_t), uniform_times, ms2)
    #uniform_times = np.arange(np.min(tvec), np.max(tvec), delta_t)
    
    # convert from number of molecules to fluorescence a.u.
    ms2 *= conversion_factor
    
    # add gaussian noise
    ms2 += ms2 * np.random.normal(scale=sigma, size=len(ms2))
    
    # simulate a detection threshold
    ms2[ms2 <= detection_threshold] = 0
    
    return ms2, uniform_times
    
    

    
    
    

    
            
    
    