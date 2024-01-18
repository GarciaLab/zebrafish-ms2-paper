#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code for a 2 component, single cell bursting model with flexible feedback functions. Implementation is a hybrid Euler-Gillespie scheme.
"""
import numpy as np
from scipy.signal import find_peaks
from scipy.integrate import solve_ivp


def simulate(p=None):
    """one cell only, so gillespie algorithm only needs one random number per event (theres only one event,
    off--> on, or on --> off"""
    if p is None:
        p = Params()

    state = np.array([p.initial_state])
    mrna = np.array([p.initial_mrna])
    protein = np.array([p.initial_protein])

    t = 0
    tvec = np.array([t])
    lots_of_random_numbers = np.random.rand(int(p.number_of_random_numbers_to_pregenerate))

    random_number_counter = 0
    while t < p.Tmax:
        if random_number_counter < p.number_of_random_numbers_to_pregenerate:
            this_random_number = lots_of_random_numbers[random_number_counter]
        else:
            print("warning: ran out of pregenerated random numbers, generating new ones")
            lots_of_random_numbers = np.random.rand(np.round(p.number_of_random_numbers_to_pregenerate))
            random_number_counter = 0
            this_random_number = lots_of_random_numbers[random_number_counter]

        state, mrna, protein, tvec, t = update(state, mrna, protein, p, tvec, t, this_random_number)
        random_number_counter += 1

    return state, mrna, protein, tvec, p


def update(state, mrna, protein, p, tvec, t, this_random_number):
    """This is an update step for a hybrid Euler-Gillespie scheme:
        1: compute time to next promoter switching event, defined as 
            min(time to next reaction from Gillespie's algorithm, 
                a user-defined p.max_time_to_next_reaction).
        2: Euler integrate the continuous variables up the new time.
        3: apply the promoter switch to the last state value."""
    
    # get current values of the simulation
    current_state = state[-1]
    current_mrna = mrna[-1]
    current_protein = protein[-1]

    """1. compute time to next switching event"""
    # determine the protein value that will influence the various rates. 
    # If there is a delay, extract the delayed protein value. 
    # Otherwise, use the current protein level.
    if p.delay > 0:
        delayed_id = compute_delayed_id(p.delay, tvec, t)
        repressing_protein = protein[delayed_id]
    else:
        repressing_protein = current_protein
        
    # compute the switching rate ("propensity", in Gillespie's language)
    if current_state:
        # currently on, switching off
        rate = p.k_off0 + p.k_off1 * (repressing_protein / p.KD) ** p.n * hill_function(repressing_protein, p.KD, p.n)
        
    else:
        # currently off, switching on
        rate = p.k_on0 + p.k_on1 * hill_function(repressing_protein, p.KD, p.n)

    # compute time to next reaction
    delta_t = get_time_to_next_reaction(rate, this_random_number, p)
    

    """2. integrate her up to t + delta_t"""
    # Euler time step
    dt = np.min((delta_t, p.dt))

    # create arrays for the new time points
    tvec_addition = np.arange(t + dt, t + delta_t + dt, dt)
    num_new_time_points = len(tvec_addition)
    
    # new mrna values
    mrna_addition = np.zeros(num_new_time_points + 1)
    mrna_addition[0] = current_mrna
    
    # new protein values
    protein_addition = np.zeros(num_new_time_points + 1)
    protein_addition[0] = current_protein
    
    # new state values
    state_addition = current_state * np.ones(num_new_time_points + 1)
    state_addition[0] = current_state

    # gaussian random numbers
    these_gaussian_numbers = np.random.normal(size=(num_new_time_points, 2))
        
    # Euler integrate up to t + delta_t
    for s in range(1, num_new_time_points + 1):
        mrna_addition[s] = mrna_addition[s - 1] + (dt * (
                (p.transcription_rate_0 + p.transcription_rate_1 * hill_function(repressing_protein, p.KD, p.n)) *  state_addition[s - 1] - p.mrna_decay_rate * mrna_addition[s - 1]) + np.sqrt(dt) * p.noise_strength * np.sqrt(p.transcription_rate_0 + p.transcription_rate_1 * hill_function(repressing_protein, p.KD, p.n) *  state_addition[s - 1] + p.mrna_decay_rate * mrna_addition[s - 1]) * these_gaussian_numbers[s - 1, 0]
                )

        protein_addition[s] = protein_addition[s - 1] + (dt * (
                p.translation_rate * mrna_addition[s - 1] - p.protein_decay_rate * protein_addition[s - 1]) + np.sqrt(dt) * p.noise_strength * np.sqrt(p.translation_rate * mrna_addition[s - 1] + p.protein_decay_rate * protein_addition[s - 1]) * these_gaussian_numbers[s - 1, 1]
                )

        if mrna_addition[s] < 0:
            mrna_addition[s] = 0
        if protein_addition[s] < 0:
            protein_addition[s] = 0

    """3. correct last state time point to reflect stochastic switch"""
    if delta_t < p.max_time_to_next_reaction:
        if current_state:
            # cell goes on --> off
            state_addition[-1] = 0
        else:
            state_addition[-1] = 1
 
    """collect output and prepare to return"""
    # append the new values to the main arrays
    tvec = np.append(tvec, tvec_addition)
    state = np.append(state, state_addition[1:])
    mrna = np.append(mrna, mrna_addition[1:])
    protein = np.append(protein, protein_addition[1:])

    # check to make sure all new values are accounted for
    assert len(tvec) == len(state)
    assert len(state) == len(mrna)
    assert len(mrna) == len(protein)

    # update current time
    t = tvec[-1]

    return state, mrna, protein, tvec, t


def compute_delayed_id(tau, tvec, t):
    delayed_time = t - tau
    delayed_id = np.int32(np.where(np.abs(tvec - delayed_time) == np.min(np.abs(tvec - delayed_time)))[0][0])

    return delayed_id


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
        self.KD = 10.0
        self.n = 2.0
        self.delay = 0.0
        self.noise_strength = 0.0
        
        
def peak_intervals(trace, tvec=None, prominence=0.25):
    trace = trace / np.max(trace)
    peaks, _ = find_peaks(trace, prominence=prominence)
    
    if tvec is not None:
        intervals = np.diff(tvec[peaks])
    else:
        intervals = np.diff(peaks)

    return intervals    
    

def sim_ms2(state, tvec, loading_rate, w, delta_t, sigma, detection_threshold):
    """given a sequence of promoter state values (0,1), simulate a realistic MS2 trace. Adapted from Nick Lammers' cpHMM code.
    The main idea is to include a memory kernel that adds the contributions of all polymerases that have been on the gene 
    since the last observation time.
    
    can pass a vector of loading_rate, representing, for instance, repression by the protein for amp reg"""
    if not isinstance(loading_rate, np.ndarray):
        loading_rate = loading_rate * np.ones_like(state)
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
        loading_rate_window = loading_rate[start_ind:end_ind]
        
        # sum the contributions of all the previous times
        ms2[i] = np.sum(loading_rate_window[1:] * state_window[1:] * dt_window)
        
    # add gaussian noise
    ms2 += ms2 * np.random.normal(scale=sigma, size=len(ms2))
    
    # simulate a detection threshold
    ms2[ms2 <= detection_threshold] = 0
    
    return ms2, uniform_times
    
    
def delay_protein(protein, tvec, tau):
    shift_id = np.where(tvec >= tau)[0][0]
    delayed_protein = np.roll(protein, shift_id)
    delayed_protein[:shift_id] = 0
    
    return delayed_protein
    
    
    
    

    
            
    
    