import numpy as np
from pydmd import BOPDMD
from tqdm.autonotebook import trange, tqdm


# time delayed stacking of z
# z is (Ntime x Nx), k is number of stacks 
# returns (Ntime-k+1 x Nx). td(z, 1) == z
from functools import reduce, partial
from itertools import chain
td1 = lambda z, k: reduce(lambda x, y: np.concatenate((x,y), axis = 1), 
      (z[slice(i, j)] for i, j in zip(range(k), chain(range(-k+1, 0, 1),(None,)))))

td = lambda z, k: np.concatenate([z[slice(i, j)] for i, j in zip(
                                          range(k), 
                                          chain(range(-k+1, 0, 1), (None,)))],
                      axis = 1)


def reshape_Psi2data(Psi, data_shape, mask = None):
    """
    Psi that we get from dmd are shaped like the time-delayed data. 
    This reshapes it properly to shape of input data

    Parameters: 
        Psi - shape (ny*nx, rank) or (n, rank) eigenfuns of Koopman, obtained form DMD
        data_shape - tuple shape of training data (N_times, ny, nx)
        mask - boolean mask shape (ny, nx). DMD was trained on
             only those pixels that are True in the mask


    Returns:
        Psi, reshaped like dmd input data    
    """
    ny, nx = data_shape[1:]
    rank = Psi.shape[-1] # rank
    
    # those not in mask will be zeros
    
    if mask is not None: 
        Psi_mask = np.zeros((ny*nx, rank))+0j
        Psi_mask[mask.flatten(), :] = Psi[:mask.sum()]

        out = Psi_mask.reshape(ny, nx, -1).transpose([2, 0, 1])
    else:
        out = Psi[:nx*ny].reshape(ny, nx, -1).transpose([2, 0, 1])

    return out

def reshape_data2dmd(X, t, time_delay = 2, mask = None, isKeepFirstTimes = True):
    """
    Convert data array of shape (N_time, ny, nx) to time-delayed dmd snapshot matrix

    Parameters:
        X - data of shape (N_time, ny, nx)
        t - associated time array of shape (N_time,)
        time_delay - time delay
        mask - boolean mask (ny, nx) which applies to each X[i], s.t. X[i][~mask] is to
                be excluded from modeling
        isKeepFirst - whether to truncate time at end of array or not

    Returns:
        X_delayed - DMD snapshot matrix obtained by flattening the data and time-delaying it
            array of shape (ny*nx*time_delay, N_time-time_delay+1)
        t_dleayed - corresponding array of time (N_time-time_delay+1, )
        data_shape - tuple (ny, nx) containing shape of original data for reshaping 
    
    """

    
    
    # (N_time, ny, nx)->(N_time, ny*nx) and select masked
    X_dmd  = X[:, mask] if mask is not None else X.reshape(len(X), -1)

    # kick out time_delay-1 elements from start or end of t
    t_delayed = t[:-time_delay+1] if isKeepFirstTimes else t[time_delay-1:]

    if time_delay == 1: t_delayed = t

    # time delay and transpose to shape (2*ny*nx, N_time-time_delay)
    X_delayed = td(X_dmd, time_delay).T
    
    return X_delayed, t_delayed, X.shape

def reshape_data2dmd_delme(X, t, time_delay = 2, isKeepFirstTimes = True):
    """
    Convert data array of shape (N_time, ny, nx) to time-delayed dmd snapshot matrix

    Parameters:
        X - data of shape (N_time, ny, nx)
        t - associated time array of shape (N_time,)
        time_delay - time delay
        isKeepFirst - whether to truncate time at end of array or not

    Returns:
        X_delayed - DMD snapshot matrix obtained by flattening the data and time-delaying it
            array of shape (ny*nx*time_delay, N_time-time_delay+1)
        t_dleayed - corresponding array of time (N_time-time_delay+1, )
        data_shape - tuple (ny, nx) containing shape of original data for reshaping 
    
    """
    
    # delayed time array
    if isKeepFirstTimes: t_delayed = t[:-time_delay+1]
    else: t_delayed = t[time_delay-1:]

    # convert (N_time, ny, nx) -> (ny*nx, N_time)
    X_dmd = X.reshape(len(X), -1)

    # time delay and transpose to shape (2*ny*nx, N_time-time_delay)
    X_delayed = td(X_dmd, time_delay).T
    
    return X_delayed, t_delayed, X.shape
    
def train_dmd(X_delayed, 
              t_delayed, 
              svd_rank=3, 
              eig_constraints={
                        "stable", # choose Re(lambda)<0
                        "conjugate_pairs", # force complex conjugate pairs
                        },
              **dmd_kwargs, 
                ):
    """
    Train dmd on snapshots X of time-delayed flattened square 2D data and times t. 
    
    Parameters: 
        X - (time_delay*ny*nx, N_time-time_delay+1): array of 
            snapshots corresponding to t time points
        t - (N_time, ) array of time points
        eig_constraints - constraints on eigenvalues -- see BOPDMD doc
        
    
    Returns: 
    The DMD fit parameters
        Lambda (rank, )
        Psi of shape (time_delay*ny*nx, rank) 
        bn (rank, )
     
        
    Note: 
    1. must perform manual time delay, because must call 
          during bagging on time-delayed data

    2. Workflow
    
    # X0 is time series of images of shape (N_time, ny, nx)
    
    # prepare time delay and reshape data as dmd input 
    X_delayed, t_delayed, data_shape = reshape_data2dmd(X0, t, time_delay = 2, 
            isKeepFirstTimes = True)
    
    # train dmd - here can train dmd with bagging by bootstrap over X_delayed!
    Lambda, Psi_, bn = train_dmd_(X_delayed, t_delayed, rank = 3)
    
    # convert modes to same shape as data
    Psi = reshape_dmd2data(Psi_, data_shape)

    """


    # DMD OBJECT
    optdmd = BOPDMD(
                    svd_rank=svd_rank, 
                    num_trials=0, # for bagging
                    eig_constraints=eig_constraints,
                    **dmd_kwargs
                                    )

    
    
    # fit dmd
    optdmd.fit(X_delayed, t_delayed)


    

    # GET DMD MODES AND EIGS
    # Get modes, cutting out the time-delay
    # Psi = optdmd.modes[:nx*ny].reshape(ny, nx, -1).transpose([2, 0, 1])

    Psi = optdmd.modes

    # Get eigenvalues:
    Lambda = optdmd.eigs

    # The b_n: IC, expressed in modes basis
    bn = optdmd.amplitudes

    return Lambda, Psi, bn


# --- FUNZIONE MODIFICATA MANTENENDO LA TUA STRUTTURA ---
def bootstrap_train_dmd(N_boot_strap, X_delayed, t_delayed, svd_rank=3, 
              eig_constraints={
                        "stable", # choose Re(lambda)<0
                        "conjugate_pairs", # force complex conjugate pairs
                        }, 
                        **dmd_kwargs):
    """
    Bootstrap over t for dmd with N_boot_strap samples 

    X_delayed: shaped as (time_delay*ny*nx, N_time-time_delay+1): 
                array of snapshots corresponding to t time points
    t_delayed: shaped as (N_time-time_delay+1, )
        
    """
    
    nx = X_delayed.shape[0]
    nt = len(t_delayed)

    L_s = np.zeros((N_boot_strap, svd_rank), dtype=complex)
    Psi_s_ = np.zeros((N_boot_strap, nx, svd_rank), dtype=complex)
    bn_s = np.zeros((N_boot_strap, svd_rank), dtype=complex)

    # perform bootstrap resampling
    i = 0
    pbar = tqdm(total=N_boot_strap, desc="Bootstrap DMD", leave=False)
    
    while i < N_boot_strap:
        # Resampling degli indici (come nella tua versione originale)
        inds = np.unique(np.sort(np.random.randint(0, nt-1, (nt,) ) ) )
        X1 = X_delayed[:, inds]
        t1 = t_delayed[inds]

        try:
            # Tentativo di training
            l, psi, bn = train_dmd(X1, 
                                   t1, 
                                   svd_rank = svd_rank, 
                                   eig_constraints = eig_constraints, 
                                   **dmd_kwargs)
            
            # Assegnazione SOLO se non fallisce
            L_s[i] = l
            Psi_s_[i] = psi
            bn_s[i] = bn
            
            # Incremento contatore solo su successo
            i += 1
            pbar.update(1)
            
        except Exception:
            # Se fallisce, il ciclo 'while' continua senza incrementare 'i'
            # quindi riproverà automaticamente con nuovi indici random
            continue
            
    pbar.close()

    return L_s, Psi_s_, bn_s

def eval_dmd(Lambda, Psi, bn, t, isPositive = True):
    """
    Assemble DMD expansion from Lambda, Psi, bn and evaluate at t
    Take real part and set neagive values to zero

    Parameters:
        Lambda - dmd eigenvals shape (rank, )
        Psi - dmd eigenvecs with shape (rank, ny, nx)
        bn - dmd expansion coefs of IC
        t - time at which to compute
    """

    dmd_expansion = lambda t, Lambda, Psi, bn: (Psi.T @ (bn[:,None]*np.exp(Lambda[:, None]*t))).T

    out = dmd_expansion(t, Lambda, Psi, bn).real
    if isPositive: out[out<0]=0.

    return out

def eval_dmd_latent(Lambda, bn, t, isPositive = True):
    """
    Assemble DMD expansion from Lambda, Psi, bn and evaluate at t
    Take real part and set neagive values to zero

    Parameters:
        Lambda - dmd eigenvals shape (rank, )
        Psi - dmd eigenvecs with shape (rank, ny, nx)
        bn - dmd expansion coefs of IC
        t - time at which to compute
    """

    dmd_expansion = lambda t, Lambda, bn: (bn[:,None]*np.exp(Lambda[:, None]*t)).T

    out = dmd_expansion(t, Lambda, bn).real
    #if isPositive: out[out<0]=0.
    return out

def eval_dmd_ensemble(L_s1, Psi_s1, bn_s1, T, isPositive = True):
    """
    same as eval_dmd, but for ensembles of lamnda, etc, stacked along leading dim
    """
    
    out = np.zeros((L_s1.shape[0], len(T), Psi_s1.shape[-1], Psi_s1.shape[-1]))

    for i, (lam, psi, bn) in tqdm(enumerate(zip(L_s1, Psi_s1, bn_s1)), total=L_s1.shape[0]):
        out[i] = eval_dmd(lam, psi, bn, T, isPositive)

    return out