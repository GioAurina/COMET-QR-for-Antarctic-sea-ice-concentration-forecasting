import time
import itertools
import torch
import numpy as np
import h5py
import scipy.io as sio
from sklearn.utils import extmath
from torch.utils.data import Dataset


class MaskManager:
    """
    The MaskManager class is used into the MultifidelityDataset class to handle the masking
    of available levels.
    Up to now, there are two strategies:
        (sequential: True) will prepare a MultifidelityDataset where the levels are masked in a sequential manner:
            if the level l is available, then all the previous levels 1, ..., l-1 will be available too.
            This strategy resembles the data preparation of the Progressive Neural Network approach.
        (sequential: False) will prepare a MultifidelityDataset where all the possible combination of the masks are
            computed. For example, if level l is available, this does not guarantee that level l-1 is available.
            In this way we hope that the model can learn different patterns in the data - this way of preparing the data
            was not possible with the PNN approach.
    """

    def __init__(self, n_levels, sequential=False):
        # Number of possible levels (n_entries of the mask)
        self.n_levels = n_levels
        # Whether the masks are sequential or complete
        self.sequential = sequential
        # Set the mappings {ids: mask} and {mask: ids}
        self.ids_to_mask, self.mask_to_ids = self._build_mask_ids_mappings()
        # Number of possible different masks
        self.n_masks = len(self.mask_to_ids.keys())

    def _build_mask_ids_mappings(self):
        ids_to_mask, mask_to_ids = (None, None)
        if not self.sequential:
            # Compute all possible combinations of mask, but remove the combination
            # where all the levels are masked
            masks = list(itertools.product([False, True], repeat=self.n_levels))
            masks.remove(tuple(True for _ in range(self.n_levels)))
            # Cast the masks to integer numbers
            int_masks = [[int(elem) for elem in mask] for mask in masks]
            # Interpret the integers masks as binary numbers and convert them to the
            # correspondent integer id
            ids = [int("".join(map(str, vec)), 2) for vec in int_masks]
        else:
            # Compute the sequential combinations for the masks
            masks = torch.triu(
                torch.ones(self.n_levels, self.n_levels, dtype=torch.bool), 1
            )
            masks = [tuple(level_mask.item() for level_mask in mask) for mask in masks]
            # Compute the ids correspondent to each mask
            ids = [sum(mask) for mask in masks]
        # Assemble two maps
        ids_to_mask = dict(zip(ids, masks))
        mask_to_ids = dict(zip(masks, ids))
        return ids_to_mask, mask_to_ids

    def masks_to_ids(self, masks: torch.Tensor) -> torch.Tensor:
        """
        This method takes as input a batch of masks (one mask for each observation
        in the batch) and returns a batch of IDs that will be fed to the embedding layer
        of the Model.
        """
        # First we need to cast the input masks (that are tensor) into tuples.
        casted_masks = [tuple(value.item() for value in mask) for mask in masks]
        # Retrieve the ids correspondent to the masks.
        ids = torch.tensor([self.mask_to_ids[mask] for mask in casted_masks])
        # Build for each observation the correspondent ids that will be passed to the model.
        # Each observation will be characterized by a sequence of n_levels ids.
        complete_ids = torch.cat(
            [
                ids.unsqueeze(1) * self.n_levels + level
                for level in range(self.n_levels)
            ],
            dim=1,
        )
        return complete_ids

    def tokenize_mask(self, mask: torch.Tensor) -> torch.Tensor:
        # First we need to cast the input mask (that is a tensor) into a tuple.
        casted_mask = tuple(value.item() for value in mask)
        # Retrieve the id correspondent to the mask.
        id = self.mask_to_ids[casted_mask]
        # Build the correspondent ids that will be passed to the model.
        # The mask is characterized by a sequence of n_levels ids.
        tokenized_mask = torch.tensor(
            [id * self.n_levels + level for level in range(self.n_levels)]
        )
        return tokenized_mask


class MultiFidelityDataset(Dataset):
    """
    features_by_levels:
        This is a dict with keys 'level_i' and the corresponding observations inside.
        Dimension for the tensors contained in each dict is (observation, sequence, features).
    targets:
        This is a tensor that contains the targets for the corresponding features.
        Dimension is (observation, sequence, features).
    """

    def __init__(self, features_by_level, targets, device, identifiers=None, sequential=False, single_mask=None):
        super().__init__()
        self.device = device
        self.n_levels = len(features_by_level)
        self.seq_len = features_by_level["level_0"].shape[1]
        self.feature_dim_by_level = {
            level: features.shape[2] for level, features in features_by_level.items()
        }
        self.dataset = []
        self.mask_manager = MaskManager(
            n_levels=self.n_levels - 1, sequential=sequential # Exclude level_0 from the mask manager (parameters are always available)
        )
        self.single_mask = single_mask  # If provided, only use this mask (no expansion)
        self._build_dataset(features_by_level, targets, identifiers)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def _build_dataset(self, features_by_levels, targets, identifiers=None):
        # Determine which masks to iterate over
        if self.single_mask is not None:
            # Use only the specified mask (no expansion into all combinations)
            # This is efficient for val/test when force_mask() will override anyway
            masks_to_use = [self.single_mask]
        else:
            # Use all possible masks from mask manager (for training)
            masks_to_use = self.mask_manager.ids_to_mask.values()
        
        # Loop on all the observations
        for i, target in enumerate(targets):
            # Loop on selected masks
            for mask in masks_to_use:
                # Build the observation
                curr_obs = {
                    f"level_{level}": (
                        features_by_levels[f"level_{level}"][i].detach().clone().to(self.device)
                        if not level_mask
                        else torch.zeros(
                            (self.seq_len, self.feature_dim_by_level[f"level_{level}"]),
                            dtype=torch.float32,
                        ).to(self.device)
                    )
                    for level, level_mask in enumerate(mask, 1)
                }
                curr_obs["level_0"] = features_by_levels["level_0"][i].detach().clone().to(self.device)
                # Target is the same
                curr_obs["target"] = target.detach().clone().to(self.device)
                # Mask the levels
                curr_obs["mask"] = torch.tensor(mask).to(self.device)
                # Build the input_ids for the embedding layer of the model
                curr_obs["input_ids"] = self.mask_manager.tokenize_mask(
                    curr_obs["mask"]
                ).to(self.device)
                # Add the identifier for this observation if needed
                if identifiers is not None:
                    curr_obs["id"] = identifiers[i]
                # Append the observation
                self.dataset.append(curr_obs)

class BaselineModelDataset(Dataset):
    """
    features_by_levels:
        This is a dict with keys 'level_i' and the corresponding observations inside.
        Dimension for the tensors contained in each dict is (observation, sequence, features).
    targets:
        This is a tensor that contains the targets for the corresponding features.
        Dimension is (observation, sequence, features).
    """

    def __init__(self, features_by_level, targets, device):
        super().__init__()
        self.device = device
        self.n_levels = len(features_by_level)
        self.seq_len = features_by_level["level_0"].shape[1]
        self.feature_dim_by_level = {
            level: features.shape[2] for level, features in features_by_level.items()
        }
        self.dataset = []
        self._build_dataset(features_by_level, targets)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def _build_dataset(self, features_by_levels, targets):
        # Loop on all the observations
        for i, target in enumerate(targets):
            # Build the observation
            curr_obs = {
                f"level_{level}": 
                    features_by_levels[f"level_{level}"][i].detach().clone().to(self.device)
                for level in range(self.n_levels)
            }
            # Target is the same
            curr_obs["target"] = target.detach().clone().to(self.device)
            # Append the observation
            self.dataset.append(curr_obs)

def compute_randomized_SVD(S, N_POD, N_h, n_channels, name="", verbose=False):
    if verbose:
        print("Computing randomized POD...")
    U = np.zeros((n_channels * N_h, N_POD))
    start_time = time.time()
    for i in range(n_channels):
        U[i * N_h : (i + 1) * N_h], Sigma, Vh = extmath.randomized_svd(
            S[i * N_h : (i + 1) * N_h, :],
            n_components=N_POD,
            transpose=False,
            flip_sign=False,
            random_state=123,
        )
        if verbose:
            print("Done... Took: {0} seconds".format(time.time() - start_time))

    if verbose:
        I = 1.0 - np.cumsum(np.square(Sigma)) / np.sum(np.square(Sigma))
        print(I[-1])

    if name:
        sio.savemat(name, {"V": U[:, :N_POD]})

    return U, Sigma


def load_navier_stokes_data(path, params, train_test, t0, T, dt, verbose=0):
    data_snap = []
    data_drag = []
    data_lift = []

    # time range
    start = int(t0 / dt)
    end = int(T / dt)

    dofs_vx = list(range(3899))
    dofs_vy = list(range(15270, 15270 + 3899))
    dovs_p = list(range(15270 * 2, 15270 * 2 + 3899))

    dofs = dofs_vx + dofs_vy  # + dovs_p
    for param in params:

        # Load snapshot data
        name_snap = (
            path + "/snapshots" + "/snap_" + str(int(param)) + "_" + train_test + ".mat"
        )
        with h5py.File(name_snap, "r") as file:
            snap = file["snapshots"][:].T
        data_snap.append(snap[start:end, :])

        # Load drag and lift data
        name_drag = (
            path + "/drag" + "/drag_" + str(int(param)) + "_" + train_test + ".mat"
        )
        with h5py.File(name_drag, "r") as file:
            drag = file["Drag"][:].T
        data_drag.append(drag[start:end, :])

        name_lift = (
            path + "/lift" + "/lift_" + str(int(param)) + "_" + train_test + ".mat"
        )
        with h5py.File(name_lift, "r") as file:
            lift = file["Lift"][:].T
        data_lift.append(lift[start:end, :])

    data_snap = np.array(data_snap)[:, :, dofs]
    data_drag = np.array(data_drag)
    data_lift = np.array(data_lift)
    if verbose:
        print("Loaded data for param = ", params)
        print("Snap shape = ", data_snap.shape)
        print("Drag shape = ", data_drag.shape)
        print("Lift shape = ", data_lift.shape)

    return data_snap, data_drag, data_lift


def sliding_windows(data_input, seq_length, freq=1, return_initial_conditions=False):
    x = []
    init_cond = []
    # Adjust the range to account for the new padded length
    for i in range(data_input.shape[0]):
        # Start from 0 to accommodate the new padded length
        for j in range(0, data_input.shape[1] - seq_length + 1, freq):
            _x = data_input[i, j : (j + seq_length), :]
            x.append(_x)
            # If it's the beginning of the sequence
            if j == 0:
                init_cond.append(_x)
            else:
                init_cond.append(np.zeros_like(_x))
    x, init_cond = np.array(x), np.array(init_cond)
    output = (x, init_cond) if return_initial_conditions else x
    return output
