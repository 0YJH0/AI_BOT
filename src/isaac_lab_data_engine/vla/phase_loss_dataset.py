"""Raw phase metadata for loss weighting; not a new policy input."""
import torch


class PhaseLossDataset:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __getitem__(self, index):
        sample = dict(self.dataset[index])
        state = sample['observation.state']
        if state.shape != (30,):
            raise ValueError('Lift loss requires raw 30D legacy state')
        phase = float(state[-1])
        if phase not in (0., 1., 2., 3., 4.):
            raise ValueError('Loss phase must be read before state normalization')
        sample['training.loss_phase'] = torch.tensor(int(phase), dtype=torch.int64)
        return sample
