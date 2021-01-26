import torch
import random
import numpy as np
import tarfile

class MaintainRandomState:
    def __enter__(self):
        self.np_state = np.random.get_state()
        self.random_state = random.getstate()
        self.torch_state = torch.get_rng_state()
        #self.torch_cuda_state = torch.cuda.get_rng_state_all()

    def __exit__(self, type, value, traceback):
        np.random.set_state(self.np_state)
        random.setstate(self.random_state)
        torch.set_rng_state(self.torch_state)
        #torch.cuda.set_rng_state_all(self.torch_cuda_state)

def untar(fname,output_dir):
    mode="r:gz" if fname.name.endswith("tar.gz") else "r:"
    with tarfile.open(fname,mode) as tar:
        tar.extractall(output_dir)