import numpy as np
from pathlib import Path
import os
import sys
sys.path.append('../')
from Auxillary.DESSCA import dessca_model

def get_dessca_samples(state_names, box_constraints, ref_pdf, num_samples):
    desc = dessca_model(box_constraints=box_constraints,
                                  state_names=state_names,
                                  bandwidth=0.1,
                                  render_online=False,
                                  reference_pdf=ref_pdf)
    samples = []
    next_sample_suggest = desc.sample_optimally()
    for i in range(num_samples-1):
        print(i, end='\r')
        samples.append(next_sample_suggest)
        desc.update_coverage_pdf(data=np.transpose([next_sample_suggest]))
        next_sample_suggest = desc.sample_optimally()
    samples.append(next_sample_suggest)
    samples_np = np.array(samples)
    return samples_np

def get_context_obs(num_samples):

    def ref_pdf(X):
        i_d = X[0]
        i_q = X[1]
        currents_circle = np.less(i_d ** 2 + i_q ** 2, 1)
        return currents_circle.astype(float)

    box_constraints=[
        [-1, 1],
        [-1, 1],
        [-1, 1],
        [-1, 1],
        [-1, 1],
        [-1, 1],
    ]
    state_names = ['i_d', 'i_q', 'omega', 'epsilon', 'u_d', 'u_q']
    samples = get_dessca_samples(state_names, box_constraints, ref_pdf, num_samples)

    return samples


def get_test_obs(num_samples):

    def ref_pdf(X):
        i_d = X[0]
        i_q = X[1]
        i_d_ref = X[3]
        i_q_ref = X[4]
        currents_circle = np.less(i_d ** 2 + i_q ** 2, 1)
        currents_circle2 = np.less(i_d_ref**2 + i_q_ref**2, 1)
        i_d_neg = np.less(i_d_ref, 0)
        f = np.logical_and(currents_circle, currents_circle2)
        g = np.logical_and(f, i_d_neg)
        return g.astype(float)

    box_constraints=[
        [-1, 1],
        [-1, 1],
        [-1, 1],
        [-1, 1],
        [-1, 1],
    ]
    state_names = ['i_d', 'i_q', 'omega', 'i_d*', 'i_q*']
    samples = get_dessca_samples(state_names, box_constraints, ref_pdf, num_samples)
    return samples




if __name__ == "__main__":
    code_path = Path(__file__).parent.absolute()
    save_path = code_path.parent.parent / "Save" / "DESSCA_Samples"
    batch_of_samples = []
    num_train_motors = 100

    for i in range(num_train_motors):
        samples = get_context_obs(1000)
        batch_of_samples.append(samples)
    np.save(save_path / 'context_samples_training', np.array(batch_of_samples))

    batch_of_samples = []
    num_test_motors = 50
    for i in range(num_train_motors):
        samples = get_context_obs(1000)
        batch_of_samples.append(samples)
    np.save(save_path / 'context_samples_test', np.array(batch_of_samples))

    samples = get_test_obs(20000)
    np.save(save_path / 'test_routine_samples', np.array(samples))