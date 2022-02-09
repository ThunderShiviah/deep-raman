from utils import *
import streamlit as st

#Intialise a random number generator
rng = np.random.default_rng(2021)
num_peak_range=(1,50)
scale_range=(4.,5.4)
shift_range=(-100, 190)
alpha_range=(.2,4.)
gamma_range=(.2,4.)
n_background_range=(0,10)
scale_background_range=(0.,4.)
shift_background_range=(-100, 190)
alpha_background_range=(.2,4.)
a_background_range=(-5.,5.)
b_background_range=(-5.,5.)
c_background_range=(-5.,5.)

rand_gen = lambda x: rng.uniform(*x)

param_factory = {
    'n': num_peak_range,
    'scale': scale_range,
    'shift': shift_range,
    'alpha': alpha_range,
    'gamma': gamma_range,
    'scale_background': scale_background_range,
    'alpha_background': alpha_background_range,
    'a': a_background_range,
    'b': b_background_range,
    'c': c_background_range,
    'shift_background': shift_background_range
}

# Need to curry the dist so it works with defaultdict
def output_default_dist():
    return rng.uniform

distributions = collections.defaultdict(output_default_dist)

distributions['n'] = rng.integers # Number of peaks is an integer.
distributions['n_background'] = rng.integers # Number of peaks is an integer.


params = {k:distributions[k](*v) for k,v in param_factory.items()}

interactive_params = {}
for param, value_range in param_factory.items():
    min_value, max_value = float(value_range[0]), float(value_range[1])
    value=float(distributions[param](*value_range))

    try:
        interactive_params[param] = st.sidebar.slider(param, min_value=min_value, max_value=max_value, value=value)
    except st.StreamlitAPIException as e:
        st.write(e)
        st.write(param, min_value, max_value, value)
        break



x = np.linspace(-200,200,1000)



noisy_raman = generate_single_raman_example(x, **interactive_params)

st.write(px.line(noisy_raman))