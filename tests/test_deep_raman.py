import main
import numpy as np

def test_main_returns_noisy_raman():
    raman_array = main.main()
    assert  isinstance(raman_array, np.ndarray), print(type(raman_array))