import main
import model
from deep_raman import metrics

import numpy as np

def test_main_returns_noisy_raman():
    raman_array = main.main()
    assert  isinstance(raman_array, np.ndarray), print(f"Expecting output of type np.ndarray but received {type(raman_array)}")

def test_model_returns_correct_outputs():
    NUM_EPOCHS = 1
    LOSS_FUNCTION = metrics.psnr_loss
    outputs = model.main(NUM_EPOCHS, LOSS_FUNCTION)

    for output in outputs:
        assert isinstance(output, np.ndarray), print(f"Expecting output of type np.ndarray but received {type(output)}")
