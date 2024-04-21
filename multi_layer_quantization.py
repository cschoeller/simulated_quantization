"""Example for a static symmetric quantizing of multiple dense layers without activations."""

import numpy as np

np.random.seed(10)
np.set_printoptions(suppress=True)


_NUM_FEATURES = 3
_HIDDEN_FEATURES = 15
_VAL_MIN = -10.
_VAL_MAX = 10.
_BATCH_INPUT = 5
_BATCH_CALIB = 10000


def apply_network(network, x):
    """Apply the network layerwise on input x."""
    y = x
    for layer in network:
        y = layer(y)
    return y
    
    
def compute_scale(x):
    """Computes a scale factor for the input tensor(s) with the min-max method."""
    if isinstance(x, np.ndarray):
        x = [x]
    max_val = max([np.max(v) for v in x])
    min_val = min([np.min(v) for v in x])
    return (max_val - min_val) / (2**7)
   
   
def quantize(x, s):
        """Quantize the input given the provided scale factor into int8."""
        signed_range = 2**7
        return np.clip(np.round(x / s ), -signed_range, signed_range - 1).astype(np.int8)


class LinLayer:
    """Simple randomly initialized linear layer without activation."""

    def __init__(self, n_in, n_out):
        """Initialize internal parameters."""
        self.W = np.random.uniform(size=(n_in, n_out), low=-1., high=1.)
        self.b = np.random.uniform(size=(n_out,), low=-0.1, high=0.1)

    def __call__(self, x):
        """Compute the layer's forward pass."""
        return  np.matmul(x, self.W) + self.b


class QuantizedLinLayer:
    """Quantization layer that converts a linear layer.
    
    Explanation:
    The computations taking place are:

    (sx * sp) * yq = (sx * xq) * (sp * Wq) + (sx * sp * bq)

    where sx is the scale factor of the quantized input xq and sp is the scale
    factor of the internal quantized parameters Wq and bq.

    As we accumulate into int32 to avoid overflows, we requantize yq into int8 yqr
    before returning:
    
    yq = sr * yqr

    Hence, the complete scale factor that dequantizes the layer's output is:

    s = sx * sp * sr
    """

    def __init__(self, linear_layer, x_calib, s_calib):
        """Copies parameters of the linear layer and calibrate scale factors."""
        # quantize internal parameters
        s_params = compute_scale([linear_layer.W, linear_layer.b])
        self.Wq = quantize(linear_layer.W, s_params)
        self.bq = quantize(linear_layer.b, s_params * s_calib)

        # compute scale factor for recalibration int float and int32
        # As we mostly get large values and a large scale factor in requantization,
        # converting the scale factor to int32 does not incur large inaccuracies. But it
        # allows us to stay in int for the whole forward pass.
        s_req_precise = self._calibrate(x_calib)
        print(s_req_precise)
        self.s_req = np.round(s_req_precise).astype(np.int32)

        # compute the final output scale factor, use the precise recalibration scale factor
        self.output_scale = s_params * s_calib * s_req_precise
        
    def _apply_linear(self, xq):
        """Apply the linear parameters and accumulate into int32."""
        return np.matmul(xq.astype(np.int32), self.Wq.astype(np.int32)) + self.bq.astype(np.int32)
    
    def __call__(self, xq):
        """Compute the layer's forward pass."""
        yq = self._apply_linear(xq)
        return quantize(yq, self.s_req) # requantize
                
    def _calibrate(self, x_calib):
        y = self._apply_linear(x_calib)
        return compute_scale(y)
    

def quantize_network(network, calib_data):
    """Quantize the network layer by layer using the calibration data.
    
    Explanation:
    To quantize a layer, we need quantized calibration (example) inputs and to know
    this input's scale factor. Based on this we can compute a single output scale
    factor that dequantizes that layer's outputs. The outputs of each layer, together with
    the computed scale factor, are then used as the calibration data for the next layer.
    This process is repated layer-by-layer until we reach the end of the network.
    """
    # quantize calibration data
    s_calib = s = compute_scale(calib_data)
    x_calib = quantize(calib_data, s_calib)
    
    quantized_network = []
    for layer in network:
        # quantize layer
        quantized_layer = QuantizedLinLayer(layer, x_calib, s_calib)
        quantized_network.append(quantized_layer)

        # update calibration data
        x_calib = quantized_layer(x_calib)
        s_calib = quantized_layer.output_scale

    return quantized_network


if __name__ == "__main__":

    # define a random input and a linear network
    x = np.random.uniform(size=(_BATCH_INPUT, _NUM_FEATURES), low=_VAL_MIN, high=_VAL_MAX)
    network = [LinLayer(_NUM_FEATURES, _HIDDEN_FEATURES),
            LinLayer(_HIDDEN_FEATURES, _HIDDEN_FEATURES),
            LinLayer(_HIDDEN_FEATURES, _NUM_FEATURES)]
    print("input: ", x)

    # compute regular forward pass and target output
    y = apply_network(network, x)
    print("\ntarget output: ", y)

    # create a large calibration set for statistical robustness
    calib_data = np.random.uniform(size=(_BATCH_CALIB, _NUM_FEATURES), low=_VAL_MIN, high=_VAL_MAX)

    # quantize the linear network using the calibration data
    quantized_network = quantize_network(network, calib_data)
    final_s = quantized_network[-1].output_scale # last layer's scale factor

    # quantize the original input and compute the compute quantized network output
    xs = compute_scale(x)
    xq = quantize(x, xs)
    yq = apply_network(quantized_network, xq)
    print("\nquantized output: ", yq)

    # dequantize and compute the error to the expected float output
    y_deq = final_s * yq.astype(np.float32)
    print("\ndequantized output: ", y_deq)
    print("\nquantization errors: ", np.abs(y - y_deq))
