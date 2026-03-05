import onnxruntime as ort


def inspect(model_path: str, n_track_features: int = 1, none_input: bool = False):
    sess_opts = ort.SessionOptions()
    sess_opts.enable_mem_reuse = False
    ort_sess = ort.InferenceSession(model_path, sess_opts=sess_opts)
    input_names = [x.name for x in ort_sess.get_inputs()]
    input_shapes = [x.shape for x in ort_sess.get_inputs()]
    output_names = [x.name for x in ort_sess.get_outputs()]
    output_shapes = [x.shape for x in ort_sess.get_outputs()]
    print("Input names: ", input_names, "with shapes: ", input_shapes)
    print("Output names: ", output_names, "with shapes: ", output_shapes)
