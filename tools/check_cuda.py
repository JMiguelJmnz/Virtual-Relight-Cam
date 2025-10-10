import onnxruntime as ort

print("ONNX Runtime version:", ort.__version__)
print("Available providers:", ort.get_available_providers())

if "CUDAExecutionProvider" in ort.get_available_providers():
    print("✅ CUDA is enabled and ready.")
else:
    print("⚠️ CUDA not detected.")
