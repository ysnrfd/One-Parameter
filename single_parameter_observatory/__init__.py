"""Parameter Studio: visual PyTorch/Transformers workflows and scalar forensics.

Typed workflows compose actual CPU models, tensor data, optimizers, training,
inference, observed outputs, and safe model checkpoints. Custom extensions use
isolated JavaScript or an interpreted tensor-forward AST. Runnable Python
project exports provide an explicit entry point outside those worker interfaces.

The original scalar observatory separately traces one real ``torch.float32``
parameter: stored bits, autograd observations, optimizer state, updates, and an
independent reconstruction. Those detailed forensic records describe the scalar
engine; multi-parameter model observations retain their own scope and provenance.
Both training paths use actual PyTorch computation rather than simulated losses.
"""

__version__ = "2.2.0"

__all__ = ["__version__"]
