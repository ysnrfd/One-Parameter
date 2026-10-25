"""Single float32 parameter observatory.

A forensic instrument for the life cycle of exactly one real, trainable
``torch.float32`` scalar: its initial stored bits, its participation in a real
autograd graph, the gradient autograd actually produced, the optimizer state the
installed PyTorch actually maintained, the bit pattern that was actually stored
after each update, and the difference between all of that and an independently
reconstructed account of the same update.

Nothing here simulates training.  The training path is ``loss.backward()`` and
``optimizer.step()`` and nothing else; every displayed value is read out of a
real tensor during a real run.
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
