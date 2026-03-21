class AxiomSyntaxError(Exception):
    """Raised when the eDSL syntax rules are violated (e.g., multiple arrows)."""
    pass

class AxiomShapeError(Exception):
    """Raised when axis sizes or tensor shapes do not mathematically align."""
    pass