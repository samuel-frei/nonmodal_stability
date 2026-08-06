"""Shared test configuration.

`nonmodal.operator` imports pyplot at module scope and the contour helpers build
real figures, so force a headless backend before anything imports matplotlib.
"""

import matplotlib

matplotlib.use('Agg')
