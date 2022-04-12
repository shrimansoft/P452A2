#!/usr/bin/python
# -*- coding: utf-8 -*-

import lib as lib

# --------------* normal chi3 fitting *----------------------------

print(lib.fitting("assign2fit.txt", "\t", 3, plot=True))

# fitting by modified chebyshev fucntion
# modified fitting code is at the end of the library

print(lib.fitting_modified("assign2fit.txt", "\t", 3, plot=True))


# ------* OUTPUT *------------
# {'Coefficients': [[0.5746586674194702], [4.72586144214381], [-11.128217777647933], [7.668677622912469]], "Pearson's r": 0.9861147039055866}
# {'Coefficients': [[1.160969479033552], [0.39351446798815237], [0.04684983209010653], [0.23964617571596986]], "Pearson's r": 0.9861147039055282}
