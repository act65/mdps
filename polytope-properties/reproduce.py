import numpy as np
import collections

# Ok. let's reproduce the figures from ...
Section3Figures = collections.namedtuple('Section3Figures', ['figa', 'figb', 'figc', 'figd'])


def main():
    section3figures = Section3Figures(
        (2, np.array([[0.01, 0.99], [0.92, 0.08], [0.08, 0.92], [0.70, 0.30]]),
         np.array([[0.06], [0.38], [-0.13], [0.64]])),
        (2, np.array([[0.96, 0.04], [0.19, 0.81], [0.43, 0.57], [0.72, 0.28]]),
        np.array([[0.88], [-0.02], [-0.98], [0.42]])),
        (3, np.array([[0.52, 0.48], [0.5, 0.5], [0.99, 0.01], [0.85, 0.15], [0.11, 0.89], [0.1, 0.9]]),
        np.array([[-0.93], [-0.49], [0.63], [0.78], [0.14], [0.41]])),
        (2, np.array([[0.7, 0.3], [0.99, 0.01], [0.2, 0.8], [0.99, 0.01]]),
        np.array([[-0.45], [-0.1], [0.5], [0.5]]))
    )


if __name__ =='__main__':
    main()
