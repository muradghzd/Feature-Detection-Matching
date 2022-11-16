import numpy as np

if __name__=='__main__':
    m1 = np.array([[0, 0, 0],
                   [1, 1, 1],
                   [2, 2, 2]])
    m2 = np.array([[1, 1, 1],
                  [2, 2, 2],
                  [3, 3, 3]])
    norm = np.linalg.norm(m2 - m1, axis=0)

    print(f"norm: {norm}")