import copy


def gauss(matrix):
    tr = matrix
    for i in range(2):
        for k in range(i + 1, 3):
            d = -tr[k][i] / tr[i][i]
            for j in range(4):
                tr[k][j] = tr[k][j] + tr[i][j] * d
    i = 2
    while i != -1:
        d = 0
        for j in range(2, i, -1):
            d = tr[i][j] * tr[j][3] + d
        tr[i][3] = (tr[i][3] - d) / tr[i][i]
        i -= 1

    return [val[3] for val in tr]


def bubble_max_row(m, col):
    max_element = m[col][col]
    max_row = col
    for i in range(col + 1, len(m)):
        if abs(m[i][col]) > abs(max_element):
            max_element = m[i][col]
            max_row = i
    if max_row != col:
        m[col], m[max_row] = m[max_row], m[col]


def gauss_with_choice(m):
    n = len(m)
    for k in range(n - 1):
        bubble_max_row(m, k)
        for i in range(k + 1, n):
            div = m[i][k] / m[k][k]
            m[i][-1] -= div * m[k][-1]
            for j in range(k, n):
                m[i][j] -= div * m[k][j]

    x = [0 for _ in range(n)]
    for k in range(n - 1, -1, -1):
        x[k] = (m[k][-1] - sum([m[k][j] * x[j] for j in range(k + 1, n)])) / m[k][k]

    return x


def jacobs_monarch(matrix, xn, eps=0.5e-4):
    iters = 0
    while True:
        xnm1 = copy.deepcopy(xn)
        xn = [
            calculate_iter(0, matrix, xn[1], xn[2]),
            calculate_iter(1, matrix, xn[0], xn[2]),
            calculate_iter(2, matrix, xn[0], xn[1])
        ]
        iters += 1
        if check_cond(xn, xnm1, eps):
            return xnm1, iters


def calculate_iter(index, matrix, x1_val, x2_val):
    return matrix[index][0] * x1_val + matrix[index][1] * x2_val + matrix[index][2]


def seidel(matrix, xn, eps=0.5e-4):
    iters = 0
    while True:
        xnm1 = copy.deepcopy(xn)
        updated_x1 = calculate_iter(0, matrix, xn[1], xn[2])
        updated_x2 = calculate_iter(1, matrix, updated_x1, xn[2])
        updated_x3 = calculate_iter(2, matrix, updated_x1, updated_x2)
        xn = [updated_x1, updated_x2, updated_x3]
        iters += 1
        if check_cond(xn, xnm1, eps):
            return xnm1, iters


def check_cond(xn, xnm1, eps):
    for i in range(len(xn)):
        if abs(xn[i] - xnm1[i]) > eps:
            return False
    return True


def preproc(matrixA, matrixB):
    pos = [0] * len(matrixA)
    for j in range(len(matrixA)):
        row = copy.copy(matrixA[j])
        sum_calc = sum(map(abs, row))
        for i in range(len(row)):
            if 2 * abs(row[i]) > sum_calc:
                pos[j] = i
                break

    return [matrixA[i] for i in pos], [matrixB[i] for i in pos]


def preproc2(A, B):
    res = []
    for j in range(len(A)):
        row = copy.copy(A[j])
        tmp = []
        for i in range(len(A)):
            if i == j:
                continue
            elem = -row[i] / row[j]
            tmp.append(elem)
        tmp.append(B[j] / row[j])
        res.append(tmp)
    return res


if __name__ == "__main__":
    extended = [[-0.1, 1, 0.1, 2.2], [1.51, -0.2, 0.4, 2.31], [-0.3, -0.2, -0.55, -2.35]]
    print(gauss(extended))
    print(gauss_with_choice(extended))
    A = [[-0.1, 1, 0.1], [1.51, -0.2, 0.4], [-0.3, -0.2, -0.55]]
    B = [2.2, 2.31, -2.35]
    A1, B1 = preproc(A, B)
    matrix = preproc2(A1, B1)
    print(jacobs_monarch(matrix, [matrix[0][2], matrix[1][2], matrix[2][2]]))
    print(seidel(matrix, [matrix[0][2], matrix[1][2], matrix[2][2]]))
