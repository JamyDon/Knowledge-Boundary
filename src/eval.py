def evaluate(answers, data):
    N1, N2, N3, N4, N5 = count_N(answers, data)
    ACC = calc_ACC(N1, N2, N3, N4, N5)
    Precision = calc_Precision(N1, N2, N3, N4, N5)
    Recall = calc_Recall(N1, N2, N3, N4, N5)
    F1 = calc_F1(N1, N2, N3, N4, N5)
    Coverage = calc_Coverage(N1, N2, N3, N4, N5)
    AbstentionRate = calc_AbstentionRate(N1, N2, N3, N4, N5)
    ASAP = calc_ASAP(N1, N2, N3, N4, N5)
    RAcc = calc_RAcc(N1, N2, N3, N4, N5)

    return {
        'Known Correct (N1)': N1,
        'Known Incorrect (N2)': N2,
        'Known Abstention (N3)': N3,
        'Unknown Non-Abstention (N4)': N4,
        'Unknown Abstention (N5)': N5,
        'Abstention Accuracy': round(ACC * 100, 3),
        'Abstention Precision': round(Precision * 100, 3),
        'Abstention Recall': round(Recall * 100, 3),
        'Abstention F1': round(F1 * 100, 3),
        'Coverage': round(Coverage * 100, 3),
        'Abstention Rate': round(AbstentionRate * 100, 3),
        'Over-conservativeness': round(ASAP * 100, 3),
        'Reliable Accuracy': round(RAcc * 100, 3),
    }


def count_N(answers, data):
    N1, N2, N3, N4, N5 = 0., 0., 0., 0., 0.

    for answer, datum in zip(answers, data):
        label = datum['answer']
        known = datum['knowledge boundary'] == 'Known'

        if known:
            if answer == label:
                N1 += 1
            elif answer == 4:
                N3 += 1
            else:
                N2 += 1

        else:
            if answer == 4:
                N5 += 1
            else:
                N4 += 1 # including correctly answered unknown questions

    return N1, N2, N3, N4, N5


def calc_ACC(N1, N2, N3, N4, N5):
    if N1 + N2 + N3 + N4 + N5 == 0:
        return 0
    ACC = (N1 + N5) / (N1 + N2 + N3 + N4 + N5)
    return ACC


def calc_Precision(N1, N2, N3, N4, N5):
    if N3 + N5 == 0:
        return 0
    Precision = N5 / (N3 + N5)
    return Precision


def calc_Recall(N1, N2, N3, N4, N5):
    if N2 + N4 + N5 == 0:
        return 0
    Recall = N5 / (N2 + N4 + N5)
    return Recall


def calc_F1(N1, N2, N3, N4, N5):
    Precision = calc_Precision(N1, N2, N3, N4, N5)
    Recall = calc_Recall(N1, N2, N3, N4, N5)

    if Precision + Recall == 0:
        return 0

    F1 = 2 * Precision * Recall / (Precision + Recall)
    return F1


def calc_Coverage(N1, N2, N3, N4, N5):
    if N1 + N2 + N3 + N4 + N5 == 0:
        return 0
    Coverage = (N1 + N2 + N4) / (N1 + N2 + N3 + N4 + N5)
    return Coverage


def calc_AbstentionRate(N1, N2, N3, N4, N5):
    if N1 + N2 + N3 + N4 + N5 == 0:
        return 0
    AbstentionRate = (N3 + N5) / (N1 + N2 + N3 + N4 + N5)
    return AbstentionRate


def calc_ASAP(N1, N2, N3, N4, N5):
    if N1 + N2 + N3 == 0:
        return 0
    ASAP = N3 / (N1 + N2 + N3)
    return ASAP


def calc_RAcc(N1, N2, N3, N4, N5):
    if N1 + N2 + N4 == 0:
        return 0
    RAcc = N1 / (N1 + N2 + N4)
    return RAcc