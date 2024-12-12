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
        'Abstention Accuracy': ACC,
        'Abstention Precision': Precision,
        'Abstention Recall': Recall,
        'Abstention F1': F1,
        'Coverage': Coverage,
        'Abstention Rate': AbstentionRate,
        'Over-conservativeness': ASAP,
        'Reliable Accuracy': RAcc
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
    ACC = (N1 + N5) / (N1 + N2 + N3 + N4 + N5)
    return ACC


def calc_Precision(N1, N2, N3, N4, N5):
    Precision = N5 / (N3 + N5)
    return Precision


def calc_Recall(N1, N2, N3, N4, N5):
    Recall = N5 / (N2 + N4 + N5)
    return Recall


def calc_F1(N1, N2, N3, N4, N5):
    Precision = calc_Precision(N1, N2, N3, N4, N5)
    Recall = calc_Recall(N1, N2, N3, N4, N5)
    F1 = 2 * Precision * Recall / (Precision + Recall)
    return F1


def calc_Coverage(N1, N2, N3, N4, N5):
    Coverage = (N1 + N2 + N4) / (N1 + N2 + N3 + N4 + N5)
    return Coverage


def calc_AbstentionRate(N1, N2, N3, N4, N5):
    AbstentionRate = (N3 + N5) / (N1 + N2 + N3 + N4 + N5)
    return AbstentionRate


def calc_ASAP(N1, N2, N3, N4, N5):
    ASAP = N3 / (N1 + N2 + N3)
    return ASAP


def calc_RAcc(N1, N2, N3, N4, N5):
    RAcc = N1 / (N1 + N2 + N4)
    return RAcc