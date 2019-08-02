import os
import shutil


def cutstr(string):
    index = 0
    for ind, char in enumerate(string):
        if char == "_":
            index = ind
            break
    i = 0
    for ind, char in enumerate(string):
        if char == ".":
            i = ind
            break
    first = int(string[0:index])
    end = int(string[index + 1:i])
    return first, end


def buildname(f_n, e_n):
    return str(f_n) + "_" + str(e_n) + ".npy"


if __name__ == "__main__":

    string = "123_12.npy"
    print(cutstr(string))

    f_l = []
    e_l = []
    for file in os.listdir("dataset"):
        f, e = cutstr(file)
        f_l.append(f)
        e_l.append(e)
    f_l = set(f_l)
    e_l = set(e_l)
    print(len(f_l))
    print(len(e_l))
    list_f = os.listdir("dataset")
    print(buildname(123, 12))
    print()
    str_lost = []
    for i in range(109):
        for j in range(200):
            str_name = buildname(i, j)
            if str_name not in list_f:
                print(str_name)
                str_lost.append(str_name)
    # o_f = os.path.join("dataset", "83_170.npy")
    # for new_name in str_lost:
    #     t_f = os.path.join("dataset", new_name)
    #     shutil.copyfile(o_f, t_f)
