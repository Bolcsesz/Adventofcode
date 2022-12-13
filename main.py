from collections import defaultdict
from string import ascii_letters

import requests
import re
import numpy as np
from numpy import ndarray


def get_input(day: str) -> requests.Response:
    url = f'https://adventofcode.com/2022/day/{day}/input'
    SESSIONID = '53616c7465645f5f9ae41403292bc0b12009b6c672b4b662c0bc22f5e4251cdf098c796a5dcd207f6cd0174d4f9df44f3351bad471c28fdb23c748843e412718'
    USER_AGENT = ''
    return requests.get(url, cookies={'session': SESSIONID}, headers={'User-Agent': USER_AGENT})


def solve_day_1(input: str):
    m_lst = sorted([sum([int(y) for y in z.split('\n') if y != '']) for z in input.split('\n\n')])
    print(m_lst[-1::][0])
    print(sum(m_lst[-3::]))


def solve_day_2(input: str):
    m_lst = [[y for y in z.split(' ')] for z in input.split('\n') if z != '']
    for x in m_lst:
        print(x, (ord(x[1]) % 88) * 3,
              (ord(x[0]) % 65 + 1) % 3 + 1 if ord(x[1]) % 88 == 2 else ord(x[0]) % 64 if ord(x[1]) % 88 == 1 else (ord(
                  x[0]) % 65 - 1) % 3 + 1)
    print(sum([(6 if (ord(x[1]) % 87 - 1) == ((ord(x[0]) % 64) % 3) else 3 if (ord(x[1]) % 87) == (
            ord(x[0]) % 64) else 0) + ord(x[1]) % 87 for x in m_lst]))
    print(sum([((ord(x[0]) % 65 + 1) % 3 + 1 if ord(x[1]) % 88 == 2 else ord(x[0]) % 64 if ord(x[1]) % 88 == 1 else (
                                                                                                                            ord(
                                                                                                                                x[
                                                                                                                                    0]) % 65 - 1) % 3 + 1) + (
                       (ord(x[1]) % 88) * 3) for x in m_lst]))


def solve_day_3(input: str):
    sum_ = 0
    src = input.split('\n')
    for i in src:
        sum_ += sum([ascii_letters.find(x[0]) + 1 for x in set(i[:int(len(i) / 2)]) & set(i[int(len(i) / 2):])])
    print(sum_)
    sum_ = 0
    for i in range(int(len(input.split('\n')) / 3)):
        sum_ += sum([ascii_letters.find(x[0]) + 1 for x in set(src[i * 3]) & set(src[i * 3 + 1]) & set(src[i * 3 + 2])])
    print(sum_)


def solve_day_4(input: str):
    m_lst = [[[int(x) for x in y.split('-')] for y in z.split(',')] for z in input.split('\n') if z != '']
    sum_ = 0
    sum2_ = 0
    for i in range(len(m_lst)):
        if (m_lst[i][0][0] <= m_lst[i][1][0] and m_lst[i][0][1] >= m_lst[i][1][1]) or (
                m_lst[i][0][0] >= m_lst[i][1][0] and m_lst[i][0][1] <= m_lst[i][1][1]):
            sum_ += 1
        if not (m_lst[i][0][0] > m_lst[i][1][1] or m_lst[i][1][0] > m_lst[i][0][1]):
            sum2_ += 1
    print(sum2_)


def init_stacks(num: int, input: str) -> list:
    char_len = 4
    stack_list = []
    for i in range(num):
        stack_list.append([])
    for i in range(int(len(input[-1]) / 4), -1, -1):
        for j in range(int(len(input[i]) / 4) + 1):
            if len(input[i][j * char_len:j * char_len + char_len].strip()) > 0:
                stack_list[j].append(input[i][j * char_len + 1:j * char_len + char_len - 2])
    return stack_list


def do_cmd(cnt: int, frm: int, to: int, stacks: list):
    for i in range(cnt):
        stacks[to - 1].append(stacks[frm - 1].pop())


def do_cmd2(cnt: int, frm: int, to: int, stacks: list):
    tmp_stack = []
    for i in range(cnt):
        tmp_stack.append(stacks[frm - 1].pop())
    for i in range(cnt):
        stacks[to - 1].append(tmp_stack.pop())


def solve_day_5(input: str, cmd):
    li = input.split('\n\n')[0].split('\n')
    print(input.split('\n\n')[0])
    cmds = input.split('\n\n')[1].split('\n')[:-1]
    stacks = init_stacks(int(len(li[-1]) / 4) + 1, li)
    print(stacks)
    for i in range(len(cmds)):
        res = re.search(r"move (\d+) from (\d+) to (\d+)", cmds[i])
        # do_cmd(int(res.group(1)),int(res.group(2)),int(res.group(3)),stacks)
        cmd(int(res.group(1)), int(res.group(2)), int(res.group(3)), stacks)
    print(''.join([stacks[i][-1] for i in range(len(stacks))]))


def solve_day_6(input: str, key_len: int):
    res = 0
    for i in range(len(input) - 4):
        if len(set(input[i:i + key_len])) == key_len:
            res = i + key_len
            break
    print(res)


def solve_day_7(input: str):
    print(input)
    path = []
    sum_ = defaultdict(int)
    for i in input.strip().split('\n'):
        line = i.split()
        if line[0] == '$':
            if line[1] == 'cd':
                if line[2] == '..':
                    path.pop()
                else:
                    path.append(line[2])
            elif line[1] == 'ls':
                continue
        elif line[0] == 'dir':
            continue
        else:
            s = int(line[0])
            for i in range(len(path) + 1):
                sum_['/'.join(path[:i])] += s
    res1 = 0
    res2 = sum_.get('/')
    totalFreeSpace = 70000000 - sum_.get('/')
    desiredFreeSpace = 30000000
    print('totalFreeSpace', totalFreeSpace)
    for k, v in sum_.items():
        if v <= 100000:
            res1 += v
        if desiredFreeSpace < totalFreeSpace + v:
            if v < res2:
                res2 = v
    print('Part 1: ', res1)
    print('Part 2: ', res2)


def isVisible(forest: [], X: int, Y: int) -> bool:
    visibleN = True
    visibleS = True
    visibleE = True
    visibleW = True
    for i in range(X):
        if forest[X - i - 1][Y] >= forest[X][Y]:
            visibleN = False
            break
    for i in range(len(forest) - X - 1):
        if forest[X + i + 1][Y] >= forest[X][Y]:
            visibleS = False
            break
    for i in range(Y):
        if forest[X][Y - i - 1] >= forest[X][Y]:
            visibleW = False
            break
    for i in range(len(forest[Y]) - Y - 1):
        if forest[X][Y + i + 1] >= forest[X][Y]:
            visibleE = False
            break
    return visibleN or visibleS or visibleE or visibleW


def visDist(forest: [], X: int, Y: int) -> int:
    visdistN: int = 0
    visdistS: int = 0
    visdistW: int = 0
    visdistE: int = 0
    for i in range(X):
        visdistN += 1
        if forest[X - i - 1][Y] >= forest[X][Y]:
            break
    for i in range(len(forest) - X - 1):
        visdistS += 1
        if forest[X + i + 1][Y] >= forest[X][Y]:
            break
    for i in range(Y):
        visdistW += 1
        if forest[X][Y - i - 1] >= forest[X][Y]:
            break
    for i in range(len(forest[Y]) - Y - 1):
        visdistE += 1
        if forest[X][Y + i + 1] >= forest[X][Y]:
            break
    return visdistN * visdistS * visdistW * visdistE


def solve_day_8(input: str):
    res = 0
    maxVisDist = 0
    forest = [list(sub) for sub in input.strip().split('\n')]
    for x in range(len(forest)):
        for y in range(len(forest[x])):
            print(forest[x][y], isVisible(forest, x, y), ' ', sep='', end='')
            res += 1 if isVisible(forest, x, y) else 0
            if visDist(forest, x, y) > maxVisDist:
                maxVisDist = visDist(forest, x, y)
        print()
    print(res)
    print(maxVisDist)


def drawCRT(tick: int, x: int):
    if x <= tick % 40 <= x + 2:
        print('#', end='')
    else:
        print('.', end='')
    if tick % 40 == 0:
        print()


def solve_day_10(input: str):
    tick = 0
    x = 1
    res1 = 0
    for i in [x.split() for x in input.strip().split('\n')]:
        if i[0] == 'noop':
            tick += 1
            drawCRT(tick, x)
            if tick % 40 == 20:
                res1 += tick * x
        if i[0] == 'addx':
            tick += 1
            drawCRT(tick, x)
            if tick % 40 == 20:
                res1 += tick * x
            tick += 1
            drawCRT(tick, x)
            if tick % 40 == 20:
                res1 += tick * x
            x += int(i[1])
    print(res1)


def moveH2T(Tarr: [], nthKnot: int, moveVect: np.ndarray, path: []):
    if nthKnot == 0:
        Tarr[nthKnot] += moveVect
    if Tarr[nthKnot][0] - Tarr[nthKnot + 1][0] > 1:
        Tarr[nthKnot + 1][0] += 1
        if Tarr[nthKnot + 1][1] - Tarr[nthKnot][1] > 1:
            Tarr[nthKnot + 1][1] -= 1
        elif Tarr[nthKnot][1] - Tarr[nthKnot + 1][1] > 1:
            Tarr[nthKnot + 1][1] += 1
        else:
            Tarr[nthKnot + 1][1] = Tarr[nthKnot][1]
    if Tarr[nthKnot][1] - Tarr[nthKnot + 1][1] > 1:
        Tarr[nthKnot + 1][1] += 1
        if Tarr[nthKnot + 1][0] - Tarr[nthKnot][0] > 1:
            Tarr[nthKnot + 1][0] -= 1
        elif Tarr[nthKnot][0] - Tarr[nthKnot + 1][0] > 1:
            Tarr[nthKnot + 1][0] += 1
        else:
            Tarr[nthKnot + 1][0] = Tarr[nthKnot][0]
    if Tarr[nthKnot + 1][0] - Tarr[nthKnot][0] > 1:
        Tarr[nthKnot + 1][0] -= 1
        if Tarr[nthKnot + 1][1] - Tarr[nthKnot][1] > 1:
            Tarr[nthKnot + 1][1] -= 1
        elif Tarr[nthKnot][1] - Tarr[nthKnot + 1][1] > 1:
            Tarr[nthKnot + 1][1] += 1
        else:
            Tarr[nthKnot + 1][1] = Tarr[nthKnot][1]
    if Tarr[nthKnot + 1][1] - Tarr[nthKnot][1] > 1:         #FEJ balra megy
        Tarr[nthKnot + 1][1] -= 1
        if Tarr[nthKnot + 1][0] - Tarr[nthKnot][0] > 1:
            Tarr[nthKnot + 1][0] -= 1
        elif Tarr[nthKnot][0] - Tarr[nthKnot + 1][0] > 1:
            Tarr[nthKnot + 1][0] += 1
        else:
            Tarr[nthKnot + 1][0] = Tarr[nthKnot][0]
    #print(nthKnot,'. csomo', Tarr[nthKnot])
    if nthKnot + 2 < len(Tarr):
        moveH2T(Tarr, nthKnot + 1, None, path)
    else:
        path.append((Tarr[nthKnot + 1][0], Tarr[nthKnot + 1][1]))


def solve_day_9(input: str, numKnots: int):
    tmp = []
    for i in range(numKnots):
        tmp.append([0, 0])
    Tarr = np.array(tmp)
    Tpath = []
    c = 0
    for i in [[y for y in x.split()] for x in input.strip().split('\n')]:
        if i[0] == 'U':
            for j in range(int(i[1])):
                moveH2T(Tarr, 0, [1, 0], Tpath)
        elif i[0] == 'R':
            for j in range(int(i[1])):
                moveH2T(Tarr, 0, [0, 1], Tpath)
        elif i[0] == 'D':
            for j in range(int(i[1])):
                moveH2T(Tarr, 0, [-1, 0], Tpath)
        elif i[0] == 'L':
            for j in range(int(i[1])):
                moveH2T(Tarr, 0, [0, -1], Tpath)
    # print(Tpath)
    print(len(set(Tpath)))


def solve_day_11(input: str):
    datTab = [x for x in input.strip().split('\n\n')]
    print(datTab)
    starting_items = {}
    operation = {}
    test = {}
    test_true = {}
    test_false = {}
    for monkey in datTab:
        monkey_number = re.search(r'^Monkey.*(\d+)', monkey).group(1)
        for dat in monkey.split('\n'):
            print(dat)
            if re.search(r'Starting items', dat) != None:
                print(re.findall(r'(\d+)', dat))
                starting_items[monkey_number] = re.findall(r'(\d+)', dat)
            if re.search(r'Operation', dat) != None:
                print(re.findall(r'[*\/\-+]\W*\d+', dat))
                operation[monkey_number] = re.findall(r'[*\/\-+]\W*\d+', dat)
            if re.search(r'Test:', dat) != None:
                print(re.findall(r'(\d+)', dat))
                test[monkey_number] = re.findall(r'(\d+)', dat)
            if re.search(r'If true:', dat) != None:
                print(re.findall(r'(\d+)', dat))
                test_true[monkey_number] = re.findall(r'(\d+)', dat)
            if re.search(r'If false:', dat) != None:
                print(re.findall(r'(\d+)', dat))
                test_false[monkey_number] = re.findall(r'(\d+)', dat)
        print(starting_items)
        print(operation)
        print(test)
        print(test_true)
        print(test_false)

if __name__ == '__main__':
    # r = get_input('5')
    # solve_day_1(get_input('1').text)
    # solve_day_2(get_input('2').text)
    # solve_day_3(get_input('3').text)
    # solve_day_4(get_input('4').text)
    # solve_day_5(get_input('5').text,do_cmd)
    # solve_day_5(get_input('5').text,do_cmd2)
    # solve_day_6(get_input('6').text, 4)
    # solve_day_6(get_input('6').text, 14)
    # solve_day_7(get_input('7').text)
    # solve_day_8(get_input('8').text)
    # solve_day_9(get_input('9').text, 2) # task 1 # 5683 jó
    # solve_day_9(get_input('9').text, 10) # task 2 # 2396 rossz | 2571 too high | 2378 nem jó
    # solve_day_10(get_input('10').text)
    # solve_day_11(get_input('11').text)
    solve_day_11('Monkey 0:\n  Starting items: 79, 98\n  Operation: new = old * 19\n  Test: divisible by 23\n    If true: throw to monkey 2\n    If false: throw to monkey 3\n\nMonkey 1:\n  Starting items: 54, 65, 75, 74\n  Operation: new = old + 6\n  Test: divisible by 19\n    If true: throw to monkey 2\n    If false: throw to monkey 0\n\nMonkey 2:\n  Starting items: 79, 60, 97\n  Operation: new = old * old\n  Test: divisible by 13\n    If true: throw to monkey 1\n    If false: throw to monkey 3\n\nMonkey 3:\n  Starting items: 74\n  Operation: new = old + 3\n  Test: divisible by 17\n    If true: throw to monkey 0\n    If false: throw to monkey 1')
