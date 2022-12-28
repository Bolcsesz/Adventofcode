import sys
from collections import defaultdict
from string import ascii_lowercase, ascii_letters
from heapq import heappop, heappush

import requests
import re
import numpy as np


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
    print(sum([((ord(x[0]) % 65 + 1) % 3 + 1 if ord(x[1]) % 88 == 2 else ord(x[0]) % 64 if ord(x[1]) % 88 == 1 else (ord(x[0]) % 65 - 1) % 3 + 1) + ((ord(x[1]) % 88) * 3) for x in m_lst]))

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
    if Tarr[nthKnot + 1][1] - Tarr[nthKnot][1] > 1:
        Tarr[nthKnot + 1][1] -= 1
        if Tarr[nthKnot + 1][0] - Tarr[nthKnot][0] > 1:
            Tarr[nthKnot + 1][0] -= 1
        elif Tarr[nthKnot][0] - Tarr[nthKnot + 1][0] > 1:
            Tarr[nthKnot + 1][0] += 1
        else:
            Tarr[nthKnot + 1][0] = Tarr[nthKnot][0]
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
    print(len(set(Tpath)))

def evaluate_cust_1(num1: int, num2: int, sign: str, chill_factor: int) -> int:
    res: int
    if sign == '+':
        res = num1 + num2
    elif sign == '-':
        res = num1 - num2
    elif sign == '*':
        res = num1 * num2
    return res // chill_factor

def evaluate_cust_2(num1: int, num2: int, sign: str, test: int) -> int:
    res: int
    if sign == '+':
        res = num1 + num2
    elif sign == '-':
        res = num1 - num2
    elif sign == '*':
        res = num1 * num2
    return res - ((res // test) * test)

def do_monkey(monkey, starting_items, operation, test, test_true, test_false, chill_factor, divider, prt1_2_indicator):
    for i in starting_items[monkey[0]]:
        worry_level = evaluate_cust_1(i, i if operation[monkey[0]][0].split()[1] == 'old' else int( operation[monkey[0]][0].split()[1]), operation[monkey[0]][0].split()[0], chill_factor) if prt1_2_indicator == 1 else evaluate_cust_2(i, i if operation[monkey[0]][0].split()[1] == 'old' else int( operation[monkey[0]][0].split()[1]), operation[monkey[0]][0].split()[0], divider)
        monkey[1] += 1
        if worry_level % int(test[monkey[0]][0]) == 0:
            starting_items[test_true[monkey[0]][0]].append(worry_level)
        else:
            starting_items[test_false[monkey[0]][0]].append(worry_level)
    starting_items[monkey[0]] = []

def solve_day_11(input: str, prt1_2_indicator):
    datTab = [x for x in input.strip().split('\n\n')]
    chill_factor = 3 if prt1_2_indicator == 1 else 1
    rounds = 20 if prt1_2_indicator == 1 else 10000
    divider = 1
    monkey_list = []
    starting_items = {}
    operation = {}
    test = {}
    test_true = {}
    test_false = {}
    for monkey in datTab:
        monkey_number = re.search(r'^Monkey.*(\d+)', monkey).group(1)
        monkey_list.append([monkey_number, 0])
        for dat in monkey.split('\n'):
            if re.search(r'Starting items', dat) is not None:
                starting_items[monkey_number[0]] = list(map(int, re.findall(r'(\d+)', dat)))
            if re.search(r'Operation', dat) is not None:
                operation[monkey_number[0]] = re.findall(r'[*\/\-+]\W*(?:\d+|old)?', dat)
            if re.search(r'Test:', dat) is not None:
                test[monkey_number[0]] = re.findall(r'(\d+)', dat)
                divider *= int(test[monkey_number[0]][0])
            if re.search(r'If true:', dat) is not None:
                test_true[monkey_number[0]] = re.findall(r'(\d+)', dat)
            if re.search(r'If false:', dat) is not None:
                test_false[monkey_number[0]] = re.findall(r'(\d+)', dat)
    for i in range(rounds):
        for j in range(len(monkey_list)):
            do_monkey(monkey_list[j], starting_items, operation, test, test_true, test_false, chill_factor, divider, prt1_2_indicator)
    print(sorted(monkey_list, key=lambda x: x[1], reverse=True)[0][1] *
          sorted(monkey_list, key=lambda x: x[1], reverse=True)[1][1])

def get_start_end(matrix: [[]]):
    S = []
    E = []
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == 'S':
                S = [i,j]
            elif matrix[i][j] == 'E':
                E = [i,j]
    return S, E

def height(pos: str) -> int:
    if pos in ascii_lowercase:
        return ascii_lowercase.index(pos)
    elif pos == 'S':
        return 0
    elif pos == 'E':
        return 25

def next_step(i, j, n, m, matrix, direction):
    for x, y in [[1, 0], [-1, 0], [0, 1], [0, -1]]:
        next_i = i + x
        next_j = j + y
        if 0 <= next_i < n and 0 <= next_j < m:
            if direction == 'up':
                if height(matrix[next_i][next_j]) <= height(matrix[i][j]) + 1:
                    yield next_i, next_j
            else:
                if height(matrix[next_i][next_j]) >= height(matrix[i][j]) - 1:
                    yield next_i, next_j

def solve_day_12(input: str, part: int):
    print(input)
    src = [[x for x in y] for y in input.strip().split('\n')]
    visited = [[False for x in y] for y in input.strip().split('\n')]
    S, E = get_start_end(src)
    if part == 1:
        path = [(0, S[0], S[1])]
    elif part == 2:
        path = [(0, E[0], E[1])]

    while True:
        steps, i, j = path.pop()
        if not visited[i][j]:
            visited[i][j] = True
            if part == 1:
                if [i, j] == E:
                    print(steps)
                    break
            elif part == 2:
                if [i, j] == S or src[i][j] == 'a':
                    print(steps)
                    break
            for x, y in next_step(i, j, len(src), len(src[0]), src, 'up' if part == 1 else 'down'):
                path.insert(0, (steps + 1, x, y))

def compare(x, y) -> int:
    if isinstance(x, int) and isinstance(y, int):
        if x < y:
            return 1
        elif x == y:
            return 0
        else:
            return -1

    if isinstance(x, list) and isinstance(y, int):
        y = [y]

    if isinstance(y, list) and isinstance(x, int):
        x = [x]

    i = 0
    while i < len(x) and i < len(y):
        res = compare(x[i], y[i])
        if res == 1:
            return 1
        if res == -1:
            return -1
        i += 1

    if i == len(x):
        if i == len(y):
            return 0
        return 1
    elif i == len(y):
        return -1
    else:
        return 0

def solve_day_13(input: str, part: int):
    res = 0
    c = [[[2]],[[6]]]
    r = [1,2]
    if part == 1:
        pairs = input.strip().split('\n\n')
        for i, _ in enumerate(pairs):
            p0, p1 = map(eval, _.strip().split('\n'))
            if compare(p0, p1) == 1:
                res += i+1
    elif part == 2:
        for i, p0 in enumerate(map(eval, list(filter(None, input.split('\n'))))):
            if compare(p0, c[0]) == 1:
                r[0] += 1
                r[1] += 1
            elif compare(p0, c[1]) == 1:
                r[1] += 1
        res = r[0]*r[1]

    print(res)

def create_map(rock_lines: list) -> set:
    res = set()
    for rock_line in rock_lines:
        rock_end_coords = [list(map(int, x.split(','))) for x in rock_line.split(' -> ')]
        for i in range(len(rock_end_coords)-1):
            frm_x, frm_y = rock_end_coords[i]
            to_x, to_y = rock_end_coords[i+1]

            if frm_x == to_x:
                for y in range(min(frm_y, to_y), max(frm_y, to_y) + 1):
                    res.add((frm_x, y))
            if frm_y == to_y:
                for x in range(min(frm_x, to_x), max(frm_x, to_x) + 1):
                    res.add((x, frm_y))
    return res

def drop_sand(waterfall_map: set, max_height:int, drop_point: tuple, part: int) -> bool:
    x = drop_point[0]
    y = drop_point[1]
    while y < max_height:
        # print(x, y)
        if part == 2 and y + 1 == max_height:
            waterfall_map.add((x, y))
            return True
        if (x, y + 1) not in waterfall_map:
            y += 1
        elif (x - 1, y + 1) not in waterfall_map:
            x -= 1
            y += 1
        elif (x + 1, y + 1) not in waterfall_map:
            x += 1
            y += 1
        elif drop_point in waterfall_map:
            return False
        else:
            waterfall_map.add((x ,y))
            # print('Stopped!')
            return True
    return False

def solve_day_14(input: str, part: int):
    rock_lines = input.strip().split('\n')
    taken_places = create_map(rock_lines)
    max_height = max([x[1] for x in taken_places]) + (2 if part == 2 else 0)
    drop_point = (500, 0)
    sand_count = 0
    has_stopped = True
    while has_stopped:
        has_stopped = drop_sand(taken_places, max_height, drop_point, part)
        if has_stopped:
            sand_count += 1
    print(sand_count)



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
    # solve_day_9(get_input('9').text, 2)
    # solve_day_9(get_input('9').text, 10)
    # solve_day_10(get_input('10').text)
    # solve_day_11(get_input('11').text, 1)
    # solve_day_11(get_input('11').text, 2)
    # solve_day_12(get_input('12').text, 1)
    # solve_day_12(get_input('12').text, 1)
    # solve_day_12(get_input('12').text, 2)
    # solve_day_13(get_input('13').text, 1)
    # solve_day_13(get_input('13').text, 2)
    # solve_day_14(get_input('14').text, 1)
    solve_day_14(get_input('14').text, 2)
    # solve_day_14('498,4 -> 498,6 -> 496,6\n503,4 -> 502,4 -> 502,9 -> 494,9', 2)
    # solve_day_13('[1,1,3,1,1]\n[1,1,5,1,1]\n\n[[1],[2,3,4]]\n[[1],4]\n\n[9]\n[[8,7,6]]\n\n[[4,4],4,4]\n[[4,4],4,4,4]\n\n[7,7,7,7]\n[7,7,7]\n\n[]\n[3]\n\n[[[]]]\n[[]]\n\n[1,[2,[3,[4,[5,6,7]]]],8,9]\n[1,[2,[3,[4,[5,6,0]]]],8,9]',2)
