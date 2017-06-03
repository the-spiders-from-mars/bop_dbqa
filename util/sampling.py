import argparse

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Run sampling")
    parser.add_argument('--input', nargs='?', help='input file')
    parser.add_argument('--output', nargs='?', help='output file')
    parser.add_argument('--number', type=int, default=100, help='number of sample')

    return parser.parse_args()


def main(args):
    in_file = open(args.input, 'r')
    out_file = open(args.output, 'w')

    total_line_num = sum(1 for _ in open(args.input, 'r'))
    p = args.number / total_line_num
    count = 0
    for line in in_file:
        r = np.random.rand()
        if r < p:
            count += 1
            out_file.write(line)
    print("saved", count, "samples")

if __name__ == '__main__':
    args = parse_args()
    main(args)
