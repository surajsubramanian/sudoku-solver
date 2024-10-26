import argparse
from sudoku_solver import sudoku_solver
from digit_recognizer import digit_recognizer
from backtracking import backtracking

def main(img_path, interactive_mode=False):
    sudoku_solver(img_path)
    board = digit_recognizer(interactive_mode)
    backtracking(board)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sudoku Solver')
    parser.add_argument('-i', '--input', type=str, help='Input image path')
    parser.add_argument('-m', '--interactive-mode', action='store_true', help='Interactive mode for digit recognition')

    args = parser.parse_args()
    main(args.input, args.interactive_mode)
