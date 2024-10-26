import numpy as np

def printBoard(b):
    for row in b:
        print(" ".join(row))

class Solution:
    def solveSudoku(self, board):
        self.board = board
        self.solve()

    def findUnassigned(self):
        for row in range(9):
            for col in range(9):
                if self.board[row][col] == ".":
                    return row, col
        return -1, -1

    def solve(self):
        row, col = self.findUnassigned()
        if row == -1 and col == -1:
            return True
        for num in "123456789":
            if self.isSafe(row, col, num):
                self.board[row][col] = num
                if self.solve():
                    return True
                self.board[row][col] = "."
        return False

    def isSafe(self, row, col, ch):
        boxrow = row - row % 3
        boxcol = col - col % 3
        return (self.checkrow(row, ch) and
                self.checkcol(col, ch) and
                self.checksquare(boxrow, boxcol, ch))

    def checkrow(self, row, ch):
        return all(self.board[row][col] != ch for col in range(9))

    def checkcol(self, col, ch):
        return all(self.board[row][col] != ch for row in range(9))

    def checksquare(self, row, col, ch):
        return all(self.board[r][c] != ch
                   for r in range(row, row + 3)
                   for c in range(col, col + 3))

def backtracking(board):
    b = np.array(board).reshape(9, 9).tolist()
    print("\nInput board\n")
    printBoard(b)
    s = Solution()
    solved = s.solveSudoku(b)  # Make sure this returns True when solved.
    print("\nOutput board\n")
    if solved:
        printBoard(s.board)
    else:
        print("No solution exists for the given Sudoku board.")
