import detecshun as ds
import numpy as np

imagi = "sample.png"
queshnn = ds.build_b0rd(imagi)
queshnn[7][8] = 7
print(queshnn)




import numpy as np

def is_valid(board, row, col, num):
    # Check if the number is not in the same row, column, or 3x3 subgrid
    for i in range(9):
        if board[row][i] == num or board[i][col] == num:
            return False
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(3):
        for j in range(3):
            if board[start_row + i][start_col + j] == num:
                return False
    return True

def solve_sudoku(board):
    # Find an empty cell (cell with 0) to start solving
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                # Try every possible number from 1 to 9
                for num in range(1, 10):
                    if is_valid(board, i, j, num):
                        # If the number is valid, fill it in the cell
                        board[i][j] = num

                        # Recursively try to solve the rest of the board
                        if solve_sudoku(board):
                            return True

                        # If the current configuration is not valid, reset the cell
                        board[i][j] = 0

                # If no valid number can be placed in this cell, backtrack
                return False

    # If all cells are filled, the Sudoku is solved
    return True

if __name__ == "__main__":
    # Example input Sudoku board (0 represents an empty cell)
    

    if solve_sudoku(queshnn):
        print("Sudoku Solved:")
        print(queshnn)
    else:
        print("No solution exists for the given Sudoku.")