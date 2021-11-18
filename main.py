from copy import deepcopy
from typing import Tuple, List
from sys import maxsize as MAX_INT
import time
from random import random
from random import randint
from math import log10, log2
from math import inf as infinity

BOARD_SIZE = 4
MAX_VALUE = 2048

MONOTONIA3 = [[5, 4, 3],
             [4, 3, 2],
             [3, 2, 1]]

MONOTONIA4 = [[7, 6, 5, 4],
             [6, 5, 4, 3],
             [5, 4, 3, 2],
             [4, 3, 2, 1]]

MONOTONIA5 = [[9, 8, 7, 6, 5],
              [8, 7, 6, 5, 4],
              [7, 6, 5, 4, 3],
              [6, 5, 4, 3, 2],
              [5, 4, 3, 2, 1]]

DEPTH = 5

HEURISTICS = 1

if BOARD_SIZE == 3:
    MONOTONIA = deepcopy(MONOTONIA3)
if BOARD_SIZE == 4:
    MONOTONIA = deepcopy(MONOTONIA4)
if BOARD_SIZE == 5:
    MONOTONIA = deepcopy(MONOTONIA5)

class Grid:

    def __init__(self, matrix):
        self.set_matrix(matrix)

    def __eq__(self, other) -> bool:
        for i in range(0, len(self.matrix), 1):
            for j in range(0, len(self.matrix[0]), 1):
                if (self.matrix[i][j] != other.matrix[i][j]):
                    return False

        return True

    def set_matrix(self, matrix):
        self.matrix = deepcopy(matrix)

    def get_matrix(self) -> List[List]:
        return deepcopy(self.matrix)

    def place_tile(self, row: int, col: int, tile: int):
        self.matrix[row][col] = tile
        
    def print_matrix(self):
        for i in self.matrix:
            print(i)
            
    #=========================================================
    #====================UTILITY FUNCTIONS====================
    #=========================================================
    def sum_of_empty_fields(self, matrix) -> int:
        empty_tiles = 0
        
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE): 
                # Sum of empty tiles
                if(matrix[i][j] == 0):
                    empty_tiles += 1
        
        return empty_tiles 
    
    def calc_tile_score(self, value):
        if value == 0:
            return 0
        else:
            return (int(log2(value))-1) * 2**int(log2(value))
    
    def score_of_all_tiles(self, matrix) -> int:
        sum = 0    
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):  
                sum += self.calc_tile_score(matrix[i][j]) 
                
        return sum
    
    def monotonia(self, matrix) -> int:
        mon = 0
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                # Monotonia
                mon += matrix[i][j] * MONOTONIA[i][j]
                
        return mon
        
    def similarity_of_neighbours(self, matrix) -> float:
        simil = 0.0
        counter = 1
        temp_simil = 0.0
        
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if matrix[i][j] != 0:
                    counter = 1
                    temp_simil = 0
                    if (j-1 >= 0) and (i-1 >= 0) and (matrix[i-1][j-1] != 0):
                        number = matrix[i][j] - matrix[i-1][j-1]
                        temp_simil += abs(number)
                        counter += 1
                    if (i-1 >= 0) and (matrix[i-1][j] != 0):
                        number = matrix[i][j] - matrix[i-1][j]
                        temp_simil += abs(number)
                        counter += 1
                    if (j+1 < BOARD_SIZE) and (i-1 >= 0) and (matrix[i-1][j+1] != 0):
                        number = matrix[i][j] - matrix[i-1][j+1]
                        temp_simil += abs(number)
                        counter += 1
                    if (j-1 >= 0) and (matrix[i][j-1] != 0):
                        number = matrix[i][j] - matrix[i][j-1]
                        temp_simil += abs(number)
                        counter += 1
                    if (j+1 < BOARD_SIZE) and (matrix[i][j+1] != 0):
                        number = matrix[i][j] - matrix[i][j+1]
                        temp_simil += abs(number)
                        counter += 1
                    if (j-1 >= 0) and (i+1 < BOARD_SIZE) and (matrix[i+1][j-1] != 0):
                        number = matrix[i][j] - matrix[i+1][j-1]
                        temp_simil += abs(number)
                        counter += 1
                    if (i+1 < BOARD_SIZE) and (matrix[i+1][j] != 0):
                        number = matrix[i][j] - matrix[i+1][j]
                        temp_simil += abs(number)
                        counter += 1
                    if (j+1 < BOARD_SIZE) and (i+1 < BOARD_SIZE) and (matrix[i+1][j+1] != 0):
                        number = matrix[i][j] - matrix[i+1][j+1]
                        temp_simil += abs(number)
                        counter += 1
                    
                    simil += (temp_simil / counter)
        
        return simil
    
    def max_tile(self, matrix) -> int:
        return max(map(max, matrix))
        
    # H1 = Score+Monotonia−Similarity+[𝑙𝑜𝑔(Score)⋅number_of_empty_cells]
    def heuristics_1(self, matrix) -> float:
        mon = self.monotonia(matrix)
        sum_of_scores = self.score_of_all_tiles(matrix)
        sum_of_empty = self.sum_of_empty_fields(matrix)
        similarity = self.similarity_of_neighbours(matrix)
        
        if sum_of_scores == 0:
            utility = sum_of_scores + mon - similarity
        else:
            utility = sum_of_scores + mon - similarity + (log10(sum_of_scores) * sum_of_empty)
            
        return utility
    
    # H2 = Monotonia−Silimarity+[𝑙𝑜𝑔(Score)⋅ number_of_empty_cells]
    def heuristics_2(self, matrix) -> float:
        mon = self.monotonia(matrix)
        sum_of_scores = self.score_of_all_tiles(matrix)
        sum_of_empty = self.sum_of_empty_fields(matrix)
        similarity = self.similarity_of_neighbours(matrix)
        
        if sum_of_scores == 0:
            utility = mon - similarity
        else:
            utility = mon - similarity + (log10(sum_of_scores) * sum_of_empty)
        
        return utility
    
    # H3 Score−Similarity+[𝑙𝑜𝑔(Score)⋅number_of_empty_cells]
    def heuristics_3(self, matrix) -> float:
        mon = self.monotonia(matrix)
        sum_of_scores = self.score_of_all_tiles(matrix)
        sum_of_empty = self.sum_of_empty_fields(matrix)
        similarity = self.similarity_of_neighbours(matrix)

        if sum_of_scores == 0:
            utility = sum_of_scores - similarity
        else:
            utility = sum_of_scores - similarity + log10(sum_of_scores) * sum_of_empty
        
        return utility
    
    # H4 Score+[number_of_empty_cells⋅𝑙𝑜𝑔2(max)]+Monotonia 
    def heuristics_4(self, matrix) -> float:
        mon = self.monotonia(matrix)
        sum_of_scores = self.score_of_all_tiles(matrix)
        sum_of_empty = self.sum_of_empty_fields(matrix)
        similarity = self.similarity_of_neighbours(matrix)
        max_tile = self.max_tile(matrix)
        
        utility = sum_of_scores + (number_of_empty * log2(max)) + mon
        
        return utility
    
    def utility(self, matrix) -> float:
        if HEURISTICS == 1:
            return self.heuristics_1(matrix)
        if HEURISTICS == 2:
            return self.heuristics_2(matrix)
        if HEURISTICS == 3:
            return self.heuristics_3(matrix)
        if HEURISTICS == 4:
            return self.heuristics_4(matrix)
        else:
            return self.heuristics_1(matrix)

        
    #==========================================================    
    # ===============MANIPULATE MATRIX FUNCTIONS===============
    #==========================================================  
    # Function to move cells when 0's are present
    def compress_cells(self, matrix) -> List[List]:
        temp_matrix = []

        changed = False

        for i in range(BOARD_SIZE):
            temp_matrix.append([0] * BOARD_SIZE)

        for i in range(BOARD_SIZE):
            pos = 0

            for j in range(BOARD_SIZE):
                if (matrix[i][j] != 0):
                    temp_matrix[i][pos] = matrix[i][j]

                    if(j != pos):
                        changed = True

                    pos += 1

        return temp_matrix, changed

    # Function to merge cells if possible
    def merge_cells(self, matrix) -> List[List]:

        changed = False

        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE-1):

                if(matrix[i][j] == matrix[i][j + 1] and matrix[i][j] != 0):
                    matrix[i][j] = matrix[i][j] * 2
                    matrix[i][j+1] = 0
                    changed = True

        return matrix, changed
    
    # reverse all lists eg [1,2] -> [2,1]
    def reverse(self, matrix) -> List[List]:
        temp_matrix = []
        for i in range(BOARD_SIZE):
            temp_matrix.append([])
            for j in range(BOARD_SIZE):
                temp_matrix[i].append(matrix[i][BOARD_SIZE - 1 - j])
        
        return temp_matrix

    # matrix transposition eg 
    # [1, 2] => [1, 3]
    # [3, 4] => [2, 4]
    def transpose(self, matrix):
        temp_matrix = []
        for i in range(BOARD_SIZE):
            temp_matrix.append([])
            for j in range(BOARD_SIZE):
                temp_matrix[i].append(matrix[j][i])
        
        return temp_matrix
    
    # ============================================
    # ===============MOVE FUNCTIONS=============== 
    # ============================================
    def can_move_left(self, matrix) -> bool:
        new_matrix, changed1 = self.compress_cells(matrix)
        new_matrix, changed2 = self.merge_cells(new_matrix)
        if (changed1 == True) or (changed2 == True):
            return True
        
        return False
    
    def can_move_right(self, matrix) -> bool:
        new_matrix = self.reverse(matrix)
        new_matrix, changed1 = self.compress_cells(new_matrix)
        new_matrix, changed2 = self.merge_cells(new_matrix)
        if (changed1 == True) or (changed2 == True):
            return True
        
        return False
    
    def can_move_up(self, matrix) -> bool:

        new_matrix = self.transpose(matrix)
        new_matrix, changed1 = self.compress_cells(new_matrix)
        new_matrix, changed2 = self.merge_cells(new_matrix)
        if (changed1 == True) or (changed2 == True):
            return True
        
        return False

    def can_move_down(self, matrix) -> bool:

        new_matrix = self.transpose(matrix)
        new_matrix = self.reverse(new_matrix)
        new_matrix, changed1 = self.compress_cells(new_matrix)
        new_matrix, changed2 = self.merge_cells(new_matrix)
        if (changed1 == True) or (changed2 == True):
            return True
        
        return False

    def prepare_move_left(self, matrix) -> List[List]:
        new_matrix, changed = self.compress_cells(matrix)
        new_matrix, changed = self.merge_cells(new_matrix)
        new_matrix, changed = self.compress_cells(new_matrix)
        return new_matrix

    def prepare_move_right(self, matrix) -> List[List]:
        new_matrix = self.reverse(matrix)
        new_matrix = self.prepare_move_left(new_matrix)
        new_matrix = self.reverse(new_matrix)
        return new_matrix 

    def prepare_move_up(self, matrix) -> List[List]:
        new_matrix = self.transpose(matrix)
        new_matrix = self.prepare_move_left(new_matrix)
        new_matrix = self.transpose(new_matrix)
        return new_matrix

    def prepare_move_down(self, matrix) -> List[List]:
        new_matrix = self.transpose(matrix)
        new_matrix = self.reverse(new_matrix)
        new_matrix = self.prepare_move_left(new_matrix)
        new_matrix = self.reverse(new_matrix)
        new_matrix = self.transpose(new_matrix)
        return new_matrix

    def move_max(self, matrix):
        self.set_matrix(matrix)
        
    def move_min(self, matrix):
        moves = self.get_available_moves_for_min(matrix)
        
        value = 0
        place = randint(0, len(moves)) -1 
        if random() < 0.75:
            value = 2
        else:
            value = 4
            
        self.place_tile(moves[place][0], moves[place][1], value)
        
    # Gives available moves for Max
    # UP -> 0
    # RIGHT -> 1
    # DOWN -> 2
    # LEFT -> 3
    def get_available_moves_for_max(self, matrix) -> List[int]:
        available_moves = []
        if(self.can_move_up(matrix)):
            available_moves.append(0)
        if(self.can_move_right(matrix)):
            available_moves.append(1)
        if(self.can_move_down(matrix)):
            available_moves.append(2)
        if(self.can_move_left(matrix)):
            available_moves.append(3)

        return available_moves

    def get_available_moves_for_min(self, matrix) -> List[Tuple[int]]:
        moves = []
        
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if matrix[i][j] == 0:
                    moves.append((i, j))
                    
        return moves

    def get_children_for_max(self, matrix):
        children = []
        moves_max = self.get_available_moves_for_max(matrix)
        
        if 0 in moves_max:
            children.append(Grid(self.prepare_move_up(matrix)))
        if 1 in moves_max:
            children.append(Grid(self.prepare_move_right(matrix)))
        if 2 in moves_max:
            children.append(Grid(self.prepare_move_down(matrix)))
        if 3 in moves_max:
            children.append(Grid(self.prepare_move_left(matrix))) 
        
        return children     
    
    def get_children_for_min(self, matrix):
        children = []        
        moves_min = self.get_available_moves_for_min(matrix)
        
        for move in moves_min:
            child = deepcopy(matrix)
            child[move[0]][move[1]] = 2
            children.append(Grid(child))
            child[move[0]][move[1]] = 4
            children.append(Grid(child))
            
        return children

# ======================================================================
# ======================TERMINATION FUNCTIONS===========================
# ======================================================================    
    

    def is_terminal(self, matrix) -> bool:
        moves_max = self.get_available_moves_for_max(matrix)
        if not moves_max:
            return True
        
        return False

    def is_game_over(self, matrix) -> bool:
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if matrix[i][j] == MAX_VALUE:
                    return True
                
        return False 


# =================================================================
# ======================MINMAX FUNCTIONS===========================
# =================================================================

def maximize(grid: Grid, alfa, beta, depth):
  
    is_terminal = grid.is_terminal(grid.matrix)
    
    if is_terminal or depth == 0:
        return (None, grid.utility(grid.matrix))
    
    (max_child, max_score) = (None, -infinity)
    
    children = grid.get_children_for_max(grid.matrix)
    
    for child in children:
        (_, score) = minimize(child, alfa, beta, depth - 1)
        if score > max_score:
            (max_child, max_score) = (child, score)
        if max_score >= beta:
            break
        if max_score > alfa:
            alfa = max_score
        

    return (max_child, max_score)


def minimize(grid: Grid, alfa, beta, depth):
       
    is_terminal = grid.is_terminal(grid.matrix)
    
    if is_terminal or depth == 0:
        return (None, grid.utility(grid.matrix))
 
    (min_child, min_score) = (None, infinity)
    
    children = grid.get_children_for_min(grid.matrix)
    
    for child in children:    
        (_, score) = maximize(child, alfa, beta, depth - 1)
        if score < min_score:
            (min_child, min_score) = (child, score)
        if min_score <= alfa:
            break
        if min_score < beta:
            beta = min_score
       
    return (min_child, min_score)


def get_best_move(grid: Grid, depth):
    (child, _) = maximize(grid, -1, MAX_INT, depth)
    return grid.move_max(child.matrix)


# ====================================================================
# ======================GAME LOOP FUNCTIONS===========================
# ====================================================================

init_matrix = []
for i in range(BOARD_SIZE):
    init_matrix.append([0] * BOARD_SIZE)

grid = Grid(init_matrix)

grid.move_min(grid.matrix)
grid.move_min(grid.matrix)

while(True):
    print("================")
    grid.print_matrix()
    
    # if grid.is_game_over(grid.matrix):
    #     print("GAME OVER - MAX WON")

    
    if grid.is_terminal(grid.matrix):
        print("GAME OVER - MIN WON")
        print("FINAL SCORE WAS ", grid.utility(grid.matrix))
        break
    # print(grid.get_children_for_max(grid.matrix))
    
    # grid.utility(grid.matrix)
    
    # key = input("Direction:")

    # UP -> 0
    # RIGHT -> 1
    # DOWN -> 2
    # LEFT -> 3
    # if (key == 'a' and 3 in moves_max):
    #     grid.move_max(grid.prepare_move_left(grid.matrix))
    # if (key == 'd' and 1 in moves_max):
    #     grid.move_max(grid.prepare_move_right(grid.matrix))
    # if (key == "w" and 0 in moves_max):
    #     grid.move_max(grid.prepare_move_up(grid.matrix))
    # if (key == "s" and 2 in moves_max):
    #     grid.move_max(grid.prepare_move_down(grid.matrix))
    # if (key == "p"):
    #     break
    
    get_best_move(grid, DEPTH)
    
    # time.sleep(0.05)
    
    grid.move_min(grid.matrix)

