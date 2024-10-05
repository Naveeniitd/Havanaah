import numpy as np
import time
from helper import fetch_remaining_time, get_valid_actions, check_win, get_neighbours, get_all_corners, get_all_edges

class AIPlayer:
    def __init__(self, player_number, timer):
        self.player_number = player_number
        self.opponent_number = 2 if player_number == 1 else 1
        self.timer = timer  
        self.type = 'ai'
        self.player_string = 'Player {}:ai'.format(player_number)
        self.dim = None  
        self.threatening_move = None  
        self.all_corners = None  
        self.all_edges = None
        
    def get_move(self, board):
        self.dim = board.shape[0]
        start_time = time.time()
        time_limit = 10
        if self.dim//2 +1  ==6:
             time_limit = 12
        if self.all_corners is None or self.all_edges is None:
            self.all_corners = np.array(get_all_corners(self.dim))
            self.all_edges = np.array(get_all_edges(self.dim))
        if time_limit <= 0:
            valid_moves = get_valid_actions(board)
            return valid_moves[0] if valid_moves else (-1, -1)

        
        opp_threat_level = self.OTL(board)
        if opp_threat_level >= 1000 and self.threatening_move in get_valid_actions(board):
            
            return self.threatening_move

        valid_moves = get_valid_actions(board)
        if not valid_moves:
            return (-1, -1)  

        best_move = valid_moves[0]  
        max_depth = 1
        MAX_DEPTH = 10  

        
        while max_depth <= MAX_DEPTH and time.time() - start_time < time_limit:
            
            depth_start_time = time.time()
            move, _ = self.minimax(board, max_depth, True, start_time, time_limit, alpha=float('-inf'), beta=float('inf'))
            
            depth_end_time = time.time()  
            print(f"{self.player_string} completed depth {max_depth} in {depth_end_time - depth_start_time:.4f} seconds")
            if move is not None:
                best_move = move
            else:
                break  
            max_depth += 1

        return best_move

    def minimax(self, board, depth, maximizing_player, start_time, time_limit, alpha, beta):
        if time.time() - start_time >= time_limit:
            return None, self.evaluate_board(board)

        if depth == 0:
            return None, self.evaluate_board(board)
        
        valid_moves = get_valid_actions(board)
        if not valid_moves:
            return None, self.evaluate_board(board)

        # Move ordering
        move_scores = []
        for move in valid_moves:
            new_board = board.copy()
            player = self.player_number if maximizing_player else self.opponent_number
            new_board[move] = player
            score = self.evaluate_board(new_board)
            move_scores.append((score, move))



        # Sort moves
        move_scores.sort(reverse=maximizing_player)
        TOP_N = 10 


        pruned_move_scores = move_scores[:TOP_N]
        move_scores = pruned_move_scores
        best_move = None

        if maximizing_player:
            max_eval = float('-inf')
            for score, move in move_scores:
                if time.time() - start_time >= time_limit:
                    break
                new_board = board.copy()
                new_board[move] = self.player_number
                win, _ = check_win(new_board, move, self.player_number)
                if win:
                    return move, float('inf')
                _, eval = self.minimax(new_board, depth - 1, False, start_time, time_limit, alpha, beta)
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return best_move, max_eval
        else:
            min_eval = float('inf')
            for score, move in move_scores:
                if time.time() - start_time >= time_limit:
                    break
                new_board = board.copy()
                new_board[move] = self.opponent_number
                win, _ = check_win(new_board, move, self.opponent_number)
                if win:
                    return move, float('-inf')
                _, eval = self.minimax(new_board, depth - 1, True, start_time, time_limit, alpha, beta)
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return best_move, min_eval

    def evaluate_board(self, board):
        score = 0
        my_pieces = np.argwhere(board == self.player_number)
        opp_pieces = np.argwhere(board == self.opponent_number)

        
        score += self.PB(board, my_pieces) * 1.0
        score += self.PF(board, my_pieces) * 1.0
        score += self.block_opponent(board, opp_pieces) * 2.0  # Increased weight
        score += self.CCP(board, my_pieces) * 0.5

        return score

    def PB(self, board, my_pieces):
        score = 0
        corners = self.all_corners
        my_corners = [corner for corner in corners if tuple(corner) in map(tuple, my_pieces)]
        if len(my_corners) >= 2:
            score += 100
        else:
            min_distance = min([self.MD(piece, corner)
                                for piece in my_pieces for corner in corners])
            score += max(0, 20 - min_distance)
        return score

    def PF(self, board, my_pieces):
        score = 0
        edges = self.all_edges
        connected_edges = set()
        for edge in edges:
            for cell in edge:
                if tuple(cell) in map(tuple, my_pieces):
                    connected_edges.add(tuple(map(tuple, edge)))
                    break
        if len(connected_edges) >= 3:
            score += 100
        else:
            for piece in my_pieces:
                for edge in edges:
                    for cell in edge:
                        distance = self.MD(piece, cell)
                        score += max(0, 10 - distance)
                        break
        return score

    def block_opponent(self, board, opp_pieces):
        score = 0
        opp_threat_level = self.OTL(board)
        score -= opp_threat_level
        return score

    def OTL(self, board):
        threat_level = 0
        valid_moves = get_valid_actions(board)
        self.threatening_move = None

        for move in valid_moves:
            new_board = board.copy()
            new_board[move] = self.opponent_number
            win, _ = check_win(new_board, move, self.opponent_number)
            if win:
                threat_level += 1000  # Immediate threat
                self.threatening_move = move  # Move to block

        return threat_level

    def CCP(self, board, my_pieces):
        score = 0
        center = self.dim // 2
        for piece in my_pieces:
            distance = abs(piece[0] - center) + abs(piece[1] - center)
            score += max(0, 10 - distance)
        return score

    def MD(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

