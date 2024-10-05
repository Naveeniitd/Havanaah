import time
import math
import random
import numpy as np
from helper import *
from typing import Tuple, List, Set, Union
from collections import deque, namedtuple

Move = namedtuple('Move', ['position', 's'])

class AIPlayer:
    def __init__(self, player_number: int, timer):
        self.player_number = player_number
        self.opponent_number = 3 - player_number
        self.type = 'ai'
        self.player_string = f'Player {player_number}: ai'
        self.timer = timer
        self.dim = None
        self.weights = self._init_weights()
    def helper2(self, state):
        my_moves = len(get_valid_actions(state))
        return my_moves * self.weights['mobility']

    def helper3(self, state):
        center = self.dim // 2
        return sum(
            self.weights['centre_area'] * -(abs(x - center) + abs(y - center))
            for x, y in np.argwhere(state == self.player_number)
        )
    def _init_weights(self):
        return {
            'fork': 1000,
            'bridge': 1000,
            'ring': 1500,
            'size': 10,
            'block_opponent': 900,
            'mobility': 5,
            'centre_area': 60,
            'threat_blocking': 1200,
            'possible': {
                'fork': 500,
                'bridge': 500,
                'ring': 700
            }
        }

    def _remaining_moves(self, state):
        e = np.count_nonzero(state == 0)
        X = (e + 1) // 2
        return X
    def move_on(self, state, m, player):
        new_state = state.copy()
        new_state[m] = player
        return new_state
    def _lose_play(self, state):
        for move in get_valid_actions(state):
            temp_state = self.move_on(state, move, self.opponent_number)
            if self._check_player_win(temp_state, self.opponent_number):
                return move
        return None
    def _time_info(self, state):
        r_time= fetch_remaining_time(self.timer, self.player_number)
        r_moves = self._remaining_moves(state)
        temp = (r_moves / (self.dim * self.dim / 2))
        situation = 1 - temp
        if situation < 0.3:
            f = 0.5 
        elif situation < 0.7:
            f = 0.7 
        else:
            f  = 0.9

        p_move = r_time/ r_moves
        time_limit = min(p_move * f, r_time- 0.1)
        
        print(f"{self.player_string} - Time limit for this move: {time_limit:.2f}s (Remaining time: {r_time:.2f}s, Estimated moves remaining: {r_moves}, Game progress: {situation:.2f})")
        
        return {
            'limit': time_limit,
            'end_time': time.perf_counter() + time_limit,
            'remaining': r_time,
            'r_moves': r_moves,
            'progress': situation
        }
    def _check_player_win(self, state, player_num):
        P = np.argwhere(state == player_num)
        board = (state == player_num)
        for i in P:
            move = tuple(i)
            if check_win(board, move, player_num)[0]:
                return True
        return False
       

    def get_move(self, state: np.array) -> Tuple[int, int]:
        self.dim = state.shape[0]
        time_info = self._time_info(state)
        
        urgent = self._lose_play(state)
        if urgent:
            print(f"{self.player_string} - Blocking opponent's winning move: {urgent}")
            return urgent

        return self._deep_search(state, time_info)


    def _deep_search(self, state, time_info):
        options = get_valid_actions(state)
        if not options:
            print(f"{self.player_string} - No valid moves available")
            return (-1, -1)

        b_move = random.choice(options)
        d = 10
        b_s = -math.inf
        d_time = []

        print(f"{self.player_string} - Starting iterative deepening")

        for depth in range(1, d + 1):
            if time.perf_counter() >= time_info['end_time']:
                print(f"{self.player_string} - Time limit reached, stopping search at depth {depth}")
                break

            try:
                s_time = time.perf_counter()
                s, move = self._minimax(state, depth, True, -math.inf, math.inf, time_info['end_time'])
                depth_time = time.perf_counter() - s_time
                d_time.append(depth_time)

                if move:
                    b_move = move
                    b_s = s
                print(f"{self.player_string} - Depth {depth}: Best move {b_move} with s {b_s}, depth time {depth_time:.2f}s")
            except TimeoutError:
                print(f"{self.player_string} - Timeout at depth {depth}")
                break

            if d_time and sum(d_time) * 1.5 > time_info['limit']:
                print(f"{self.player_string} - Not enough time for deeper search, stopping at depth {depth}")
                break

        print(f"{self.player_string} - Returning best move: {b_move} with s {b_s}")
        return b_move

    def _minimax(self, state, depth, state_player, alpha, beta, end_time):
        if time.perf_counter() >= end_time:
            raise TimeoutError

        if depth == 0:
            return self.heuristic_value(state), None
        if self.terminal_state(state):
            return self.heuristic_value(state), None
        M = self._sorted_moves(state, state_player)
        b_move = None
        
        if state_player:
            max_eval = -math.inf
            for move in M:
                new_state = self.move_on(state, move.position, self.player_number)
                eval, _ = self._minimax(new_state, depth - 1, False, alpha, beta, end_time)
                if eval > max_eval:
                    max_eval, b_move = eval, move.position
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval, b_move
        else:
            min_eval = math.inf
            for move in M:
                new_state = self.move_on(state, move.position, self.opponent_number)
                eval, _ = self._minimax(new_state, depth - 1, True, alpha, beta, end_time)
                if eval < min_eval:
                    min_eval, b_move = eval, move.position
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval, b_move

    def _sorted_moves(self, state, state_player):
        if state_player:
            player = self.player_number  
        else :
            player = self.opponent_number
        M = []
        for m in get_valid_actions(state):
            new_state = self.move_on(state, m, player)
            s = self.heuristic_value(new_state)
            M.append(Move(m, s))
        return sorted(M, key=lambda x: x.s, reverse=state_player)

    def terminal_state(self, state):
        if self._check_player_win(state, self.player_number):
            return True
        if self._check_player_win(state, self.opponent_number):
            return True
        if len(get_valid_actions(state)) == 0:
            return True
        return False

    

    def heuristic_value(self, state):
        A = self.player_number
        B = self.opponent_number
        if self._check_player_win(state, A):
            return float('inf')
        if self._check_player_win(state, B):
            return float('-inf')

        a = self.helper_func(state, A)
        b = self.helper_func(state, B)
        score = a - b
        score += self.helper2(state)
        score += self.helper3(state)
        
        return score

    def helper_func(self, state, player_num):
        c = self.neig_options(state, player_num)
        return sum(self._evaluate_component(state, comp, player_num) for comp in c)

    def neig_options(self, state, player_num):
        board = (state == player_num)
        visited = set()
        c = []
        for x in range(self.dim):
            for y in range(self.dim):
                if board[x, y] and (x, y) not in visited:
                    collections = self._bfs(board, (x, y), visited)
                    c.append(collections)
        return c

    def _bfs(self, board, start, visited):
        queue = deque([start])
        collections = set([start])
        visited.add(start)
        while queue:
            current = queue.popleft()
            for nx, ny in get_neighbours(self.dim, current):
                if is_valid(nx, ny, self.dim) and board[nx, ny] and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
                    collections.add((nx, ny))
        return collections

    def _evaluate_component(self, state, collections, player_num):
        s = 0
        is_opponent = (player_num != self.player_number)

        s += len(collections) * self.weights['size']

        ed_c = set()
        cor_con = set()

        for i in collections:
            edge = get_edge(i, self.dim)
            if edge != -1:
                ed_c.add(edge)
            corner = get_corner(i, self.dim)
            if corner != -1:
                cor_con.add(corner)

        if collections:
            first_vertex = next(iter(collections))
            player_board = (state == player_num)
            if check_ring(player_board, first_vertex):
                s += self.weights['ring']
            if check_fork(player_board, first_vertex):
                s += self.weights['fork']
            if check_bridge(player_board, first_vertex):
                s += self.weights['bridge']

        # possible patterns
        s += self.wining_conditi(ed_c, cor_con, is_opponent, len(collections))
        return s

    def wining_conditi(self, e_cn, c_cn, is_opponent, c_size):
        score = 0

        # Evaluate potential Fork opportunities
        if len(e_cn) >= 2:
            score += (len(e_cn) / 6) * self.weights['possible']['fork']
            if is_opponent and len(e_cn) >= 3:
                score -= 4 * self.weights['possible']['fork']

        # Evaluate potential Bridge opportunities
        if len(c_cn) >= 1:
            score += (len(c_cn) / 2) * self.weights['possible']['bridge']
            if is_opponent and len(c_cn) >= 2:
                score -= 4 * self.weights['possible']['bridge']

        # Evaluate potential Ring opportunities
        if c_size >= 4:
            score += self.weights['possible']['ring']
            if is_opponent and c_size >= 8:
                score -= 4 * self.weights['possible']['ring']

        return score
        

    