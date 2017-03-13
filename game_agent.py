"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    # TODO: finish this function!
    # if game.is_winner(self):
    #     return self.score(game,self),game.get_player_location(self)
    #
    # if game.is_loser(self):
    #     return self.score(game,self),(-1,-1)

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves - 0.5 * opp_moves)


def argmin(seq, fn):
    """Return an element with lowest fn(seq[i]) score; tie goes to first one.    >>> argmin(['one', 'to', 'three'], len)
    'to'
    """
    best = seq[0];
    best_score = fn(best)
    for x in seq:
        x_score = fn(x)
        if x_score < best_score:
            best, best_score = x, x_score
    return best


def argmax(seq, fn):
    """Return an element with highest fn(seq[i]) score; tie goes to first one. >>> argmax(['one', 'to', 'three'], len)
    'three'
    """
    return argmin(seq, lambda x: -fn(x))

import sys
class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # TODO: finish this function!

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves
        if not legal_moves:
            return (-1, -1)
        move = legal_moves[0]
        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            # if self.method == "minimax" and self.iterative:
            # result, move = self.minimax(game, self.search_depth)
            if not self.iterative:
                if self.method == "alphabeta":
                    result, move = self.alphabeta(game, self.search_depth)
                if self.method == "minimax":
                    result, move = self.minimax(game, self.search_depth)
            if self.iterative:
                depth = 1
                win_or_done=False
                while not win_or_done:
                    prev_move=move
                    if self.method == "minimax":
                        result, move = self.minimax(game, depth)
                    if self.method == "alphabeta":
                        result, move = self.alphabeta(game, depth)
                    depth += 1
                    if result == float('inf'):
                        win_or_done=True
                    if result == float('-inf'):
                        move=prev_move
                        win_or_done=True
        except Timeout:
            # Handle any actions required at timeout, if necessary
            if depth % 2 == 0:
                print("\r* {0} Depth:{1}".format(self.method,depth),flush=True,end="")
            else:
                print("\r  {0} Depth:{1}".format(self.method, depth), flush=True, end="")
            pass

        # Return the best move from the last completed search iteration
        return move

    def minimax_v1(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()
        if depth <= 0:
            return self.score(game, self), game.get_player_location(self)
        legal_moves = game.get_legal_moves()
        if len(legal_moves) <= 0:
            return self.score(game, self), (-1, -1)
        best_move = (-1, -1)

        if maximizing_player:
            best_score = float('-inf')
            for move in legal_moves:
                new_game = game.forecast_move(move)
                score, next_move = self.minimax(new_game, depth - 1, False)
                if score > best_score:
                    best_score = score
                    best_move = move
            return best_score, best_move
        else:
            best_score = float('inf')
            for move in legal_moves:
                new_game = game.forecast_move(move)
                score, next_move = self.minimax(new_game, depth - 1, True)
                if score < best_score:
                    best_score = score
                    best_move = move
            return best_score, best_move

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """

        # if self.time_left() < self.TIMER_THRESHOLD:
        #     raise Timeout()

        def max_value(game_in, depth_in):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise Timeout()
            if depth_in == 0:
                return self.score(game_in, self)
            my_best_score = float('-inf')
            for mov in game_in.get_legal_moves():
                my_next_score = min_value(game_in.forecast_move(mov), depth_in - 1)
                my_best_score = max(my_next_score, my_best_score)
            return my_best_score

        def min_value(game_in, depth_in):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise Timeout()
            if depth_in == 0:
                return self.score(game_in, self)
            my_best_score = float('inf')
            for mov in game_in.get_legal_moves():
                my_next_score = max_value(game_in.forecast_move(mov), depth_in - 1)
                my_best_score = min(my_next_score, my_best_score)
            return my_best_score

        moves = game.get_legal_moves()
        best_move = moves[0]
        best_score = float('-inf')

        for move in moves:
            score = min_value(game.forecast_move(move), depth - 1)
            if score > best_score:
                best_score = score
                best_move = move
        return best_score, best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # TODO: finish this function!
        def max_value(game_in, depth_in, alpha_in, beta_in):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise Timeout()
            if depth_in == 0:
                return self.score(game_in, self)
            v = float('-inf')
            for mov in game_in.get_legal_moves():
                v = max(v, min_value(game_in.forecast_move(mov), depth_in - 1, alpha_in, beta_in))
                if v >= beta_in:
                    return v
                alpha_in = max(alpha_in, v)
            return v

        def min_value(game_in, depth_in, alpha_in, beta_in):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise Timeout()
            if depth_in == 0:
                return self.score(game_in, self)
            v = float('inf')
            for mov in game_in.get_legal_moves():
                v = min(v, max_value(game_in.forecast_move(mov), depth_in - 1, alpha_in, beta_in))
                if v <= alpha_in:
                    return v
                beta_in = min(beta_in, v)
            return v

        moves = game.get_legal_moves()
        best_move = moves[0]
        best_score = float('-inf')

        for move in moves:
            score = min_value(game.forecast_move(move), depth - 1, best_score, beta)
            if score > best_score:
                best_score = score
                best_move = move
        return best_score, best_move

#
#
# """
#
#         # if depth == 1:
#         #     if maximizing_player:
#         #         return max(
#         #             {self.score(game.forecast_move(move), self): move for move in game.get_legal_moves()}.items())
#         #     if not maximizing_player:
#         #         return min(
#         #             {self.score(game.forecast_move(move), self): move for move in game.get_legal_moves()}.items())
#
#
#
#         def max_value(game_in, depth_in):
#             my_best_score = float('-inf')
#             my_best_move = (-1, -1)
#             if depth_in == 0 or len(legal_moves())==0:
#                 return self.score(game_in, self), (-1,-1)
#             moves = game_in.get_legal_moves()
#             for mov in moves:
#                 max_new_game = game_in.forecast_move(mov)
#                 my_next_score,my_next_move = min_value(max_new_game, depth_in - 1)
#                 print("min,my_next_score_move=", my_next_score)
#                 if my_next_score > my_best_score:
#                     my_best_score = my_next_score
#                     my_best_move = mov
#             return my_best_score, my_best_move
#
#         def min_value(game_in, depth_in):
#             if depth_in == 0 or len(legal_moves())==0:
#                 return self.score(game_in, self), (-1,-1)
#             my_best_score = float('inf')
#             my_best_move = (-1, -1)
#             moves = game_in.get_legal_moves()
#             for mov in moves:
#                 max_new_game = game_in.forecast_move(mov)
#                 my_next_score, my_next_move = max_value(max_new_game, depth_in - 1)
#                 print("min,my_next_score_move=", my_next_score)
#                 if my_next_score < my_best_score:
#                     my_best_score = my_next_score
#                     my_best_move = mov
#             return my_best_score, my_best_move
#
#
#         legal_moves = game.get_legal_moves()
#         tbest_move = (-1,-1)
#         tbest_score = float('-inf')
#
#         for move in legal_moves:
#             new_game = game.forecast_move(move)
#             mscore , mmove = min_value(new_game, depth-1)
#             if mscore > tbest_score:
#                 tbest_score = mscore
#                 tbest_move = mmove
#         return tbest_score, tbest_move
# """
#         # if depth_in == 0:
#         #     if len(game_in.get_legal_moves())<=0:
#         #         return float('-inf'),(-1,-1)
#         #     return self.score(game_in, self), (-1, -1)
#         # return min((self.score(game_in, self), mov) for mov in game_in.get_legal_moves())
#
#
#         # if len(game_in.get_legal_moves())<=0:
#         #     return float('-inf'),(-1,-1)
#         # return max((self.score(game_in, self), mov) for mov in game_in.get_legal_moves())
#
#
#         # player = game.active_player() #game.to_move()
#         #
#         # if maximizing_player:
#         #     score_move = self.max_value(self, legal_moves, depth)
#         #     return score_move
#         #
#         # if not maximizing_player:
#         #     score_move = self.min_value(self, legal_moves, depth)
#         #     return score_move
#         #
#         # return best_score, best_move
#
#
#         # if depth == 1:
#         #     if maximizing_player:
#         #         return max(
#         #             {self.score(game.forecast_move(move), self): move for move in game.get_legal_moves()}.items())
#         #     if not maximizing_player:
#         #         return min(
#         #             {self.score(game.forecast_move(move), self): move for move in game.get_legal_moves()}.items())
#
#
#         # best_move = moves[0]
#         # if self.time_left() < self.TIMER_THRESHOLD or currentDepth >= depth:
#
#         # return min([self.score(game.forecast_move(m), self), m] for m in game.get_legal_moves())
#
#         # max_move_score = max(moves_scores.items())
#
#         # maxscore2 = max([self.score(game.forecast_move(move), self)] for move in legal_moves)
#         # if maxscore2[0] != max_move_score[0]:
#         #     print(*maxscore2, max_move_score[0])
#         #     print(type(*maxscore2), type(max_move_score[0]))
#         #     print("different")
#
#
#         # return max_move_score[0],max_move_score[1]
#         # return max([self.score(game.forecast_move(move), self), move] for move in legal_moves)
#
#         # return max(score_all_moves)
#         # score_all_moves = ([self.score(game.forecast_move(move), self), move] for move in legal_moves)
#         # return max(score_all_moves)
#
#
#         # print("max_move=", max_move_score[0])
#         # max_score = max_move_score[1]
#         # # print("maxMove=",max_move)
#         # # print("maxScore=",max_score)
#
#         # for move in game.get_legal_moves():
#         #     print("move=", move, "score=", self.score(game.forecast_move(move), self))
#
#
#         #
#         # player = game.active_player() #game.to_move()
#         #
#         # def max_value(state):
#         #     # if game.terminal_test(state):
#         #     #     return game.utility(state, player)
#         #     if  game.terminal_test(state):
#         #         return game.utility(state, player)
#         #     maxv = float("-inf") #-infinity
#         #     for a in game.actions(state):
#         #         maxv = max(maxv, min_value(game.result(state, a)))
#         #     return maxv
#         #
#         # def min_value(state):
#         #     if game.terminal_test(state):
#         #         return game.utility(state, player)
#         #     v = float("inf")
#         #     for a in game.actions(state):
#         #         v = min(v, max_value(game.result(state, a)))
#         #     return v
#         #
#         # # Body of minimax_decision:
#         # return argmax(game.actions(state),
#         #               key=lambda a: min_value(game.result(state, a)))
