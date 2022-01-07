/*
 * This file is part of Connect4 Game Solver <http://connect4.gamesolver.org>
 * Copyright (C) 2017-2019 Pascal Pons <contact@gamesolver.org>
 *
 * Connect4 Game Solver is free software: you can redistribute it and/or
 * modify it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * Connect4 Game Solver is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with Connect4 Game Solver. If not, see <http://www.gnu.org/licenses/>.
 */

#include "Solver.hpp"
#include <iostream>
#include "Position.hpp"
#include <string>
#include <unordered_set>


using namespace GameSolver::Connect4;
std::unordered_set<uint64_t> visited;
int positions_explored = 0;
int positions_ignored = 0;

/**
 * Main function.
 * Reads Connect 4 positions, line by line, from standard input
 * and writes one line per position to standard output containing:
 *  - score of the position
 *  - number of nodes explored
 *  - time spent in microsecond to solve the position.
 *
 *  Any invalid position (invalid sequence of move, or already won game)
 *  will generate an error message to standard error and an empty line to standard output.
 */

void explore2(Solver& solver, const Position &P, char* pos_str, const int depth) 
{
  // uint64_t key = P.key3();
  int nb_moves = P.nbMoves();

  // std::cerr << "Entering explore2 with depth " << depth << " str " << pos_str << " num_moves " << nb_moves << std::endl;
  if(nb_moves > depth) return;  // do not explore at further depth

  // if(!visited.insert(key).second) {
  //   return; // already explored position
  // }

  for (int i=0; i<Position::WIDTH; i++) {
    if(P.canPlay(i) && !P.isWinningMove(i)) {
      Position P2(P);
      P2.playCol(i);
      uint64_t key = P2.key3();
      if(!visited.insert(key).second) {
        // std::cerr << "key " << key << " exists" << std::endl;
        continue; // already explored position
      }
      
      positions_explored++;
      std::vector<int> scores = solver.analyze(P2, false);
      int best = -1000;
      int best_move = -1;
      for(int j = 0; j < Position::WIDTH; j++) {
        if (scores[solver.columnOrder[j]] > best) {
          best = scores[solver.columnOrder[j]];
          best_move = solver.columnOrder[j];
        }
      }
      pos_str[nb_moves] = '1' + i;
      pos_str[nb_moves + 1] = 0; 
      if (best >= (Position::WIDTH * Position::HEIGHT + 1 - P.nbMoves()) / 2 - 5) {
        // std::cerr << "score " << best << " for " << pos_str << " ignore\n";
        positions_ignored++;
        continue;   //ignore all positions that can be searched up
      }
      
      std::cout << pos_str << " " << best_move << std::endl;
      pos_str[nb_moves+1] = '1' + best_move;
      pos_str[nb_moves+2] = 0;
      P2.playCol(best_move);
      // std::cerr << "Calling explore2 with depth " << depth << " str " << pos_str << std::endl;
      explore2(solver, P2, pos_str, depth);
    }
  }
}
int main(int argc, char** argv) {
  Solver solver;
  bool weak = false;
  bool analyze = false;
  int depth =24;
  bool generate = false;

  std::string opening_book = "7x6.book";
  for(int i = 1; i < argc; i++) {
    if(argv[i][0] == '-') {
      if(argv[i][1] == 'w') weak = true; // parameter -w: use weak solver
      else if(argv[i][1] == 'b') { // paramater -b: define an alternative opening book
        if(++i < argc) opening_book = std::string(argv[i]);
      }
      else if(argv[i][1] == 'a') { // paramater -a: make an analysis of all possible moves
        analyze = true;
      }
      else if (argv[i][1] == 'g') { // paraneter -g: generate opening book for best moves
        generate = true;
      }
    }
  }
  solver.loadBook(opening_book);

  if (!generate) {
    std::string line;

    for(int l = 1; std::getline(std::cin, line); l++) {
      Position P;
      if(P.play(line) != line.size()) {
        std::cerr << "Line " << l << ": Invalid move " << (P.nbMoves() + 1) << " \"" << line << "\"" << std::endl;
      } else {
        std::cout << line;
        // std::cout << P.key3();
        if(analyze) {
          std::vector<int> scores = solver.analyze(P, weak);
          for(int i = 0; i < Position::WIDTH; i++) std::cout << " " << scores[i];
        }
        else {
          int score = solver.solve(P, weak);
          std::cout << " " << score;
        }
        std::cout << std::endl;
      }
    }
  } else {

    char pos_str[depth + 1]; // = {0};
    memset(pos_str, 0, (depth+1)*sizeof(char));
    pos_str[0] = '4';
    pos_str[1] = 0;
    Position P;
    P.play(pos_str);
    explore2(solver, P, pos_str, depth);   //First player
    pos_str[0] = 0;
    //for second player, we should hard code first 2 moves
    //12 or 14, 23, 33 or 34, 44
    explore2(solver, Position(), pos_str, depth); //Second player

    std::cerr << "Positions explored: " << positions_explored << " Positions ignore: " << positions_ignored << std::endl;

  }
}
