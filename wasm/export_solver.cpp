#include <algorithm>
#include <cctype>
#include <memory>
#include <string>
#include <vector>

#include <emscripten/bind.h>

#include "../game.hpp"
#include "../heuristics.hpp"
#include "../strategies/ExpectimaxDepthStrategy.hpp"
#include "../strategies/ExpectimaxProbabilityStrategy.hpp"
#include "../strategies/MonteCarloPlayer.hpp"
#include "../strategies/RandomPlayer.hpp"
#include "../strategies/RandomTrialsStrategy.hpp"

const std::array<row_t, ROWS> GameSimulator::shift = generate_shift();
const std::array<uint8_t, EMPTY_TILE_POSITIONS> GameSimulator::empty_tiles = generate_empty_tiles();
const std::array<int, EMPTY_MASKS> GameSimulator::empty_index = generate_empty_index();

namespace {

constexpr int kBoardTileCount = 16;
constexpr int kTileBitWidth = 4;
constexpr int kMaxExponent = 0xF;

std::string toLowerCopy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return value;
}

heuristic_t resolveHeuristic(const std::string& name) {
  const std::string key = toLowerCopy(name);

  if (key == "score") return heuristics::score_heuristic;
  if (key == "merge") return heuristics::merge_heuristic;
  if (key == "corner" || key == "corner_bias") return heuristics::corner_heuristic;
  if (key == "wall" || key == "strict_wall") return heuristics::strict_wall_heuristic;
  if (key == "wall_gap") return heuristics::wall_gap_heuristic;
  if (key == "full_wall") return heuristics::full_wall_heuristic;
  if (key == "skewed_corner") return heuristics::skewed_corner_heuristic;
  if (key == "monotonicity") return heuristics::monotonicity_heuristic;

  return heuristics::corner_heuristic;
}

int clampExponent(int value) {
  if (value < 0) return 0;
  if (value > kMaxExponent) return kMaxExponent;
  return value;
}

GameSimulator& simulator() {
  static GameSimulator instance;
  return instance;
}

std::unique_ptr<Strategy> makeStrategy(const std::string& type,
                                       heuristic_t evaluator,
                                       int depth,
                                       double probability,
                                       int trials) {
  const std::string key = toLowerCopy(type);

  if (key == "expectimax-depth" || key == "expectimax") {
    const int effectiveDepth = depth <= 0 ? 4 : depth;
    return std::make_unique<ExpectimaxDepthStrategy>(effectiveDepth, evaluator);
  }

  if (key == "expectimax-probability") {
    float prob = static_cast<float>(probability);
    if (!(prob > 0.0f)) {
      prob = 0.001f;
    }
    return std::make_unique<ExpectimaxProbabilityStrategy>(prob, evaluator);
  }

  if (key == "monte-carlo") {
    int iterations = trials > 0 ? trials : std::max(128, depth * 128);
    return std::make_unique<MonteCarloPlayer>(iterations);
  }

  if (key == "random-trials") {
    const int branchDepth = depth > 0 ? depth : 3;
    const int gamesPerMove = trials > 0 ? trials : 32;
    return std::make_unique<RandomTrialsStrategy>(gamesPerMove, branchDepth, 2);
  }

  if (key == "random") {
    return std::make_unique<RandomPlayer>();
  }

  // Fallback: expectimax depth strategy.
  const int fallbackDepth = depth <= 0 ? 4 : depth;
  return std::make_unique<ExpectimaxDepthStrategy>(fallbackDepth, evaluator);
}

}  // namespace

board_t boardFromArray(const std::vector<int>& tiles) {
  board_t board = 0;
  const size_t limit = std::min<size_t>(tiles.size(), kBoardTileCount);
  for (size_t i = 0; i < limit; ++i) {
    const board_t exponent = static_cast<board_t>(clampExponent(tiles[i]));
    board |= (exponent & kMaxExponent) << (i * kTileBitWidth);
  }
  return board;
}

std::vector<int> arrayFromBoard(board_t board) {
  std::vector<int> tiles(kBoardTileCount, 0);
  for (int i = 0; i < kBoardTileCount; ++i) {
    tiles[i] = static_cast<int>((board >> (i * kTileBitWidth)) & kMaxExponent);
  }
  return tiles;
}

int getScore(board_t board) {
  const eval_t score = heuristics::score_heuristic(board);
  return static_cast<int>(score);
}

int getMaxTile(board_t board) {
  const int maxExponent = get_max_tile(board);
  if (maxExponent <= 0) {
    return 0;
  }
  const int capped = std::min(maxExponent, kMaxExponent);
  return 1 << capped;
}

bool isGameOver(board_t board) {
  return simulator().game_over(board);
}

bool isValidMove(board_t board, int direction) {
  if (direction < 0 || direction > 3) {
    return false;
  }
  const GameSimulator& sim = simulator();
  return board != sim.make_move(board, direction);
}

board_t makeMove(board_t board, int direction) {
  if (direction < 0 || direction > 3) {
    return board;
  }
  return simulator().make_move(board, direction);
}

class StrategyWrapper {
 public:
  StrategyWrapper(const std::string& type,
                  const std::string& heuristic,
                  int depth,
                  double probability)
      : strategy_type_(toLowerCopy(type)),
        heuristic_name_(toLowerCopy(heuristic)) {
    configure(strategy_type_, heuristic_name_, depth, probability);
  }

  void configure(const std::string& type,
                 const std::string& heuristic,
                 int depth,
                 double probability) {
    strategy_type_ = toLowerCopy(type);
    heuristic_name_ = toLowerCopy(heuristic);
    depth_ = depth;
    probability_ = probability;
    evaluator_ = resolveHeuristic(heuristic_name_);
    strategy_ = makeStrategy(strategy_type_, evaluator_, depth_, probability_, trials_);
  }

  void setTrials(int trials) {
    trials_ = trials;
    strategy_ = makeStrategy(strategy_type_, evaluator_, depth_, probability_, trials_);
  }

  int pickMove(board_t board) {
    ensureStrategy();
    return strategy_->pick_move(board);
  }

  double evaluateBoard(board_t board) const {
    return static_cast<double>(evaluator_(board));
  }

 private:
  void ensureStrategy() {
    if (!strategy_) {
      strategy_ = makeStrategy(strategy_type_, evaluator_, depth_, probability_, trials_);
    }
  }

  std::unique_ptr<Strategy> strategy_;
  heuristic_t evaluator_ = heuristics::corner_heuristic;
  std::string strategy_type_ = "expectimax-depth";
  std::string heuristic_name_ = "corner";
  int depth_ = 4;
  double probability_ = 0.0025;
  int trials_ = 256;
};

EMSCRIPTEN_BINDINGS(solver_module) {
  emscripten::function("boardFromArray", &boardFromArray);
  emscripten::function("arrayFromBoard", &arrayFromBoard);
  emscripten::function("getScore", &getScore);
  emscripten::function("getMaxTile", &getMaxTile);
  emscripten::function("isGameOver", &isGameOver);
  emscripten::function("isValidMove", &isValidMove);
  emscripten::function("makeMove", &makeMove);

  emscripten::class_<StrategyWrapper>("StrategyWrapper")
      .constructor<const std::string&, const std::string&, int, double>()
      .function("configure", &StrategyWrapper::configure)
      .function("setTrials", &StrategyWrapper::setTrials)
      .function("pickMove", &StrategyWrapper::pickMove)
      .function("evaluateBoard", &StrategyWrapper::evaluateBoard);

  emscripten::register_vector<int>("IntVector");
}
