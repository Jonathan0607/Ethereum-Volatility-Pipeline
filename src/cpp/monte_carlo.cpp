#include <algorithm>
#include <cmath>
#include <map>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <random>
#include <vector>

namespace py = pybind11;

std::map<std::string, double> simulate_gbm(double current_price,
                                           double predicted_vol,
                                           int num_sims = 10000,
                                           int steps = 24) {
  std::random_device rd;
  std::mt19937_64 rng(rd());
  std::normal_distribution<double> norm(0.0, 1.0);
  std::vector<double> terminal_prices(num_sims);
  double dt = 1.0;
  double drift = -0.5 * std::pow(predicted_vol, 2);

  for (int i = 0; i < num_sims; ++i) {
    double price = current_price;
    for (int t = 0; t < steps; ++t) {
      double z = norm(rng);
      price = price * std::exp(drift * dt + predicted_vol * std::sqrt(dt) * z);
    }
    terminal_prices[i] = price;
  }

  std::sort(terminal_prices.begin(), terminal_prices.end());

  std::map<std::string, double> results;
  results["lower_bound"] = terminal_prices[static_cast<int>(num_sims * 0.025)];
  results["median"] = terminal_prices[static_cast<int>(num_sims * 0.500)];
  results["upper_bound"] = terminal_prices[static_cast<int>(num_sims * 0.975)];

  return results;
}

PYBIND11_MODULE(monte_carlo, m) {
  m.def("simulate_gbm", &simulate_gbm, "Runs GBM paths and returns 95% CI",
        py::arg("current_price"), py::arg("predicted_vol"),
        py::arg("num_sims") = 10000, py::arg("steps") = 24);
}
