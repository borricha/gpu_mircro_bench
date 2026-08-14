// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <nvml.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace powertelemetry {

using Clock = std::chrono::steady_clock;

// One observation collected by the CPU-side NVML sampler thread. This thread
// is independent of CUDA kernel launches: kernels run asynchronously on the
// GPU while the host periodically records board power, clocks and throttling.
struct Sample {
  double t_s = 0.0;
  double power_w = NAN;
  unsigned int util_pct = 0;
  unsigned int sm_clock_mhz = 0;
  unsigned int mem_clock_mhz = 0;
  unsigned int temp_c = 0;
  unsigned long long throttle_reasons = 0;
};

struct Summary {
  double arithmetic_avg_w = NAN;
  double time_weighted_avg_w = NAN;
  double time_weighted_energy_j = NAN;
  double first_sample_delay_ms = NAN;
  double util_avg_pct = NAN;
  double sm_clock_avg_mhz = NAN;
  unsigned int sm_clock_min_mhz = 0;
  unsigned int sm_clock_max_mhz = 0;
  double mem_clock_avg_mhz = NAN;
  unsigned int temp_max_c = 0;
  double throttle_any_pct = 0.0;
  double throttle_sw_power_pct = 0.0;
  double throttle_hw_slowdown_pct = 0.0;
  double throttle_sw_thermal_pct = 0.0;
  double throttle_hw_thermal_pct = 0.0;
  double throttle_app_clock_pct = 0.0;
  unsigned long long throttle_reasons_or = 0;
  size_t sample_count = 0;
};

inline double mean(const std::vector<double>& values) {
  if (values.empty()) return NAN;
  return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

class Sampler {
 public:
  Sampler(nvmlDevice_t device, unsigned int interval_ms)
      : device_(device), interval_ms_(interval_ms) {}

  ~Sampler() { stop(); }

  void start(Clock::time_point origin) {
    if (running_) return;
    origin_ = origin;
    stop_.store(false, std::memory_order_relaxed);
    running_ = true;
    worker_ = std::thread([this] { loop(); });
  }

  void stop() {
    if (!running_) return;
    stop_.store(true, std::memory_order_relaxed);
    if (worker_.joinable()) worker_.join();
    running_ = false;
  }

  const std::vector<Sample>& samples() const { return samples_; }

  Summary summarize(double measurement_end_s) const {
    // First summarize diagnostic fields. Clock min/max and throttling are
    // essential validity checks: a requested locked clock can still fall when
    // a power or thermal safety limit takes precedence.
    Summary out;
    out.sample_count = samples_.size();
    if (samples_.empty()) return out;
    out.first_sample_delay_ms = 1000.0 * samples_.front().t_s;

    std::vector<double> powers;
    powers.reserve(samples_.size());
    double util_sum = 0.0, sm_sum = 0.0, mem_sum = 0.0;
    out.sm_clock_min_mhz = samples_.front().sm_clock_mhz;
    out.sm_clock_max_mhz = samples_.front().sm_clock_mhz;
    size_t any = 0, sw_power = 0, hw_slow = 0, sw_thermal = 0;
    size_t hw_thermal = 0, app_clock = 0;

    for (const auto& s : samples_) {
      if (std::isfinite(s.power_w)) powers.push_back(s.power_w);
      util_sum += s.util_pct;
      sm_sum += s.sm_clock_mhz;
      mem_sum += s.mem_clock_mhz;
      out.sm_clock_min_mhz = std::min(out.sm_clock_min_mhz, s.sm_clock_mhz);
      out.sm_clock_max_mhz = std::max(out.sm_clock_max_mhz, s.sm_clock_mhz);
      out.temp_max_c = std::max(out.temp_max_c, s.temp_c);
      out.throttle_reasons_or |= s.throttle_reasons;
      const unsigned long long active =
          s.throttle_reasons & ~nvmlClocksThrottleReasonGpuIdle;
      if (active) ++any;
      if (s.throttle_reasons & nvmlClocksThrottleReasonSwPowerCap) ++sw_power;
      if (s.throttle_reasons & nvmlClocksThrottleReasonHwSlowdown) ++hw_slow;
      if (s.throttle_reasons & nvmlClocksThrottleReasonSwThermalSlowdown) ++sw_thermal;
      if (s.throttle_reasons & nvmlClocksThrottleReasonHwThermalSlowdown) ++hw_thermal;
      if (s.throttle_reasons & nvmlClocksThrottleReasonApplicationsClocksSetting) ++app_clock;
    }

    out.arithmetic_avg_w = mean(powers);
    out.util_avg_pct = util_sum / samples_.size();
    out.sm_clock_avg_mhz = sm_sum / samples_.size();
    out.mem_clock_avg_mhz = mem_sum / samples_.size();
    const double denom = static_cast<double>(samples_.size());
    out.throttle_any_pct = 100.0 * any / denom;
    out.throttle_sw_power_pct = 100.0 * sw_power / denom;
    out.throttle_hw_slowdown_pct = 100.0 * hw_slow / denom;
    out.throttle_sw_thermal_pct = 100.0 * sw_thermal / denom;
    out.throttle_hw_thermal_pct = 100.0 * hw_thermal / denom;
    out.throttle_app_clock_pct = 100.0 * app_clock / denom;

    // Zero-order hold integration using actual sample timestamps. Extend the
    // first observed value backward to measurement start (t=0), otherwise a
    // slow first NVML call would leave [0, first_timestamp) unaccounted while
    // the result is still divided by the full measurement duration.
    double energy_j = 0.0;
    const auto first_finite = std::find_if(
        samples_.begin(), samples_.end(),
        [](const Sample& sample) { return std::isfinite(sample.power_w); });
    if (first_finite != samples_.end()) {
      const double leading_dt =
          std::clamp(first_finite->t_s, 0.0, measurement_end_s);
      energy_j += first_finite->power_w * leading_dt;
    }
    for (size_t i = 0; i < samples_.size(); ++i) {
      if (!std::isfinite(samples_[i].power_w)) continue;
      const double next_t = (i + 1 < samples_.size())
                                ? std::min(samples_[i + 1].t_s, measurement_end_s)
                                : measurement_end_s;
      const double dt = std::max(0.0, next_t - samples_[i].t_s);
      energy_j += samples_[i].power_w * dt;
    }
    out.time_weighted_energy_j = energy_j;
    if (measurement_end_s > 0.0)
      out.time_weighted_avg_w = energy_j / measurement_end_s;
    return out;
  }

  void write_csv(const std::string& path) const {
    // Preserve every timestamped sample so averages can be audited or
    // recomputed without rerunning the benchmark.
    if (path.empty()) return;
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot open telemetry output: " + path);
    out << "sample_index,timestamp_s,power_w,util_pct,sm_clock_mhz,"
           "mem_clock_mhz,temp_c,throttle_reasons\n";
    out << std::fixed << std::setprecision(9);
    for (size_t i = 0; i < samples_.size(); ++i) {
      const auto& s = samples_[i];
      out << i << ',' << s.t_s << ',' << s.power_w << ',' << s.util_pct << ','
          << s.sm_clock_mhz << ',' << s.mem_clock_mhz << ',' << s.temp_c << ','
          << s.throttle_reasons << '\n';
    }
  }

 private:
  void loop() {
    // The host thread requests samples at interval_ms_. sleep_until() avoids
    // accumulating drift, but OS scheduling still makes intervals nonuniform;
    // summarize() therefore integrates using the timestamps actually observed.
    auto next = Clock::now();
    while (!stop_.load(std::memory_order_relaxed)) {
      Sample s;
      unsigned int value = 0;
      nvmlUtilization_t util{};
      if (nvmlDeviceGetPowerUsage(device_, &value) == NVML_SUCCESS)
        s.power_w = value / 1000.0;
      if (nvmlDeviceGetUtilizationRates(device_, &util) == NVML_SUCCESS)
        s.util_pct = util.gpu;
      if (nvmlDeviceGetClockInfo(device_, NVML_CLOCK_SM, &value) == NVML_SUCCESS)
        s.sm_clock_mhz = value;
      if (nvmlDeviceGetClockInfo(device_, NVML_CLOCK_MEM, &value) == NVML_SUCCESS)
        s.mem_clock_mhz = value;
      if (nvmlDeviceGetTemperature(device_, NVML_TEMPERATURE_GPU, &value) == NVML_SUCCESS)
        s.temp_c = value;
      nvmlDeviceGetCurrentClocksThrottleReasons(device_, &s.throttle_reasons);
      const auto stamped = Clock::now();
      s.t_s = std::chrono::duration<double>(stamped - origin_).count();
      samples_.push_back(s);
      next += std::chrono::milliseconds(interval_ms_);
      std::this_thread::sleep_until(next);
    }
  }

  nvmlDevice_t device_;
  unsigned int interval_ms_;
  Clock::time_point origin_{};
  std::atomic<bool> stop_{false};
  bool running_ = false;
  std::thread worker_;
  std::vector<Sample> samples_;
};

}  // namespace powertelemetry
