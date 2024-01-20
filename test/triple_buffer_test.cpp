#include <chrono>
#include <thread>

#include <gtest/gtest.h>

#include <fmt/format.h>

#include "irmv_detection/triple_buffer.hpp"

std::atomic<bool> stop_program = false;
void stop_program_callback(int sig)
{
  (void)sig;
  stop_program = true;
}

TEST(irmv_detection, triple_buffer_basic)
{
  std::array<int, 3> buffers = {0, 0, 0};
  irmv_detection::TripleBuffer triple_buffer(buffers);
  auto producer = [&triple_buffer]() {
    int i = 0;
    while (!stop_program) {
      auto buffer = triple_buffer.get_producer_buffer();
      *buffer = i;
      triple_buffer.producer_commit();
      fmt::print("Producer: {}\n", i);
      using namespace std::chrono_literals;
      std::this_thread::sleep_for(10ms);
      i++;
    }
  };
  auto consumer = [&triple_buffer]() {
    while (!stop_program) {
      auto buffer = triple_buffer.get_consumer_buffer();
      fmt::print("Consumer: {}\n", *buffer);
      using namespace std::chrono_literals;
      std::this_thread::sleep_for(25ms);
    }
  };
  std::jthread producer_thread(producer);
  std::jthread consumer_thread(consumer);
}

TEST(irmv_detection, triple_buffer_fps)
{
  constexpr int producer_fps = 100;
  constexpr int consumer_fps = 100;
  std::array<int, 3> buffers = {0, 0, 0};
  irmv_detection::TripleBuffer triple_buffer(buffers);
  auto producer = [&triple_buffer]() {
    namespace chrono = std::chrono;
    auto interval = chrono::milliseconds(1000 / producer_fps);
    auto start_time = chrono::system_clock::now();
    int counter = 0;
    while (!stop_program) {
      auto buffer = triple_buffer.get_producer_buffer();
      *buffer = counter;
      triple_buffer.producer_commit();
      counter++;
      if (counter % 100 == 0) {
        auto cur_time = chrono::system_clock::now();
        fmt::print(
          "Producer FPS: {}\n", 100 / (chrono::duration<double>(cur_time - start_time).count()));
        ASSERT_GE(
          100 / (chrono::duration<double>(cur_time - start_time).count()), producer_fps * 0.9);
        ASSERT_LE(
          100 / (chrono::duration<double>(cur_time - start_time).count()), producer_fps * 1.1);
        start_time = cur_time;
      }
      using namespace std::chrono_literals;
      std::this_thread::sleep_for(interval);
    }
  };
  auto consumer = [&triple_buffer]() {
    namespace chrono = std::chrono;
    auto interval = chrono::milliseconds(1000 / consumer_fps);
    auto start_time = chrono::system_clock::now();
    int counter = 0;
    while (!stop_program) {
      auto buffer = triple_buffer.get_consumer_buffer();
      (void)buffer;
      // ASSERT_EQ(*buffer, i);
      counter++;
      if (counter % 100 == 0) {
        auto cur_time = chrono::system_clock::now();
        fmt::print(
          "Consumer FPS: {}\n", 100 / (chrono::duration<double>(cur_time - start_time).count()));
        ASSERT_GE(
          100 / (chrono::duration<double>(cur_time - start_time).count()), consumer_fps * 0.9);
        ASSERT_LE(
          100 / (chrono::duration<double>(cur_time - start_time).count()), consumer_fps * 1.1);
        start_time = cur_time;
      }
      using namespace std::chrono_literals;
      std::this_thread::sleep_for(interval);
    }
  };
  std::jthread producer_thread(producer);
  std::jthread consumer_thread(consumer);
}

int main(int argc, char ** argv)
{
  signal(SIGINT, stop_program_callback);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}