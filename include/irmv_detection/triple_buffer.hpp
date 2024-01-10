// A simple triple buffer implementation for producer-consumer pattern
// Credit to https://github.com/remis-thoughts/blog/blob/master/triple-buffering/src/main/md/triple-buffering.md
// This implementation is completely lock-free.
// The producer will never be blocked by the consumer,
// and consumer will only be blocked when the producer haven't produce anything new yet (which in this case waiting is desired).

#ifndef TRIPLE_BUFFER_HPP_
#define TRIPLE_BUFFER_HPP_

#include <array>
#include <atomic>

namespace irmv_detection
{
template <typename Buffer>
class TripleBuffer
{
public:
  explicit TripleBuffer(std::array<Buffer, 3> & buffers)
  : writing_(&buffers[0]), ready_(&buffers[1]), reading_(&buffers[2])
  {
  }

  Buffer * get_producer_buffer() { return writing_; }

  void producer_commit()
  {
    // swap writing and ready,
    writing_ = ready_.exchange(writing_);
    consumer_ready_.store(true);
    consumer_ready_.notify_one();
  }

  Buffer * get_consumer_buffer()
  {
    consumer_ready_.wait(false);
    reading_ = ready_.exchange(reading_);
    consumer_ready_.store(false);
    return reading_;
  }

private:
  std::atomic<Buffer *> writing_;
  std::atomic<Buffer *> ready_;
  std::atomic<Buffer *> reading_;
  // It's possible that the consumer is faster than the producer, so we need to wait for the producer to produce something.
  // Note that we used std::atomic waiting from C++20 here, so we don't need to use a mutex and a condition variable.
  std::atomic<bool> consumer_ready_ = false;
};
}  // namespace irmv_detection

#endif  // TRIPLE_BUFFER_HPP_