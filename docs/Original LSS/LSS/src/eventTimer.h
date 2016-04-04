#ifndef EVENT_TIMER_CPP
#define EVENT_TIMER_CPP

#include <cassert>

class EventTimer
{
public:
	EventTimer() : mStarted(false), mStopped(false)
	{
		cudaEventCreate(&mStart);
		cudaEventCreate(&mStop);
	}
	~EventTimer()
	{
		cudaEventDestroy(mStart);
		cudaEventDestroy(mStop);
	}
	void start(cudaStream_t s = 0)
	{
		cudaEventRecord(mStart, s);
		mStarted = true; mStopped = false;
	}
	void stop(cudaStream_t s = 0)
	{
		assert(mStarted);
		cudaEventRecord(mStop, s);
		mStarted = false; mStopped = true;
	}
	float elapsed()
	{
		assert(mStopped);
		if (!mStopped) return 0;
		cudaEventSynchronize(mStop);
		float elapsed = 0;
		cudaEventElapsedTime(&elapsed, mStart, mStop);
		return elapsed / 1000;
	}

private:
	bool mStarted, mStopped;
	cudaEvent_t mStart, mStop;
};

#endif
