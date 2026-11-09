// Experimental, opt-in diagnostics for the Windows HIP verification cliff.
#pragma once

#if defined(_WIN32) && defined(GGML_USE_HIP) && !defined(__HIP_DEVICE_COMPILE__)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <dxgi1_4.h>
#endif

static bool cliff_profile_enabled() {
    static const bool enabled = getenv("GGML_CUDA_CLIFF_PROFILE") != nullptr;
    return enabled;
}

static bool cliff_op_profile_enabled() {
    static const bool enabled = getenv("GGML_CUDA_CLIFF_OPS") != nullptr;
    return enabled;
}

static void cliff_memory_snapshot(const char * phase) {
#if defined(_WIN32) && defined(GGML_USE_HIP) && !defined(__HIP_DEVICE_COMPILE__)
    static IDXGIFactory1 * factory = nullptr;
    static bool initialized = false;
    if (!initialized) {
        initialized = true;
        HMODULE module = LoadLibraryA("dxgi.dll");
        if (module) {
            auto create = reinterpret_cast<HRESULT (WINAPI *)(REFIID, void **)>(GetProcAddress(module, "CreateDXGIFactory1"));
            if (create) {
                create(__uuidof(IDXGIFactory1), reinterpret_cast<void **>(&factory));
            }
        }
    }
    if (factory) {
        for (UINT i = 0;; ++i) {
            IDXGIAdapter1 * adapter = nullptr;
            if (factory->EnumAdapters1(i, &adapter) != S_OK) {
                break;
            }
            DXGI_ADAPTER_DESC1 desc = {};
            adapter->GetDesc1(&desc);
            IDXGIAdapter3 * adapter3 = nullptr;
            if (desc.VendorId == 0x1002 && SUCCEEDED(adapter->QueryInterface(__uuidof(IDXGIAdapter3), reinterpret_cast<void **>(&adapter3)))) {
                DXGI_QUERY_VIDEO_MEMORY_INFO local = {}, shared = {};
                const HRESULT hr_local = adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &local);
                const HRESULT hr_shared = adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &shared);
                GGML_LOG_INFO("CLIFF memory phase=%s adapter=%u dedicated=%llu local_budget=%llu local_usage=%llu shared_budget=%llu shared_usage=%llu hr=%ld,%ld\n",
                    phase, i, (unsigned long long) desc.DedicatedVideoMemory,
                    (unsigned long long) local.Budget, (unsigned long long) local.CurrentUsage,
                    (unsigned long long) shared.Budget, (unsigned long long) shared.CurrentUsage, (long) hr_local, (long) hr_shared);
                adapter3->Release();
            }
            adapter->Release();
        }
    }
#else
    GGML_UNUSED(phase);
#endif
}

static std::mutex cliff_alloc_mutex;
static std::map<void *, size_t> cliff_allocations;
static size_t cliff_live = 0, cliff_peak = 0;

static void cliff_allocation(void * ptr, size_t size, const char * category) {
    if (!cliff_profile_enabled()) {
        return;
    }
    std::lock_guard<std::mutex> lock(cliff_alloc_mutex);
    cliff_allocations[ptr] = size;
    cliff_live += size;
    cliff_peak = std::max(cliff_peak, cliff_live);
    GGML_LOG_INFO("CLIFF alloc ptr=%p bytes=%zu category=%s live=%zu peak=%zu\n", ptr, size, category, cliff_live, cliff_peak);
    cliff_memory_snapshot("alloc");
}

static cudaError_t cliff_device_free(void * ptr) {
    if (cliff_profile_enabled()) {
        std::lock_guard<std::mutex> lock(cliff_alloc_mutex);
        auto it = cliff_allocations.find(ptr);
        if (it != cliff_allocations.end()) {
            cliff_live -= it->second;
            GGML_LOG_INFO("CLIFF free ptr=%p bytes=%zu live=%zu\n", ptr, it->second, cliff_live);
            cliff_allocations.erase(it);
        }
    }
    return cudaFree(ptr);
}

struct cliff_op_timer {
    const ggml_tensor * node;
    cudaStream_t stream;
    int64_t started = 0;
    struct events {
        cudaEvent_t start, stop;
        events() {
            CUDA_CHECK(cudaEventCreateWithFlags(&start, 0));
            CUDA_CHECK(cudaEventCreateWithFlags(&stop, 0));
        }
        ~events() {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
    };
    static events & get_events() {
        static thread_local events value;
        return value;
    }
    cliff_op_timer(const ggml_tensor * node, cudaStream_t stream) : node(node), stream(stream) {
        if (cliff_op_profile_enabled()) {
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaEventRecord(get_events().start, stream));
            started = ggml_time_us();
        }
    }
    ~cliff_op_timer() {
        if (started) {
            const int64_t dispatch = ggml_time_us() - started;
            CUDA_CHECK(cudaEventRecord(get_events().stop, stream));
            CUDA_CHECK(cudaDeviceSynchronize());
            float gpu_ms = 0;
#if defined(GGML_USE_HIP)
            CUDA_CHECK(hipEventElapsedTime(&gpu_ms, get_events().start, get_events().stop));
#elif defined(GGML_USE_MUSA)
            CUDA_CHECK(musaEventElapsedTime(&gpu_ms, get_events().start, get_events().stop));
#else
            CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, get_events().start, get_events().stop));
#endif
            GGML_LOG_INFO("CLIFF op name=%s type=%s us=%lld dispatch_us=%lld event_ms=%.4f ne=%lld,%lld,%lld,%lld\n", node->name, ggml_op_name(node->op), (long long) (ggml_time_us() - started), (long long) dispatch, gpu_ms,
                (long long) node->ne[0], (long long) node->ne[1], (long long) node->ne[2], (long long) node->ne[3]);
        }
    }
};
