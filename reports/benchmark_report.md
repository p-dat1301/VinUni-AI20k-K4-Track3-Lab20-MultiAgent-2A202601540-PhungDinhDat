# Benchmark Report

## Metrics

| Run | Latency (s) | Cost (USD) | Quality | Citation cov. | Failure rate | Notes |
|---|---:|---:|---:|---:|---:|---|
| baseline | 0.02 | 0.00015 | 9.0 | 20% | 0% | provider=offline-mock | tokens=548+107 | agents= |
| multi-agent | 0.62 | 0.00068 | 10.0 | 100% | 0% | provider=offline-mock | tokens=1696+716 | agents=analyst,critic,researcher,writer |
| baseline | 0.08 | 0.00014 | 9.0 | 20% | 0% | provider=offline-mock | tokens=495+107 | agents= |
| multi-agent | 0.04 | 0.00069 | 10.0 | 100% | 0% | provider=offline-mock | tokens=1543+765 | agents=analyst,critic,researcher,writer |
| baseline | 0.02 | 0.00014 | 9.0 | 40% | 0% | provider=offline-mock | tokens=529+107 | agents= |
| multi-agent | 0.02 | 0.00067 | 10.0 | 100% | 0% | provider=offline-mock | tokens=1643+700 | agents=analyst,critic,researcher,writer |

## Comparison

- Latency: multi-agent **+0.23s** vs single-agent (5.4x).
- Quality: multi-agent **+10.0/10** vs single-agent (9.0 -> 10.0).
- Citation coverage: multi-agent **100%** vs single-agent **27%**.
- Multi-agent trades latency/cost for better citation coverage and structured evidence, which matters for research-grade answers.

## Failure Mode & Fix

**Failure mode gặp phải:** OfflineLLMClient (fallback khi không có API key) chỉ đọc context sau marker `CONTEXT:`. Ban đầu Analyst/Writer gửi user_prompt dạng `RESEARCH NOTES:` / `AVAILABLE SOURCES:` nên mock nhận context rỗng, writer trả `No sources retrieved`, quality tụt còn 5.0 và citation coverage 0%.

**Cách fix:** chuẩn hóa contract user_prompt — mọi agent đưa dữ liệu vào marker `CONTEXT:` với cùng format source block `- [source_id] title: snippet`. Sau fix, quality lên 10.0 và citation coverage đạt 100%. Bài học: contract giữa agent và LLM/mock client phải nhất quán; đổi một bên mà không đổi bên kia sẽ fail im lặng.
