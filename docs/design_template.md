# Design Template

## Problem

Xây dựng research assistant nhận câu hỏi nghiên cứu dài, tìm bằng chứng, phân tích và viết câu trả lời cuối cùng có trích dẫn nguồn. Hệ thống phải benchmark được single-agent vs multi-agent theo latency, cost, quality, citation coverage, failure rate.

## Why multi-agent?

Single-agent làm toàn bộ chỉ gọi LLM một lần: nhanh, rẻ nhưng dễ "một mình tự biên tự diễn" — không tách rõ nhiệm vụ tìm kiếm, phân tích và viết, dẫn tới trích dẫn lỏng lẻo hoặc bịa nguồn. Tách thành Supervisor + Researcher + Analyst + Writer + Critic giúp mỗi agent giữ một responsibility, shared state ghi rõ từng bước, và Critic kiểm tra citation coverage độc lập — phù hợp cho câu trả lời dạng nghiên cứu cần truy vết.

## Agent roles

| Agent | Responsibility | Input | Output | Failure mode |
|---|---|---|---|---|
| Supervisor | Quyết định worker tiếp theo, dừng khi đủ | ResearchState (route_history, iteration) | next route (researcher/analyst/writer/critic/done) | Vòng lặp vô hạn → chặn bằng max_iterations |
| Researcher | Tìm nguồn + viết research notes có citation | query, SearchClient | sources + research_notes | Search/LLM fail → fallback notes + ghi errors |
| Analyst | Trích claims, so sánh viewpoint, gắn cờ weak evidence | research_notes | analysis_notes | Thiếu nguồn → analysis "insufficient evidence" |
| Writer | Viết final answer có citation [source_id] | analysis_notes + sources | final_answer | Không có nguồn → báo rõ, không bịa citation |
| Critic | Đếm citation coverage, phát hiện hallucinated reference | final_answer + sources | findings (coverage, hallucinated list) | Không có final_answer → coverage 0% |

## Shared state

`ResearchState` truyền xuyên graph: `request`, `iteration`, `route_history`, `sources`, `research_notes`, `analysis_notes`, `final_answer`, `agent_results`, `trace`, `errors`, `total_input_tokens`, `total_output_tokens`, `total_cost_usd`, `llm_provider`. Cần `trace` để giải thích ai làm gì, `errors` để đo failure rate, token/cost để benchmark. Mỗi agent chỉ ghi đúng field của mình → handoff không mất context.

## Routing policy

```
supervisor
  ├─ research_notes is None ──────> researcher
  ├─ analysis_notes is None ──────> analyst
  ├─ final_answer is None ────────> writer
  ├─ critic chưa chạy ────────────> critic
  └─ else / iteration >= max ─────> done (END)
```

Mỗi worker chạy xong quay lại supervisor. Supervisor dùng deterministic rule (không cần LLM) nên luôn dừng được.

## Guardrails

- Max iterations: `MAX_ITERATIONS` (mặc định 6) — supervisor hard stop.
- Timeout: `TIMEOUT_SECONDS` truyền vào OpenAI client; retry `tenacity` (3 lần, exponential backoff).
- Retry: OpenAI client tự retry transient lỗi.
- Fallback: mỗi agent bọc try/except → ghi `errors`, đặt giá trị fallback để graph không crash.
- Validation: ResearchQuery dùng Pydantic (min_length, bounds); ResearchState validate khi khởi tạo.

## Benchmark plan

Query: 3 câu lấy từ `configs/lab_default.yaml` (so sánh kiến trúc, guardrails, role specialization — khớp corpus offline).

| Metric | Cách đo |
|---|---|
| Latency | wall-clock quanh runner |
| Cost | tích lũy từ LLM response (model pricing map) |
| Quality | heuristic 0-10: có answer, độ sâu, có sources, có citation |
| Citation coverage | số [source_id] hợp lệ trong final_answer / tổng sources |
| Failure rate | 1 nếu có lỗi trong state.errors, ngược lại 0 |

Expected: multi-agent quality +1, citation coverage 100% vs baseline ~27%, nhưng latency ~5.8x và cost cao hơn do nhiều LLM call.

## Failure mode & cách fix

**Failure mode (đã gặp):** OfflineLLMClient (fallback khi không có API key) chỉ đọc context sau marker `CONTEXT:`. Lúc đầu Analyst/Writer gửi user_prompt dạng `RESEARCH NOTES:` / `AVAILABLE SOURCES:` nên mock nhận context rỗng → writer trả "No sources retrieved" → quality 5.0, citation coverage 0%.

**Cách fix:** chuẩn hóa contract user_prompt — mọi agent đưa dữ liệu vào marker `CONTEXT:` và dùng cùng format source block `- [source_id] title: snippet`. Sau đó quality 10.0, citation coverage 100%. Bài học: contract giữa agent và LLM/mock client phải nhất quán; đổi một bên mà không đổi bên kia sẽ fail im lặng.

## Exit ticket

**1. Case nào nên dùng multi-agent? Vì sao?**

Nên dùng khi nhiệm vụ có nhiều giai đoạn cần chuyên môn khác nhau và cần truy vết bằng chứng — ví dụ research-report: tìm nguồn (Researcher) → phân tích claims (Analyst) → viết (Writer) → kiểm tra citation (Critic). Vì tách role giúp mỗi agent làm đúng một việc, shared state ghi rõ bước nào xảy ra, và Critic độc lập phát hiện bịa nguồn — kết quả benchmark cho thấy citation coverage 100% so với 27% của single-agent. Phù hợp khi sai sót tốn kém (báo cáo, tài chính, pháp lý).

**2. Case nào không nên dùng multi-agent? Vì sao?**

Không nên dùng cho tác vụ đơn giản, trả lời nhanh, một bước — ví dụ chat Q&A ngắn, paraphrase, classify. Vì multi-agent thêm latency (~6x) và cost (nhiều LLM call) mà không tăng chất lượng đáng kể; benchmark cho thấy single-agent đã đạt quality 9/10 và nhanh hơn hẳn. Quy tắc: chỉ thêm agent khi có lý do rõ ràng về role/responsibility, đúng như lab guide "không thêm agent nếu không có lý do".
